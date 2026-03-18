"""
Audio pipeline diagnostic — processes synthetic speech through the full
effects chain chunk-by-chunk (exactly like the real-time stream), then
analyses the output for crackling, discontinuities, and artifacts.

Run:  python scripts/diagnose_audio.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.engine.pipeline import AudioPipeline
from src.engine.effects import (
    NoiseGate, PitchShifter, FormantShifter, ReverbEffect, VoiceDisguise
)

# ── Settings (match real app) ─────────────────────────────────────────────────
SR = 48000
CHUNK = 2048
DURATION = 3.0          # seconds of test signal
INPUT_GAIN = 5.0

# deep_male profile
PITCH   = -8.0
FORMANT = -5.0
REVERB  = 0.2
GATE_DB = -50.0
GAIN    = 1.3
DISGUISE = 0.6

# ── Generate speech-like test signal ──────────────────────────────────────────
n_total = int(SR * DURATION)
t = np.arange(n_total, dtype=np.float32) / SR

# Simulate voiced speech: fundamental + harmonics + amplitude envelope
f0 = 150  # male fundamental
signal = np.zeros(n_total, dtype=np.float32)
for h in range(1, 8):
    signal += (0.5 ** h) * np.sin(2 * np.pi * f0 * h * t + np.random.rand() * 2 * np.pi)

# Amplitude modulation (~4 Hz syllable rate)
envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)
signal *= envelope
# Normalize to typical mic level
signal *= 0.02  # RMS ~0.01 (quiet mic like Galaxy Buds)

print(f"Test signal: {DURATION}s, SR={SR}, chunks of {CHUNK}")
print(f"Input RMS: {np.sqrt(np.mean(signal**2)):.6f}")
print()

# ── Build pipeline (same as app) ──────────────────────────────────────────────
pipeline = AudioPipeline()
pipeline.input_gain = INPUT_GAIN

if GATE_DB > -80.0:
    pipeline.add_effect(NoiseGate(threshold_db=GATE_DB))
if PITCH != 0.0:
    pipeline.add_effect(PitchShifter(semitones=PITCH))
if FORMANT != 0.0:
    pipeline.add_effect(FormantShifter(semitones=FORMANT))
if DISGUISE > 0.0:
    pipeline.add_effect(VoiceDisguise(intensity=DISGUISE))
if REVERB > 0.0:
    pipeline.add_effect(ReverbEffect(wet_level=REVERB))

print("Effects chain:", [e.name for e in pipeline.get_effects()])
print()

# ── Process chunk by chunk ────────────────────────────────────────────────────
output_chunks = []
n_chunks = 0
for start in range(0, n_total, CHUNK):
    chunk = signal[start : start + CHUNK]
    if len(chunk) < CHUNK:
        chunk = np.pad(chunk, (0, CHUNK - len(chunk)))
    out = pipeline.process(chunk, SR)
    # Flatten to 1D if needed
    if out.ndim == 2:
        out = out[:, 0]
    output_chunks.append(out)
    n_chunks += 1

output = np.concatenate(output_chunks)
print(f"Processed {n_chunks} chunks → {len(output)} samples")
print(f"Output RMS: {np.sqrt(np.mean(output**2)):.6f}")
print()

# ── ANALYSIS ──────────────────────────────────────────────────────────────────

# 1. Check for boundary clicks at chunk edges
print("=" * 60)
print("1. BOUNDARY DISCONTINUITY CHECK (at chunk edges)")
print("=" * 60)
clicks = 0
max_jump = 0.0
worst_pos = 0
for i in range(1, n_chunks):
    pos = i * CHUNK
    if pos >= len(output):
        break
    # Jump between last sample of prev chunk and first of next
    jump = abs(float(output[pos]) - float(output[pos - 1]))
    if jump > max_jump:
        max_jump = jump
        worst_pos = pos
    if jump > 0.05:  # audible click threshold
        clicks += 1

if clicks == 0:
    print(f"  ✓ No clicks detected (max jump: {max_jump:.6f})")
else:
    print(f"  ✗ {clicks} clicks at chunk boundaries!")
    print(f"    Worst jump: {max_jump:.6f} at sample {worst_pos}")
print()

# 2. Check for overall discontinuities (not just at boundaries)
print("=" * 60)
print("2. OVERALL WAVEFORM DISCONTINUITY CHECK")
print("=" * 60)
diff = np.abs(np.diff(output))
p99 = np.percentile(diff, 99)
p999 = np.percentile(diff, 99.9)
big_jumps = np.sum(diff > 0.1)
huge_jumps = np.sum(diff > 0.3)
print(f"  Sample-to-sample diff:  p99={p99:.6f}  p99.9={p999:.6f}")
print(f"  Jumps > 0.1: {big_jumps}   Jumps > 0.3: {huge_jumps}")
if huge_jumps > 0:
    positions = np.where(diff > 0.3)[0]
    # Show which chunk boundaries they fall on
    boundary_count = sum(1 for p in positions if p % CHUNK == 0 or (p+1) % CHUNK == 0)
    mid_count = len(positions) - boundary_count
    print(f"    At chunk boundaries: {boundary_count}   Mid-chunk: {mid_count}")
    # Show first 5
    for p in positions[:5]:
        print(f"    Sample {p}: {output[p]:.4f} → {output[p+1]:.4f} (Δ={diff[p]:.4f})")
print()

# 3. Check for clipping/saturation
print("=" * 60)
print("3. CLIPPING / SATURATION CHECK")
print("=" * 60)
clipped_hi = np.sum(output >= 0.99)
clipped_lo = np.sum(output <= -0.99)
peak = float(np.max(np.abs(output)))
print(f"  Peak amplitude: {peak:.6f}")
print(f"  Samples at ≥0.99: {clipped_hi}   Samples at ≤-0.99: {clipped_lo}")
clip_pct = 100.0 * (clipped_hi + clipped_lo) / len(output)
if clip_pct > 1.0:
    print(f"  ✗ {clip_pct:.1f}% clipping — this causes audible distortion!")
elif clip_pct > 0.01:
    print(f"  ⚠ {clip_pct:.3f}% clipping — minor but noticeable")
else:
    print(f"  ✓ No significant clipping")
print()

# 4. Spectral analysis — check for robotic artifacts
print("=" * 60)
print("4. SPECTRAL ARTIFACT CHECK (chunk-periodic patterns)")
print("=" * 60)
# The chunk repetition rate is SR/CHUNK Hz. If we see a peak there,
# it means there's a periodic click pattern = robotic sound.
from numpy.fft import rfft, rfftfreq
spec = np.abs(rfft(output))
freqs = rfftfreq(len(output), 1.0 / SR)
chunk_freq = SR / CHUNK
# Look for energy spike at chunk_freq and its harmonics
chunk_freq_idx = int(round(chunk_freq * len(output) / SR))
band = 3  # bins around the target frequency
local_energy = float(np.mean(spec[max(0, chunk_freq_idx - band):chunk_freq_idx + band + 1]))
# Compare to median energy in nearby region
wide_start = max(0, chunk_freq_idx - 50)
wide_end = min(len(spec), chunk_freq_idx + 50)
median_energy = float(np.median(spec[wide_start:wide_end]))
ratio = local_energy / (median_energy + 1e-10)
print(f"  Chunk repetition freq: {chunk_freq:.1f} Hz")
print(f"  Energy at {chunk_freq:.0f}Hz: {local_energy:.2f}  (median nearby: {median_energy:.2f})")
print(f"  Spike ratio: {ratio:.2f}x")
if ratio > 5.0:
    print(f"  ✗ Strong chunk-periodic artifact detected! This = robotic sound")
elif ratio > 2.0:
    print(f"  ⚠ Mild chunk-periodic artifact")
else:
    print(f"  ✓ No chunk-periodic artifact")
print()

# 5. tanh saturation analysis
print("=" * 60)
print("5. TANH SATURATION ANALYSIS (input gain)")
print("=" * 60)
# Apply the gain and check how much tanh compresses
gained = signal * INPUT_GAIN
tanh_out = np.tanh(gained)
# How much does tanh differ from linear?
linear_clipped = np.clip(gained, -1, 1)
max_diff = float(np.max(np.abs(tanh_out - gained)))
# Check how much signal is in the saturated zone (|x| > 0.8)
pct_saturated = 100.0 * np.sum(np.abs(gained) > 0.8) / len(gained)
pct_deep_sat = 100.0 * np.sum(np.abs(gained) > 2.0) / len(gained)
print(f"  After gain={INPUT_GAIN}x: peak={float(np.max(np.abs(gained))):.3f}")
print(f"  Samples with |x| > 0.8 (light saturation): {pct_saturated:.1f}%")
print(f"  Samples with |x| > 2.0 (heavy saturation): {pct_deep_sat:.1f}%")
if pct_deep_sat > 5:
    print(f"  ✗ Heavy tanh saturation adds harmonic distortion (warmth → buzz)")
elif pct_saturated > 20:
    print(f"  ⚠ Significant saturation zone — may color the sound")
else:
    print(f"  ✓ Saturation level acceptable")
print()

# 6. Per-effect artifact isolation
print("=" * 60)
print("6. PER-EFFECT ARTIFACT ISOLATION")
print("=" * 60)
effects_to_test = [
    ("PitchShifter (-8st)", PitchShifter(semitones=PITCH)),
    ("FormantShifter (-5st)", FormantShifter(semitones=FORMANT)),
    ("VoiceDisguise (0.6)", VoiceDisguise(intensity=DISGUISE)),
    ("ReverbEffect (0.2)", ReverbEffect(wet_level=REVERB)),
]
for name, effect in effects_to_test:
    # Process a clean signal through just this one effect
    test_sig = signal * INPUT_GAIN  # apply gain like pipeline does
    test_sig = np.tanh(test_sig).astype(np.float32)
    out_chunks = []
    for start in range(0, n_total, CHUNK):
        chunk = test_sig[start : start + CHUNK]
        if len(chunk) < CHUNK:
            chunk = np.pad(chunk, (0, CHUNK - len(chunk)))
        o = effect.process(chunk, SR)
        if o.ndim == 2:
            o = o[:, 0]
        out_chunks.append(o)
    eff_out = np.concatenate(out_chunks)
    
    # Boundary analysis
    eff_jumps = 0
    eff_max_jump = 0.0
    for i in range(1, len(out_chunks)):
        pos = i * CHUNK
        if pos >= len(eff_out):
            break
        j = abs(float(eff_out[pos]) - float(eff_out[pos - 1]))
        if j > eff_max_jump:
            eff_max_jump = j
        if j > 0.05:
            eff_jumps += 1
    
    # Overall diff
    d = np.abs(np.diff(eff_out))
    huge = int(np.sum(d > 0.3))
    clip = int(np.sum(np.abs(eff_out) >= 0.99))
    
    status = "✓" if eff_jumps == 0 and huge == 0 else "✗"
    print(f"  {status} {name:30s}  boundary_clicks={eff_jumps}  max_jump={eff_max_jump:.4f}"
          f"  huge_jumps={huge}  clipped={clip}")

print()
print("=" * 60)
print("DONE — review above for specific issues")
print("=" * 60)
