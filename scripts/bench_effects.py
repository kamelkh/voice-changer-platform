"""Quick per-effect benchmark."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.engine.effects import (
    PitchShifter, FormantShifter, VoiceDisguise, ReverbEffect, NoiseGate
)
from src.engine.pipeline import AudioPipeline

SR = 48000
CHUNK = 2048
ITERS = 100
mono = np.random.randn(CHUNK).astype(np.float32) * 0.01
budget_ms = CHUNK / SR * 1000.0

print(f"Chunk={CHUNK}  SR={SR}  Budget={budget_ms:.1f} ms  Iters={ITERS}")
print()

effects = [
    ("NoiseGate",      NoiseGate(threshold_db=-50.0)),
    ("PitchShifter",   PitchShifter(semitones=-8.0)),
    ("FormantShifter", FormantShifter(semitones=-5.0)),
    ("VoiceDisguise",  VoiceDisguise(intensity=0.6)),
    ("ReverbEffect",   ReverbEffect(wet_level=0.2)),
]

# Warm up
for _, eff in effects:
    for _ in range(3):
        eff.process(mono.copy(), SR)

for name, eff in effects:
    times = []
    for _ in range(ITERS):
        d = mono.copy()
        t0 = time.perf_counter()
        eff.process(d, SR)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    avg = sum(times) / len(times)
    mx = max(times)
    print(f"  {name:20s}  avg={avg:6.2f} ms   max={mx:6.2f} ms")

# Full pipeline
print()
pipeline = AudioPipeline()
pipeline.input_gain = 5.0
for _, eff in effects:
    pipeline.add_effect(eff)
for _ in range(3):
    pipeline.process(mono.copy(), SR)

times = []
for _ in range(ITERS):
    t0 = time.perf_counter()
    pipeline.process(mono.copy(), SR)
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000.0)
avg = sum(times) / len(times)
mx = max(times)
print(f"  {'FULL PIPELINE':20s}  avg={avg:6.2f} ms   max={mx:6.2f} ms")
print(f"  {'Budget':20s}  {budget_ms:6.1f} ms")
pct = avg / budget_ms * 100
print(f"  Utilization: {pct:.0f}%")
if pct > 100:
    print(f"  *** OVERRUN by {avg - budget_ms:.1f} ms — will drop chunks! ***")
