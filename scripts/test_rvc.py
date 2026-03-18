"""Test RVC engine end-to-end with all 3 models."""
import time
import numpy as np
import torch
from pathlib import Path

# Ensure we can import from the project
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine.rvc_engine import RVCEngine

SR = 48000
CHUNK = 2048
DURATION_S = 1.0  # test with 1 second of audio for F0 extraction

models_dir = Path("models")
models = [
    ("da7ee7.pth", None),
    ("Elissa.pth", None),
    ("fairuz.pth", None),
]

# Generate test signal (speech-like: 200 Hz sine with amplitude modulation)
n_samples = int(DURATION_S * SR)
t = np.arange(n_samples) / SR
test_audio = (0.3 * np.sin(2 * np.pi * 200 * t) *
              (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)

print(f"Test audio: {n_samples} samples at {SR} Hz ({DURATION_S}s)")
print(f"Device: {'CUDA ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print()

engine = RVCEngine(use_gpu=True)

for model_name, idx_path in models:
    model_path = models_dir / model_name
    if not model_path.exists():
        print(f"  SKIP {model_name} — file not found")
        continue

    print(f"{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    # Load model
    t0 = time.perf_counter()
    ok = engine.load_model(model_path, idx_path)
    load_time = time.perf_counter() - t0

    if not ok:
        print(f"  FAILED to load {model_name}")
        continue
    print(f"  Loaded in {load_time:.1f}s")

    # Convert
    t0 = time.perf_counter()
    output = engine.convert(test_audio, SR)
    conv_time = time.perf_counter() - t0

    print(f"  Convert time: {conv_time*1000:.0f} ms")
    print(f"  Input  shape: {test_audio.shape}  RMS: {np.sqrt(np.mean(test_audio**2)):.4f}")
    print(f"  Output shape: {output.shape}  RMS: {np.sqrt(np.mean(output**2)):.4f}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")

    # Check output is different from input (actual conversion happened)
    correlation = np.corrcoef(test_audio.flatten()[:len(output.flatten())],
                              output.flatten()[:len(test_audio.flatten())])[0, 1]
    print(f"  Input-output correlation: {correlation:.3f}")
    if abs(correlation) < 0.99:
        print(f"  ✓ Voice conversion IS happening (correlation < 0.99)")
    else:
        print(f"  ✗ Output too similar to input — conversion may not be working")

    engine.unload_model()
    print()

print("Done!")
