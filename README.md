# 🎙️ AI Voice Changer Platform

> Real-time AI-powered voice conversion for Windows.  
> Speak in any voice on WhatsApp Web, Zoom, Discord, and more.

---

## ✨ Features

- **Real-time processing** — target latency < 50 ms
- **RVC v2 support** — load custom `.pth` voice models
- **GPU acceleration** — CUDA-powered inference with automatic CPU fallback
- **Built-in effects** — pitch shift, formant shift, reverb, noise gate, compressor, gain
- **Voice profiles** — one-click presets (Deep Male, Female, Robot, AI Clone)
- **Custom profiles** — create, import, and export your own presets
- **VB-Cable integration** — route output to any app that accepts a microphone
- **Dark-themed GUI** — clean Tkinter interface with live volume meters

---

## 🖥️ System Requirements

### Hardware
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| RAM | 8 GB | 16 GB |
| GPU | — (CPU mode) | NVIDIA GTX 1060+ (CUDA) |
| Microphone | Any USB/3.5mm mic | Low-latency USB mic |

### Software
| Dependency | Version | Notes |
|------------|---------|-------|
| Windows | 10 / 11 | 64-bit |
| Python | 3.10 – 3.12 | https://www.python.org/ |
| CUDA Toolkit | 11.8 / 12.x | Optional, for GPU acceleration |
| VB-Audio Virtual Cable | Latest | Required for routing to apps |

---

## 🚀 Installation

### Step 1 — Install VB-Audio Virtual Cable

Download and install from: https://vb-audio.com/Cable/

After installation you will see two new audio devices:
- **CABLE Input (VB-Audio Virtual Cable)** — write processed audio here
- **CABLE Output (VB-Audio Virtual Cable)** — apps read your "microphone" from here

### Step 2 — Install Python dependencies

```bat
scripts\install_dependencies.bat
```

Or manually:

```bat
pip install -r requirements.txt
```

GPU users — install PyTorch with CUDA support first:

```bat
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 3 — Launch the application

```bat
scripts\run.bat
```

Or directly:

```bat
python -m src.main
```

---

## 🎮 How to Use

### Basic Usage

1. **Select Input** — choose your physical microphone
2. **Select Output** — choose `CABLE Input (VB-Audio Virtual Cable)`
3. **Choose a Profile** — click a voice card (Deep Male, Female, Robot, or AI Clone)
4. **Adjust sliders** — fine-tune pitch, formant, reverb, noise gate, and gain
5. **Click START** — voice conversion begins immediately

### Using with WhatsApp Web / Zoom / Discord

In each application, navigate to audio/microphone settings and select:

**Microphone: CABLE Output (VB-Audio Virtual Cable)**

| App | Path to setting |
|-----|----------------|
| WhatsApp Web | Browser microphone permission → select CABLE Output |
| Zoom | Settings → Audio → Microphone → CABLE Output |
| Discord | User Settings → Voice & Video → Input Device → CABLE Output |
| Google Meet | Settings → Audio → Microphone → CABLE Output |
| Microsoft Teams | Settings → Devices → Microphone → CABLE Output |

---

## 🤖 Adding Custom AI Voice Models (RVC)

1. Download an RVC v2 `.pth` model file
2. Copy it to the `models/` directory (add matching `.index` file for better quality)
3. Edit `config/profiles/custom_ai.json`:

```json
{
    "use_ai": true,
    "ai_model_path": "models/your_model.pth"
}
```

4. Click the **AI Voice Clone** card in the app

**Model sources:**
- https://www.weights.gg — community voice models
- https://huggingface.co/lj1995/VoiceConversionWebUI — official RVC weights
- Train your own with RVC-WebUI

---

## 🎨 Creating Custom Profiles

Create a JSON file in `config/profiles/`:

```json
{
    "name": "My Custom Voice",
    "description": "A unique voice effect",
    "icon": "🎵",
    "pitch_shift": -2.0,
    "formant_shift": 1.0,
    "reverb_level": 0.15,
    "noise_gate_threshold": -40,
    "gain": 1.1,
    "use_ai": false
}
```

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `pitch_shift` | float | -12 to +12 | Pitch in semitones |
| `formant_shift` | float | -6 to +6 | Formant/timbre change |
| `reverb_level` | float | 0.0 to 1.0 | Reverb wet mix |
| `noise_gate_threshold` | float | -80 to 0 | Noise cutoff (dBFS) |
| `gain` | float | 0.0 to 4.0 | Volume multiplier |
| `use_ai` | bool | — | Enable RVC AI conversion |
| `ai_model_path` | string | — | Path to .pth file |
| `ai_f0_method` | string | rmvpe/harvest/crepe | F0 extraction method |

---

## 🏗️ Architecture

```
Physical Mic
     |
     v
AudioCapture (sounddevice InputStream)
     |   callback(chunk: np.ndarray)
     v
AudioStream (pipeline thread)
     |
     |-- RVCEngine.convert()      <- AI voice conversion (optional)
     |      |-- HuBERT feature extraction
     |      |-- F0 extraction (rmvpe / harvest)
     |      +-- SynthesizerTrnMs768NSFsid (PyTorch)
     |
     +-- AudioPipeline.process()  <- Effects chain
            |-- NoiseGate
            |-- PitchShifter
            |-- FormantShifter
            |-- ReverbEffect
            +-- VolumeControl
     |
     v
AudioOutput (sounddevice OutputStream)
     |
     v
CABLE Input (VB-Audio Virtual Cable)
     |
     v
CABLE Output -> WhatsApp / Zoom / Discord
```

### Module Map

```
src/
├── main.py                 Entry point
├── app.py                  Application controller
├── audio/
│   ├── devices.py          Device enumeration & VB-Cable detection
│   ├── capture.py          Mic capture (non-blocking callback)
│   ├── output.py           Virtual cable output (queue-based)
│   └── stream.py           Capture -> pipeline -> output
├── engine/
│   ├── effects.py          IAudioEffect + all effect classes
│   ├── pipeline.py         Chainable effect pipeline
│   ├── rvc_engine.py       RVC v2 inference wrapper
│   └── model_manager.py    .pth file discovery & download
├── profiles/
│   ├── profile.py          VoiceProfile dataclass
│   └── profile_manager.py  Load / save / switch / import / export
├── ui/
│   ├── main_window.py      Tkinter root window
│   ├── device_selector.py  Input/Output dropdowns
│   ├── profile_selector.py Profile card grid
│   ├── effect_controls.py  Parameter sliders
│   └── status_bar.py       Latency + volume meters
└── utils/
    ├── constants.py        App-wide constants
    └── logger.py           Logging (file + console)
```

---

## 🧪 Running Tests

```bat
python -m pytest tests/ -v
```

---

## 🔧 Configuration (config/settings.json)

```json
{
    "audio": {
        "sample_rate": 44100,
        "channels": 1,
        "chunk_size": 1024,
        "buffer_size": 4096
    },
    "processing": {
        "use_gpu": true,
        "max_latency_ms": 50
    }
}
```

Latency tips: reduce `chunk_size` to 512 or 256 for lower latency.

---

## 📋 Troubleshooting

| Problem | Solution |
|---------|----------|
| No audio devices listed | Install drivers; run as Administrator |
| VB-Cable not detected | Install from vb-audio.com; restart |
| High latency | Reduce chunk_size in settings.json |
| CUDA out of memory | Set use_gpu: false in settings.json |
| AI model not loading | Check path in profile JSON |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Follow PEP 8; add type hints and docstrings to all functions
4. Add tests in `tests/`
5. Open a pull request

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- RVC Project — voice conversion framework
- VB-Audio — Virtual Cable driver
- sounddevice — cross-platform audio I/O
- PyTorch — deep learning framework
- fairseq — HuBERT content encoder
