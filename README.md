# RF Signal Classifier with Adaptive Power Control

**Resume project: Power + AI for RF/Electrical Engineering**

A real-time RF signal classifier running on a HackRF One SDR, with a
closed-loop adaptive gain controller driven by CNN inference confidence.
Targets defense/telecom RF engineering roles.

---

## System architecture

```
HackRF One
    │ IQ samples (2 MSPS, complex64)
    ▼
Preprocessing
    │ reshape → (2, 1024) float32 tensor
    ▼
1D-CNN Classifier (PyTorch)
    │ softmax confidence score
    ▼
Adaptive Gain Controller
    │ low confidence → increase gain
    │ high confidence → decrease gain
    ▼
SoapySDR gain API → HackRF LNA/VGA gain registers
```

## Signal classes

| Class | Frequency | Modulation | Notes |
|-------|-----------|------------|-------|
| FM broadcast | 88–108 MHz | WBFM | Wideband stereo |
| ADS-B | 1090 MHz | OOK/PPM | Aircraft transponders |
| NOAA weather | 162.4–162.55 MHz | NFM | Continuous |
| Noise floor | 433 MHz (quiet) | — | Baseline class |

## Project phases

| Phase | Description | Duration |
|-------|-------------|----------|
| 1 | IQ data capture & dataset construction | Weeks 1–3 |
| 2 | CNN classifier training & evaluation | Weeks 4–7 |
| 3 | Adaptive gain control loop | Weeks 8–11 |
| 4 | Integration, demo, resume packaging | Weeks 12–14 |

---

## Phase 1 — Getting started

### Hardware requirements
- HackRF One (1 MHz–6 GHz, 2 MSPS minimum)
- Antenna appropriate for each signal class:
  - FM/NOAA: simple monopole or dipole (~75 cm)
  - ADS-B: vertical antenna at 1090 MHz

### Software requirements

**System packages (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3-soapysdr soapysdr-module-hackrf hackrf
```

**Python packages:**
```bash
pip install numpy scipy h5py tqdm matplotlib
```

**Verify HackRF is detected:**
```bash
hackrf_info
# Should print: Found HackRF One...
```

### Capture workflow

**Step 1 — Edit signal frequencies for your area:**
```
config/signals.py → SIGNAL_CLASSES[0]["center_freq_hz"]
```
Find a strong local FM station with:
```bash
# Scan FM band first — tune around 88–108 MHz in SDR++ or SDR#
# Pick the strongest station (RSSI > -60 dBm)
```

**Step 2 — Capture one class at a time:**
```bash
cd data/
python capture.py --class fm_broadcast
python capture.py --class noaa_wx
python capture.py --class ads_b
python capture.py --class noise_floor
```

**Or capture all classes in sequence:**
```bash
python capture.py
```

**Step 3 — Verify dataset:**
```bash
python capture.py --verify
python inspect_dataset.py --balance
python inspect_dataset.py --psd
```

**Alternative: use RadioML open dataset (no hardware needed for Phase 2):**
```bash
python download_radioml.py
```

---

## Dataset format

The HDF5 dataset (`data/rf_dataset.h5`) contains:

| Dataset | Shape | dtype | Description |
|---------|-------|-------|-------------|
| `iq_frames` | (N, 2, 1024) | float32 | I and Q channels, 1024 samples/frame |
| `labels` | (N,) | int32 | Class index (0–3) |
| `gains` | (N,) | float32 | HackRF gain at capture time (dB) |

Attributes on `iq_frames`:
- `class_names`: list of class label strings
- `sample_rate`: 2000000 (2 MSPS)
- `frame_size`: 1024
- `gain_levels`: [10, 20, 30, 40]

**Loading in PyTorch (Phase 2 preview):**
```python
import h5py, numpy as np, torch
from torch.utils.data import Dataset

class RFDataset(Dataset):
    def __init__(self, path):
        f = h5py.File(path, 'r')
        self.X = torch.tensor(f['iq_frames'][:], dtype=torch.float32)
        self.y = torch.tensor(f['labels'][:],    dtype=torch.long)
        f.close()
    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]
```

---

## Target metrics (Phase 1 exit criteria)

- [ ] ≥ 10,000 frames per class (2,500 per gain level × 4 gain levels)
- [ ] 4 gain levels captured: 10, 20, 30, 40 dB
- [ ] IQ power balance ratio within 0.8–1.2 (no severe DC offset)
- [ ] PSD plots show distinct spectral character per class
- [ ] Constellation diagram shows expected IQ distribution shape

---

## Resume bullets (fill in after Phase 2 metrics)

> "Designed and trained a 1D-CNN RF signal classifier in PyTorch achieving
> __% accuracy across 4 signal classes (FM, ADS-B, NOAA, noise) on
> captured HackRF One IQ data at SNRs from −5 to +20 dB"

> "Implemented a real-time adaptive gain control loop using CNN inference
> confidence as feedback signal, reducing average receive gain by __ dB
> while maintaining >__% classification accuracy"

---

## File structure

```
rf_classifier/
├── README.md
├── config/
│   └── signals.py          ← signal class definitions, frequencies, gain levels
├── data/
│   ├── capture.py          ← Phase 1: HackRF IQ capture script
│   ├── download_radioml.py ← Phase 1 alt: RadioML open dataset downloader
│   ├── inspect_dataset.py  ← Phase 1 QA: plots, PSD, constellation diagrams
│   ├── rf_dataset.h5       ← captured dataset (created by capture.py)
│   └── plots/              ← QA plots output directory
├── model/                  ← Phase 2 (coming)
│   ├── train.py
│   ├── evaluate.py
│   └── rf_classifier.pt
├── control/                ← Phase 3 (coming)
│   └── adaptive_gain.py
└── notebooks/
    └── results.ipynb       ← Phase 4 (coming)
```
