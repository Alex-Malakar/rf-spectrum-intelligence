"""
inference.py — Phase 3: Real-Time RF Signal Classifier + Adaptive Gain Control
Project: NN_RF
Scalars: PTM, IFV (instantaneous frequency variance), SK (spectral kurtosis)
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from scipy.stats import kurtosis as scipy_kurtosis

os.environ['SOAPY_SDR_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/SoapySDR/modules0.8'
import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
CHECKPOINT_PATH = "models/rf_cnn_best.pt"
FRAME_SIZE      = 1024
SAMPLE_RATE     = 2.048e6
DC_OFFSET_HZ    = 100e3
DEFAULT_FREQ_HZ = 99.3e6

GAIN_LEVELS = [14.4, 25.4, 36.4, 48.0, 49.6]

AGC_HIGH_CONFIDENCE = 0.90
AGC_LOW_CONFIDENCE  = 0.60
AGC_CLIP_ENERGY     = 0.85

VOTE_WINDOW  = 10
CLASS_NAMES  = ["fm_broadcast", "ads_b", "noaa_wx", "noise_floor", "unknown"]
NUM_CLASSES  = 5
NUM_SCALARS  = 4

FREQ_GATE = [
    (88e6,   108e6,  [0]),
    (162e6,  163e6,  [2]),
    (1089e6, 1091e6, [1]),
    (398e6,  402e6,  [3]),
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────
# SCALAR EXTRACTION
# ─────────────────────────────────────────
def extract_scalars(complex_iq_np: np.ndarray) -> np.ndarray:
    amp = np.abs(complex_iq_np)

    # PTM
    ptm = float(amp.max() / (amp.mean() + 1e-9)) / 30.0

    # IFV — instantaneous frequency variance
    phase     = np.unwrap(np.angle(complex_iq_np))
    inst_freq = np.diff(phase)
    ifv_norm  = float(np.clip(np.var(inst_freq) / 10.0, 0.0, 1.0))

    # SK — spectral kurtosis (Fisher, Gaussian=0)
    spectrum = np.abs(np.fft.fft(complex_iq_np))
    sk       = float(scipy_kurtosis(spectrum, fisher=True))
    sk_norm  = float(np.clip((sk + 3.0) / 30.0, 0.0, 1.0))

    # CNR — peak bin vs mean; no dominant carrier in noise floor (~1.0)
    peak      = float(spectrum.max())
    cnr       = peak / (float(spectrum.mean()) + 1e-9)
    cnr_norm  = float(np.clip(cnr / 100.0, 0.0, 1.0))

    return np.array([ptm, ifv_norm, sk_norm, cnr_norm], dtype=np.float32)


# ─────────────────────────────────────────
# FREQ GATE
# ─────────────────────────────────────────
def apply_freq_gate(probs: torch.Tensor, freq_hz: float) -> torch.Tensor:
    for low, high, allowed in FREQ_GATE:
        if low <= freq_hz <= high:
            mask = torch.zeros_like(probs)
            for cls in allowed:
                mask[cls] = 1.0
            gated = probs * mask
            total = gated.sum()
            return gated / total if total > 0 else gated
    # Outside known bands — block known signal classes, let noise_floor/unknown compete
    mask = torch.ones_like(probs)
    mask[0] = 0.0
    mask[1] = 0.0
    mask[2] = 0.0
    gated = probs * mask
    total = gated.sum()
    return gated / total if total > 0 else gated


# ─────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel//2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

    def forward(self, x):
        return self.block(x)


class RFClassifier(nn.Module):
    def __init__(self, num_classes=5, num_scalars=3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(2, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        fc_in = 256 + num_scalars
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_in, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, scalars):
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = torch.cat([x, scalars], dim=1)
        return self.classifier(x)


# ─────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────
def preprocess_frame(iq_np: np.ndarray):
    iq         = torch.tensor(iq_np, dtype=torch.float32)
    complex_iq = iq[0] + 1j * iq[1]
    spectrum   = torch.log1p(torch.abs(torch.fft.fft(complex_iq)))
    energy     = torch.log1p(torch.abs(complex_iq))
    x          = torch.stack([spectrum, energy], dim=0).unsqueeze(0).to(device)

    scalars    = extract_scalars(complex_iq.numpy())
    scalars_t  = torch.tensor(scalars, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 3)

    return x, scalars_t


# ─────────────────────────────────────────
# SDR
# ─────────────────────────────────────────
def init_sdr(freq_hz: float, gain_db: float):
    devs = SoapySDR.Device.enumerate({"driver": "rtlsdr"})
    if not devs:
        print("ERROR: No RTL-SDR device found.")
        sys.exit(1)
    sdr = SoapySDR.Device(devs[0])
    sdr.setSampleRate(SOAPY_SDR_RX, 0, SAMPLE_RATE)
    sdr.setFrequency(SOAPY_SDR_RX, 0, freq_hz + DC_OFFSET_HZ)
    sdr.setGainMode(SOAPY_SDR_RX, 0, False)
    sdr.setGain(SOAPY_SDR_RX, 0, gain_db)
    stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    sdr.activateStream(stream)
    print(f"SDR: {freq_hz/1e6:.3f} MHz | Gain: {gain_db}dB | SR: {SAMPLE_RATE/1e6:.3f} MSPS")
    return sdr, stream


def set_gain(sdr, gain_db: float):
    sdr.setGain(SOAPY_SDR_RX, 0, gain_db)


def capture_frame(sdr, stream):
    buf = np.zeros(FRAME_SIZE, dtype=np.complex64)
    sr  = sdr.readStream(stream, [buf], FRAME_SIZE)
    if sr.ret != FRAME_SIZE:
        return None
    return np.stack([buf.real, buf.imag], axis=0)


# ─────────────────────────────────────────
# ADAPTIVE GAIN CONTROL
# ─────────────────────────────────────────
class AdaptiveGainController:
    def __init__(self, initial_gain_idx: int = 1):
        self.gain_idx     = initial_gain_idx
        self.current_gain = GAIN_LEVELS[self.gain_idx]
        self.hold_counter = 0
        self.HOLD_FRAMES  = 5

    def step(self, confidence: float, smoothed_idx: int,
             frame_energy_norm: float, sdr) -> tuple:
        if self.hold_counter > 0:
            self.hold_counter -= 1
            return self.current_gain, "HOLD"

        action = "HOLD"

        if frame_energy_norm > AGC_CLIP_ENERGY and self.gain_idx > 0:
            self.gain_idx -= 1
            action = "DOWN (clip)"
        elif confidence < AGC_LOW_CONFIDENCE and self.gain_idx < len(GAIN_LEVELS) - 1:
            self.gain_idx += 1
            action = "UP (low conf)"
        elif smoothed_idx == 3 and confidence > AGC_HIGH_CONFIDENCE and self.gain_idx > 0:
            self.gain_idx -= 1
            action = "DOWN (noise floor)"

        if action != "HOLD":
            self.current_gain = GAIN_LEVELS[self.gain_idx]
            set_gain(sdr, self.current_gain)
            self.hold_counter = self.HOLD_FRAMES

        return self.current_gain, action


# ─────────────────────────────────────────
# INFERENCE LOOP
# ─────────────────────────────────────────
def run_inference(freq_hz: float, initial_gain_idx: int = 1):
    print(f"\nLoading model from {CHECKPOINT_PATH}...")
    model = RFClassifier(NUM_CLASSES, NUM_SCALARS).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    print("Model loaded.")

    agc         = AdaptiveGainController(initial_gain_idx)
    sdr, stream = init_sdr(freq_hz, agc.current_gain)
    softmax     = nn.Softmax(dim=1)

    for low, high, allowed in FREQ_GATE:
        if low <= freq_hz <= high:
            print(f"Freq gate active: only {[CLASS_NAMES[i] for i in allowed]} "
                  f"allowed at {freq_hz/1e6:.3f} MHz")
            break
    else:
        print("Freq gate: noise_floor + unknown compete (unknown band)")

    print("\n── Real-Time Inference Running (Ctrl+C to stop) ──\n")
    print(f"{'Frame':>6} | {'Class':<14} | {'Conf':>6} | {'Gain':>6} | "
          f"{'AGC Action':<20} | PTM   IFV   SK")
    print("─" * 100)

    frame_count = 0
    vote_buffer = deque(maxlen=VOTE_WINDOW)

    try:
        while True:
            iq = capture_frame(sdr, stream)
            if iq is None:
                continue

            tensor, scalars = preprocess_frame(iq)

            with torch.no_grad():
                logits = model(tensor, scalars)
                probs  = softmax(logits)[0]

            probs      = apply_freq_gate(probs, freq_hz)
            pred_idx   = probs.argmax().item()
            confidence = probs[pred_idx].item()
            pred_class = CLASS_NAMES[pred_idx]

            vote_buffer.append(pred_idx)
            smoothed_idx   = max(set(vote_buffer), key=vote_buffer.count)
            smoothed_class = CLASS_NAMES[smoothed_idx]

            complex_iq        = iq[0] + 1j * iq[1]
            frame_energy_norm = min(float(np.mean(np.abs(complex_iq))) / 0.5, 1.0)

            current_gain, agc_action = agc.step(
                confidence, smoothed_idx, frame_energy_norm, sdr
            )

            s = scalars[0].cpu().numpy()
            raw_tag = f"  [raw: {pred_class}]" if smoothed_class != pred_class else ""
            print(f"{frame_count:>6} | {smoothed_class:<14} | {confidence:>5.1%} | "
                  f"{current_gain:>5.1f}dB | {agc_action:<20} | "
                  f"{s[0]:.2f}  {s[1]:.2f}  {s[2]:.2f}{raw_tag}")

            frame_count += 1
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    finally:
        sdr.deactivateStream(stream)
        sdr.closeStream(stream)
        print(f"Stream closed. Total frames: {frame_count}")


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NN_RF Real-Time Inference")
    parser.add_argument("--freq",     type=float, default=DEFAULT_FREQ_HZ / 1e6)
    parser.add_argument("--gain-idx", type=int,   default=1, choices=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    print(f"\nNN_RF Inference Pipeline")
    print(f"Device:    {device}")
    print(f"Frequency: {args.freq} MHz")
    print(f"Init gain: {GAIN_LEVELS[args.gain_idx]} dB")

    run_inference(freq_hz=args.freq * 1e6, initial_gain_idx=args.gain_idx)