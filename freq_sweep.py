"""
freq_sweep.py — Frequency sweep inference plot
Project: NN_RF
Scalars: PTM, IFV (instantaneous frequency variance), SK (spectral kurtosis)
Top panel:    Mean PSD (dB) with class prediction dots overlaid
Bottom panel: Stacked softmax probability area chart
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import kurtosis as scipy_kurtosis

os.environ['SOAPY_SDR_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/SoapySDR/modules0.8'
import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32

# ── CONFIG ────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "models/rf_cnn_best.pt"
FRAME_SIZE      = 1024
SAMPLE_RATE     = 2.048e6
DC_OFFSET_HZ    = 100e3
GAIN_DB         = 36.4
FRAMES_PER_FREQ = 10
DWELL_MS        = 50
FREQ_START_MHZ  = 70.0
FREQ_STOP_MHZ   = 1500.0
FREQ_STEP_MHZ   = 1.0

CLASS_NAMES  = ["fm_broadcast", "ads_b", "noaa_wx", "noise_floor", "unknown"]
CLASS_COLORS = ["#2196F3", "#F44336", "#4CAF50", "#9E9E9E", "#FF9800"]
NUM_CLASSES  = 5
NUM_SCALARS  = 4

FREQ_GATE = [
    (88e6,   108e6,  [0]),
    (162e6,  163e6,  [2]),
    (1089e6, 1091e6, [1]),
    (398e6,  402e6,  [3]),
]

BAND_SHADES = [
    (88,   108,  "#2196F3", "FM"),
    (162,  163,  "#4CAF50", "NOAA"),
    (1089, 1091, "#F44336", "ADS-B"),
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── SCALAR EXTRACTION ─────────────────────────────────────────────────────────
def extract_scalars(complex_iq_np: np.ndarray) -> np.ndarray:
    amp = np.abs(complex_iq_np)
    ptm = float(amp.max() / (amp.mean() + 1e-9)) / 30.0
    phase     = np.unwrap(np.angle(complex_iq_np))
    inst_freq = np.diff(phase)
    ifv_norm  = float(np.clip(np.var(inst_freq) / 10.0, 0.0, 1.0))
    spectrum  = np.abs(np.fft.fft(complex_iq_np))
    sk        = float(scipy_kurtosis(spectrum, fisher=True))
    sk_norm   = float(np.clip((sk + 3.0) / 30.0, 0.0, 1.0))

    # CNR — peak bin vs mean; no dominant carrier in noise floor (~1.0)
    peak      = float(spectrum.max())
    cnr       = peak / (float(spectrum.mean()) + 1e-9)
    cnr_norm  = float(np.clip(cnr / 100.0, 0.0, 1.0))

    return np.array([ptm, ifv_norm, sk_norm, cnr_norm], dtype=np.float32)


# ── MODEL ─────────────────────────────────────────────────────────────────────
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


# ── PREPROCESSING ─────────────────────────────────────────────────────────────
def preprocess_frame(iq_np):
    iq         = torch.tensor(iq_np, dtype=torch.float32)
    complex_iq = iq[0] + 1j * iq[1]
    spectrum   = torch.log1p(torch.abs(torch.fft.fft(complex_iq)))
    energy     = torch.log1p(torch.abs(complex_iq))
    x          = torch.stack([spectrum, energy], dim=0).unsqueeze(0).to(device)
    scalars    = extract_scalars(complex_iq.numpy())
    scalars_t  = torch.tensor(scalars, dtype=torch.float32).unsqueeze(0).to(device)
    return x, scalars_t


def frame_psd_db(iq_np):
    complex_iq = iq_np[0] + 1j * iq_np[1]
    spectrum   = np.abs(np.fft.fft(complex_iq)) ** 2
    return 10 * np.log10(np.mean(spectrum) + 1e-12)


# ── FREQ GATE ─────────────────────────────────────────────────────────────────
def apply_freq_gate(probs, freq_hz):
    for low, high, allowed in FREQ_GATE:
        if low <= freq_hz <= high:
            mask = torch.zeros_like(probs)
            for cls in allowed:
                mask[cls] = 1.0
            gated = probs * mask
            total = gated.sum()
            return gated / total if total > 0 else gated
    mask = torch.ones_like(probs)
    mask[0] = 0.0
    mask[1] = 0.0
    mask[2] = 0.0
    gated = probs * mask
    total = gated.sum()
    return gated / total if total > 0 else gated


# ── SDR ───────────────────────────────────────────────────────────────────────
def init_sdr(freq_hz, gain_db):
    devs = SoapySDR.Device.enumerate({"driver": "rtlsdr"})
    if not devs:
        print("ERROR: No RTL-SDR found.")
        sys.exit(1)
    sdr = SoapySDR.Device(devs[0])
    sdr.setSampleRate(SOAPY_SDR_RX, 0, SAMPLE_RATE)
    sdr.setFrequency(SOAPY_SDR_RX, 0, freq_hz + DC_OFFSET_HZ)
    sdr.setGainMode(SOAPY_SDR_RX, 0, False)
    sdr.setGain(SOAPY_SDR_RX, 0, gain_db)
    stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    sdr.activateStream(stream)
    return sdr, stream


def capture_frame(sdr, stream):
    buf = np.zeros(FRAME_SIZE, dtype=np.complex64)
    sr  = sdr.readStream(stream, [buf], FRAME_SIZE)
    if sr.ret != FRAME_SIZE:
        return None
    return np.stack([buf.real, buf.imag], axis=0)


# ── SWEEP ─────────────────────────────────────────────────────────────────────
def run_sweep():
    print(f"Loading model from {CHECKPOINT_PATH}...")
    model = RFClassifier(NUM_CLASSES, NUM_SCALARS).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    softmax = nn.Softmax(dim=1)

    freqs_mhz = np.arange(FREQ_START_MHZ, FREQ_STOP_MHZ + FREQ_STEP_MHZ, FREQ_STEP_MHZ)
    n_freqs   = len(freqs_mhz)
    est_secs  = n_freqs * (FRAMES_PER_FREQ * FRAME_SIZE / SAMPLE_RATE + DWELL_MS / 1000)
    print(f"Sweep: {FREQ_START_MHZ}–{FREQ_STOP_MHZ} MHz | Step: {FREQ_STEP_MHZ} MHz | "
          f"Steps: {n_freqs} | Est: {est_secs:.0f}s\n")

    sdr, stream = init_sdr(freqs_mhz[0] * 1e6, GAIN_DB)
    results = []

    try:
        for freq_mhz in freqs_mhz:
            freq_hz = freq_mhz * 1e6
            sdr.setFrequency(SOAPY_SDR_RX, 0, freq_hz + DC_OFFSET_HZ)
            time.sleep(DWELL_MS / 1000)

            all_probs = []
            psd_vals  = []

            for _ in range(FRAMES_PER_FREQ):
                iq = capture_frame(sdr, stream)
                if iq is None:
                    continue
                x, scalars = preprocess_frame(iq)
                with torch.no_grad():
                    probs = softmax(model(x, scalars))[0]
                all_probs.append(probs.cpu().numpy())
                psd_vals.append(frame_psd_db(iq))

            if not all_probs:
                continue

            mean_probs_raw = np.mean(all_probs, axis=0)
            mean_tensor    = torch.tensor(mean_probs_raw, dtype=torch.float32)
            mean_gated     = apply_freq_gate(mean_tensor, freq_hz).numpy()
            pred_idx       = mean_gated.argmax()
            confidence     = mean_gated[pred_idx]
            psd_db         = float(np.mean(psd_vals))

            results.append((freq_mhz, pred_idx, confidence, mean_gated, psd_db))
            print(f"  {freq_mhz:7.1f} MHz → {CLASS_NAMES[pred_idx]:<14} "
                  f"({confidence:.1%})  PSD: {psd_db:.1f} dB")

    except KeyboardInterrupt:
        print("\nSweep interrupted — plotting partial results...")
    finally:
        sdr.deactivateStream(stream)
        sdr.closeStream(stream)

    plot_sweep(results)


# ── PLOT ──────────────────────────────────────────────────────────────────────
def plot_sweep(results):
    freqs      = np.array([r[0] for r in results])
    pred_class = np.array([r[1] for r in results])
    probs_all  = np.array([r[3] for r in results])
    psd_db     = np.array([r[4] for r in results])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8), sharex=True)

    ax1.plot(freqs, psd_db, color='#1a1a2e', linewidth=0.8, alpha=0.85,
             label='Mean PSD (dB)', zorder=2)

    psd_min = psd_db.min() - 2
    psd_max = psd_db.max() + 4
    for low, high, color, label in BAND_SHADES:
        ax1.axvspan(low, high, alpha=0.18, color=color, zorder=1)
        ax1.text((low + high) / 2, psd_max - 1, label,
                 ha='center', fontsize=7, color='black', zorder=4)

    dot_y = psd_db + (psd_max - psd_min) * 0.04
    for cls_idx in range(NUM_CLASSES):
        mask = pred_class == cls_idx
        if mask.any():
            ax1.scatter(freqs[mask], dot_y[mask],
                        c=CLASS_COLORS[cls_idx], s=25, zorder=5,
                        label=CLASS_NAMES[cls_idx], edgecolors='none', alpha=0.85)

    ax1.set_ylabel("Mean Power (dB)")
    ax1.set_title(f"NN_RF Frequency Sweep  |  {freqs[0]:.0f}–{freqs[-1]:.0f} MHz  |  "
                  f"Step={FREQ_STEP_MHZ} MHz  |  Gain={GAIN_DB} dB")
    ax1.set_ylim(psd_min, psd_max + 2)
    ax1.legend(loc='upper right', fontsize=8, ncol=3)
    ax1.grid(True, alpha=0.3)

    ax2.stackplot(freqs,
                  probs_all[:, 0], probs_all[:, 1],
                  probs_all[:, 2], probs_all[:, 3], probs_all[:, 4],
                  labels=CLASS_NAMES, colors=CLASS_COLORS, alpha=0.75)
    ax2.set_xlabel("Frequency (MHz)")
    ax2.set_ylabel("Class Probability")
    ax2.set_title("Softmax Probabilities Across Sweep")
    ax2.legend(loc='upper right', fontsize=8, ncol=3)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    out = "results/freq_sweep.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nSaved: {out}")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",  type=float, default=FREQ_START_MHZ)
    parser.add_argument("--stop",   type=float, default=FREQ_STOP_MHZ)
    parser.add_argument("--step",   type=float, default=FREQ_STEP_MHZ)
    parser.add_argument("--gain",   type=float, default=GAIN_DB)
    parser.add_argument("--frames", type=int,   default=FRAMES_PER_FREQ)
    args = parser.parse_args()

    FREQ_START_MHZ  = args.start
    FREQ_STOP_MHZ   = args.stop
    FREQ_STEP_MHZ   = args.step
    GAIN_DB         = args.gain
    FRAMES_PER_FREQ = args.frames

    run_sweep()