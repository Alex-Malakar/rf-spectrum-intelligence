#!/usr/bin/env python3
"""
Phase 1 capture script — RF Signal Classifier project
------------------------------------------------------
Captures labeled IQ frames from RTL-SDR V3 across multiple
signal classes and gain levels, saving to an HDF5 dataset.

Usage:
    python capture.py                        # capture all classes
    python capture.py --class fm_broadcast   # capture one class only
    python capture.py --class ads_b          # ADS-B with burst gate
    python capture.py --verify               # print dataset stats
    python capture.py --freq 99.3 --class fm_broadcast  # override frequency

Changes from previous version:
    - 7 gain levels (was 4)
    - Frequency jitter per gain level per class
    - NOAA second station (162.550 MHz)
    - Noise floor moved to 400 MHz
    - Unknown class added (5 classes total)
    - Metadata correctly written on dataset creation
    - Capture quality report after each class
    - Jitter logged per capture
"""

import argparse
import sys
import os
import time
import numpy as np
import h5py
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.signals import (
    SIGNAL_CLASSES, GAIN_LEVELS_DB,
    FRAMES_SINGLE_STATION, FRAME_SIZE, SAMPLE_RATE_HZ
)

DATASET_PATH    = os.path.join(os.path.dirname(__file__), "rf_dataset.h5")
BURST_THRESHOLD = 13.0  # dB peak-to-mean — ADS-B burst detection


# ─── SoapySDR wrapper ─────────────────────────────────────────────────────────

def init_rtlsdr(center_freq_hz: float, sample_rate_hz: float, gain_db: float):
    os.environ['SOAPY_SDR_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/SoapySDR/modules0.8'
    import SoapySDR
    from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32

    devs = SoapySDR.Device.enumerate({"driver": "rtlsdr"})
    if not devs:
        raise RuntimeError("No RTL-SDR found. Run: usbipd attach --wsl --busid 1-5")
    sdr = SoapySDR.Device(devs[0])
    sdr.setSampleRate(SOAPY_SDR_RX, 0, sample_rate_hz)
    sdr.setFrequency(SOAPY_SDR_RX, 0, center_freq_hz + 100e3)  # DC offset fix
    sdr.setGain(SOAPY_SDR_RX, 0, gain_db)
    rxStream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    sdr.activateStream(rxStream)
    time.sleep(0.3)
    return sdr, rxStream


def teardown_sdr(sdr, rxStream):
    sdr.deactivateStream(rxStream)
    sdr.closeStream(rxStream)


def capture_frames(sdr, rxStream, n_frames: int, frame_size: int,
                   flush: bool = True) -> np.ndarray:
    """
    Capture n_frames IQ frames.
    flush=True only on first call after SDR init — False for burst gate loop.
    """
    buf    = np.zeros(frame_size, dtype=np.complex64)
    frames = np.empty((n_frames, 2, frame_size), dtype=np.float32)

    if flush:
        flush_buf = np.zeros(frame_size, dtype=np.complex64)
        for _ in range(20):
            sdr.readStream(rxStream, [flush_buf], frame_size, timeoutUs=1_000_000)

    for i in range(n_frames):
        sr = sdr.readStream(rxStream, [buf], frame_size, timeoutUs=1_000_000)
        if sr.ret != frame_size:
            sr = sdr.readStream(rxStream, [buf], frame_size, timeoutUs=1_000_000)
        frames[i, 0, :] = buf.real
        frames[i, 1, :] = buf.imag

    return frames


# ─── Burst energy gate ────────────────────────────────────────────────────────

def has_burst(frame: np.ndarray) -> bool:
    """
    Returns True if frame contains a detectable burst.
    frame: (2, 1024) float32 [I, Q]
    """
    complex_iq = frame[0] + 1j * frame[1]
    energy     = np.abs(complex_iq)
    peak       = energy.max()
    mean       = energy.mean()
    if mean < 1e-9:
        return False
    return 20 * np.log10(peak / mean) > BURST_THRESHOLD

def has_signal(frame: np.ndarray, ptm_threshold_db: float = 5.0) -> bool:
    """
    Returns True if frame contains a real signal above noise floor.
    Used to gate unknown class — discards dead-air frames.
    frame: (2, 1024) float32 [I, Q]
    """
    complex_iq = frame[0] + 1j * frame[1]
    energy     = np.abs(complex_iq)
    peak       = energy.max()
    mean       = energy.mean()
    if mean < 1e-9:
        return False
    return 20 * np.log10(peak / mean) > ptm_threshold_db


# ─── Dataset helpers ──────────────────────────────────────────────────────────

def open_or_create_dataset(path: str) -> h5py.File:
    if os.path.exists(path):
        print(f"  Appending to existing dataset: {path}")
        return h5py.File(path, "a")

    print(f"  Creating new dataset: {path}")
    f = h5py.File(path, "w")
    f.create_dataset("iq_frames",
        shape=(0, 2, FRAME_SIZE), maxshape=(None, 2, FRAME_SIZE),
        dtype=np.float32, chunks=(256, 2, FRAME_SIZE),
        compression="gzip", compression_opts=4)
    f.create_dataset("labels",
        shape=(0,), maxshape=(None,), dtype=np.int32, chunks=(1024,))
    f.create_dataset("gains",
        shape=(0,), maxshape=(None,), dtype=np.float32, chunks=(1024,))

    # Metadata — correctly written on creation
    class_names = [s["label"] for s in SIGNAL_CLASSES]
    f["iq_frames"].attrs["class_names"] = class_names
    f["iq_frames"].attrs["sample_rate"] = float(SAMPLE_RATE_HZ)
    f["iq_frames"].attrs["frame_size"]  = int(FRAME_SIZE)
    f["iq_frames"].attrs["gain_levels"] = GAIN_LEVELS_DB
    return f


def append_to_dataset(f: h5py.File, frames: np.ndarray,
                      label_id: int, gain_db: float):
    n       = frames.shape[0]
    current = f["iq_frames"].shape[0]
    f["iq_frames"].resize(current + n, axis=0)
    f["labels"].resize(current + n, axis=0)
    f["gains"].resize(current + n, axis=0)
    f["iq_frames"][current:current + n] = frames
    f["labels"][current:current + n]    = label_id
    f["gains"][current:current + n]     = gain_db


def count_captured(f: h5py.File, label_id: int, gain_db: float) -> int:
    if "labels" not in f or f["labels"].shape[0] == 0:
        return 0
    labels = f["labels"][:]
    gains  = f["gains"][:]
    return int(np.sum((labels == label_id) & np.isclose(gains, gain_db)))


# ─── Capture quality report ───────────────────────────────────────────────────

def quality_report(f: h5py.File, label_id: int, label: str):
    """Print IQ balance, mean power, and PSD peak for a class after capture."""
    labels = f["labels"][:]
    mask   = labels == label_id
    if not mask.any():
        return

    sample_idx = np.where(mask)[0][:200]
    frames     = f["iq_frames"][sample_idx]

    i_power   = float(np.mean(frames[:, 0, :] ** 2))
    q_power   = float(np.mean(frames[:, 1, :] ** 2))
    iq_ratio  = i_power / max(q_power, 1e-9)

    complex_iq   = frames[:, 0, :] + 1j * frames[:, 1, :]
    spectra      = np.abs(np.fft.fft(complex_iq, axis=1)) ** 2
    mean_psd     = np.mean(spectra, axis=0)
    peak_bin     = int(mean_psd.argmax())
    peak_khz     = (peak_bin - FRAME_SIZE // 2) * SAMPLE_RATE_HZ / FRAME_SIZE / 1e3

    print(f"\n  ── Quality report: {label} ──")
    print(f"     IQ balance ratio : {iq_ratio:.3f}  (ideal=1.0)")
    print(f"     Mean I power     : {i_power:.5f}")
    print(f"     Mean Q power     : {q_power:.5f}")
    print(f"     PSD peak offset  : {peak_khz:+.1f} kHz from center")
    status = "✓ PASS" if 0.8 <= iq_ratio <= 1.2 else "✗ WARN — IQ imbalance"
    print(f"     Status           : {status}")


# ─── Capture one (freq, gain, jitter) combo ───────────────────────────────────

def capture_combo(f, label_id, label, freq_hz, gain_db,
                  frames_per_station, station_idx, total_stations,
                  burst_gate=False, jitter_hz=0):
    """
    Capture frames_per_station frames for one station+gain combo.
    Applies random frequency jitter within ±jitter_hz of freq_hz.
    Uses station slot logic to prevent skipping later stations.
    """
    existing_total = count_captured(f, label_id, gain_db)
    station_start  = station_idx * frames_per_station
    station_end    = (station_idx + 1) * frames_per_station

    if existing_total >= station_end:
        print(f"    [skip] Station {station_idx+1}/{total_stations} already captured "
              f"({freq_hz/1e6:.3f} MHz gain={gain_db}dB)")
        return

    already_in_slot = max(0, existing_total - station_start)
    needed = frames_per_station - already_in_slot

    if already_in_slot > 0:
        print(f"    [partial] {already_in_slot}/{frames_per_station} — capturing {needed} more")

    # Apply frequency jitter
    jitter   = np.random.uniform(-jitter_hz, jitter_hz) if jitter_hz > 0 else 0
    actual_freq = freq_hz + jitter
    jitter_str  = f" (jitter={jitter/1e3:+.1f} kHz)" if jitter_hz > 0 else ""

    if burst_gate:
        print(f"  [burst gate ON] threshold={BURST_THRESHOLD}dB")

    print(f"  RTL-SDR: {actual_freq/1e6:.3f} MHz @ {SAMPLE_RATE_HZ/1e6:.3f} MSPS "
          f"gain={gain_db}dB{jitter_str}")

    sdr, rxStream = init_rtlsdr(actual_freq, SAMPLE_RATE_HZ, gain_db)

    with tqdm(total=needed,
              desc=f"  {label} {freq_hz/1e6:.1f}MHz gain={gain_db}dB",
              unit="frames", ncols=80) as pbar:
        captured = 0
        scanned  = 0
        first    = True

        while captured < needed:
            frame = capture_frames(sdr, rxStream, 1, FRAME_SIZE, flush=first)[0]
            first    = False
            scanned += 1

            if burst_gate and not has_burst(frame):
                continue

            # For the unknown class to discard dead air frames
            # if label == "unknown" and not has_signal(frame):
            #     continue

            append_to_dataset(f, frame[np.newaxis], label_id, float(gain_db))
            f.flush()
            captured += 1
            pbar.update(1)

    teardown_sdr(sdr, rxStream)
    if burst_gate:
        hit_rate = captured / scanned * 100 if scanned > 0 else 0
        print(f"  Done. {captured} burst frames from {scanned} scanned ({hit_rate:.1f}% hit rate)")
    else:
        print(f"  Done. {needed} frames captured.")


# ─── Main capture flow ────────────────────────────────────────────────────────

def capture_all(target_class: str = None, freq_override_hz: float = None):
    classes_to_capture = [
        s for s in SIGNAL_CLASSES
        if target_class is None or s["label"] == target_class
    ]

    if not classes_to_capture:
        print(f"ERROR: class '{target_class}' not found.")
        print(f"Available: {[s['label'] for s in SIGNAL_CLASSES]}")
        sys.exit(1)

    f = open_or_create_dataset(DATASET_PATH)

    try:
        for sig in classes_to_capture:
            label           = sig["label"]
            label_id        = sig["label_id"]
            freqs           = sig["center_freq_hz"]
            frames_per_gain = sig.get("frames_per_gain", FRAMES_SINGLE_STATION)
            jitter_hz       = sig.get("jitter_hz", 0)
            burst_gate      = sig.get("burst_gate", False)

            # Normalize to list
            freq_list = freqs if isinstance(freqs, list) else [freqs]

            # CLI --freq override for single-freq classes only
            if freq_override_hz and len(freq_list) == 1:
                print(f"Overriding {label} frequency to {freq_override_hz/1e6:.3f} MHz")
                freq_list = [freq_override_hz]

            print(f"\n{'='*65}")
            print(f"Signal class : {label}  (ID={label_id})")
            print(f"Frequencies  : {[f'{hz/1e6:.3f} MHz' for hz in freq_list]}")
            print(f"Frames/gain  : {frames_per_gain} per station")
            print(f"Gain levels  : {GAIN_LEVELS_DB} dB")
            print(f"Jitter       : ±{jitter_hz/1e3:.0f} kHz" if jitter_hz > 0 else "Jitter       : none")
            print(f"Burst gate   : {'ON (' + str(BURST_THRESHOLD) + ' dB)' if burst_gate else 'OFF'}")
            print(f"Notes        : {sig['notes']}")
            print(f"{'='*65}")
            input("  Press ENTER when ready...")

            for s_idx, freq_hz in enumerate(freq_list):
                print(f"\n  ── Station {s_idx+1}/{len(freq_list)}: {freq_hz/1e6:.3f} MHz ──")
                for gain in GAIN_LEVELS_DB:
                    capture_combo(
                        f, label_id, label, freq_hz, gain,
                        frames_per_gain, s_idx, len(freq_list),
                        burst_gate=burst_gate, jitter_hz=jitter_hz
                    )

            # Quality report after class completes
            quality_report(f, label_id, label)

    except KeyboardInterrupt:
        print("\n\nCapture interrupted — partial data saved.")
    finally:
        f.close()

    print(f"\nDataset saved: {DATASET_PATH}")
    verify_dataset()


# ─── Verification ─────────────────────────────────────────────────────────────

def verify_dataset():
    if not os.path.exists(DATASET_PATH):
        print(f"No dataset found at {DATASET_PATH}")
        return

    with h5py.File(DATASET_PATH, "r") as f:
        total  = f["iq_frames"].shape[0]
        labels = f["labels"][:]
        gains  = f["gains"][:]
        class_names = list(f["iq_frames"].attrs.get("class_names", []))
        sr  = float(f["iq_frames"].attrs.get("sample_rate", 0))
        fs  = int(f["iq_frames"].attrs.get("frame_size", 0))

        print(f"\n{'='*60}")
        print(f"Dataset: {DATASET_PATH}")
        print(f"  Total frames  : {total:,}")
        print(f"  Shape         : {f['iq_frames'].shape}")
        print(f"  Sample rate   : {sr/1e6:.3f} MSPS" if sr > 0 else "  Sample rate   : N/A")
        print(f"  Frame size    : {fs} samples  ({fs/sr*1000:.3f} ms)" if sr > 0 and fs > 0 else "  Frame size    : N/A")
        print(f"\n  Class breakdown:")

        for i, name in enumerate(class_names if class_names else [s["label"] for s in SIGNAL_CLASSES]):
            mask   = labels == i
            n      = int(np.sum(mask))
            g_this = gains[mask]
            g_counts = {g: int(np.sum(np.isclose(g_this, g))) for g in GAIN_LEVELS_DB}
            g_str  = "  ".join(f"{g}:{c}" for g, c in g_counts.items())
            bar    = "█" * (n // 500)
            print(f"    [{i}] {name:<15} {n:>6} frames  |  {g_str}")
            print(f"         {bar}")

        sample   = f["iq_frames"][:100]
        i_power  = float(np.mean(sample[:, 0, :] ** 2))
        q_power  = float(np.mean(sample[:, 1, :] ** 2))
        ratio    = i_power / max(q_power, 1e-9)
        print(f"\n  IQ balance ratio (first 100 frames): {ratio:.3f}  (ideal=1.0)")
        print(f"{'='*60}\n")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RTL-SDR IQ capture for RF classifier")
    parser.add_argument("--class",  dest="signal_class", default=None,
                        help="Capture only this signal class")
    parser.add_argument("--verify", action="store_true",
                        help="Print dataset stats only")
    parser.add_argument("--freq",   type=float, default=None,
                        help="Override center frequency in MHz (single-freq classes only)")
    args = parser.parse_args()

    if args.verify:
        verify_dataset()
    else:
        freq_override = args.freq * 1e6 if args.freq else None
        capture_all(target_class=args.signal_class, freq_override_hz=freq_override)