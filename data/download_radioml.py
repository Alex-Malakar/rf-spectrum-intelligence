#!/usr/bin/env python3
"""
RadioML 2016.10a dataset downloader & converter
------------------------------------------------
Downloads the DeepSig RadioML 2016.10a open dataset and converts
it to the same HDF5 format as our live capture script.

This gives you 11 modulation classes × 20 SNR levels × 1000 frames = 220,000
labeled IQ frames instantly — useful to:
  1. Train and validate your CNN in Phase 2 before capture is complete
  2. Supplement your captured dataset with more SNR diversity
  3. Benchmark your model against published results

Usage:
    python download_radioml.py              # download and convert
    python download_radioml.py --stats      # stats only (if already downloaded)

The converted file is saved alongside rf_dataset.h5 as radioml_dataset.h5
with identical format: shape (N, 2, 1024), labels as integers.

RadioML class mapping (integer → modulation):
    0=8PSK  1=AM-DSB  2=AM-SSB  3=BPSK  4=CPFSK  5=GFSK
    6=PAM4  7=QAM16   8=QAM64   9=QPSK  10=WBFM

Dataset paper: O'Shea & West (2016), "Radio Machine Learning Dataset Generation
with GNU Radio", GNU Radio Conference.
"""

import os
import sys
import pickle
import numpy as np
import h5py

RADIOML_URL  = "https://raw.githubusercontent.com/radioML/dataset/master/RML2016.10a_dict.pkl"
OUTPUT_PATH  = os.path.join(os.path.dirname(__file__), "radioml_dataset.h5")
PICKLE_PATH  = os.path.join(os.path.dirname(__file__), "RML2016.10a_dict.pkl")

MODULATION_CLASSES = [
    "8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK",
    "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"
]

FRAME_SIZE = 128   # RadioML uses 128 samples per frame (not 1024)
                   # We'll zero-pad to 1024 to match our capture format


def download_radioml():
    """Download the RadioML pickle file."""
    try:
        import urllib.request
        print(f"Downloading RadioML 2016.10a from GitHub...")
        print(f"  URL: {RADIOML_URL}")
        print(f"  Destination: {PICKLE_PATH}")
        print("  This may take a few minutes (~55 MB)...")
        urllib.request.urlretrieve(RADIOML_URL, PICKLE_PATH,
            reporthook=lambda b, bs, ts: print(f"\r  {b*bs/1e6:.1f}/{ts/1e6:.0f} MB", end="", flush=True)
        )
        print("\n  Download complete.")
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print("\nManual download:")
        print("  1. Visit: https://www.deepsig.ai/datasets")
        print("  2. Download RML2016.10a.tar.bz2")
        print("  3. Extract RML2016.10a_dict.pkl to rf_classifier/data/")
        sys.exit(1)


def convert_to_hdf5():
    """Convert RadioML pickle to our HDF5 format."""
    print(f"\nLoading pickle: {PICKLE_PATH}")
    with open(PICKLE_PATH, "rb") as pf:
        data = pickle.load(pf, encoding="latin1")

    # data keys are tuples: (modulation_str, snr_int)
    # Each value: ndarray shape (1000, 2, 128) — 1000 frames, I/Q, 128 samples
    mods = sorted(set(k[0] for k in data.keys()))
    snrs = sorted(set(k[1] for k in data.keys()))
    print(f"  Modulations: {mods}")
    print(f"  SNR range:   {snrs[0]} to {snrs[-1]} dB")

    total_frames = sum(data[k].shape[0] for k in data.keys())
    print(f"  Total frames: {total_frames:,}")

    # Build label map: modulation string → integer
    label_map = {m: i for i, m in enumerate(MODULATION_CLASSES)}

    print(f"\nConverting to HDF5: {OUTPUT_PATH}")
    with h5py.File(OUTPUT_PATH, "w") as f:
        # Zero-pad 128-sample frames to 1024 to match our capture format
        PADDED = 1024
        iq_ds = f.create_dataset(
            "iq_frames",
            shape=(total_frames, 2, PADDED),
            dtype=np.float32,
            chunks=(256, 2, PADDED),
            compression="gzip",
            compression_opts=4,
        )
        lbl_ds = f.create_dataset("labels", shape=(total_frames,), dtype=np.int32)
        snr_ds = f.create_dataset("snrs",   shape=(total_frames,), dtype=np.int32)

        # Metadata
        iq_ds.attrs["class_names"]  = MODULATION_CLASSES
        iq_ds.attrs["sample_rate"]  = 200_000   # RadioML uses 200 kSPS
        iq_ds.attrs["frame_size"]   = PADDED
        iq_ds.attrs["original_frame_size"] = FRAME_SIZE
        iq_ds.attrs["snr_levels"]   = snrs

        idx = 0
        for mod in MODULATION_CLASSES:
            for snr in snrs:
                key = (mod, snr)
                if key not in data:
                    continue
                batch = data[key].astype(np.float32)  # (1000, 2, 128)
                n = batch.shape[0]
                # Zero-pad from 128 → 1024
                padded = np.zeros((n, 2, PADDED), dtype=np.float32)
                padded[:, :, :FRAME_SIZE] = batch
                iq_ds[idx:idx+n]  = padded
                lbl_ds[idx:idx+n] = label_map.get(mod, -1)
                snr_ds[idx:idx+n] = snr
                idx += n
                print(f"  {mod:<10} SNR={snr:+3d} dB   {n} frames   total={idx:,}", end="\r")

        print(f"\n\nConversion complete. {idx:,} frames written.")
        print_stats(f)


def print_stats(f=None):
    """Print dataset statistics."""
    close_after = f is None
    if f is None:
        if not os.path.exists(OUTPUT_PATH):
            print(f"No file at {OUTPUT_PATH}")
            return
        f = h5py.File(OUTPUT_PATH, "r")

    total = f["iq_frames"].shape[0]
    labels = f["labels"][:]
    class_names = list(f["iq_frames"].attrs.get("class_names", []))

    print(f"\n{'='*50}")
    print(f"RadioML dataset: {OUTPUT_PATH}")
    print(f"  Total frames: {total:,}")
    print(f"  Shape:        {f['iq_frames'].shape}")
    print(f"\n  Per class:")
    for i, name in enumerate(class_names):
        n = int(np.sum(labels == i))
        print(f"    [{i:2d}] {name:<10}  {n:>6} frames")
    print(f"{'='*50}\n")

    if close_after:
        f.close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--stats", action="store_true", help="Print stats only")
    args = p.parse_args()

    if args.stats:
        print_stats()
    else:
        if not os.path.exists(PICKLE_PATH):
            download_radioml()
        convert_to_hdf5()
