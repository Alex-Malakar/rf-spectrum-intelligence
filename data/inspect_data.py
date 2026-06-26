import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

DATASET_PATH = 'data/rf_dataset.h5'
CLASS_NAMES  = ["fm_broadcast", "ads_b", "noaa_wx", "noise_floor", "unknown"]
GAIN_LEVELS  = [14.4, 25.4, 36.4, 48.0, 49.6]

os.makedirs('data/plots', exist_ok=True)

with h5py.File(DATASET_PATH, 'r') as f:
    iq     = f['iq_frames'][:]
    labels = f['labels'][:]
    gains  = f['gains'][:]

# ── Summary ───────────────────────────────────────────────
print(f"Total frames:  {len(labels)}")
print(f"IQ shape:      {iq.shape}")
print()
for i, name in enumerate(CLASS_NAMES):
    count = int(np.sum(labels == i))
    print(f"  {name:<15} {count:>6} frames")
print()

# IQ balance ratio per class
print("IQ balance ratio (target 0.8 – 1.2):")
for i, name in enumerate(CLASS_NAMES):
    idx = np.where(labels == i)[0]
    if len(idx) == 0:
        print(f"  {name:<15} NO DATA")
        continue
    frames = iq[idx]
    i_power = np.mean(frames[:, 0, :] ** 2)
    q_power = np.mean(frames[:, 1, :] ** 2)
    ratio   = i_power / (q_power + 1e-12)
    flag    = "" if 0.8 <= ratio <= 1.2 else "  ← CHECK"
    print(f"  {name:<15} {ratio:.3f}{flag}")
print()

# ── Balance plot ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
x       = np.arange(len(CLASS_NAMES))
width   = 0.15

for g_idx, gain in enumerate(GAIN_LEVELS):
    counts = [int(np.sum((labels == i) & np.isclose(gains, gain)))
              for i in range(len(CLASS_NAMES))]
    ax.bar(x + g_idx * width, counts, width=width, label=f'{gain} dB')

ax.set_xticks(x + width * 2)
ax.set_xticklabels(CLASS_NAMES, rotation=15, ha='right')
ax.set_ylabel('Frame count')
ax.set_title('Dataset balance per class per gain level')
ax.legend(title='Gain')
plt.tight_layout()
plt.savefig('data/plots/balance.png', dpi=150)
plt.close()
print("Saved: data/plots/balance.png")

# ── PSD plot ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes      = axes.flatten()
freqs_khz = np.linspace(-1024, 1024, 1024)

for i, (name, ax) in enumerate(zip(CLASS_NAMES, axes)):
    idx = np.where(labels == i)[0]
    if len(idx) == 0:
        ax.set_title(f"{name} — NO DATA")
        ax.set_visible(True)
        continue
    sample_idx = np.random.choice(idx, min(100, len(idx)), replace=False)
    frames     = iq[sample_idx]
    complex_iq = frames[:, 0, :] + 1j * frames[:, 1, :]
    spectra    = np.abs(np.fft.fftshift(np.fft.fft(complex_iq, axis=1))) ** 2
    mean_psd   = 10 * np.log10(np.mean(spectra, axis=0) + 1e-12)
    ax.plot(freqs_khz, mean_psd, linewidth=0.8)
    ax.set_title(name)
    ax.set_xlabel('Frequency offset (kHz)')
    ax.set_ylabel('Power (dB)')
    ax.grid(True, alpha=0.3)

# hide unused 6th panel
axes[-1].set_visible(False)

plt.suptitle('Mean PSD per class — up to 100 frames averaged', fontsize=13)
plt.tight_layout()
plt.savefig('data/plots/psd_per_class.png', dpi=150)
plt.close()
print("Saved: data/plots/psd_per_class.png")