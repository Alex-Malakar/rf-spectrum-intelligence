"""
Signal class definitions for RF capture.
RTL-SDR V3 (R820T tuner) specific configuration.

Classes: fm_broadcast, ads_b, noaa_wx, noise_floor, unknown (5 total)
Gain levels: 5 (expanded from 4 for SNR variation proxy)
Frequency jitter: applied per gain level for position generalization
Frame budget: ~10,000 frames per class
"""

SAMPLE_RATE_HZ = 2_048_000
FRAME_SIZE     = 1024
TARGET_FRAMES_PER_CLASS = 10_000


GAIN_LEVELS_DB = [14.4, 25.4, 36.4, 48.0, 49.6]


FRAMES_SINGLE_STATION = 2000
FRAMES_TWO_STATIONS   = 1000
FRAMES_FOUR_STATIONS  = 500

# Frequency jitter — applied per gain level capture
# Randomizes where signal sits within 2 MHz capture window
JITTER_FM_HZ      = 300_000   # ±300 kHz
JITTER_NOAA_HZ    = 200_000   # ±200 kHz — narrower, less deviation
JITTER_ADSB_HZ    = 500_000   # ±500 kHz — burst visible anywhere in window
JITTER_NOISE_HZ   = 500_000   # ±500 kHz — flat signal, position irrelevant
JITTER_UNKNOWN_HZ = 0         # no jitter — unknown class needs stable reference

SIGNAL_CLASSES = [
    {
        "label":          "fm_broadcast",
        "label_id":       0,
        "center_freq_hz": [
            99_300_000,    # 99.3 MHz
            98_900_000,    # 98.9 MHz
            96_900_000,    # 96.9 MHz
            102_700_000,   # 102.7 MHz
        ],
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "frames_per_gain": FRAMES_FOUR_STATIONS,
        "jitter_hz":      JITTER_FM_HZ,
        "burst_gate":     False,
        "notes":          "FM stereo broadcast — 4 OKC stations, ±300 kHz jitter"
    },
    {
        "label":          "ads_b",
        "label_id":       1,
        "center_freq_hz": 1_090_000_000,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "frames_per_gain": FRAMES_SINGLE_STATION,
        "jitter_hz":      JITTER_ADSB_HZ,
        "burst_gate":     True,
        "notes":          "Aircraft transponder bursts — 13dB burst gate, ±500 kHz jitter"
    },
    {
        "label":          "noaa_wx",
        "label_id":       2,
        "center_freq_hz": [
            162_400_000,   # OKC primary
            162_550_000,   # OKC secondary
        ],
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "frames_per_gain": FRAMES_TWO_STATIONS,
        "jitter_hz":      JITTER_NOAA_HZ,
        "burst_gate":     False,
        "notes":          "NOAA weather radio — 2 OKC channels, ±200 kHz jitter"
    },
    {
        "label":          "noise_floor",
        "label_id":       3,
        "center_freq_hz": 400_000_000,   #cleaner ISM-free band
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "frames_per_gain": FRAMES_SINGLE_STATION,
        "jitter_hz":      JITTER_NOISE_HZ,
        "burst_gate":     False,
        "notes":          "Wideband noise — 400 MHz, Faraday cage recommended, ±500 kHz jitter"
    },
    {
        "label":          "unknown",
        "label_id":       4,
        "center_freq_hz": [
            118_000_000,   # aviation voice ATC — keep, active during day
            144_200_000,   # amateur 2m SSB calling frequency — replace 155
            462_562_500,   # GMRS channel 1 — keep roughly, adjust to exact channel
            851_000_000,   # public safety 800 MHz trunked — replace 906
        ],
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "frames_per_gain": FRAMES_FOUR_STATIONS,
        "jitter_hz":      JITTER_UNKNOWN_HZ,
        "burst_gate":     False,
        "notes":          "Mixed unknown signals — teaches model to recognize what it does not know"
    },
]