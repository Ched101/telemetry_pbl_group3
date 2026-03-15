import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.modulation_lead.utils import (
    load_signal_csv,
    load_segment_from_combined,
    create_demo_signal,
    normalize_signal,
    ensure_dir,
)
from src.modulation_lead.channel import add_awgn
from src.modulation_lead.metrics import mse, correlation, snr_db


def moving_average(x, window):
    if window < 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def fm_modulate(message, t, fc, kf, fs):
    integral = np.cumsum(message) / fs
    s = np.cos(2 * np.pi * fc * t + 2 * np.pi * kf * integral)
    return s


def fm_demodulate(received, fs):
    diff_signal = np.diff(received, prepend=received[0])
    recovered = moving_average(diff_signal, window=max(5, int(fs * 0.02)))
    recovered = recovered - np.mean(recovered)
    recovered = normalize_signal(recovered)
    return recovered


def main():
    parser = argparse.ArgumentParser(description="FM modulation on Student 2 telemetry-ready segments")
    parser.add_argument("--input", type=str, default="", help="Path to a single Student 2 segment CSV or the combined file")
    parser.add_argument("--combined", action="store_true", help="Use when --input points to high_priority_segments_combined.csv")
    parser.add_argument("--pollutant", type=str, default=None, help="Pollutant to select from combined file, e.g. NO2")
    parser.add_argument("--sensor_id", type=str, default=None, help="Sensor to select from combined file, e.g. S10")
    parser.add_argument("--segment_id", type=int, default=None, help="Segment ID to select from combined file, e.g. 2")
    parser.add_argument("--fc", type=float, default=0.1, help="Carrier frequency in cycles per sample-time")
    parser.add_argument("--kf", type=float, default=0.05, help="Frequency sensitivity")
    parser.add_argument("--snr", type=float, default=20.0, help="Channel SNR in dB")
    args = parser.parse_args()

    ensure_dir("results/figures")
    ensure_dir("results/modulation")

    if args.input:
        if args.combined:
            t, x_raw, fs, meta = load_segment_from_combined(
                args.input,
                pollutant=args.pollutant,
                sensor_id=args.sensor_id,
                segment_id=args.segment_id,
                signal_col="value_sg",
            )
        else:
            t, x_raw, fs, meta = load_signal_csv(args.input, signal_col="value_sg")
        print(f"Using real segment: {meta}")
    else:
        print("No real input provided. Using demo signal.")
        t, x_raw, fs = create_demo_signal()
        meta = {"pollutant": "DEMO", "sensor_id": "DEMO", "segment_id": 0}

    x = normalize_signal(x_raw)

    s_tx = fm_modulate(x, t, fc=args.fc, kf=args.kf, fs=fs)
    s_rx = add_awgn(s_tx, snr_db=args.snr)
    x_hat = fm_demodulate(s_rx, fs=fs)

    label = f"{meta.get('pollutant', 'UNK')}_{meta.get('sensor_id', 'UNK')}_seg{meta.get('segment_id', 'X')}"

    plt.figure(figsize=(12, 8))

    plt.subplot(4, 1, 1)
    plt.plot(t, x)
    plt.title(f"Original Signal ({label})")
    plt.xlabel("Time [samples / hours]")
    plt.ylabel("Normalized amplitude")

    plt.subplot(4, 1, 2)
    plt.plot(t, s_tx)
    plt.title(f"FM Modulated Signal ({label})")
    plt.xlabel("Time [samples / hours]")
    plt.ylabel("Amplitude")

    plt.subplot(4, 1, 3)
    plt.plot(t, s_rx)
    plt.title(f"Signal After Channel Noise ({label})")
    plt.xlabel("Time [samples / hours]")
    plt.ylabel("Amplitude")

    plt.subplot(4, 1, 4)
    plt.plot(t, x, label="Original")
    plt.plot(t, x_hat, label="Recovered", alpha=0.8)
    plt.title(f"Demodulated Signal Recovery ({label})")
    plt.xlabel("Time [samples / hours]")
    plt.ylabel("Normalized amplitude")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"results/figures/fig_modulation_fm_waveform_{label}.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(t, s_tx)
    plt.title(f"FM Modulated Signal ({label})")
    plt.xlabel("Time [samples / hours]")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig("results/figures/fig_modulation_fm_waveform.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(t, x, label="Original")
    plt.plot(t, x_hat, label="Recovered", alpha=0.8)
    plt.title(f"Demodulated Signal Recovery - FM ({label})")
    plt.xlabel("Time [samples / hours]")
    plt.ylabel("Normalized amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/figures/fig_demodulation_signal_recovery_fm_{label}.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(t, x, label="Original")
    plt.plot(t, x_hat, label="Recovered", alpha=0.8)
    plt.title(f"Demodulated Signal Recovery - FM ({label})")
    plt.xlabel("Time [samples / hours]")
    plt.ylabel("Normalized amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/fig_demodulation_signal_recovery_fm.png", dpi=300)
    plt.close()

    results = {
        "scheme": "FM",
        "pollutant": meta.get("pollutant", "DEMO"),
        "sensor_id": meta.get("sensor_id", "DEMO"),
        "segment_id": meta.get("segment_id", 0),
        "mse": mse(x, x_hat),
        "correlation": correlation(x, x_hat),
        "recovered_snr_db": snr_db(x, x_hat),
        "channel_snr_db": args.snr,
        "carrier_hz": args.fc,
        "kf": args.kf,
        "fs_hz": fs,
        "num_samples": len(x),
    }

    pd.DataFrame([results]).to_csv(f"results/modulation/fm_metrics_{label}.csv", index=False)
    pd.DataFrame([results]).to_csv("results/modulation/fm_metrics.csv", index=False)

    print("FM modulation complete.")
    print(pd.DataFrame([results]))


if __name__ == "__main__":
    main()