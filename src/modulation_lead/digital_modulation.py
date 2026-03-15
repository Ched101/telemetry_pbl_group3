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
from src.modulation_lead.metrics import bit_error_rate


def signal_to_bits(x, n_bits=8):
    x = normalize_signal(x)
    x_scaled = ((x + 1) / 2) * (2**n_bits - 1)
    x_quantized = np.clip(np.round(x_scaled), 0, 2**n_bits - 1).astype(np.uint8)
    bits = np.unpackbits(x_quantized)
    return bits


def ask_modulate(bits, samples_per_bit, fc, fs):
    symbols = bits.astype(float)
    x = np.repeat(symbols, samples_per_bit)
    t = np.arange(len(x)) / fs
    carrier = np.cos(2 * np.pi * fc * t)
    return t, x * carrier


def ask_demodulate(received, samples_per_bit, fc, fs):
    t = np.arange(len(received)) / fs
    mixed = received * np.cos(2 * np.pi * fc * t)
    recovered_bits = []
    for i in range(0, len(mixed), samples_per_bit):
        chunk = mixed[i:i + samples_per_bit]
        recovered_bits.append(1 if np.mean(chunk) > 0.2 else 0)
    return np.array(recovered_bits[: len(mixed) // samples_per_bit], dtype=int)


def fsk_modulate(bits, samples_per_bit, f0, f1, fs):
    bit_wave = []
    for bit in bits:
        t_local = np.arange(samples_per_bit) / fs
        freq = f1 if bit == 1 else f0
        bit_wave.append(np.cos(2 * np.pi * freq * t_local))
    signal = np.concatenate(bit_wave)
    t = np.arange(len(signal)) / fs
    return t, signal


def fsk_demodulate(received, bits_len, samples_per_bit, f0, f1, fs):
    recovered_bits = []
    for i in range(bits_len):
        chunk = received[i * samples_per_bit:(i + 1) * samples_per_bit]
        t_local = np.arange(len(chunk)) / fs
        ref0 = np.cos(2 * np.pi * f0 * t_local)
        ref1 = np.cos(2 * np.pi * f1 * t_local)
        score0 = np.sum(chunk * ref0)
        score1 = np.sum(chunk * ref1)
        recovered_bits.append(1 if score1 > score0 else 0)
    return np.array(recovered_bits, dtype=int)


def psk_modulate(bits, samples_per_bit, fc, fs):
    bit_wave = []
    for bit in bits:
        t_local = np.arange(samples_per_bit) / fs
        phase = 0 if bit == 1 else np.pi
        bit_wave.append(np.cos(2 * np.pi * fc * t_local + phase))
    signal = np.concatenate(bit_wave)
    t = np.arange(len(signal)) / fs
    return t, signal


def psk_demodulate(received, bits_len, samples_per_bit, fc, fs):
    recovered_bits = []
    for i in range(bits_len):
        chunk = received[i * samples_per_bit:(i + 1) * samples_per_bit]
        t_local = np.arange(len(chunk)) / fs
        ref = np.cos(2 * np.pi * fc * t_local)
        score = np.sum(chunk * ref)
        recovered_bits.append(1 if score >= 0 else 0)
    return np.array(recovered_bits, dtype=int)


def main():
    parser = argparse.ArgumentParser(description="ASK/FSK/PSK modulation on Student 2 telemetry-ready segments")
    parser.add_argument("--input", type=str, default="", help="Path to a single Student 2 segment CSV or the combined file")
    parser.add_argument("--combined", action="store_true", help="Use when --input points to high_priority_segments_combined.csv")
    parser.add_argument("--pollutant", type=str, default=None, help="Pollutant to select from combined file")
    parser.add_argument("--sensor_id", type=str, default=None, help="Sensor to select from combined file")
    parser.add_argument("--segment_id", type=int, default=None, help="Segment ID to select from combined file")
    parser.add_argument("--scheme", type=str, choices=["ASK", "FSK", "PSK"], default="ASK")
    parser.add_argument("--snr", type=float, default=15.0)
    parser.add_argument("--samples_per_bit", type=int, default=20)
    args = parser.parse_args()

    ensure_dir("results/figures")
    ensure_dir("results/modulation")

    if args.input:
        if args.combined:
            _, x_raw, _, meta = load_segment_from_combined(
                args.input,
                pollutant=args.pollutant,
                sensor_id=args.sensor_id,
                segment_id=args.segment_id,
                signal_col="value_sg",
            )
        else:
            _, x_raw, _, meta = load_signal_csv(args.input, signal_col="value_sg")
        print(f"Using real segment: {meta}")
    else:
        print("No real input provided. Using demo signal.")
        _, x_raw, _ = create_demo_signal()
        meta = {"pollutant": "DEMO", "sensor_id": "DEMO", "segment_id": 0}

    bits = signal_to_bits(x_raw[:128], n_bits=8)
    fs = 10.0
    label = f"{meta.get('pollutant', 'UNK')}_{meta.get('sensor_id', 'UNK')}_seg{meta.get('segment_id', 'X')}"

    if args.scheme == "ASK":
        fc = 1.0
        t, s_tx = ask_modulate(bits, args.samples_per_bit, fc, fs)
        s_rx = add_awgn(s_tx, snr_db=args.snr)
        bits_hat = ask_demodulate(s_rx, args.samples_per_bit, fc, fs)
        figure_name = f"fig_modulation_ask_waveform_{label}.png"

    elif args.scheme == "FSK":
        f0, f1 = 0.5, 1.5
        t, s_tx = fsk_modulate(bits, args.samples_per_bit, f0, f1, fs)
        s_rx = add_awgn(s_tx, snr_db=args.snr)
        bits_hat = fsk_demodulate(s_rx, len(bits), args.samples_per_bit, f0, f1, fs)
        figure_name = f"fig_modulation_fsk_waveform_{label}.png"

    else:
        fc = 1.0
        t, s_tx = psk_modulate(bits, args.samples_per_bit, fc, fs)
        s_rx = add_awgn(s_tx, snr_db=args.snr)
        bits_hat = psk_demodulate(s_rx, len(bits), args.samples_per_bit, fc, fs)
        figure_name = f"fig_modulation_psk_waveform_{label}.png"

    ber = bit_error_rate(bits, bits_hat)

    plt.figure(figsize=(12, 4))
    plt.plot(t[:1000], s_tx[:1000])
    plt.title(f"{args.scheme} Waveform ({label})")
    plt.xlabel("Time [samples]")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(f"results/figures/{figure_name}", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(t[:1000], s_tx[:1000])
    plt.title(f"{args.scheme} Waveform ({label})")
    plt.xlabel("Time [samples]")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(f"results/figures/fig_modulation_{args.scheme.lower()}_waveform.png", dpi=300)
    plt.close()

    results = {
        "scheme": args.scheme,
        "pollutant": meta.get("pollutant", "DEMO"),
        "sensor_id": meta.get("sensor_id", "DEMO"),
        "segment_id": meta.get("segment_id", 0),
        "ber": ber,
        "channel_snr_db": args.snr,
        "samples_per_bit": args.samples_per_bit,
        "num_bits": len(bits),
    }

    pd.DataFrame([results]).to_csv(
        f"results/modulation/{args.scheme.lower()}_metrics_{label}.csv", index=False
    )
    pd.DataFrame([results]).to_csv(
        f"results/modulation/{args.scheme.lower()}_metrics.csv", index=False
    )

    print(f"{args.scheme} modulation complete.")
    print(pd.DataFrame([results]))


if __name__ == "__main__":
    main()