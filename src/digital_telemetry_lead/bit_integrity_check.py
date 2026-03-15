import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class UniformQuantizer:
    def __init__(self, bits=8, x_min=None, x_max=None):
        if bits <= 0:
            raise ValueError("bits must be a positive integer")
        self.bits = bits
        self.levels = 2 ** bits
        self.x_min = x_min
        self.x_max = x_max
        self.delta = None

    def fit_range(self, signal):
        signal = np.asarray(signal, dtype=float)

        if signal.size == 0:
            raise ValueError("Input signal is empty")

        if self.x_min is None:
            self.x_min = float(np.min(signal))
        if self.x_max is None:
            self.x_max = float(np.max(signal))

        if self.x_max == self.x_min:
            raise ValueError("x_max and x_min cannot be equal")

        self.delta = (self.x_max - self.x_min) / (self.levels - 1)

    def quantize(self, signal):
        signal = np.asarray(signal, dtype=float)
        self.fit_range(signal)

        q_indices = np.round((signal - self.x_min) / self.delta).astype(int)
        q_indices = np.clip(q_indices, 0, self.levels - 1)

        reconstructed = self.x_min + q_indices * self.delta
        error = signal - reconstructed

        return q_indices, reconstructed, error


def load_signal_from_csv(csv_path, pollutant_name, signal_column="value"):
    df = pd.read_csv(csv_path)

    if "pollutant" not in df.columns:
        raise ValueError("CSV must contain a 'pollutant' column")

    if signal_column not in df.columns:
        raise ValueError(
            f"Column '{signal_column}' not found. Available columns: {list(df.columns)}"
        )

    df = df[df["pollutant"] == pollutant_name].copy()
    df = df.dropna(subset=[signal_column])

    signal = df[signal_column].to_numpy(dtype=float)

    if len(signal) == 0:
        raise ValueError(f"No data found for pollutant '{pollutant_name}'")

    return signal, df


def pcm_encode_indices(indices, bits=8):
    pcm_words = [format(int(idx), f"0{bits}b") for idx in indices]
    bitstream = "".join(pcm_words)
    return pcm_words, bitstream


def add_even_parity(data_bits):
    ones_count = data_bits.count("1")
    parity_bit = "0" if ones_count % 2 == 0 else "1"
    return data_bits + parity_bit, parity_bit


def check_even_parity(codeword):
    ones_count = codeword.count("1")
    return ones_count % 2 == 0


def flip_bit(bitstring, index):
    flipped = "1" if bitstring[index] == "0" else "0"
    return bitstring[:index] + flipped + bitstring[index + 1:]


def save_integrity_figure(pass_clean, pass_error, save_path, pollutant_name):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    labels = ["Clean Transmission", "Transmission with Bit Error"]
    values = [1 if pass_clean else 0, 1 if pass_error else 0]

    plt.figure(figsize=(8, 4))
    bars = plt.bar(labels, values)
    plt.ylim(0, 1.2)
    plt.ylabel("Integrity Check Result")
    plt.title(f"Bit Integrity / Error Summary ({pollutant_name})")
    plt.yticks([0, 1], ["Fail", "Pass"])
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.03,
            "Pass" if value == 1 else "Fail",
            ha="center"
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    csv_path = "data/processed/turdata_psd_ready.csv"
    signal_column = "value"
    pollutants = ["NO2", "O3", "PM10", "PM2_5"]
    bits = 8

    for pollutant_name in pollutants:
        try:
            signal, df = load_signal_from_csv(
                csv_path=csv_path,
                pollutant_name=pollutant_name,
                signal_column=signal_column
            )

            quantizer = UniformQuantizer(bits=bits)
            quantized_indices, reconstructed_signal, quantization_error = quantizer.quantize(signal)

            pcm_words, bitstream = pcm_encode_indices(quantized_indices, bits=bits)

            data_bits = pcm_words[0]
            codeword, parity_bit = add_even_parity(data_bits)
            clean_pass = check_even_parity(codeword)

            error_codeword = flip_bit(codeword, 2)
            error_pass = check_even_parity(error_codeword)

            figure_path = f"results/figures/fig_digital_bit_integrity_summary_{pollutant_name}.png"
            save_integrity_figure(clean_pass, error_pass, figure_path, pollutant_name)

            print(f"{pollutant_name}: integrity figure saved to {figure_path}")

        except Exception as e:
            print(f"{pollutant_name}: skipped -> {e}")