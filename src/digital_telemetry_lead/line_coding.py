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


def nrz_encode(bitstream):
    return np.array([1 if bit == "1" else -1 for bit in bitstream], dtype=int)


def save_line_coding_figure(bitstream, encoded_signal, save_path, pollutant_name, n_bits=32):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    bits_to_plot = bitstream[:n_bits]
    signal_to_plot = encoded_signal[:n_bits]

    x = np.arange(len(signal_to_plot) + 1)
    y = np.append(signal_to_plot, signal_to_plot[-1])

    plt.figure(figsize=(10, 4))
    plt.step(x, y, where="post", linewidth=2)
    plt.ylim(-1.5, 1.5)
    plt.xlabel("Bit Index")
    plt.ylabel("Amplitude")
    plt.title(f"NRZ Line Coding Waveform ({pollutant_name})")
    plt.grid(True)

    for i, bit in enumerate(bits_to_plot):
        plt.text(i + 0.35, 1.15, bit, fontsize=9)

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
            encoded_signal = nrz_encode(bitstream)

            figure_path = f"results/figures/fig_digital_line_coding_{pollutant_name}.png"
            save_line_coding_figure(
                bitstream,
                encoded_signal,
                figure_path,
                pollutant_name=pollutant_name,
                n_bits=32
            )

            print(f"{pollutant_name}: line coding figure saved to {figure_path}")

        except Exception as e:
            print(f"{pollutant_name}: skipped -> {e}")