import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class UniformQuantizer:
    """
    Uniform quantizer for telemetry signals.
    """

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


def load_signal_from_csv(csv_path, pollutant_name="NO2", signal_column="value"):
    """
    Load one pollutant signal from CSV.
    """
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
    """
    Convert quantized indices into fixed-length PCM binary words.
    """
    pcm_words = [format(int(idx), f"0{bits}b") for idx in indices]
    bitstream = "".join(pcm_words)
    return pcm_words, bitstream


def print_pcm_summary(original_signal, quantized_indices, pcm_words, n_show=10):
    """
    Print a few sample values with their quantized indices and PCM words.
    """
    print("\n--- PCM Encoding Summary ---")
    print("Sample\tOriginal Value\tQuantized Index\tPCM Word")
    for i in range(min(n_show, len(original_signal))):
        print(f"{i}\t{original_signal[i]:.4f}\t\t{quantized_indices[i]}\t\t{pcm_words[i]}")


def save_pcm_encoding_figure(original_signal, quantized_indices, pcm_words, save_path, n_show=10):
    """
    Save Figure 17: PCM Encoding Example
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(min(n_show, len(original_signal))):
        rows.append([
            i,
            f"{original_signal[i]:.4f}",
            int(quantized_indices[i]),
            pcm_words[i]
        ])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=["Sample", "Signal Value", "Quantized Index", "PCM Word"],
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    plt.title("PCM Encoding Example, NO2 (8-bit Quantization)", pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # ==============================
    # EDIT ONLY IF NEEDED
    # ==============================
    csv_path = "data/processed/turdata_psd_ready.csv"
    pollutant_name = "NO2"
    signal_column = "value"
    bits = 8

    # ==============================
    # LOAD SIGNAL
    # ==============================
    signal, df = load_signal_from_csv(
        csv_path=csv_path,
        pollutant_name=pollutant_name,
        signal_column=signal_column
    )

    # ==============================
    # QUANTIZE SIGNAL
    # ==============================
    quantizer = UniformQuantizer(bits=bits)
    quantized_indices, reconstructed_signal, quantization_error = quantizer.quantize(signal)

    # ==============================
    # PCM ENCODE
    # ==============================
    pcm_words, bitstream = pcm_encode_indices(quantized_indices, bits=bits)

    # ==============================
    # PRINT SUMMARY
    # ==============================
    print_pcm_summary(signal, quantized_indices, pcm_words, n_show=10)

    print(f"\nBits per sample : {bits}")
    print(f"Total samples   : {len(signal)}")
    print(f"Total bitstream length : {len(bitstream)} bits")

    # ==============================
    # SAVE FIGURE 17
    # ==============================
    figure_path = "results/figures/fig_digital_pcm_encoding_example.png"
    save_pcm_encoding_figure(
        signal,
        quantized_indices,
        pcm_words,
        figure_path,
        n_show=10
    )

    print(f"\nFigure saved to: {figure_path}")