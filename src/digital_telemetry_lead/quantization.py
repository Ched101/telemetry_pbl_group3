import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class UniformQuantizer:
    """
    Uniform quantizer for environmental telemetry signals.
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

    def print_summary(self, signal, q_indices, reconstructed, error):
        print("\n--- Quantization Summary ---")
        print(f"Bits              : {self.bits}")
        print(f"Levels            : {self.levels}")
        print(f"x_min             : {self.x_min:.4f}")
        print(f"x_max             : {self.x_max:.4f}")
        print(f"Step size (delta) : {self.delta:.6f}")
        print(f"Signal length     : {len(signal)}")
        print(f"Mean Abs Error    : {np.mean(np.abs(error)):.6f}")
        print(f"Mean Sq Error     : {np.mean(error ** 2):.6f}")

        print("\nFirst 10 samples:")
        print("Original\tIndex\tReconstructed\tError")
        for i in range(min(10, len(signal))):
            print(
                f"{signal[i]:.4f}\t{q_indices[i]}\t{reconstructed[i]:.4f}\t\t{error[i]:.4f}"
            )


def load_signal_from_csv(csv_path, signal_column):
    """
    Load one signal column from a CSV file.
    """
    df = pd.read_csv(csv_path)
    df = df[df["pollutant"] == "NO2"]
    signal = df[signal_column]

    if signal_column not in df.columns:
        raise ValueError(
            f"Column '{signal_column}' not found in CSV. Available columns: {list(df.columns)}"
        )

    signal = df[signal_column].dropna().to_numpy(dtype=float)
    return signal, df


def save_quantization_figure(original_signal, quantized_signal, save_path):
    """
    Save Figure 16: Original Signal vs Quantized Signal
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    n_samples = 500

    plt.plot(original_signal[:n_samples], label="Original Signal", linewidth=2)
    plt.step(
        np.arange(n_samples),
        quantized_signal[:n_samples],
        where="mid",
        label="Quantized Signal",
        linewidth=1.8
    )
    plt.xlabel("Sample Index")
    plt.ylabel("Signal Value")
    plt.title("Original Signal vs Quantized Signal (NO2)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def quantize_signal(signal, bits=8, x_min=None, x_max=None):
    """
    Quantize a signal and return all useful outputs.
    """
    quantizer = UniformQuantizer(bits=bits, x_min=x_min, x_max=x_max)
    q_indices, reconstructed, error = quantizer.quantize(signal)
    quantizer.print_summary(signal, q_indices, reconstructed, error)

    return {
        "original_signal": np.asarray(signal, dtype=float),
        "quantized_indices": q_indices,
        "reconstructed_signal": reconstructed,
        "quantization_error": error,
        "bits": bits,
        "levels": quantizer.levels,
        "x_min": quantizer.x_min,
        "x_max": quantizer.x_max,
        "delta": quantizer.delta,
    }


if __name__ == "__main__":
    # ==============================
    # EDIT THESE TWO LINES ONLY
    # ==============================
    csv_path = "data/processed/turdata_psd_ready.csv"
    signal_column = "value"

    # Optional: use fixed telemetry range instead of automatic min/max
    use_fixed_range = False
    fixed_min = 0
    fixed_max = 200

    bits = 8

    # ==============================
    # LOAD SIGNAL
    # ==============================
    signal, df = load_signal_from_csv(csv_path, signal_column)

    # ==============================
    # QUANTIZE SIGNAL
    # ==============================
    if use_fixed_range:
        results = quantize_signal(signal, bits=bits, x_min=fixed_min, x_max=fixed_max)
    else:
        results = quantize_signal(signal, bits=bits)

    # ==============================
    # SAVE FIGURE 16
    # ==============================
    figure_path = "results/figures/fig_digital_original_vs_quantized.png"
    save_quantization_figure(
        results["original_signal"],
        results["reconstructed_signal"],
        figure_path
    )

    print(f"\nFigure saved to: {figure_path}")