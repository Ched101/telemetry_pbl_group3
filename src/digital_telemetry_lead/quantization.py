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


def save_quantization_figure(original_signal, quantized_signal, save_path, pollutant_name, n_samples=500):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = min(n_samples, len(original_signal))

    plt.figure(figsize=(10, 5))
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
    plt.title(f"Original vs Quantized Signal ({pollutant_name}, First {n_samples} Samples)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def quantize_signal(signal, bits=8, x_min=None, x_max=None):
    quantizer = UniformQuantizer(bits=bits, x_min=x_min, x_max=x_max)
    q_indices, reconstructed, error = quantizer.quantize(signal)

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

            results = quantize_signal(signal, bits=bits)

            figure_path = f"results/figures/fig_digital_original_vs_quantized_{pollutant_name}.png"
            save_quantization_figure(
                results["original_signal"],
                results["reconstructed_signal"],
                figure_path,
                pollutant_name=pollutant_name,
                n_samples=500
            )

            print(f"{pollutant_name}: quantization figure saved to {figure_path}")

        except Exception as e:
            print(f"{pollutant_name}: skipped -> {e}")