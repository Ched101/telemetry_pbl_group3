import numpy as np


def add_awgn(signal, snr_db, rng_seed=42):
    signal = np.asarray(signal, dtype=float)

    rng = np.random.default_rng(rng_seed)

    signal_power = np.mean(signal ** 2)
    if np.isclose(signal_power, 0.0):
        return signal.copy()

    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = rng.normal(0, np.sqrt(noise_power), size=signal.shape)

    return signal + noise