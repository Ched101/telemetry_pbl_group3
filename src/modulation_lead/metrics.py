import numpy as np


def mse(x_true, x_pred):
    x_true = np.asarray(x_true, dtype=float)
    x_pred = np.asarray(x_pred, dtype=float)
    n = min(len(x_true), len(x_pred))
    return float(np.mean((x_true[:n] - x_pred[:n]) ** 2))


def correlation(x_true, x_pred):
    x_true = np.asarray(x_true, dtype=float)
    x_pred = np.asarray(x_pred, dtype=float)
    n = min(len(x_true), len(x_pred))

    if n < 2:
        return 0.0

    c = np.corrcoef(x_true[:n], x_pred[:n])[0, 1]
    if np.isnan(c):
        return 0.0
    return float(c)


def snr_db(reference, test):
    reference = np.asarray(reference, dtype=float)
    test = np.asarray(test, dtype=float)
    n = min(len(reference), len(test))

    ref = reference[:n]
    err = ref - test[:n]

    signal_power = np.mean(ref ** 2)
    noise_power = np.mean(err ** 2)

    if np.isclose(noise_power, 0.0):
        return float("inf")

    return float(10 * np.log10(signal_power / noise_power))


def bit_error_rate(bits_true, bits_pred):
    bits_true = np.asarray(bits_true, dtype=int)
    bits_pred = np.asarray(bits_pred, dtype=int)
    n = min(len(bits_true), len(bits_pred))

    if n == 0:
        return 0.0

    return float(np.mean(bits_true[:n] != bits_pred[:n]))