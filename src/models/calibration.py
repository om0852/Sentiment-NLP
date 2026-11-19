"""Temperature scaling (step 5): makes a model's confidence match how often it is actually right.

A temperature T > 1 softens over-confident probabilities; T < 1 sharpens under-confident ones.
It never changes which label wins, only how confident the model claims to be.
"""
import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    p = softmax(logits, temperature)
    return float(-np.log(p[np.arange(len(labels)), labels] + 1e-12).mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Grid + golden-section search for the temperature minimising negative log-likelihood."""
    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 60))
    best = min(grid, key=lambda t: nll(logits, labels, t))
    lo, hi = best / 1.3, best * 1.3
    g = (np.sqrt(5) - 1) / 2
    for _ in range(40):
        a, b = hi - g * (hi - lo), lo + g * (hi - lo)
        if nll(logits, labels, a) < nll(logits, labels, b):
            hi = b
        else:
            lo = a
    return float((lo + hi) / 2)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    conf = probs.max(axis=-1)
    correct = probs.argmax(axis=-1) == labels
    ece = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)
