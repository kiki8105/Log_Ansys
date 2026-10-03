# src/engines/signal_filters.py
"""
PlotJuggler-style data filters for time-series signals.

All filter functions accept NumPy arrays (x, y) and return transformed
(x_out, y_out) arrays.  The original data is never mutated.
"""

import numpy as np


class SignalFilters:
    """Static methods implementing 7 signal filter transforms."""

    # ── Absolute ──────────────────────────────────────────────────
    @staticmethod
    def absolute(x, y):
        """Return |y| with the same x axis."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        return x.copy(), np.abs(y)

    # ── Derivative ────────────────────────────────────────────────
    @staticmethod
    def derivative(x, y):
        """Numerical derivative dy/dx using central differences on interior
        points and forward/backward differences at the boundaries so the
        output length equals the input length."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if len(x) < 2:
            return x.copy(), np.zeros_like(y)
        dy_dx = np.gradient(y, x)
        return x.copy(), dy_dx

    # ── Integral ──────────────────────────────────────────────────
    @staticmethod
    def integral(x, y):
        """Cumulative trapezoidal integration.  Output length == input length
        (first element is 0)."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if len(x) < 2:
            return x.copy(), np.zeros_like(y)
        dx = np.diff(x)
        y_avg = 0.5 * (y[:-1] + y[1:])
        cumulative = np.concatenate(([0.0], np.cumsum(y_avg * dx)))
        return x.copy(), cumulative

    # ── Moving Average ────────────────────────────────────────────
    @staticmethod
    def moving_average(x, y, window=10):
        """Simple moving-average filter.  Window is clamped to [1, len(y)].
        Output length == input length (uses 'same' convolution with edge
        padding so the axes stay aligned)."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        window = max(1, min(int(window), len(y)))
        if window <= 1:
            return x.copy(), y.copy()
        kernel = np.ones(window, dtype=np.float64) / window
        # Pad edges to keep same length and avoid NaN tails.
        pad_left = window // 2
        pad_right = window - 1 - pad_left
        y_padded = np.pad(y, (pad_left, pad_right), mode='edge')
        y_avg = np.convolve(y_padded, kernel, mode='valid')
        return x.copy(), y_avg[:len(x)]

    # ── Moving Root Mean Squared ──────────────────────────────────
    @staticmethod
    def moving_rms(x, y, window=10):
        """Moving RMS (root-mean-square) filter.  Same length semantics as
        moving_average."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        window = max(1, min(int(window), len(y)))
        if window <= 1:
            return x.copy(), np.abs(y)
        kernel = np.ones(window, dtype=np.float64) / window
        pad_left = window // 2
        pad_right = window - 1 - pad_left
        y2_padded = np.pad(y ** 2, (pad_left, pad_right), mode='edge')
        mean_sq = np.convolve(y2_padded, kernel, mode='valid')
        rms = np.sqrt(np.maximum(mean_sq[:len(x)], 0.0))
        return x.copy(), rms

    # ── Outlier Removal ───────────────────────────────────────────
    @staticmethod
    def outlier_removal(x, y, threshold=3.0):
        """Remove outliers beyond *threshold* standard deviations from the
        mean.  Outlier y values are replaced with NaN (so x/y length is
        preserved and the plot simply gaps over them)."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).copy()
        threshold = max(0.1, float(threshold))
        finite_mask = np.isfinite(y)
        if not np.any(finite_mask):
            return x.copy(), y
        mean_val = float(np.mean(y[finite_mask]))
        std_val = float(np.std(y[finite_mask]))
        if std_val < 1e-15:
            return x.copy(), y
        z_scores = np.abs((y - mean_val) / std_val)
        y[z_scores > threshold] = np.nan
        return x.copy(), y

    # ── Scale / Offset ────────────────────────────────────────────
    @staticmethod
    def scale_offset(x, y, time_offset=0.0, value_offset=0.0, multiplier=1.0):
        """Apply linear transform:
            x_new = x + time_offset
            y_new = y * multiplier + value_offset
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        x_new = x + float(time_offset)
        y_new = y * float(multiplier) + float(value_offset)
        return x_new, y_new
