from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(slots=True)
class Standardizer:
    """Store mean/std so training, validation, test, and online data scale identically.

    In simple words: most models train better when every feature has roughly
    mean 0 and standard deviation 1.  ``fit`` learns those numbers on training
    rows only, ``transform`` applies them before the model, and ``inverse``
    converts model outputs back to physical qpos/qvel units.
    """

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        """Estimate feature-wise mean/std from training data only."""
        # x is a 2-D matrix: rows are time bins, columns are features.  For
        # example, x could be neural+phase features and y could be 32 qpos dims.
        mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = x.std(axis=0, dtype=np.float64).astype(np.float32)
        # Constant columns cannot be divided by zero; scale 1 leaves them as
        # zero after centering.
        scale[scale < 1e-6] = 1.0
        return cls(mean=mean, scale=scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Convert raw values to standardized values."""
        return (x - self.mean) / self.scale

    def inverse(self, x: np.ndarray) -> np.ndarray:
        """Convert standardized predictions back to original units."""
        return x * self.scale + self.mean


def _fit_ridge_readout(h: np.ndarray, y_scaled: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Fit a ridge-regularized affine readout from features to scaled targets.

    This solves:

        y_scaled ~= h @ weights + bias

    ``alpha`` is the ridge penalty.  Larger alpha means smaller/smoother weights
    and less overfitting, but too much alpha can underfit.
    """
    h = np.asarray(h, dtype=np.float32)
    y_scaled = np.asarray(y_scaled, dtype=np.float32)
    # Add a column of ones so the bias can be solved in the same linear system
    # as the weights.
    ones = np.ones((h.shape[0], 1), dtype=np.float32)
    hb = np.concatenate([h, ones], axis=1)
    gram = hb.T @ hb
    reg = np.eye(gram.shape[0], dtype=np.float32) * np.float32(alpha)
    # Do not regularize the bias row/column.
    reg[-1, -1] = 0.0
    rhs = hb.T @ y_scaled
    try:
        wb = np.linalg.solve(gram + reg, rhs)
    except np.linalg.LinAlgError:
        wb = np.linalg.pinv(gram + reg) @ rhs
    return wb[:-1].astype(np.float32), wb[-1].astype(np.float32)

