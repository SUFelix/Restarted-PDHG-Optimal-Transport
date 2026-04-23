"""
Loader for DOTmark CSV instances.

This loader:
  - reads the DOTmark CSV files (which contain the *real* 32x32 or 64x64 arrays)
  - normalizes them so their total mass = 1
  - flattens them for the OT LP
"""
from pathlib import Path

import numpy as np


def normalize_to_mass_one(arr: np.ndarray) -> np.ndarray:

    total = arr.sum()
    if total == 0:
        raise ValueError("Array has total sum 0, cannot normalize.")

    return arr / total


def load_dotmark_csv(path: Path) -> np.ndarray:

    arr = np.loadtxt(path, delimiter=",", dtype=np.float64)
    arr2 = normalize_to_mass_one(arr)

    return arr2

def load_dotmark_instance_csv(path_A: Path, path_B: Path):

    A2 = load_dotmark_csv(path_A)
    B2 = load_dotmark_csv(path_B)

    a = A2.flatten()
    b = B2.flatten()

    if not np.isclose(a.sum(), 1.0):
        raise RuntimeError("Normalization failed for image A")
    if not np.isclose(b.sum(), 1.0):
        raise RuntimeError("Normalization failed for image B")

    return A2, B2, a, b
