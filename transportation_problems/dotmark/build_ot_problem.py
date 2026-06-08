import numpy as np
import scipy.sparse as sp
from algorithms.models import LPProblem

def pixel_coordinates(H: int, W: int) -> np.ndarray:

    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    return np.column_stack([ys.ravel(), xs.ravel()])


def build_cost_vector(coords: np.ndarray, cost_type: str = "squared_euclidean") -> np.ndarray:

    if cost_type == "squared_euclidean":
        sq = np.sum(coords ** 2, axis=1)                     # (N,)
        dist2 = sq[:, None] + sq[None, :] - 2.0 * (coords @ coords.T)  # (N,N)
        np.clip(dist2, 0.0, None, out=dist2)                 # remove numerical negative values
        return dist2.ravel()

    elif cost_type == "manhattan":
        d = (np.abs(coords[:, None, 0] - coords[None, :, 0])
           + np.abs(coords[:, None, 1] - coords[None, :, 1]))
        return d.ravel().astype(np.float64)

    else:
        raise ValueError(f"Unknown cost_type '{cost_type}'")


def build_constraint_matrix(n: int) -> sp.csr_matrix:
    """
    Baue Sparse-Constraintmatrix A (2n × n²) für das OT-LP.

    Variable x_ij hat Spaltenindex k = i*n + j:
      - Supply-Zeile i:    A[i, k] = 1  für alle j
      - Demand-Zeile n+j:  A[n+j, k] = 1  für alle i
    """
    n_vars = n * n
    rows_supply = np.repeat(np.arange(n), n)        # [0,0,..,0, 1,1,..,1, ..., n-1,..]
    rows_demand = n + np.tile(np.arange(n), n)      # [n,n+1,..,2n-1, n,n+1,..]
    rows = np.concatenate([rows_supply, rows_demand])
    cols = np.concatenate([np.arange(n_vars), np.arange(n_vars)])
    data = np.ones(2 * n_vars, dtype=np.float64)
    return sp.coo_matrix((data, (rows, cols)), shape=(2 * n, n_vars)).tocsr()


# -------------------------------------------------------------
#  Build complete Optimal Transport Problem
# -------------------------------------------------------------
def build_ot_lp(
    A_vec: np.ndarray,
    B_vec: np.ndarray,
    H: int,
    W: int,
    name: str,
    cost_type: str = "squared_euclidean",
) -> LPProblem:
    """
    Baue das OT-LP aus zwei normierten Histogrammen (sum = 1).

    Rückgabe: LPProblem mit skaliertem c ∈ [0,1] und skaliertem b (Einträge O(1)).
    meta['objective_scale'] = c_max / n  →  W2_wahr = objective_scale * (c^T x).
    """
    n = H * W
    assert A_vec.shape[0] == n, f"A_vec hat {A_vec.shape[0]} Einträge, erwartet {n}"
    assert B_vec.shape[0] == n, f"B_vec hat {B_vec.shape[0]} Einträge, erwartet {n}"

    if not np.isclose(A_vec.sum(), B_vec.sum(), atol=1e-8):
        raise ValueError(f"Massenungleichgewicht: sum(A)={A_vec.sum():.6g} vs sum(B)={B_vec.sum():.6g}")

    coordinates = pixel_coordinates(H, W)

    c_raw = build_cost_vector(coordinates, cost_type)
    c_max = float(c_raw.max()) if c_raw.max() > 0.0 else 1.0
    c = c_raw / c_max                               # c ∈ [0, 1]

    A = build_constraint_matrix(n)

    b_scale = float(n)
    b = np.concatenate([A_vec, B_vec]) * b_scale


    return LPProblem(
        A=A,
        b=b,
        c=c,
        name=name,
        A_shape=tuple(A.shape),
        height=H,
        width=W,
        sum_A = float(A_vec.sum()),
        sum_b = float(B_vec.sum()),
        cost_type=cost_type,
        c_max = c_max,
        dotmark_type= name.split("data")[0]
    )
