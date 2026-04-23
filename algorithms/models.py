from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy
import numpy as np

@dataclass
class SolverResult:
    status:str
    x: np.ndarray                 # (n,)
    y: np.ndarray                 # (m,)
    gaps: Optional[np.ndarray] = None
    primal_res_seq: Optional[List[float]] = None
    dual_res_seq: Optional[List[float]] = None
    primal_obj_seq: Optional[List[float]] = None
    dual_obj_seq: Optional[np.ndarray] = None
    iterations:Optional[int] = None
    restarts:Optional[int] = None
    restart_indices: Optional[List[int]] = None


@dataclass
class LPProblem:
    A: Any
    b: np.ndarray
    c: np.ndarray
    name: str
    A_shape:tuple
    height:int
    width:int
    sum_A:float
    sum_b:float
    cost_type:str
    c_max:float
    dotmark_type:str

@dataclass
class RPDHGParams:
    tau: Optional[float] = None  # will be calculated in pdhg
    sigma: Optional[float] = None
    theta: float = 1.05          # acceleration per step
    alpha:float = 1.0
    rebalancing_threshhold:float = 1.5
    step_shrinkage:float = 0.75
    restart_check: str = "adaptive"
    min_epoch_length: int = 500
    max_iter: int = 100_000
    fixed_iter_restart:int = 2000
    tol_primal: float = 1e-2
    tol_dual: float = 1e-2
    tol_gap: float = 1e-2
    tau_sigma_preconditioned: bool = True
    rebalance_tau_sigma: bool = False
    diagnostik_i: int = min_epoch_length            # residuals stored every diagnostik_i iter


@dataclass
class GurobiParams:
    method: int = -1           # -1=auto, 0=primal simplex, 1=dual, 2=barrier
    tol_primal: float = 1e-9   # FeasibilityTol
    tol_dual: float = 1e-9     # OptimalityTol
    tol_gap: float = 1e-9
    time_limit: Optional[float] = 600
    presolve: int = -1
    num_threads: int = 0       # 0 = all available cores


@dataclass
class SolverConfig:
    solver_name: str    # "rPDHG" | "Gurobi"
    params: Union[RPDHGParams, GurobiParams]

@dataclass
class CPUInfo:
    processor: str
    cores_physical: int
    cores_logical: int
    frequency_mhz: Union[float, str]

@dataclass
class RAMInfo:
    total_gb: float
    available_at_start_gb: float

@dataclass
class GPUInfo:
    name: str
    vram_total_mb: int
    vram_available_at_start_mb: int
    driver_version: str

@dataclass
class EnvironmentInfo:
    platform: str
    python_version: str
    numpy_version: str
    scipy_version: str
    git_commit: Optional[str]
    cpu: CPUInfo
    ram: RAMInfo
    gpu: Optional[List[GPUInfo]] = None       # None  → no NVIDIA GPU detected
    gurobi_version: Optional[str] = None

@dataclass
class EvaluationResult:
    solver: SolverConfig
    environment: EnvironmentInfo
    status: str   #important to sort out the not completed (max_iter reached)

    # Timing & convergence
    runtime_seconds: float
    iterations: int
    num_restarts: int  # From Optional to int (0 as default)
    restart_indices: List[int]

    # Final normalized metrics
    primal_obj: float
    dual_obj: float
    rel_gap: float
    rel_primal_res: float
    rel_dual_res: float

    history_primal_res: List[float]
    history_dual_res: List[float]
    history_gap: List[float]
    history_primal_obj: Optional[List[float]] = None
    history_dual_obj: Optional[List[float]] = None