from __future__ import annotations
import time
from pathlib import Path
from typing import Any, Callable, Optional, Union
import numpy as np
import scipy.sparse as sp
from algorithms.models import LPProblem, SolverConfig, SolverResult, EvaluationResult, GurobiParams, RPDHGParams
from experiments.utils_io import gather_environment_info, write_experiment_run

try:
    import cupy as cp
except ImportError:
    cp = None


def warmup_rpdhg_gpu(grid_size: int = 4, max_iter: int = 60) -> None:
    """
    Run pdhg_restarted_cu once on a tiny synthetic OT problem so that CUDA
    context init and CuPy kernel JIT compilation happen here, not inside the
    first timed evaluate_solver() call. Call this once per process before
    any timed rPDHG runs. No-op if CuPy/GPU is unavailable.
    """
    if cp is None:
        return

    from algorithms.pdhg_restarted_cu import pdhg_restarted_cu
    from transportation_problems.dotmark.build_ot_problem import (
        pixel_coordinates, build_cost_vector, build_constraint_matrix,
    )

    n = grid_size * grid_size
    rng = np.random.default_rng(0)
    a = rng.random(n) + 1e-3
    a /= a.sum()
    d = rng.random(n) + 1e-3
    d /= d.sum()

    c_raw = build_cost_vector(pixel_coordinates(grid_size, grid_size))
    c_max = float(c_raw.max()) if c_raw.max() > 0.0 else 1.0
    c = c_raw / c_max
    A = build_constraint_matrix(n)
    b = np.concatenate([a, d]) * n

    dummy_problem = LPProblem(
        A=A, b=b, c=c, name="__gpu_warmup__", A_shape=tuple(A.shape),
        height=grid_size, width=grid_size, sum_A=float(a.sum()), sum_b=float(d.sum()),
        cost_type="squared_euclidean", c_max=c_max, dotmark_type="warmup",
    )

    # diagnostik_i <= max_iter so the diagnostic/restart-check kernels (which
    # only run every diagnostik_i iters in the real solver) also get JIT'd here.
    warmup_params = RPDHGParams(max_iter=max_iter, diagnostik_i=max(1, max_iter // 3))
    pdhg_restarted_cu(dummy_problem, warmup_params)
    cp.cuda.Stream.null.synchronize()


def compute_primal_residual(A, x: np.ndarray, b: np.ndarray) -> float:
    r = A.dot(x) - b if sp.issparse(A) else (A @ x - b)
    return float(np.linalg.norm(r))


def compute_dual_residual(A, y: np.ndarray, c: np.ndarray) -> float:
    # Dual feasibility for min c^T x s.t. Ax=b, x>=0 is A^T y <= c;
    # violation is the positive part of (A^T y - c).
    s = A.T.dot(y) - c if sp.issparse(A) else (A.T @ y - c)
    return float(np.linalg.norm(np.maximum(0.0, s)))


def _downsample(seq: Optional[list], max_points: int = 1000) -> Optional[list]:

    if seq is None or len(seq) <= max_points:
        return seq
    step = max(1, len(seq) // max_points)

    return seq[::step]


def evaluate_solver(
    problem: LPProblem,
    solver_fn: Callable[[LPProblem], SolverResult],
    solver_config: Optional[SolverConfig],
    write_run: bool = False,
    exp_dir: Optional[Union[str, Path]] = None,
    experiment_name: Optional[str] = None,
    csv_export: bool = False,
) -> EvaluationResult:

    A, b, c = problem.A, problem.b, problem.c
    m, n = A.shape

    # Drain any pending async GPU work from a previous call before starting the
    # clock, and make sure the timed solver's own GPU work has actually
    # finished before stopping it (CuPy kernel launches are async).
    if cp is not None:
        cp.cuda.Stream.null.synchronize()

    t0 = time.perf_counter()
    solver_result = solver_fn(problem)

    if cp is not None:
        cp.cuda.Stream.null.synchronize()
    runtime = time.perf_counter() - t0

    if not isinstance(solver_result, SolverResult):
        raise TypeError(
            f"{solver_config.solver_name}: solver_fn must return SolverResult, got {type(solver_result)}"
        )

    x = np.asarray(solver_result.x).ravel()
    y = np.asarray(solver_result.y).ravel()

    if x.shape != (n,):
        raise ValueError(f"{solver_config.solver_name}: x.shape={x.shape}, expected ({n},)")
    if y.shape != (m,):
        raise ValueError(f"{solver_config.solver_name}: y.shape={y.shape}, expected ({m},)")

    # --- Compute metrics ---
    primal_obj = float(c @ x)
    dual_obj   = float(b @ y)
    norm_b     = float(np.linalg.norm(b))
    norm_c     = float(np.linalg.norm(c))

    r_primal = compute_primal_residual(A, x, b)
    r_dual   = compute_dual_residual(A, y, c)

    rel_gap = abs(abs(primal_obj) - abs(dual_obj)) / (1.0 + abs(primal_obj))
    rel_primal_res = r_primal                   / (1.0 + norm_b)
    rel_dual_res   = r_dual                     / (1.0 + norm_c)


    if solver_config is None:
        raise ValueError("solver_config darf nicht None sein")

    history_primal_residuum = _downsample(
        list(solver_result.primal_res_seq) if solver_result.primal_res_seq is not None else None
    )
    history_dual_residuum = _downsample(
        list(solver_result.dual_res_seq) if solver_result.dual_res_seq is not None else None
    )
    history_gap = _downsample(
        list(solver_result.gaps) if solver_result.gaps is not None else None
    )


    history_primal_obj = _downsample(
        list(solver_result.primal_obj_seq) if hasattr(solver_result, 'primal_obj_seq') and solver_result.primal_obj_seq is not None else None
    )
    history_dual_obj = _downsample(
        list(solver_result.dual_obj_seq) if hasattr(solver_result, 'dual_obj_seq') and solver_result.dual_obj_seq is not None else None
    )

    evaluation_result = EvaluationResult(
        solver =solver_config,
        environment =gather_environment_info(),
        status = solver_result.status,
        runtime_seconds =runtime,
        iterations =solver_result.iterations,
        num_restarts =solver_result.restarts,
        restart_indices = solver_result.restart_indices,
        primal_obj =primal_obj,
        dual_obj =dual_obj,
        rel_gap =rel_gap,
        rel_primal_res =rel_primal_res,
        rel_dual_res =rel_dual_res,
        history_primal_res = history_primal_residuum,
        history_dual_res = history_dual_residuum,
        history_gap = history_gap ,
        history_primal_obj= history_primal_obj,
        history_dual_obj= history_dual_obj
    )

    # ── Write to disk ─────────────────────────────────────────────────────────
    if write_run:
        if exp_dir is None:
            raise ValueError("exp_dir is required when write_run=True")
        if experiment_name is None:
            raise ValueError("experiment_name is required when write_run=True")


        metrics = {
            "runtime_seconds": runtime,
            "iterations": evaluation_result.iterations,
            "num_restarts": evaluation_result.num_restarts,
            "status": solver_result.status,
            "primal_obj": primal_obj,
            "dual_obj": dual_obj,
            "rel_gap": rel_gap,
            "rel_primal_res": rel_primal_res,
            "rel_dual_res": rel_dual_res
        }

        sequences = {
            "primal_residual": history_primal_residuum,
            "dual_residual":   history_dual_residuum,
            "duality_gap": _downsample(
                list(solver_result.gaps) if solver_result.gaps is not None else None
            ),
            "restart_indices": solver_result.restart_indices,
            "tau": _downsample(
                list(solver_result.tau_seq) if getattr(solver_result, 'tau_seq', None) is not None else None
            ),
            "sigma": _downsample(
                list(solver_result.sigma_seq) if getattr(solver_result, 'sigma_seq', None) is not None else None
            ),
        }

        if True: #(verbose)

            print(f"Primal objective: {primal_obj:.6e}")

            print(f"Primal residual:  {r_primal:.6e}")
            print(f"Dual residual:    {r_dual:.6e}")

        write_experiment_run(
            exp_dir=exp_dir,
            experiment_name = experiment_name,
            solver_name = solver_config.solver_name,
            solver_params = solver_config.params,
            problem = problem,
            environment = evaluation_result.environment,
            metrics = metrics,
            sequences = sequences,
            csv_export = csv_export,
        )

    return evaluation_result
