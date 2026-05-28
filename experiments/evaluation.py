from __future__ import annotations
import time
from pathlib import Path
from typing import Any, Callable, Optional, Union
import numpy as np
import scipy.sparse as sp
from algorithms.models import LPProblem, SolverConfig, SolverResult, EvaluationResult, GurobiParams, RPDHGParams
from experiments.utils_io import gather_environment_info, write_experiment_run


def compute_primal_residual(A, x: np.ndarray, b: np.ndarray) -> float:
    r = A.dot(x) - b if sp.issparse(A) else (A @ x - b)
    return float(np.linalg.norm(r))


def compute_dual_residual(A, y: np.ndarray, c: np.ndarray) -> float:
    s = A.T.dot(y) + c if sp.issparse(A) else (A.T @ y + c)
    return float(np.linalg.norm(np.minimum(0.0, s)))


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

    t0 = time.perf_counter()
    solver_result = solver_fn(problem)
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
    dual_obj   = float(-b @ y)
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
