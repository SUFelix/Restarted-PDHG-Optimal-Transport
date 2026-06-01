from __future__ import annotations
import time
import cupy as cp
import numpy as np
from algorithms.models import LPProblem, SolverResult, RPDHGParams
from algorithms.onestep import  one_pdhg_step_gpu_inplace

def compute_ot_preconditioner_cu(
        N: int,
        norm_b:float,
        norm_c: float,
        alpha: float = 1.0, #dual/primal overweight
        safety_margin:float = 0.999 #make sure tau * sigma * ||A||^2 < 1
):
    # Basierend auf Pock-Chambolle (2011) für OT-Struktur:
    # Each column has 2 ones -> primal_scale = 1 / 2^alpha
    # Each row has N ones  -> dual_scale   = 1 / N^(2-alpha)

    tau_base = 1.0 / (2.0 ** alpha)
    sigma_base = 1.0 / (N ** (2.0 - alpha))

    omega = norm_c / norm_b

    tau = tau_base * cp.sqrt(omega)
    sigma = sigma_base / cp.sqrt(omega)

    return tau * safety_margin, sigma * safety_margin


def pdhg_restarted_cu(
    problem: LPProblem,
    params:RPDHGParams
) -> SolverResult:

    global tau, sigma, k, mu_curr, x_restart, y_restart, mu_best
    A, b, c = problem.A, cp.asarray(problem.b) , cp.asarray(problem.c)
    m, n = A.shape
    N = m // 2

    norm_b = cp.linalg.norm(b)
    norm_c = cp.linalg.norm(c)

    if params.tau is None or params.sigma is None:

        if params.tau_sigma_preconditioned:

            tau, sigma = compute_ot_preconditioner_cu(N = N,norm_b = norm_b, norm_c = norm_c, alpha=params.alpha)
            tau, sigma = float(tau), float(sigma)

        else:
            # ||A||² = 2N, normalised per-element -> 1/sqrt(2)
            tau = sigma = float(1.0 / cp.sqrt(2.0))

        if params.theta is not None:
            tau *= params.theta
            sigma *= params.theta

    x_0 = cp.zeros(n)
    y_0 = cp.zeros(m)
    x_1 = cp.zeros(n)
    y_1 = cp.zeros(m)

    grad_buffer = cp.empty(n)
    x_tilde_buf = cp.empty((N, N))
    dx_buf = cp.empty(n)
    dy_buf = cp.empty(m)
    row_sum_dx_buf = cp.empty(N)
    col_sum_dx_buf = cp.empty(N)

    # Initializing the pointers
    x_curr, y_curr = x_0, y_0
    x_next, y_next = x_1, y_1

    x_avg  = cp.zeros(n)   # Epoch means (Welford)
    y_avg  = cp.zeros(m)
    x_ref = cp.zeros(n)
    y_ref = cp.zeros(m)

    epoch_k = 0  # will increment before calculation of new avergage

    gaps: list[float] = []
    primal_residual_seq: list[float] = []
    dual_residual_seq: list[float] = []
    restart_indices: list[int] = []
    restart_count:int = 0

    mu_prev_restart = cp.inf
    mu_last_check = cp.inf
    iters_since_last_restart = 0

    for k in range(params.max_iter):

        tau *= params.theta
        sigma *= params.theta

        success = False
        for attempt in range(10):  # try up to 10x, fallback newly preconditioned tau sigma

            one_pdhg_step_gpu_inplace(
                x_curr, y_curr, b, c, tau, sigma, N,
                x_next, y_next,
                grad_buffer, x_tilde_buf
            )

            #--- calc step acceptance criteria---

            cp.subtract(x_next, x_curr, out=dx_buf)
            cp.subtract(y_next, y_curr, out=dy_buf)

            dx_mat = dx_buf.reshape(N, N)
            dx_mat.sum(axis=1, out=row_sum_dx_buf)
            dx_mat.sum(axis=0, out=col_sum_dx_buf)

            # interaction = <dy, A*dx>
            interaction = cp.dot(dy_buf[:N], row_sum_dx_buf) + cp.dot(dy_buf[N:], col_sum_dx_buf)

            norm_sq = (1.0 / tau) * cp.dot(dx_buf, dx_buf) + (1.0 / sigma) * cp.dot(dy_buf, dy_buf)

            if 2.0 * cp.abs(interaction) <= 0.95 * norm_sq:
                success = True
                break

            # step to big:
            tau *= params.step_shrinkage
            sigma *= params.step_shrinkage
            if tau < 1e-15: break

        if not success:

            tau, sigma = compute_ot_preconditioner_cu(N=N, norm_b=norm_b, norm_c=norm_c, alpha=params.alpha)
            tau, sigma = float(tau), float(sigma)

            # use safe step size
            one_pdhg_step_gpu_inplace(
                x_curr, y_curr, b, c, tau, sigma, N,
                x_next, y_next,
                grad_buffer, x_tilde_buf
            )

        # --- WELFORD (In-place auf x_avg/y_avg) --- counter epoch local
        epoch_k += 1
        ik = 1.0 / epoch_k
        x_avg *= (1.0 - ik)
        y_avg *= (1.0 - ik)
        cp.add(x_avg, x_next * ik, out=x_avg)
        cp.add(y_avg, y_next * ik, out=y_avg)

        # --- BUFFER SWAP  ---
        x_curr, x_next = x_next, x_curr
        y_curr, y_next = y_next, y_curr

        do_restart = False

        # --- cheap restart checks ---

        if params.restart_check == "adaptive":
            pass  # will be set in diagnostic block
        elif params.restart_check == "none":
            pass
        elif params.restart_check == "fixed":
            if (k > 0) and (k % params.fixed_iter_restart == 0):
                do_restart = True

        else:
            raise ValueError(f"Unknown restart_check: {params.restart_check}")

        # --- diagnostic (only every k Iterations, because expensive to calculate) ---
        if (k % params.diagnostik_i == 0) and (k > 0):

            def calculate_metrics(x_p, y_p):
                # Primal Objective & Gap
                primal_obj = float(cp.dot(c, x_p))
                dual_obj = float(cp.dot(b, y_p))
                absolute_gap = abs(primal_obj - dual_obj)

                # Primal Infeasibility (Ax - b)
                X_m = x_p.reshape(N, N)
                r_s = X_m.sum(axis=1)
                c_s = X_m.sum(axis=0)
                primal_residual = float(cp.sqrt(cp.sum((r_s - b[:N]) ** 2) + cp.sum((c_s - b[N:]) ** 2)))

                # Dual Infeasibility (A^T y - c)_+
                ATy_p = (y_p[:N, cp.newaxis] + y_p[cp.newaxis, N:]).ravel()
                dual_residual = float(cp.linalg.norm(cp.maximum(0.0, ATy_p - c)))

                return absolute_gap, primal_residual, dual_residual, primal_obj, dual_obj

            abs_gap_curr, primal_res_curr, dual_res_curr, p_obj_curr, d_obj_curr = calculate_metrics(x_curr, y_curr)

            abs_gap_avg, primal_res_avg, dual_res_avg, p_obj_avg, d_obj_avg = calculate_metrics(x_avg, y_avg)

            rel_gap = abs_gap_curr / (1.0 + abs(p_obj_curr) + abs(d_obj_curr))
            rel_primal_res = primal_res_curr / (1.0 + norm_b)
            rel_dual_res = dual_res_curr / (1.0 + norm_c)

            gaps.append(abs_gap_curr)
            primal_residual_seq.append(primal_res_curr)
            dual_residual_seq.append(dual_res_curr)

            print(
                f"Iter {k} | Primal: {p_obj_curr:.4e} | relPRes: {rel_primal_res:.2e} | relDRes: {rel_dual_res:.2e} | relGap: {rel_gap:.2e}|restartcheck: {params.restart_check} | tau: {tau}")

            # --- stop criteria ---
            if rel_gap < params.tol_gap and rel_primal_res < params.tol_primal and rel_dual_res < params.tol_dual:
                break

            if params.restart_check == "adaptive":

                # special metric dist_z from Practical Large-Scale Linear Programming using Primal-Dual Hybrid Gradient, Applegate, omega=1
                omega = cp.sqrt(tau/sigma)
                dist_z_curr = max(cp.sqrt(cp.sum((x_curr - x_ref) ** 2) * omega + cp.sum((y_curr - y_ref) ** 2) / omega), 1e-15)
                mu_curr = (abs_gap_curr / dist_z_curr) + primal_res_curr + dual_res_curr

                dist_z_avg = max(cp.sqrt(cp.sum((x_avg - x_ref) ** 2) * omega + cp.sum((y_avg - y_ref) ** 2) / omega),1e-15)
                mu_avg_pt = (abs_gap_avg / dist_z_avg) + primal_res_avg + dual_res_avg

                if mu_curr < mu_avg_pt:
                    x_restart = x_curr.copy()
                    y_restart = y_curr.copy()
                    mu_best = mu_curr
                else:
                    x_restart = x_avg.copy()
                    y_restart = y_avg.copy()
                    mu_best = mu_avg_pt

                if mu_best <= 0.9 * mu_prev_restart:
                    if iters_since_last_restart >= params.min_epoch_length:
                        do_restart = True

                elif 0.1 * mu_prev_restart >= mu_best > mu_last_check:
                    if iters_since_last_restart >= params.min_epoch_length:
                        do_restart = True
                elif iters_since_last_restart >= params.fixed_iter_restart:
                    do_restart = True

                if not do_restart:
                    mu_last_check = mu_best

            # --- rebalance step sizes (optional) ---
            if params.rebalance_tau_sigma and (rel_primal_res < params.tol_primal or rel_dual_res < params.tol_dual):
                res_imbalance = rel_primal_res / (rel_dual_res + 1e-15)
                if res_imbalance > params.rebalancing_threshhold or res_imbalance < (1.0/params.rebalancing_threshhold):
                    factor = float(cp.sqrt(res_imbalance))
                    factor = max(0.8, min(1.25, factor))
                    tau /= factor
                    sigma *= factor


        if do_restart:
            print(f"*** RESTART bei Iteration {k} ***")

            iters_since_improvement , iters_since_last_restart = 0,0
            restart_count += 1
            restart_indices.append(k)

            x_curr = x_restart
            y_curr = y_restart

            if params.restart_check == "adaptive":
                mu_prev_restart = mu_best
                mu_last_check = mu_best

            x_ref = x_curr.copy()
            y_ref = y_curr.copy()
            epoch_k = 0

        iters_since_last_restart += 1

    final_x_cpu = x_curr.get()
    final_y_cpu = y_curr.get()

    final_gaps = np.array([g.get() if hasattr(g, 'get') else g for g in gaps])

    status = 'completed' if k< params.max_iter - 1 else 'max_iter_reached'

    return SolverResult(
        status=status,
        x=final_x_cpu,
        y=final_y_cpu,
        gaps=final_gaps,
        iterations= k + 1 if params.max_iter > 0 else 0,
        restarts=restart_count,
        restart_indices=restart_indices,
        primal_res_seq= primal_residual_seq,
        dual_res_seq=dual_residual_seq
    )