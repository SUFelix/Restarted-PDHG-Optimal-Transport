import numpy as np
import gurobipy as gp
from gurobipy import GRB
from algorithms.models import SolverResult, LPProblem, GurobiParams
import scipy.sparse as sp

def solve_lp_gurobi(
        problem: LPProblem,
        params: GurobiParams,
        sense: str = "min",
        verbose: bool = True,
        model_name: str = "ot_lp",
) -> SolverResult:
    A, b, c = problem.A, problem.b, problem.c

    # --- Shape & type handling ---
    if sp is not None and sp.issparse(A):
        A_mat = A.tocsr()
        m, n = A_mat.shape
        is_sparse = True
    else:
        A_mat = np.asarray(A, dtype=float)
        m, n = A_mat.shape
        is_sparse = False

    b = np.asarray(b, dtype=float).ravel()
    c = np.asarray(c, dtype=float).ravel()

    assert b.shape[0] == m, f"b has shape {b.shape}, but A has {m} rows"
    assert c.shape[0] == n, f"c has shape {c.shape}, but A has {n} columns"

    env = gp.Env(empty=True)
    if not verbose:
        env.setParam("OutputFlag", 0)

    env.setParam("Method", params.method)
    env.setParam("FeasibilityTol", params.tol_primal)
    env.setParam("OptimalityTol", params.tol_dual)
    env.setParam("BarConvTol", params.tol_gap)
    env.setParam("Presolve", params.presolve)
    env.setParam("Threads", params.num_threads)

    if params.time_limit is not None:
        env.setParam("TimeLimit", float(params.time_limit))

    env.start()

    model = gp.Model(model_name, env=env)

    n = c.shape[0]
    m = b.shape[0]

    x = model.addMVar(shape=n, lb=0.0, vtype=GRB.CONTINUOUS, name="x")

    if sense == "min":
        model.setObjective(c @ x, GRB.MINIMIZE)
    else:
        model.setObjective(c @ x, GRB.MAXIMIZE)

    model.addMConstr(A, x, '=', b, name="constraints")

    model.optimize()

    status = model.Status

    info = {
        "status": status,
        "runtime": model.Runtime,
        "iter_count": getattr(model, "IterCount", 0),
        "bar_iter_count": getattr(model, "BarIterCount", 0),
    }

    x_opt = np.zeros(n)
    y_opt = np.zeros(m)

    if status == GRB.OPTIMAL:
        x_opt = x.X

        try:
            y_opt = model.getAttr(GRB.Attr.Pi, model.getConstrs())
        except:
            y_opt = np.zeros(m)

        info["obj_val"] = model.ObjVal
    else:
        info["obj_val"] = None

    return SolverResult(
        x=x_opt,
        y=y_opt,
        status=str(status),
        primal_obj_seq=[model.ObjVal],
        gaps=None
    )


