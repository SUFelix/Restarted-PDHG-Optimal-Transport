from __future__ import annotations
import cupy as cp

def one_pdhg_step_gpu_inplace(
        x, y, b, c, tau, sigma, N,  # x,y: Current primal and dual variables (iteration k)
        x_plus, y_plus,             # The target buffers for this iteration
        grad_buffer, x_tilde_buf
):

    # --- 1. Primal Update ---
    #  ATy
    cp.add(y[:N, cp.newaxis], y[cp.newaxis, N:], out=x_tilde_buf)

    # grad = c - ATy
    cp.subtract(c.reshape(N, N), x_tilde_buf, out=x_tilde_buf)

    # x_plus = x - tau * grad,
    cp.multiply(x_tilde_buf, tau, out=x_tilde_buf)
    cp.subtract(x.reshape(N, N), x_tilde_buf, out=x_plus.reshape(N, N))

    # projektion, negativ -> 0
    x_plus.clip(0, None, out=x_plus)

    # --- 2. Dual Update --- (y_plus = y + sigma * (b - A (2 * x_plus - x))))

    # x_tilde = 2 * x_plus - x
    cp.multiply(x_plus, 2.0, out=grad_buffer)
    cp.subtract(grad_buffer, x, out=grad_buffer)

    x_tilde_view = grad_buffer.reshape(N, N)

    # y_plus[:N] (Supply)
    row_sums = x_tilde_view.sum(axis=1)
    y_plus[:N] = y[:N] + sigma * (b[:N] - row_sums)

    # y_plus[N:] (Demand)
    col_sums = x_tilde_view.sum(axis=0)
    y_plus[N:] = y[N:] + sigma * (b[N:] - col_sums)
