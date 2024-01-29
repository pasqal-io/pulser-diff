from __future__ import annotations

import torch


def _fix_border_vals(deriv: torch.Tensor, border_indices: list, dt) -> torch.Tensor:
    prev_idx = 0
    with torch.no_grad():
        for idx in border_indices:
            if idx == 0:
                deriv[0] = deriv[2] - ((deriv[2] - deriv[1]) / dt) * 2 * dt
                prev_idx = idx
            else:
                if (idx - prev_idx) != 1 or idx+3>=len(deriv):
                    deriv[idx-1] = deriv[idx-3] + ((deriv[idx-2] - deriv[idx-3]) / dt) * 2 * dt
                    deriv[idx] = deriv[idx-2] + ((deriv[idx-1] - deriv[idx-2]) / dt) * 2 * dt
                else:
                    deriv[idx] = deriv[idx+2] - ((deriv[idx+2] - deriv[idx+1]) / dt) * 2 * dt
                prev_idx = idx
    return deriv


def derivative(f: torch.Tensor, 
               x: torch.Tensor, 
               mode = "time", 
               pulse_endtimes: list | None = None, 
               sample_rate: float = 0.1) -> torch.Tensor:

    step = int(1/sample_rate)
    if mode == "time":
        res = torch.autograd.grad(f, x, torch.ones_like(f))[0]
        if pulse_endtimes is not None:
            dt = x[1] - x[0]
            res = _fix_border_vals(res, pulse_endtimes, dt)
        res = res[::step]

    elif mode == "param":
        # derivative with respect to parameter
        def dfdp(v):
            return torch.autograd.grad(f, x, v)[0]
        
        rem = len(f) % step
        basis_vectors = torch.zeros((len(f)//step + rem, len(f)))
        for i in range(len(f)//step):
            basis_vectors[i, i*step] = 1.0
        res = torch.vmap(dfdp)(basis_vectors)
    else:
        raise ValueError("Derivative mode not supported.")

    return res