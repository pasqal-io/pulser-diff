from __future__ import annotations

from typing import Any, get_args

import torch

from pulser_diff.dq._utils import check_time_tensor, obj_type_str
from pulser_diff.dq.solvers.options import Options
from pulser_diff.dq.solvers.result import Result
from pulser_diff.dq.solvers.utils.utils import to_time_operator
from pulser_diff.dq.time_tensor import TimeTensor
from pulser_diff.dq.utils.tensor_types import ArrayLike, to_tensor
from pulser_diff.dq.sesolve.adaptive import SEDormandPrince5


def sesolve(
    H: ArrayLike | TimeTensor,
    psi0: ArrayLike,
    tsave: ArrayLike,
    *,
    exp_ops: list[ArrayLike] | None = None,
    options: dict[str, Any] | None = None,
) -> Result:
    r"""Solve the Schrödinger equation.

    This function computes the evolution of the state vector $\ket{\psi(t)}$ at time
    $t$, starting from an initial state $\ket{\psi(t=0)}$, according to the Schrödinger
    equation (with $\hbar=1$):
    $$
        \frac{\dd\ket{\psi(t)}}{\dt} = -i H(t) \ket{\psi(t)},
    $$
    where $H(t)$ is the system's Hamiltonian at time $t$.

    Quote: Time-dependent Hamiltonian
        If the Hamiltonian depends on time, it can be passed as a function with
        signature `H(t: float) -> Tensor`.

    Quote: Running multiple simulations concurrently
        Both the Hamiltonian `H` and the initial state `psi0` can be batched to
        solve multiple Schrödinger equations concurrently. All other arguments are
        common to every batch.

    Args:
        H _(array-like or function, shape (n, n) or (bH, n, n))_: Hamiltonian. For
            time-dependent problems, provide a function with signature
            `H(t: float) -> Tensor` that returns a tensor (batched or not) for any
            given time between `t = 0.0` and `t = tsave[-1]`.
        psi0 _(array-like, shape (n, 1) or (bpsi, n, 1))_: Initial state vector.
        tsave _(1d array-like)_: Times at which the states and expectation values are
            saved. The Schrödinger equation is solved from `t = 0.0` to `t = tsave[-1]`.
        exp_ops _(list of 2d array-like, each with shape (n, n), optional)_: List of
            operators for which the expectation value is computed. Defaults to `None`.
        options _(dict, optional)_: Generic options (see the list below). Defaults to
            `None`.

    Note-: Available options
        - **save_states** _(bool, optional)_ – If `True`, the state is saved at every
            time in `tsave`. If `False`, only the final state is returned. Defaults to
            `True`.
        - **verbose** _(bool, optional)_ – If `True`, print a progress bar during the
            integration. Defaults to `True`.
        - **dtype** _(torch.dtype, optional)_ – Complex data type to which all
            complex-valued tensors are converted. `tsave` is converted to a real data
            type of the corresponding precision. Defaults to the complex data type set
            by `torch.set_default_dtype`.
        - **device** _(torch.device, optional)_ – Device on which the tensors are
            stored. Defaults to the device set by `torch.set_default_device`.

    Returns:
        Object holding the results of the Schrödinger equation integration. It has the
            following attributes:

            - **states** _(Tensor)_ – Saved states with shape
                _(bH?, bpsi?, len(tsave), n, 1)_.
            - **expects** _(Tensor, optional)_ – Saved expectation values with shape
                _(bH?, bpsi?, len(exp_ops), len(tsave))_.
            - **tsave** or **times** _(Tensor)_ – Times for which states and expectation
                values were saved.
            - **start_datetime** _(datetime)_ – Start date and time of the integration.
            - **end_datetime** _(datetime)_ – End date and time of the integration.
            - **total_time** _(timedelta)_ – Total duration of the integration.
            - **solver** (Solver) –  Solver used.
            - **gradient** (Gradient) – Gradient used.
            - **options** _(dict)_  – Options used.
    """
    # === options
    options = Options(options=options)

    # === check exp_ops
    if exp_ops is not None and not isinstance(exp_ops, list):
        raise TypeError(
            'Argument `exp_ops` must be `None` or a list of array-like objects, but has'
            f' type {obj_type_str(exp_ops)}.'
        )

    # === convert and batch H, y0, E
    kw = dict(dtype=options.cdtype, device=options.device)

    # convert and batch H
    if not isinstance(H, (*get_args(ArrayLike), TimeTensor)):
        raise TypeError(
            'Argument `H` must be an array-like object or a `TimeTensor`, but has type'
            f' {obj_type_str(H)}.'
        )
    H = to_time_operator(H, 'H', **kw)  # (bH?, n, n)

    # convert and batch y0
    y0 = to_tensor(psi0, **kw)  # (by?, n, 1)

    # convert E
    E = to_tensor(exp_ops, **kw)  # (nE, n, n)

    # === convert tsave and init tmeas
    kw = dict(dtype=options.rdtype, device='cpu')
    tsave = to_tensor(tsave, **kw)
    check_time_tensor(tsave, arg_name='tsave')
    tmeas = torch.empty(0, **kw)

    # === define the solver
    solver = SEDormandPrince5(H, y0, tsave, tmeas, E, options)

    # === compute the result
    result = solver.run()

    return result
