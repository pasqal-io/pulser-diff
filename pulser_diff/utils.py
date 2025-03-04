from __future__ import annotations

from functools import lru_cache, reduce
from math import prod

import torch
from pyqtorch.matrices import IMAT, ZMAT
from torch import Tensor, log2
from torch.linalg import eigvalsh


def kron(*args: Tensor) -> Tensor:
    if not all([t.is_sparse for t in args]):
        # converting to dense tensors and calculating Kronecker product
        args = tuple([t.to_dense() if t.is_sparse else t for t in args])
        return reduce(torch.kron, args)

    mat1: Tensor = args[0]
    if len(args) == 1:
        # return last matrix
        return mat1
    else:
        # calculate Kronecker product recursively
        mat2 = kron(*args[1:])

    # get tensor sizes and calculate resulting tensor size
    size_mat1 = torch.as_tensor(mat1.size())
    size_mat2 = torch.as_tensor(mat2.size())
    new_size = torch.Size(size_mat1 * size_mat2)

    # calculate tensor product of 2 matrices
    new_indices = []
    new_values = []
    for idx, val in zip(mat1.indices().T, mat1.values()):
        new_idxs = idx * size_mat2 + mat2.indices().T
        new_indices.append(new_idxs)
        new_vals = val * mat2.values()
        new_values.append(new_vals)

    # create resulting sparse tensor
    mat_prod = torch.sparse_coo_tensor(
        torch.vstack(new_indices).T, torch.hstack(new_values), tuple(new_size)
    ).coalesce()
    return mat_prod


@lru_cache
def total_magnetization(n_qubits: int) -> Tensor:
    zero_sparse = torch.sparse_coo_tensor(
        torch.as_tensor([[0], [0]]),
        [0],
        (2**n_qubits, 2**n_qubits),
        dtype=torch.complex128,
    )

    # create sparse total magnetization observable
    obs = []
    for i in range(n_qubits):
        tprod = [IMAT.to_sparse() for _ in range(n_qubits)]
        tprod[i] = ZMAT.to_sparse()
        obs.append(kron(*tprod))
    return sum(obs, start=zero_sparse)


def expect(obs: Tensor, states: Tensor) -> Tensor:
    if obs.is_sparse:
        if states.size(-1) == 1:
            states = states.squeeze(-1)
            exp_val = torch.matmul(states.conj(), torch.matmul(obs, states.T)).to_dense().diag()
        elif states.size(-1) == states.size(-2):
            exp_val = trace(torch.matmul(obs, states))
    else:
        if len(states.shape) == 3:
            # ket tensor of shape (n_tsteps, 2**N, n_batch)
            exp_val = torch.einsum("...ij,jk,...kl->...", states.mH, obs, states)  # <x|O|x>
        elif len(states.shape) == 4:
            # density matrix tensor of shape (n_tsteps, 2**N, 2**N, n_batch)
            exp_val = torch.einsum("ij,...jik->...", obs, states)  # tr(Ox)

    return exp_val


def trace(mat: Tensor) -> Tensor:
    """calculate de trace of a 2D sparse tensor"""
    n_qbit = int(torch.log2(torch.tensor([mat.shape[-1]])))
    tensprod_list = [IMAT.to_sparse() for n in range(n_qbit)]
    sparse_identity = kron(*tensprod_list)
    return (mat * sparse_identity).sum(dim=(-2, -1)).to_dense()


def vn_entropy(rho: Tensor) -> Tensor:
    """calculate the Von Neumann entropy of a density matrix"""
    ev = eigvalsh(rho)
    vne = torch.tensor(0.0)
    for k in range(rho.shape[0]):
        if ev[k] > 0:
            vne += -ev[k] * log2(ev[k])

    return vne


def basis_state(dim: int | tuple[int, ...], number: int | tuple[int, ...]) -> Tensor:
    r"""Returns the ket of a Fock state or the ket of a tensor product of Fock states.

    Args:
        dim _(int or tuple of ints)_: Dimension of the Hilbert space of each mode.
        number _(int or tuple of ints)_: Fock state number of each mode.
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(n, 1)_ Ket of the Fock state or tensor product of Fock states.
    """
    # convert integer inputs to tuples by default, and check dimensions match
    dim = (dim,) if isinstance(dim, int) else dim
    number = (number,) if isinstance(number, int) else number
    if len(dim) != len(number):
        raise ValueError(
            "Arguments `number` must have the same length as `dim` of length"
            f" {len(dim)}, but has length {len(number)}."
        )

    # compute the required basis state
    n = 0
    for d, s in zip(dim, number):
        n = d * n + s
    ket = torch.zeros(prod(dim), 1)
    ket[n] = 1.0
    return ket
