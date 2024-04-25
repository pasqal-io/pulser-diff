from __future__ import annotations

from enum import Enum
from functools import lru_cache

import torch
from torch import Tensor

import pulser_diff.dq as dq


def kron(*args: Tensor) -> Tensor:
    if not all([t.is_sparse for t in args]):
        raise ValueError("All arguments must be sparse tensors")

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

    # rearange indices and values tensors
    new_indices = torch.vstack(new_indices).T
    new_values = torch.hstack(new_values)

    # create resulting sparse tensor
    mat_prod = torch.sparse_coo_tensor(
        new_indices, new_values, tuple(new_size)
    ).coalesce()
    return mat_prod


@lru_cache
def total_magnetization(n_qubits: int) -> Tensor:
    zero_sparse = torch.sparse_coo_tensor(
        [[0], [0]],
        [0],
        (2**n_qubits, 2**n_qubits),
        dtype=torch.complex128,
    )

    # create sparse total magnetization observable
    obs = []
    for i in range(n_qubits):
        tprod = [dq.eye(2).to_sparse() for _ in range(n_qubits)]
        tprod[i] = dq.sigmaz().to_sparse()
        obs.append(kron(*tprod))
    obs = sum(obs, start=zero_sparse)
    return obs


def expect(obs: Tensor, state: Tensor) -> Tensor:
    if obs.is_sparse:
        if dq.isket(state):
            state = state.squeeze(-1)
            exp_val = (
                torch.matmul(state.conj(), torch.matmul(obs, state.T)).to_dense().diag()
            )
        elif dq.isdm(state):
            exp_val = trace(torch.matmul(obs, state))
    else:
        exp_val = dq.expect(obs, state)

    return exp_val


def trace(mat: Tensor) -> Tensor:
    """calculate de trace of a 2D sparse tensor"""
    n_qbit = int(torch.log2(torch.tensor([mat.shape[0]])))
    tensprod_list = [dq.eye(2).to_sparse() for n in range(n_qbit)]
    sparse_identity = kron(*tensprod_list)
    return (mat * sparse_identity).sum()


class StrEnum(str, Enum):
    def __str__(self) -> str:
        """Used when dumping enum fields in a schema."""
        ret: str = self.value
        return ret

    @classmethod
    def list(cls) -> list[str]:
        return list(map(lambda c: c.value, cls))


class SolverType(StrEnum):
    DQ = "dq"
    """Uses dynamiqs solver"""

    DQ_ME = "dq_me"
    """Uses dynamiqs master equation solver"""

    KRYLOV = "krylov"
    """Uses the krylov solver"""
