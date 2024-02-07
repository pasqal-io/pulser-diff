from __future__ import annotations

import torch
from torch import Tensor


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
