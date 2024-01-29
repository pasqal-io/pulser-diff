from __future__ import annotations

from typing import Union, get_args

import numpy as np
import torch
from qutip import Qobj
from torch import Tensor

from .._utils import hdim, obj_type_str
from .utils import isbra, isket, isop

__all__ = ['to_tensor', 'to_numpy', 'to_qutip', 'from_qutip']

Number = Union[int, float, complex]

# type for objects convertible to a torch.Tensor using `to_tensor`
ArrayLike = Union[tuple, list, np.ndarray, Tensor, Qobj]

# data type conversion dictionaries
DTYPE_TO_REAL = {torch.complex64: torch.float32, torch.complex128: torch.float64}
DTYPE_TO_COMPLEX = {torch.float32: torch.complex64, torch.float64: torch.complex128}


def get_cdtype(
    dtype: torch.complex64 | torch.complex128 | None = None,
) -> torch.complex64 | torch.complex128:
    if dtype is None:
        # the default dtype for complex tensors is determined by the default
        # floating point dtype (torch.complex128 if default is torch.float64,
        # otherwise torch.complex64)
        if torch.get_default_dtype() is torch.float64:
            return torch.complex128
        else:
            return torch.complex64
    elif dtype not in (torch.complex64, torch.complex128):
        raise TypeError(
            'Argument `dtype` must be `torch.complex64`, `torch.complex128` or `None`'
            f' for a complex tensor, but is `{dtype}`.'
        )
    return dtype


def get_rdtype(
    dtype: torch.float32 | torch.float64 | None = None,
) -> torch.float32 | torch.float64:
    if dtype is None:
        return torch.get_default_dtype()
    elif dtype not in (torch.float32, torch.float64):
        raise TypeError(
            'Argument `dtype` must be `torch.float32`, `torch.float64` or `None` for'
            f' a real-valued tensor, but is `{dtype}`.'
        )
    return dtype


def dtype_complex_to_real(
    dtype: torch.complex64 | torch.complex128,
) -> torch.float32 | torch.float64:
    return DTYPE_TO_REAL[dtype]


def dtype_real_to_complex(
    dtype: torch.float32 | torch.float64,
) -> torch.complex64 | torch.complex128:
    return DTYPE_TO_COMPLEX[dtype]


def to_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.ones(1).device  # default device
    elif isinstance(device, str):
        return torch.device(device)
    elif isinstance(device, torch.device):
        return device
    else:
        raise TypeError(
            f'Argument `device` ({device}) must be a string, a `torch.device` object or'
            ' `None`.'
        )


def to_tensor(
    x: ArrayLike | list[ArrayLike] | None,
    *,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    """Convert an array-like object or a list of array-like objects to a tensor.

    Args:
        x: QuTiP quantum object or NumPy array or Python list or Python tuple or PyTorch
            tensor or list of these types. If `None` returns an empty tensor of shape
            _(0)_.
        dtype: Data type of the returned tensor.
        device: Device on which the returned tensor is stored.

    Returns:
        Output tensor.

    Examples:
        >>> import numpy as np
        >>> import qutip as qt
        >>> dq.to_tensor(qt.fock(3, 1))
        tensor([[0.+0.j],
                [1.+0.j],
                [0.+0.j]])
        >>> dq.to_tensor(np.array([[1, 2, 3], [4, 5, 6]]))
        tensor([[1, 2, 3],
                [4, 5, 6]])
        >>> dq.to_tensor([qt.fock(3, 1), qt.fock(3, 2)])
        tensor([[[0.+0.j],
                 [1.+0.j],
                 [0.+0.j]],
        <BLANKLINE>
                [[0.+0.j],
                 [0.+0.j],
                 [1.+0.j]]])
    """
    if x is None:
        return torch.tensor([], dtype=dtype, device=device)
    elif isinstance(x, tuple):
        if len(x) == 0:
            return torch.tensor([], dtype=dtype, device=device)
        else:
            return torch.as_tensor(x, dtype=dtype, device=device)
    elif isinstance(x, list):
        if len(x) == 0:
            return torch.tensor([], dtype=dtype, device=device)
        if not isinstance(x[0], get_args(ArrayLike)):
            return torch.as_tensor(x, dtype=dtype, device=device)
        else:
            return torch.stack([to_tensor(el, dtype=dtype, device=device) for el in x])
    elif isinstance(x, Qobj):
        return from_qutip(x, dtype=get_cdtype(dtype), device=device)
    elif isinstance(x, (np.ndarray, Tensor)):
        return torch.as_tensor(x, dtype=dtype, device=device)
    else:
        raise TypeError(
            'Argument `x` must be an array-like object or a list of array-like objects,'
            f' but has type {obj_type_str(x)}.'
        )


def to_numpy(x: ArrayLike | list[ArrayLike]) -> np.ndarray:
    """Convert an array-like object or a list of array-like objects to a NumPy array.

    Args:
        x: QuTiP quantum object or NumPy array or Python list or Python tuple or
            PyTorch tensor or list of these types.

    Returns:
        Output NumPy array.

    Examples:
        >>> import qutip as qt
        >>> dq.to_numpy(dq.fock(3, 1))
        array([[0.+0.j],
               [1.+0.j],
               [0.+0.j]], dtype=complex64)
        >>> dq.to_numpy([qt.fock(3, 1), qt.fock(3, 2)])
        array([[[0.+0.j],
                [1.+0.j],
                [0.+0.j]],
        <BLANKLINE>
               [[0.+0.j],
                [0.+0.j],
                [1.+0.j]]])
    """
    if isinstance(x, tuple):
        if len(x) == 0:
            return np.array([])
        else:
            return np.array(x)
    elif isinstance(x, list):
        if len(x) == 0:
            return np.array([])
        if not isinstance(x[0], get_args(ArrayLike)):
            return np.array(x)
        else:
            return np.array([to_numpy(el) for el in x])
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, Tensor):
        return x.numpy(force=True)
    elif isinstance(x, Qobj):
        return x.full()
    else:
        raise TypeError(
            f'Argument `x` must be an array-like object but has type {obj_type_str(x)}.'
        )


def to_qutip(x: Tensor, dims: tuple[int, ...] | None = None) -> Qobj | list[Qobj]:
    r"""Convert a PyTorch tensor to a QuTiP quantum object (or a list of QuTiP quantum
    object if the tensor is batched).

    Args:
        x: PyTorch tensor.
        dims _(tuple of ints)_: Dimensions of each subsystem in a composite system
            Hilbert space tensor product, defaults to `None` (a single system with the
            same dimension as `x`).

    Returns:
        QuTiP quantum object or list of QuTiP quantum object.

    Examples:
        >>> psi = dq.fock(3, 1)
        >>> psi
        tensor([[0.+0.j],
                [1.+0.j],
                [0.+0.j]])
        >>> dq.to_qutip(psi)
        Quantum object: dims = [[3], [1]], shape = (3, 1), type = ket
        Qobj data =
        [[0.]
         [1.]
         [0.]]

        For a batched tensor:
        >>> rhos = torch.stack([dq.coherent_dm(16, i) for i in range(5)])
        >>> rhos.shape
        torch.Size([5, 16, 16])
        >>> len(dq.to_qutip(rhos))
        5

        Note that the tensor product structure is not inferred automatically, it must be
        specified with the `dims` argument:
        >>> I = dq.eye(3, 2)
        >>> dq.to_qutip(I)
        Quantum object: dims = [[6], [6]], shape = (6, 6), type = oper, isherm = True
        Qobj data =
        [[1. 0. 0. 0. 0. 0.]
         [0. 1. 0. 0. 0. 0.]
         [0. 0. 1. 0. 0. 0.]
         [0. 0. 0. 1. 0. 0.]
         [0. 0. 0. 0. 1. 0.]
         [0. 0. 0. 0. 0. 1.]]
        >>> dq.to_qutip(I, (3, 2))
        Quantum object: dims = [[3, 2], [3, 2]], shape = (6, 6), type = oper, isherm = True
        Qobj data =
        [[1. 0. 0. 0. 0. 0.]
         [0. 1. 0. 0. 0. 0.]
         [0. 0. 1. 0. 0. 0.]
         [0. 0. 0. 1. 0. 0.]
         [0. 0. 0. 0. 1. 0.]
         [0. 0. 0. 0. 0. 1.]]
    """  # noqa: E501
    if x.ndim > 2:
        return [to_qutip(sub_x) for sub_x in x]
    else:
        if dims is None:
            dims = [hdim(x)]
        dims = list(dims)
        if isket(x):  # [[3], [1]] or for composite systems [[3, 4], [1, 1]]
            dims = [dims, [1] * len(dims)]
        elif isbra(x):  # [[1], [3]] or for composite systems [[1, 1], [3, 4]]
            dims = [[1] * len(dims), dims]
        elif isop(x):  # [[3], [3]] or for composite systems [[3, 4], [3, 4]]
            dims = [dims, dims]
        return Qobj(x.numpy(force=True), dims=dims)


def from_qutip(
    x: Qobj,
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    r"""Convert a QuTiP quantum object to a PyTorch tensor.

    Args:
        x _(QuTiP quantum object)_: Input quantum object.
        dtype: Complex data type of the returned tensor.
        device: Device on which the returned tensor is stored.

    Returns:
        Output tensor.

    Examples:
        >>> import qutip as qt
        >>> omega = 2.0
        >>> a = qt.destroy(4)
        >>> H = omega * a.dag() * a
        >>> H
        Quantum object: dims = [[4], [4]], shape = (4, 4), type = oper, isherm = True
        Qobj data =
        [[0. 0. 0. 0.]
         [0. 2. 0. 0.]
         [0. 0. 4. 0.]
         [0. 0. 0. 6.]]
        >>> dq.from_qutip(H)
        tensor([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 2.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 4.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 6.+0.j]])
    """
    return torch.from_numpy(x.full()).to(dtype=get_cdtype(dtype), device=device)
