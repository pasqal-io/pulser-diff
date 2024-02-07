from __future__ import annotations

from abc import abstractmethod, abstractproperty

import torch
from torch import Tensor

from pulser_diff.dq.utils.utils import cache, obj_type_str, type_str
from pulser_diff.dq.utils.tensor_types import (
    ArrayLike,
    Number,
    get_cdtype,
    to_device,
)

__all__ = ["totime"]


def totime(
    x: (
        ArrayLike
        | callable[[float], Tensor]
        | tuple[ArrayLike, ArrayLike, ArrayLike]
        | tuple[callable[[float], Tensor], ArrayLike]
    ),
    *,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
) -> TimeTensor:
    dtype = dtype or get_cdtype(dtype)  # assume complex by default
    device = to_device(device)

    if callable(x):
        return _factory_callable(x, dtype=dtype, device=device)
    else:
        raise TypeError(
            "For time-dependent tensors, argument `x` must be one of 4 types: (1)"
            " ArrayLike; (2) 2-tuple with type (function, ArrayLike) where function"
            " has signature (t: float) -> Tensor; (3) 3-tuple with type (ArrayLike,"
            " ArrayLike, ArrayLike); (4) function with signature (t: float) -> Tensor."
            f" The provided `x` has type {obj_type_str(x)}."
        )


def _factory_callable(
    x: callable[[float], Tensor], *, dtype: torch.dtype, device: torch.device
) -> CallableTimeTensor:
    f0 = x(0.0)

    # check type, dtype and device match
    if not isinstance(f0, Tensor):
        raise TypeError(
            f"The time-dependent operator must be a {type_str(Tensor)}, but has"
            f" type {obj_type_str(f0)}. The provided function must return a tensor,"
            " to avoid costly type conversion at each time solver step."
        )
    elif f0.dtype != dtype:
        raise TypeError(
            f"The time-dependent operator must have dtype `{dtype}`, but has dtype"
            f" `{f0.dtype}`. The provided function must return a tensor with the"
            " same `dtype` as provided to the solver, to avoid costly dtype"
            " conversion at each solver time step."
        )
    elif f0.device != device:
        raise TypeError(
            f"The time-dependent operator must be on device `{device}`, but is on"
            f" device `{f0.device}`. The provided function must return a tensor on"
            " the same device as provided to the solver, to avoid costly device"
            " transfer at each solver time step."
        )

    return CallableTimeTensor(x, f0)


class TimeTensor:
    # Subclasses should implement:
    # - the properties: dtype, device, shape
    # - the methods: __call__, view, adjoint, __neg__, __mul__, __add__

    # Special care should be taken when implementing `__call__` for caching to work
    # properly. The `@cache` decorator checks the tensor `__hash__`, which is
    # implemented as its address in memory. Thus, when two consecutive calls to a
    # `TimeTensor` should return a tensor with the same values, these two tensors must
    # not only be equal, they should be the same object in memory.

    # Note that a subclass implementation of `__add__` only need to support addition
    # with `Tensor`, `ConstantTimeTensor` and the subclass type itself.

    @abstractproperty
    def dtype(self) -> torch.dtype:
        """Data type."""
        pass

    @abstractproperty
    def device(self) -> torch.device:
        """Device."""
        pass

    @abstractproperty
    def shape(self) -> torch.Size:
        """Shape."""
        pass

    @abstractmethod
    def __call__(self, t: float) -> Tensor:
        """Evaluate at a given time"""
        pass

    @abstractmethod
    def view(self, *shape: int) -> TimeTensor:
        """Returns a new tensor with the same data but of a different shape."""
        pass

    @abstractmethod
    def adjoint(self) -> TimeTensor:
        pass

    @property
    def mH(self) -> TimeTensor:
        return self.adjoint()

    @abstractmethod
    def __neg__(self) -> TimeTensor:
        pass

    @abstractmethod
    def __mul__(self, other: Number | Tensor) -> TimeTensor:
        pass

    def __rmul__(self, other: Number | Tensor) -> TimeTensor:
        return self * other

    @abstractmethod
    def __add__(self, other: Tensor | TimeTensor) -> TimeTensor:
        pass

    def __radd__(self, other: Tensor | TimeTensor) -> TimeTensor:
        return self + other

    def __sub__(self, other: Tensor | TimeTensor) -> TimeTensor:
        return self + (-other)

    def __rsub__(self, other: Tensor | TimeTensor) -> TimeTensor:
        return other + (-self)

    def __repr__(self) -> str:
        return str(type(self).__name__)

    def __str__(self) -> str:
        return self.__repr__()

    def size(self, dim: int) -> int:
        """Size along a given dimension."""
        return self.shape[dim]

    def dim(self) -> int:
        """Get the number of dimensions."""
        return len(self.shape)

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self.dim()


class CallableTimeTensor(TimeTensor):
    def __init__(self, f: callable[[float], Tensor], f0: Tensor):
        # f0 carries all the transformation on the shape
        self.f = f
        self.f0 = f0

    @property
    def dtype(self) -> torch.dtype:
        return self.f0.dtype

    @property
    def device(self) -> torch.device:
        return self.f0.device

    @property
    def shape(self) -> torch.Size:
        return self.f0.shape

    @cache
    def __call__(self, t: float) -> Tensor:
        # cached if called twice with the same time, otherwise we recompute `f(t)`
        return self.f(t)

    def view(self) -> TimeTensor:
        f = self.f
        f0 = self.f0.unsqueeze(0)
        return CallableTimeTensor(f, f0)

    @abstractmethod
    def adjoint(self) -> TimeTensor:
        def f(t):
            return self.f(t).adjoint()

        f0 = self.f0.adjoint()
        return CallableTimeTensor(f, f0)

    def __neg__(self) -> TimeTensor:
        def f(t):
            return -self.f(t)

        f0 = -self.f0
        return CallableTimeTensor(f, f0)

    def __mul__(self, other: Number | Tensor) -> TimeTensor:
        def f(t):
            return self.f(t) * other

        f0 = self.f0 * other
        return CallableTimeTensor(f, f0)

    def __add__(self, other: Tensor | TimeTensor) -> TimeTensor:
        if isinstance(other, Tensor):

            def f(t):
                return self.f(t) + other

            f0 = self.f0 + other
            return CallableTimeTensor(f, f0)
        elif isinstance(other, CallableTimeTensor):

            def f(t):
                return self.f(t) + other.f(t)

            f0 = self.f0 + other.f0
            return CallableTimeTensor(f, f0)
        else:
            return NotImplemented
