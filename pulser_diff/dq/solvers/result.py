from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from torch import Tensor

from pulser_diff.dq.solver import Dopri5
from pulser_diff.dq.solvers.options import Options


def memory_bytes(x: Tensor) -> int:
    return x.element_size() * x.numel()


def memory_str(x: Tensor) -> str:
    mem = memory_bytes(x)
    if mem < 1024**2:
        return f"{mem / 1024:.2f} Kb"
    elif mem < 1024**3:
        return f"{mem / 1024**2:.2f} Mb"
    else:
        return f"{mem / 1024**3:.2f} Gb"


def tensor_str(x: Tensor) -> str:
    return f"Tensor {tuple(x.shape)} | {memory_str(x)}"


class Result:
    def __init__(
        self,
        options: Options,
        ysave: Tensor,
        tsave: Tensor,
        Esave: Tensor | None,
        Lmsave: Tensor | None = None,
        tmeas: Tensor | None = None,
    ):
        self._options = options
        self.ysave = ysave
        self.tsave = tsave
        self.Esave = Esave
        self.Lmsave = Lmsave
        self.tmeas = tmeas
        self.start_time: float | None = None
        self.end_time: float | None = None

    @property
    def options(self) -> dict[str, Any]:
        return self._options.options

    @property
    def states(self) -> Tensor:
        # alias for ysave
        return self.ysave

    @property
    def times(self) -> Tensor:
        # alias for tsave
        return self.tsave

    @property
    def expects(self) -> Tensor | None:
        # alias for Esave
        return self.Esave

    @property
    def measurements(self) -> Tensor | None:
        # alias for Lmsave
        return self.Lmsave

    @property
    def start_datetime(self) -> datetime | None:
        if self.start_time is None:
            return None
        return datetime.fromtimestamp(self.start_time)

    @property
    def end_datetime(self) -> datetime | None:
        if self.end_time is None:
            return None
        return datetime.fromtimestamp(self.end_time)

    @property
    def total_time(self) -> timedelta | None:
        if self.start_datetime is None or self.end_datetime is None:
            return None
        return self.end_datetime - self.start_datetime

    def __str__(self) -> str:
        parts = {
            "Solver": type(Dopri5).__name__,
            "Start": self.start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            "End": self.end_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            "Total time": f"{self.total_time.total_seconds():.2f} s",
            "States": tensor_str(self.states),
            "Expects": tensor_str(self.expects) if self.expects is not None else None,
            "Measurements": (
                tensor_str(self.measurements) if self.measurements is not None else None
            ),
        }
        parts = {k: v for k, v in parts.items() if v is not None}
        padding = max(len(k) for k in parts.keys()) + 1
        parts_str = "\n".join(f"{k:<{padding}}: {v}" for k, v in parts.items())
        return "==== Result ====\n" + parts_str

    def to_qutip(self) -> Result:
        raise NotImplementedError

    def to_numpy(self) -> Result:
        raise NotImplementedError

    def save(self, filename: str):
        raise NotImplementedError

    def load(self, filename: str) -> Result:
        raise NotImplementedError
