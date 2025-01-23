from __future__ import annotations

from typing import Callable

from pulser.parametrized import ParamObj
from pulser.parametrized.variable import Variable


def constant_waveform(
    ti: ParamObj,
    tf: ParamObj,
    value: Variable,
    edge_steepness: float = 1.0,
) -> Callable:

    def pulse_envelope(t: int) -> ParamObj:
        if ti == 0:
            fn = value * 0.5 * (1.0 + (edge_steepness * (-(t - tf * 1000))).tanh())  # type: ignore [operator]
        else:
            fn = value * (
                (0.5 * (1.0 + (edge_steepness * (t - ti * 1000)).tanh()))  # type: ignore [operator]
                + (0.5 * (1.0 + (edge_steepness * (-(t - tf * 1000))).tanh()))  # type: ignore [operator]
                - 1.0
            )
        return fn

    return pulse_envelope
