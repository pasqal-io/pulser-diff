from __future__ import annotations

from typing import Callable

from pulser.parametrized import ParamObj
from pulser.parametrized.variable import VariableItem


def constant_waveform(
    ti: VariableItem,
    tf: VariableItem,
    value: VariableItem,
    edge_steepness: float = 1.0,
) -> Callable:

    def pulse_envelope(t: int) -> ParamObj:
        if ti == 0:
            fn = value * 0.5 * (1.0 + (edge_steepness * (-(t - tf))).tanh())  # type: ignore [operator]
        else:
            fn = value * (
                (0.5 * (1.0 + (edge_steepness * (t - ti)).tanh()))  # type: ignore [operator]
                + (0.5 * (1.0 + (edge_steepness * (-(t - tf))).tanh()))  # type: ignore [operator]
                - 1.0
            )
        return fn

    return pulse_envelope
