from __future__ import annotations

import torch
from metrics import ATOL_ENV
from pulser.parametrized import Variable

from pulser_diff.waveform_funcs import constant_waveform


def test_constant_pulse() -> None:
    # create variables
    ti = Variable("ti", dtype=float)
    tf = Variable("tf", dtype=float)
    value = Variable("value", dtype=float)

    # define constant pulse envelope
    const_pulse = constant_waveform(ti, tf, value)  # type: ignore [arg-type]

    # generate random numerical values for variables
    ti_val = torch.rand(1)
    tf_val = ti_val + torch.rand(1) + 0.3
    value_val = torch.rand(1) * 5 + 1

    # assign numerical values to variables
    ti._assign(ti_val)
    tf._assign(tf_val)
    value._assign(value_val)

    # generate waveform
    wf = torch.tensor(
        [float(const_pulse(t).build()) for t in range(int(ti_val * 1000), int(tf_val * 1000))]
    )

    assert abs(value_val - wf.mean()) < ATOL_ENV
