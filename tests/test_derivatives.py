from __future__ import annotations

import pytest
import torch
from scipy import interpolate
from torch import Tensor

from pulser_diff import DynamiqsEmulator
from pulser_diff.derivative import deriv_param, deriv_time
from pulser_diff.pulser import Pulse, Register, Sequence
from pulser_diff.pulser.devices import MockDevice
from pulser_diff.pulser.waveforms import (
    BlackmanWaveform,
    ConstantWaveform,
    KaiserWaveform,
    RampWaveform,
)
from pulser_diff.pulser_simulation import QutipEmulator

ATOL_WF = 1e-2
ATOL_EXPV = 1e-3
ATOL_DERIV_TIME = 2e-2
ATOL_DERIV_PARAM = 1e-4
EPS_PARAM = 1e-4


def add_pulses(
    seq: Sequence,
    duration: int,
    const_val: Tensor,
    ramp_val_start: Tensor,
    ramp_val_end: Tensor,
    blackman_area: Tensor,
    kaiser_area: Tensor,
) -> Sequence:
    # create pulses and add to sequence
    const_wf = ConstantWaveform(duration, const_val)
    ramp_wf = RampWaveform(duration, ramp_val_start, ramp_val_end)
    blackman_wf = BlackmanWaveform(duration, blackman_area)
    kaiser_wf = KaiserWaveform(duration, kaiser_area)
    seq.add(Pulse(const_wf, ramp_wf, 0), "rydberg_global")
    seq.target("q1", "rydberg_local")
    seq.add(Pulse(blackman_wf, const_wf, 0), "rydberg_local")
    seq.add(Pulse(kaiser_wf, ramp_wf, 0), "rydberg_global")
    return seq


def test_wavefunction(
    seq: Sequence,
    duration: int,
    const_val: Tensor,
    ramp_vals: tuple[Tensor, Tensor],
    blackman_area: Tensor,
    kaiser_area: Tensor,
) -> None:
    seq = add_pulses(
        seq, duration, const_val, ramp_vals[0], ramp_vals[1], blackman_area, kaiser_area
    )

    # simulate with dynamiqs
    sim_dq = DynamiqsEmulator.from_sequence(seq, sampling_rate=1.0)
    results_dq = sim_dq.run()

    # simulate with qutip
    sim_qt = QutipEmulator.from_sequence(seq, sampling_rate=1.0)
    results_qt = sim_qt.run()

    assert torch.allclose(
        results_dq.states[-1],
        torch.as_tensor(results_qt.states[-1].full()),
        atol=ATOL_WF,
    )


def test_expectation(
    seq: Sequence,
    duration: int,
    const_val: Tensor,
    ramp_vals: tuple[Tensor, Tensor],
    blackman_area: Tensor,
    kaiser_area: Tensor,
    total_magnetization_dq: Tensor,
    total_magnetization_qt: Tensor,
) -> None:
    seq = add_pulses(
        seq, duration, const_val, ramp_vals[0], ramp_vals[1], blackman_area, kaiser_area
    )

    # simulate with dynamiqs
    sim_dq = DynamiqsEmulator.from_sequence(seq, sampling_rate=1.0)
    results_dq = sim_dq.run()
    exp_val_dq = results_dq.expect([total_magnetization_dq])[0].real

    # simulate with qutip
    sim_qt = QutipEmulator.from_sequence(seq, sampling_rate=1.0)
    results_qt = sim_qt.run()
    exp_val_qt = results_qt.expect([total_magnetization_qt])[0].real

    assert torch.allclose(exp_val_dq, torch.as_tensor(exp_val_qt), atol=ATOL_EXPV)


@pytest.mark.flaky(max_runs=5)
def test_time_derivative(
    seq: Sequence,
    duration: int,
    const_val: Tensor,
    ramp_vals: tuple[Tensor, Tensor],
    blackman_area: Tensor,
    kaiser_area: Tensor,
    total_magnetization_dq: Tensor,
) -> None:
    seq = add_pulses(
        seq, duration, const_val, ramp_vals[0], ramp_vals[1], blackman_area, kaiser_area
    )

    # simulate with dynamiqs
    sim = DynamiqsEmulator.from_sequence(seq, sampling_rate=1.0)
    results = sim.run(time_grad=True)
    exp_val = results.expect([total_magnetization_dq])[0].real

    # calculate derivative with torch autograd
    eval_times = sim.evaluation_times
    pulse_endtimes = sim.endtimes
    dfdt_autograd = deriv_time(
        f=exp_val, times=eval_times, pulse_endtimes=pulse_endtimes
    )

    # calculate exact derivative with respect to time
    x = eval_times.detach().numpy()
    y = exp_val.detach().numpy()
    interp_fx = interpolate.UnivariateSpline(x, y, k=5, s=0)
    dfdt_exact = interp_fx.derivative()(x)

    assert (
        torch.abs(dfdt_autograd - torch.as_tensor(dfdt_exact)).mean() < ATOL_DERIV_TIME
    )


def test_pulse_param_derivative(
    reg: Register,
    duration: int,
    const_val: Tensor,
    ramp_vals: tuple[Tensor, Tensor],
    blackman_area: Tensor,
    kaiser_area: Tensor,
    total_magnetization_dq: Tensor,
) -> None:
    def run_sequence(
        const_val: Tensor,
        ramp_val_start: Tensor,
        ramp_val_end: Tensor,
        blackman_area: Tensor,
        kaiser_area: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # create sequence and add pulses
        seq = Sequence(reg, MockDevice)
        seq.declare_channel("rydberg_global", "rydberg_global")
        seq.declare_channel("rydberg_local", "rydberg_local")
        seq = add_pulses(
            seq,
            duration,
            const_val,
            ramp_val_start,
            ramp_val_end,
            blackman_area,
            kaiser_area,
        )

        # simulate with dynamiqs
        sim = DynamiqsEmulator.from_sequence(seq, sampling_rate=1.0)
        results = sim.run()
        exp_vals = results.expect([total_magnetization_dq])[0].real

        return exp_vals, sim.evaluation_times

    # compare autograd gradients vs finite difference gradients
    diff_params = [const_val, ramp_vals[0], ramp_vals[1], blackman_area, kaiser_area]
    exp_vals_auto, eval_times = run_sequence(*diff_params)
    for i, param in enumerate(diff_params):
        # autograd
        grad_auto = deriv_param(
            f=exp_vals_auto, x=param, times=eval_times, t=1000 * eval_times[-1]
        )

        # finite difference
        exp_vals = torch.tensor([0.0])
        for p in [1.0, -1.0]:
            diff_params_new = diff_params.copy()
            diff_params_new[i] = diff_params_new[i] + p * EPS_PARAM
            ev = run_sequence(*diff_params_new)[0]
            exp_vals += p * ev[-1]
        grad_fd = exp_vals / (2 * EPS_PARAM)

        assert torch.isclose(grad_auto, grad_fd, atol=ATOL_DERIV_PARAM)


def test_register_coords_derivative(
    duration: int,
    q0_coords: Tensor,
    q1_coords: Tensor,
    const_val: Tensor,
    ramp_vals: tuple[Tensor, Tensor],
    blackman_area: Tensor,
    kaiser_area: Tensor,
    total_magnetization_dq: Tensor,
) -> None:
    def run_sequence(q0_coords: Tensor, q1_coords: Tensor) -> Tensor:
        # create register
        reg = Register({"q0": q0_coords, "q1": q1_coords})

        # create sequence and add pulses
        seq = Sequence(reg, MockDevice)
        seq.declare_channel("rydberg_global", "rydberg_global")
        seq.declare_channel("rydberg_local", "rydberg_local")
        seq = add_pulses(
            seq,
            duration,
            const_val,
            ramp_vals[0],
            ramp_vals[1],
            blackman_area,
            kaiser_area,
        )

        # simulate with dynamiqs
        sim = DynamiqsEmulator.from_sequence(seq, sampling_rate=1.0)
        results = sim.run(dist_grad=True)
        exp_vals = results.expect([total_magnetization_dq])[0].real

        return exp_vals

    # compare autograd gradients vs finite difference gradients
    diff_params = [q0_coords, q1_coords]
    exp_vals_auto = run_sequence(*diff_params)
    for i, param in enumerate(diff_params):
        # autograd
        grad_auto = deriv_param(f=exp_vals_auto, x=param)

        # finite difference
        exp_vals = torch.tensor([0.0])
        for p in [1.0, -1.0]:
            diff_params_new = diff_params.copy()
            diff_params_new[i] = diff_params_new[i] + p * EPS_PARAM
            ev = run_sequence(*diff_params_new)
            exp_vals += p * ev[-1]
        grad_fd = exp_vals / (2 * EPS_PARAM)

        assert torch.isclose(grad_auto.sum(), grad_fd, atol=ATOL_DERIV_PARAM)
