from __future__ import annotations

import pytest
import torch
from metrics import (
    ATOL_EXPV_DQ,
    ATOL_EXPV_KRYLOV,
    ATOL_OPTIM,
    ATOL_WF,
)
from torch import Tensor

from pulser_diff.model import QuantumModel
from pulser_diff.pulser import Pulse, Sequence
from pulser_diff.pulser.waveforms import (
    BlackmanWaveform,
    ConstantWaveform,
    KaiserWaveform,
    RampWaveform,
)
from pulser_diff.pulser_simulation import QutipEmulator


def add_parameterized_pulses(
    seq: Sequence,
    duration: int,
) -> Sequence:
    # define variables
    const_val_var = seq.declare_variable("const_val")
    phase_val_var = seq.declare_variable("phase_val")
    ramp_val_start_var = seq.declare_variable("ramp_val_start")
    ramp_val_end_var = seq.declare_variable("ramp_val_end")
    blackman_area_var = seq.declare_variable("blackman_area")
    kaiser_area_var = seq.declare_variable("kaiser_area")

    # create pulses and add to sequence
    const_wf = ConstantWaveform(duration, const_val_var)
    # ramp_wf = ConstantWaveform(duration, 0.0)
    ramp_wf = RampWaveform(duration, ramp_val_start_var, ramp_val_end_var)
    blackman_wf = BlackmanWaveform(duration, blackman_area_var)
    kaiser_wf = KaiserWaveform(duration, kaiser_area_var)
    seq.add(Pulse(const_wf, ramp_wf, phase_val_var), "rydberg_global")
    seq.target("q1", "rydberg_local")
    seq.add(Pulse(blackman_wf, const_wf, 0), "rydberg_local")
    seq.add(Pulse(kaiser_wf, ramp_wf, 0), "rydberg_global")
    return seq


@pytest.mark.parametrize("solver", ["dq", "krylov"])
def test_model_wavefunction(
    solver: str,
    seq: Sequence,
    duration: int,
    const_val: Tensor,
    phase_val: Tensor,
    ramp_vals: tuple[Tensor, Tensor],
    blackman_area: Tensor,
    kaiser_area: Tensor,
) -> None:
    seq = add_parameterized_pulses(
        seq,
        duration,
    )

    trainable_params = {
        "const_val": const_val,
        "phase_val": phase_val,
        "ramp_val_start": ramp_vals[0],
        "ramp_val_end": ramp_vals[1],
        "blackman_area": blackman_area,
        "kaiser_area": kaiser_area,
    }
    model = QuantumModel(seq, trainable_params, sampling_rate=1.0, solver=solver)

    # simulate with quantum model
    states_torch = model()[1]

    # simulate with qutip
    seq_built = seq.build(**trainable_params)
    sim_qt = QutipEmulator.from_sequence(seq_built, sampling_rate=1.0)
    results_qt = sim_qt.run()

    assert torch.allclose(
        states_torch[-1],
        torch.as_tensor(results_qt.states[-1].full()),
        atol=ATOL_WF,
    )


@pytest.mark.flaky(max_runs=5)
@pytest.mark.parametrize("solver", ["dq", "krylov"])
def test_model_expectation(
    solver: str,
    seq: Sequence,
    duration: int,
    const_val: Tensor,
    phase_val: Tensor,
    ramp_vals: tuple[Tensor, Tensor],
    blackman_area: Tensor,
    kaiser_area: Tensor,
    total_magnetization_torch: Tensor,
    total_magnetization_qt: Tensor,
) -> None:
    seq = add_parameterized_pulses(
        seq,
        duration,
    )

    trainable_params = {
        "const_val": const_val,
        "phase_val": phase_val,
        "ramp_val_start": ramp_vals[0],
        "ramp_val_end": ramp_vals[1],
        "blackman_area": blackman_area,
        "kaiser_area": kaiser_area,
    }
    model = QuantumModel(seq, trainable_params, sampling_rate=1.0, solver=solver)

    # simulate with quantum model
    exp_val_torch = model.expectation(total_magnetization_torch)[1].real

    # simulate with qutip
    seq_built = seq.build(**trainable_params)
    sim_qt = QutipEmulator.from_sequence(seq_built, sampling_rate=1.0)
    results_qt = sim_qt.run()
    exp_val_qt = results_qt.expect([total_magnetization_qt])[0].real

    atol = ATOL_EXPV_DQ if solver == "dq" else ATOL_EXPV_KRYLOV
    assert torch.allclose(exp_val_torch, torch.as_tensor(exp_val_qt), atol=atol)


# @pytest.mark.flaky(max_runs=5)
@pytest.mark.parametrize("solver", ["dq", "krylov"])
def test_model_training(
    solver: str,
    seq: Sequence,
    duration: int,
    const_val: Tensor,
    phase_val: Tensor,
    ramp_vals: tuple[Tensor, Tensor],
    blackman_area: Tensor,
    kaiser_area: Tensor,
    total_magnetization_torch: Tensor,
) -> None:
    seq = add_parameterized_pulses(
        seq,
        duration,
    )

    trainable_params = {
        "const_val": const_val,
        "phase_val": phase_val,
        "ramp_val_start": ramp_vals[0],
        "ramp_val_end": ramp_vals[1],
        "blackman_area": blackman_area,
        "kaiser_area": kaiser_area,
    }
    model = QuantumModel(seq, trainable_params, sampling_rate=1.0, solver=solver)

    # define loss function
    loss_fn = torch.nn.MSELoss()

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    epochs = 50
    target_value = torch.tensor(-0.5)
    for t in range(epochs):
        # calculate current function value and loss
        _, exp_val = model.expectation(total_magnetization_torch)
        loss = loss_fn(exp_val.real[-1], target_value)

        # backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # rebuild sequence with updated parameter values
        model.update_sequence()

        if torch.sqrt(loss) < ATOL_OPTIM:
            break

    assert torch.isclose(exp_val[-1].real, target_value, atol=ATOL_OPTIM)
