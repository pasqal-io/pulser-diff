from __future__ import annotations

from typing import Callable

import pytest
import torch
from metrics import ATOL_EXPV_DP, ATOL_EXPV_KRYLOV, ATOL_OPTIM, ATOL_OPTIM_COORD, ATOL_WF
from pulser import MockDevice, Pulse, Register, Sequence
from pulser.parametrized import ParamObj
from pulser.waveforms import (
    BlackmanWaveform,
    ConstantWaveform,
    CustomWaveform,
    KaiserWaveform,
    RampWaveform,
)
from pulser_simulation import QutipEmulator
from pyqtorch.utils import SolverType
from torch import Tensor

from pulser_diff.model import QuantumModel


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
    ramp_wf = RampWaveform(duration, ramp_val_start_var, ramp_val_end_var)
    blackman_wf = BlackmanWaveform(duration, blackman_area_var)
    kaiser_wf = KaiserWaveform(duration, kaiser_area_var)
    seq.add(Pulse(const_wf, ramp_wf, phase_val_var), "rydberg_global")
    seq.target("q1", "rydberg_local")
    seq.add(Pulse(blackman_wf, const_wf, 0), "rydberg_local")
    seq.add(Pulse(kaiser_wf, ramp_wf, 0), "rydberg_global")
    return seq


def add_var_duration_pulses(seq: Sequence, duration: int) -> Sequence:
    # define duration variables
    dur1_var = seq.declare_variable("dur1")
    dur2_var = seq.declare_variable("dur2")

    # create and add pulses to sequence
    pulse1 = Pulse.ConstantPulse(dur1_var, 5.0, 1.0, 0.4)
    pulse2 = Pulse.ConstantPulse(dur2_var, 3.0, 1.0, 0.0)
    pulse3 = Pulse.ConstantPulse(duration, 3.0, 1.0, 0.0)
    seq.add(pulse1, "rydberg_global")
    seq.add(pulse2, "rydberg_global")
    seq.add(pulse3, "rydberg_global")
    return seq


def test_add_pulse_trainable_params(
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
    model = QuantumModel(seq, trainable_params)

    # check if all tranable params are registeredin QuantumModel
    assert set(trainable_params.keys()) == set(
        [p[0].split(".")[-1] for p in model.named_parameters()]
    )

    # check if trainable params values are correctly processed in the QuantumModel
    for name, param in model.named_parameters():
        name = name.split(".")[-1]
        assert param.data == trainable_params[name]


def test_add_duration_trainable_params(seq: Sequence, duration: int) -> None:
    seq = add_var_duration_pulses(seq, duration)

    trainable_params = {
        "dur1": torch.tensor([0.4], requires_grad=True),
        "dur2": torch.tensor([0.2], requires_grad=True),
    }
    model = QuantumModel(seq, trainable_params)

    # check if all tranable params are registeredin QuantumModel
    assert set(trainable_params.keys()) == set(
        [p[0].split(".")[-1] for p in model.named_parameters()]
    )

    # check if trainable params values are correctly processed in the QuantumModel
    for name, param in model.named_parameters():
        name = name.split(".")[-1]
        assert param.data == trainable_params[name]


def test_add_register_trainable_params(
    q0_coords: Tensor,
    q1_coords: Tensor,
) -> None:
    # create register
    reg = Register({"q0": q0_coords, "q1": q1_coords})

    # create sequence and add pulses
    seq = Sequence(reg, MockDevice)
    seq.declare_channel("rydberg_global", "rydberg_global")
    pulse = Pulse.ConstantPulse(100, 5.0, 2.0, 0.0)
    seq.add(pulse, "rydberg_global")

    trainable_params = {"q0": q0_coords, "q1": q1_coords}
    model = QuantumModel(seq, trainable_params)

    # check if all tranable params are registeredin QuantumModel
    assert set(trainable_params.keys()) == set(
        [p[0].split(".")[-1] for p in model.named_parameters()]
    )

    # check if trainable params values are correctly processed in the QuantumModel
    for name, param in model.named_parameters():
        name = name.split(".")[-1]
        assert torch.allclose(param.data, trainable_params[name])


def test_create_abstract_repr(seq: Sequence, duration: int) -> None:
    seq = add_var_duration_pulses(seq, duration)
    amp_var = seq.declare_variable("amp")
    det_var = seq.declare_variable("det")
    phase_var = seq.declare_variable("phase")
    seq.add(Pulse.ConstantPulse(duration, amp_var, det_var, phase_var), "rydberg_global")

    trainable_params = {
        "dur1": torch.tensor([0.4], requires_grad=True),
        "dur2": torch.tensor([0.2], requires_grad=True),
        "amp": torch.tensor([5.0], requires_grad=True),
        "det": torch.tensor([1.0], requires_grad=True),
        "phase": torch.tensor([0.5], requires_grad=True),
    }
    model = QuantumModel(seq, trainable_params)

    # check duration key
    assert all(["duration" in pulse for pulse in model.seq_abs_repr])
    assert model.seq_abs_repr[0]["duration"].name == "dur1"
    assert model.seq_abs_repr[0]["duration"].value is None
    assert "dur_var" in model.seq_abs_repr[2]["duration"].name
    assert model.seq_abs_repr[2]["duration"].value == duration / 1000

    # check amplitude key
    assert all(["amplitude" in pulse for pulse in model.seq_abs_repr])
    assert model.seq_abs_repr[3]["amplitude"]["value"].name == "amp"
    assert model.seq_abs_repr[3]["amplitude"]["value"].value is None
    assert "amp_var" in model.seq_abs_repr[0]["amplitude"]["value"].name
    assert model.seq_abs_repr[0]["amplitude"]["value"].value == 5.0

    # check detuning key
    assert all(["detuning" in pulse for pulse in model.seq_abs_repr])
    assert model.seq_abs_repr[3]["detuning"]["value"].name == "det"
    assert model.seq_abs_repr[3]["detuning"]["value"].value is None
    assert "det_var" in model.seq_abs_repr[0]["detuning"]["value"].name
    assert model.seq_abs_repr[0]["detuning"]["value"].value == 1.0

    # check phase key
    assert all(["phase" in pulse for pulse in model.seq_abs_repr])
    assert model.seq_abs_repr[3]["phase"].name == "phase"
    assert model.seq_abs_repr[3]["phase"].value is None
    assert "phase_var" in model.seq_abs_repr[0]["phase"].name
    assert model.seq_abs_repr[0]["phase"].value == 0.4


def test_optimizable_duration(seq: Sequence, duration: int) -> None:
    seq = add_var_duration_pulses(seq, duration)

    trainable_params = {
        "dur1": torch.tensor([0.4], requires_grad=True),
        "dur2": torch.tensor([0.2], requires_grad=True),
    }
    model = QuantumModel(seq, trainable_params)
    assert model.optimize_duration

    total_duration = model._get_total_duration(trainable_params)
    pulse_durations = [duration] + [int(val * 1000) for val in trainable_params.values()]
    assert total_duration == sum(pulse_durations) + 5


def test_check_constraints(
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

    min_vals = {name: float(torch.rand(1)) * 5.0 for name in trainable_params.keys()}
    constraints = {
        name: {"min": min_vals[name], "max": min_vals[name] + 2.0}
        for name in trainable_params.keys()
    }
    model = QuantumModel(seq, trainable_params, constraints)
    model.check_constraints()

    for name, param in model.named_parameters():
        name = name.split(".")[-1]
        assert (param.data >= constraints[name]["min"]) and (param.data <= constraints[name]["max"])


def test_pass_unparametrized_seq(seq: Sequence) -> None:
    pulse = Pulse.ConstantPulse(100, 5.0, 2.0, 0.0)
    seq.add(pulse, "rydberg_global")

    model = QuantumModel(seq)
    assert model.built_seq == seq


def test_create_discretized_seq(seq: Sequence, duration: int) -> None:
    seq = add_var_duration_pulses(seq, duration)

    trainable_params = {
        "dur1": torch.tensor([0.4], requires_grad=True),
        "dur2": torch.tensor([0.2], requires_grad=True),
    }
    model = QuantumModel(seq, trainable_params)

    # check if there is a correct number of pulses in discretized sequence
    assert (
        len(model._seq_opt._to_build_calls)
        == int(trainable_params["dur1"] * 1000)
        + int(trainable_params["dur2"] * 1000)
        + duration
        + 5
    )

    # check if each pulse has 1 ns duration
    assert all([pulse.args[0].args[0].args[0] == 1 for pulse in model._seq_opt._to_build_calls])

    # check if amplitude, detuning and phase are parameters
    assert all(
        [
            all(isinstance(arg, ParamObj) for arg in pulse.args[0].args[:3])
            for pulse in model._seq_opt._to_build_calls
        ]
    )


def test_create_register(q0_coords: Tensor, q1_coords: Tensor) -> None:
    # create register
    reg = Register({"q0": q0_coords, "q1": q1_coords})

    # create sequence and add pulses
    seq = Sequence(reg, MockDevice)
    seq.declare_channel("rydberg_global", "rydberg_global")
    pulse = Pulse.ConstantPulse(100, 5.0, 2.0, 0.0)
    seq.add(pulse, "rydberg_global")

    trainable_params = {"q0": q0_coords, "q1": q1_coords}
    model = QuantumModel(seq, trainable_params)
    assert model._construct_register() == reg


@pytest.mark.parametrize("solver", [SolverType.DP5_SE, SolverType.KRYLOV_SE])
def test_model_wavefunction(
    solver: SolverType,
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
    seq_built = seq.build(qubits=None, **trainable_params)
    sim_qt = QutipEmulator.from_sequence(seq_built, sampling_rate=1.0)
    results_qt = sim_qt.run()

    assert torch.allclose(
        states_torch[-1],
        torch.as_tensor(results_qt.states[-1].full()),
        atol=ATOL_WF,
    )


@pytest.mark.flaky(max_runs=5)
@pytest.mark.parametrize("solver", [SolverType.DP5_SE, SolverType.KRYLOV_SE])
def test_model_expectation(
    solver: SolverType,
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
    seq_built = seq.build(qubits=None, **trainable_params)
    sim_qt = QutipEmulator.from_sequence(seq_built, sampling_rate=1.0)
    results_qt = sim_qt.run()
    exp_val_qt = torch.as_tensor(results_qt.expect([total_magnetization_qt])[0]).real

    atol = ATOL_EXPV_DP if solver == SolverType.DP5_SE else ATOL_EXPV_KRYLOV
    assert torch.allclose(exp_val_torch, exp_val_qt, atol=atol)


@pytest.mark.flaky(max_runs=5)
@pytest.mark.parametrize("solver", [SolverType.DP5_SE, SolverType.KRYLOV_SE])
def test_pulse_param_training(
    solver: SolverType,
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
    target_value = torch.tensor(-0.5, dtype=torch.float64)
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


@pytest.mark.flaky(max_runs=5)
@pytest.mark.parametrize("solver", [SolverType.DP5_SE, SolverType.KRYLOV_SE])
def test_duration_param_training(
    solver: SolverType,
    seq: Sequence,
    duration: int,
    total_magnetization_torch: Tensor,
) -> None:
    seq = add_var_duration_pulses(seq, duration)

    trainable_params = {
        "dur1": torch.tensor([0.4], requires_grad=True),
        "dur2": torch.tensor([0.2], requires_grad=True),
    }
    model = QuantumModel(seq, trainable_params, sampling_rate=1.0, solver=solver)

    # define loss function
    loss_fn = torch.nn.MSELoss()

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    epochs = 50
    target_value = torch.tensor(-0.5, dtype=torch.float64)
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


@pytest.mark.flaky(max_runs=5)
@pytest.mark.parametrize("solver", [SolverType.DP5_SE, SolverType.KRYLOV_SE])
def test_register_param_training(
    solver: SolverType,
    q0_coords: Tensor,
    q1_coords: Tensor,
    total_magnetization_torch: Tensor,
) -> None:
    # create register
    reg = Register({"q0": q0_coords, "q1": q1_coords})

    # create sequence and add pulses
    seq = Sequence(reg, MockDevice)
    seq.declare_channel("rydberg_global", "rydberg_global")
    pulse = Pulse.ConstantPulse(200, 5.0, 1.0, 0.0)
    seq.add(pulse, "rydberg_global")

    trainable_params = {"q0": q0_coords, "q1": q1_coords}
    model = QuantumModel(seq, trainable_params, sampling_rate=1.0, solver=solver)

    # define loss function
    loss_fn = torch.nn.MSELoss()

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

    epochs = 50
    target_value = torch.tensor(-1.089, dtype=torch.float64)
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

        if torch.sqrt(loss) < ATOL_OPTIM_COORD:
            break

    assert torch.isclose(exp_val[-1].real, target_value, atol=ATOL_OPTIM_COORD)


@pytest.mark.flaky(max_runs=5)
@pytest.mark.parametrize("solver", [SolverType.DP5_SE, SolverType.KRYLOV_SE])
def test_training_with_constraints(
    solver: SolverType,
    seq: Sequence,
    const_val: Tensor,
    blackman_area: Tensor,
    total_magnetization_torch: Tensor,
) -> None:
    # declare sequence variables
    omega_param = seq.declare_variable("omega")
    area_param = seq.declare_variable("area")

    # create pulses
    pulse_const = Pulse.ConstantPulse(1000, omega_param, 0.0, 0.0)
    amp_wf = BlackmanWaveform(800, area_param)
    det_wf = RampWaveform(800, 5.0, 0.0)
    pulse_td = Pulse(amp_wf, det_wf, 0)

    # add pulses
    seq.add(pulse_const, "rydberg_global")
    seq.add(pulse_td, "rydberg_global")

    trainable_params = {"omega": const_val, "area": blackman_area}
    constraints = {"omega": {"min": 4.5, "max": 5.5}}
    model = QuantumModel(
        seq, trainable_params, constraints=constraints, sampling_rate=1.0, solver=solver
    )

    # define loss function
    loss_fn = torch.nn.MSELoss()

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    epochs = 50
    target_value = torch.tensor(-0.5, dtype=torch.float64)
    for t in range(epochs):
        # calculate current function value and loss
        _, exp_val = model.expectation(total_magnetization_torch)
        loss = loss_fn(exp_val.real[-1], target_value)

        # backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # enforce constraints on optimizable parameters
        model.check_constraints()

        # rebuild sequence with updated parameter values
        model.update_sequence()

        if torch.sqrt(loss) < ATOL_OPTIM:
            break

    assert torch.isclose(exp_val[-1].real, target_value, atol=ATOL_OPTIM)

    model_params = {name.split(".")[-1]: value.data for name, value in model.named_parameters()}
    assert (model_params["omega"] >= 4.5) and (model_params["omega"] <= 5.5)


@pytest.mark.flaky(max_runs=5)
@pytest.mark.parametrize("solver", [SolverType.DP5_SE, SolverType.KRYLOV_SE])
def test_custom_waveform_training(
    solver: SolverType,
    seq: Sequence,
    const_val: Tensor,
    blackman_area: Tensor,
    custom_wf_func: Callable,
    custom_pulse_duration: int,
    total_magnetization_torch: Tensor,
) -> None:
    # declare sequence variables
    omega_param = seq.declare_variable("omega")
    area_param = seq.declare_variable("area")

    # create pulses with predefined shapes
    pulse_const = Pulse.ConstantPulse(1000, omega_param, 0.0, 0.0)
    amp_wf = BlackmanWaveform(800, area_param)
    det_wf = RampWaveform(800, 5.0, 0.0)
    pulse_td = Pulse(amp_wf, det_wf, 0)

    # create custom-shaped pulse
    omega_custom_param = seq.declare_variable("omega_custom", size=custom_pulse_duration)
    cust_amp = CustomWaveform(omega_custom_param)  # type: ignore [arg-type]
    cust_det = ConstantWaveform(custom_pulse_duration, 1.5)
    pulse_custom = Pulse(cust_amp, cust_det, 0.0)

    # add pulses
    seq.add(pulse_const, "rydberg_global")
    seq.add(pulse_td, "rydberg_global")
    seq.add(pulse_custom, "rydberg_global")

    param1 = torch.tensor(6.0, requires_grad=True)
    param2 = torch.tensor(2.0, requires_grad=True)

    # create quantum model from sequence
    trainable_params = {
        "omega": const_val,
        "area": blackman_area,
        "omega_custom": ((param1, param2), custom_wf_func),
    }
    model = QuantumModel(seq, trainable_params, sampling_rate=1.0, solver=solver)  # type: ignore [arg-type]

    # define loss function
    loss_fn = torch.nn.MSELoss()

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

    epochs = 50
    target_value = torch.tensor(-0.5, dtype=torch.float64)
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
