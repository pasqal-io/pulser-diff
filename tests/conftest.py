from __future__ import annotations

from typing import Callable, cast

import numpy as np
import pytest
import torch
from pulser import Pulse, Register, Sequence
from pulser.devices import MockDevice
from pulser.waveforms import Waveform
from pulser_simulation import QutipEmulator
from qutip import Qobj
from torch import Tensor

from pulser_diff import SimConfig, TorchEmulator
from pulser_diff.utils import IMAT, ZMAT, kron


@pytest.fixture
def reg() -> Register:
    return Register.rectangle(2, 1, spacing=8, prefix="q")


@pytest.fixture
def duration() -> int:
    return int(torch.randint(200, 300, (1,)))


@pytest.fixture
def q0_coords() -> Tensor:
    return torch.tensor([-3.0, -1.0], requires_grad=True)


@pytest.fixture
def q1_coords() -> Tensor:
    return torch.tensor([4.0, 3.0], requires_grad=True)


@pytest.fixture
def const_val() -> Tensor:
    return torch.rand(1, requires_grad=True) * 10.0 + 4.0


@pytest.fixture
def phase_val() -> Tensor:
    return torch.rand(1, requires_grad=True) + 0.5


@pytest.fixture
def ramp_vals() -> tuple[Tensor, Tensor]:
    start = torch.rand(1, requires_grad=True) * 10.0 + 4.0
    stop = torch.rand(1, requires_grad=True) * 10.0 + 4.0
    return start, stop


@pytest.fixture
def blackman_area() -> Tensor:
    return torch.rand(1, requires_grad=True) * torch.pi + 1.0


@pytest.fixture
def kaiser_area() -> Tensor:
    return torch.rand(1, requires_grad=True) * torch.pi + 1.0


@pytest.fixture
def param1() -> Tensor:
    return torch.rand(1, requires_grad=True) * 5.0 + 2.0


@pytest.fixture
def param2() -> Tensor:
    return torch.rand(1, requires_grad=True) * 5.0 + 2.0


@pytest.fixture
def custom_pulse_duration() -> int:
    return int(torch.randint(200, 300, (1,)))


@pytest.fixture
def custom_wf_func(custom_pulse_duration: int) -> Callable:
    def custom_wf(param1: Tensor, param2: Tensor) -> Tensor:
        x = torch.arange(custom_pulse_duration) / custom_pulse_duration
        return param1 * torch.sin(torch.pi * x) * torch.exp(-param2 * x)

    return custom_wf


@pytest.fixture
def seq(reg: Register) -> Sequence:
    # create sequence and declare channels
    seq = Sequence(reg, MockDevice)
    seq.declare_channel("rydberg_global", "rydberg_global")
    seq.declare_channel("rydberg_local", "rydberg_local")
    return seq


@pytest.fixture
def total_magnetization_torch(reg: Register) -> Tensor:
    # create total magnetization observable with dynamiqs/torch
    n_qubits = len(reg._coords)
    total_magnetization = []
    for i in range(n_qubits):
        tprod = [IMAT for _ in range(n_qubits)]
        tprod[i] = ZMAT
        total_magnetization.append(kron(*tprod))
    return cast(Tensor, sum(total_magnetization))


@pytest.fixture
def total_magnetization_qt(total_magnetization_torch: Tensor) -> Qobj:
    # create total magnetization observable for qutip
    total_magnetization = Qobj(total_magnetization_torch.numpy())
    return total_magnetization


def sequence(reg: Register, amp_wf: Waveform, det_wf: Waveform) -> Sequence:
    seq = Sequence(reg, MockDevice)
    seq.declare_channel("rydberg_global", "rydberg_global")
    seq.add(Pulse(amp_wf, det_wf, 0.0), "rydberg_global")
    return seq


@pytest.fixture
def torch_sim(reg: Register) -> Callable:
    def callable_sim(amp_wf: Waveform, det_wf: Waveform, noise_config: SimConfig) -> TorchEmulator:
        sim = TorchEmulator.from_sequence(sequence(reg, amp_wf, det_wf))
        sim.set_evaluation_times(torch.linspace(0, 0.8, 3))
        sim.set_config(noise_config)
        return sim

    return callable_sim


@pytest.fixture
def qt_sim(reg: Register) -> Callable:
    def callable_sim(amp_wf: Waveform, det_wf: Waveform, noise_config: SimConfig) -> QutipEmulator:
        sim = QutipEmulator.from_sequence(sequence(reg, amp_wf, det_wf))
        sim.set_evaluation_times(np.linspace(0, 0.8, 3))
        sim.set_config(noise_config)
        return sim

    return callable_sim


@pytest.fixture
def hermitian() -> Tensor:
    vec = torch.rand(16, 1, dtype=torch.complex128)
    return vec @ vec.mH
