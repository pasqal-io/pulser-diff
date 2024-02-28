from __future__ import annotations

import pytest
import torch
from qutip import Qobj
from torch import Tensor

import pulser_diff.dq as dq
from pulser_diff.pulser import Register, Sequence
from pulser_diff.pulser.devices import MockDevice


@pytest.fixture
def reg() -> Register:
    return Register.rectangle(2, 1, spacing=8, prefix="q")


@pytest.fixture
def duration() -> int:
    return int(torch.randint(200, 800, (1,)))


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
def seq(reg: Register) -> Sequence:
    # create sequence and declare channels
    seq = Sequence(reg, MockDevice)
    seq.declare_channel("rydberg_global", "rydberg_global")
    seq.declare_channel("rydberg_local", "rydberg_local")
    return seq


@pytest.fixture
def total_magnetization_dq(reg: Register) -> Tensor:
    # create total magnetization observable with dynamiqs/torch
    n_qubits = len(reg._coords)
    total_magnetization = []
    for i in range(n_qubits):
        tprod = [dq.eye(2) for _ in range(n_qubits)]
        tprod[i] = dq.sigmaz()
        total_magnetization.append(dq.tensprod(*tprod))
    total_magnetization = sum(total_magnetization)
    return total_magnetization


@pytest.fixture
def total_magnetization_qt(total_magnetization_dq: Tensor) -> Qobj:
    # create total magnetization observable for qutip
    total_magnetization = Qobj(total_magnetization_dq.numpy())
    return total_magnetization
