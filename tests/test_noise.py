from __future__ import annotations

from typing import Callable

import pytest
import torch
from metrics import ATOL_NOISE, RTOL_NOISE
from pulser import Pulse, Register, Sequence
from pulser.devices import MockDevice
from pulser.waveforms import (
    BlackmanWaveform,
    ConstantWaveform,
    KaiserWaveform,
    Waveform,
)
from pulser_simulation import QutipEmulator

# from pulser_simulation import SimConfig as SC
from pyqtorch.utils import SolverType
from torch import Tensor

from pulser_diff import TorchEmulator
from pulser_diff.simconfig import SimConfig
from pulser_diff.utils import XMAT, expect, total_magnetization, trace, vn_entropy


@pytest.mark.parametrize(
    "amp_wf, det_wf",
    [
        (ConstantWaveform(800, 5.0), ConstantWaveform(800, 0)),
        (BlackmanWaveform(800, 2 * torch.pi), ConstantWaveform(800, 2.5)),
        (KaiserWaveform(800, 2 * torch.pi), ConstantWaveform(800, 5.0)),
    ],
)
@pytest.mark.parametrize(
    "cfg",
    [
        SimConfig(noise="dephasing", dephasing_rate=1.0),
        SimConfig(noise="depolarizing", depolarizing_rate=1.0),
        SimConfig(
            noise="eff_noise",
            eff_noise_opers=[XMAT],
            eff_noise_rates=[1.0],
        ),
    ],
)
def test_linblad_noise(
    torch_sim: Callable,
    qt_sim: Callable,
    cfg: SimConfig,
    amp_wf: Waveform,
    det_wf: Waveform,
) -> None:
    torch_results = torch_sim(amp_wf, det_wf, cfg).run(solver=SolverType.DP5_ME)
    qt_results = qt_sim(amp_wf, det_wf, cfg.to_pulser()).run()

    for idx, qt_state in enumerate(qt_results.states):
        torch_state_tensor = torch_results.states[idx].squeeze(-1)
        qt_state_tensor = torch.tensor(qt_state.data.toarray())
        assert torch.allclose(torch_state_tensor, qt_state_tensor, rtol=RTOL_NOISE, atol=ATOL_NOISE)


@pytest.mark.parametrize(
    "amp_wf, det_wf",
    [
        (ConstantWaveform(800, 5.0), ConstantWaveform(800, 0)),
        (BlackmanWaveform(800, 2 * torch.pi), ConstantWaveform(800, 2.5)),
    ],
)
def test_laser_waist(
    torch_sim: Callable, qt_sim: Callable, amp_wf: Waveform, det_wf: Waveform
) -> None:
    cfg = SimConfig(
        noise="amplitude",
        amp_sigma=0.0,
        laser_waist=100.0,
    )

    torch_results = torch_sim(amp_wf, det_wf, cfg).run(solver=SolverType.DP5_SE)
    qt_results = qt_sim(amp_wf, det_wf, cfg).run()

    for idx, qt_state in enumerate(qt_results.states):
        torch_state_tensor = torch_results.states[idx]
        qt_state_tensor = torch.tensor(qt_state.data.toarray())
        assert torch.allclose(torch_state_tensor, qt_state_tensor, rtol=RTOL_NOISE, atol=ATOL_NOISE)


@pytest.mark.flaky(max_runs=10)
@pytest.mark.parametrize(
    "cfg",
    [SimConfig(noise="doppler", runs=100), SimConfig(noise="amplitude", runs=100)],
)
def test_stochastic_noise(torch_sim: Callable, qt_sim: Callable, cfg: SimConfig) -> None:
    amp_wf, det_wf = ConstantWaveform(800, 5.0), ConstantWaveform(800, 0)

    qt_results = qt_sim(amp_wf, det_wf, cfg).run()
    torch_results = torch_sim(amp_wf, det_wf, cfg).run(solver=SolverType.DP5_SE)

    assert torch_results.states[0].shape == (4, 4)

    obs = total_magnetization(2)

    assert torch_results.expect([obs])[0].real.size() == torch.Size([3])
    assert torch_results._basis_name == "ground-rydberg"
    assert torch_results._size == 2
    assert len(torch_results._sim_times) == 3

    assert torch.allclose(
        torch_results.states[-1].diag().real,
        torch.tensor(qt_results.states[-1].diag()),
        0.1,
        0.1,
    )

    for state in torch_results.states:
        assert torch.allclose(trace(state), torch.tensor([1.0 + 0j], dtype=torch.complex128))

    ent = vn_entropy(torch_results.states[-1])
    assert ent > 0


def test_expect_sparse_dm(hermitian: Tensor) -> None:
    density_matrix = hermitian / trace(hermitian)
    sparse_density_matrix = density_matrix.to_sparse()
    obs = total_magnetization(4)
    assert torch.allclose(expect(obs, density_matrix), expect(obs, sparse_density_matrix))


def test_trace(hermitian: Tensor) -> None:
    sparse_H = hermitian.to_sparse()
    assert torch.allclose(hermitian.trace(), trace(sparse_H))


def test_1qbit() -> None:
    reg = Register({"q0": torch.tensor([0.0, 0.0])})
    seq = Sequence(reg, MockDevice)
    seq.declare_channel("rydberg_global", "rydberg_global")
    seq.add(Pulse.ConstantPulse(100, 2.0, 1.0, 0.0), "rydberg_global")

    sim_dq = TorchEmulator.from_sequence(seq)
    sim_qt = QutipEmulator.from_sequence(seq)

    res_dq = sim_dq.run()
    res_qt = sim_qt.run()

    assert torch.allclose(
        res_dq.states[-1],
        torch.tensor(res_qt.states[-1].data.toarray()),
        ATOL_NOISE,
        RTOL_NOISE,
    )
