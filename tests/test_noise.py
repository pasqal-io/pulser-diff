from __future__ import annotations

import pytest
import torch
from metrics import ATOL_NOISE, RTOL_NOISE

import pulser_diff.dq as dq
from pulser_diff.pulser.waveforms import (
    BlackmanWaveform,
    ConstantWaveform,
    KaiserWaveform,
)
from pulser_diff.pulser_simulation import SimConfig
from pulser_diff.utils import expect, total_magnetization, trace, vn_entropy


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
        SimConfig(noise="dephasing", dephasing_rate=torch.tensor([1.0])),
        SimConfig(noise="depolarizing", depolarizing_rate=torch.tensor([1.0])),
        SimConfig(
            noise="eff_noise",
            eff_noise_opers=[dq.sigmax()],
            eff_noise_rates=[torch.tensor([1.0])],
        ),
    ],
)
def test_linblad_noise(dq_sim, qt_sim, cfg, amp_wf, det_wf):
    dq_results = dq_sim(amp_wf, det_wf, cfg).run(solver="dq_me")
    qt_results = qt_sim(amp_wf, det_wf, cfg).run()

    for idx, qt_state in enumerate(qt_results.states):
        dq_state_tensor = dq_results.states[idx]
        qt_state_tensor = torch.tensor(qt_state.data.toarray())
        assert torch.allclose(
            dq_state_tensor, qt_state_tensor, rtol=RTOL_NOISE, atol=ATOL_NOISE
        )


@pytest.mark.parametrize(
    "amp_wf, det_wf",
    [
        (ConstantWaveform(800, 5.0), ConstantWaveform(800, 0)),
        (BlackmanWaveform(800, 2 * torch.pi), ConstantWaveform(800, 2.5)),
    ],
)
def test_laser_waist(dq_sim, qt_sim, amp_wf, det_wf):
    cfg = SimConfig(
        noise="amplitude",
        amp_sigma=torch.tensor([0.0]),
        laser_waist=torch.tensor([100.0]),
    )

    dq_results = dq_sim(amp_wf, det_wf, cfg).run(solver="dq")
    qt_results = qt_sim(amp_wf, det_wf, cfg).run()

    for idx, qt_state in enumerate(qt_results.states):
        dq_state_tensor = dq_results.states[idx]
        qt_state_tensor = torch.tensor(qt_state.data.toarray())
        assert torch.allclose(
            dq_state_tensor, qt_state_tensor, rtol=RTOL_NOISE, atol=ATOL_NOISE
        )


@pytest.mark.flaky(max_runs=10)
@pytest.mark.parametrize(
    "cfg",
    [SimConfig(noise="doppler", runs=100), SimConfig(noise="amplitude", runs=100)],
)
def test_stochastic_noise(dq_sim, qt_sim, cfg):
    amp_wf, det_wf = ConstantWaveform(800, 5.0), ConstantWaveform(800, 0)

    dq_results = dq_sim(amp_wf, det_wf, cfg).run(solver="dq")
    qt_results = qt_sim(amp_wf, det_wf, cfg).run()

    assert dq_results.states[0].shape == (4, 4)

    obs = total_magnetization(2)

    assert dq_results.expect([obs])[0].real.size() == torch.Size([3])
    assert dq_results._basis_name == "ground-rydberg"
    assert dq_results._size == 2
    assert len(dq_results._sim_times) == 3

    assert torch.allclose(
        dq_results.states[-1].diag().real,
        torch.tensor(qt_results.states[-1].diag()),
        0.1,
        0.1,
    )

    for state in dq_results.states:
        assert torch.allclose(trace(state), torch.tensor([1.0 + 0j]))

    ent = vn_entropy(dq_results.states[-1])
    assert ent > 0


def test_expect_sparse_dm(hermitian):
    density_matrix = hermitian / trace(hermitian)
    sparse_density_matrix = density_matrix.to_sparse()
    obs = total_magnetization(4)
    assert torch.allclose(
        expect(obs, density_matrix), expect(obs, sparse_density_matrix)
    )


def test_trace(hermitian):
    sparse_H = hermitian.to_sparse()
    assert torch.allclose(hermitian.trace(), trace(sparse_H))
