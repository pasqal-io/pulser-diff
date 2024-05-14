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
from pulser_diff.utils import expect, total_magnetization, trace


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
