from __future__ import annotations

from typing import Any

import torch
from pulser import Sequence
from pyqtorch.utils import SolverType
from torch import Tensor
from torch.nn import Module

from pulser_diff.backend import TorchEmulator
from pulser_diff.simresults import SimulationResults
from pulser_diff.utils import total_magnetization


class QuantumModel(Module):
    def __init__(
        self,
        seq: Sequence,
        trainable_params: dict[str, Tensor] | None = None,
        sampling_rate: float = 1.0,
        solver: SolverType = SolverType.DP5_SE,
        time_grad: bool = False,
        dist_grad: bool = False,
        **options: Any,
    ) -> None:
        """`torch` module wrapper for a `pulser_diff` sequence. Makes sequence pulse parameters
        trainable using standard `torch` training loop code.

        Args:
            seq (Sequence): parameterized sequence
            trainable_params (dict[str, Tensor]): dict containing tensor values for
            pulse parameters
            sampling_rate (float, optional): sampling rate for creating
            amplitude/detuning/phase samples. Defaults to 1.0.
            solver (SolverType, optional): solver to use in state vector simulation.
            Defaults to "DP5_SE".
            time_grad (bool, optional): whether to enable differentiability of model output
            with respect to time. Defaults to False.
            dist_grad (bool, optional): whether to enable differentiability of model output
            with respect to inter-qubit distances. Defaults to False.
            **options (Any, optional): optional keyword arguments passed directly to underlying
            solver.
        """

        super().__init__()

        self._seq = seq
        self.sampling_rate = sampling_rate
        self.solver = solver
        self.time_grad = time_grad
        self.dist_grad = dist_grad
        self.options = options

        # register trainable parameters
        if trainable_params is not None:
            self.trainable_params = torch.nn.ParameterDict(
                {
                    name: torch.nn.Parameter(val, requires_grad=True)
                    for name, val in trainable_params.items()
                }
            )
        else:
            self.trainable_params = torch.nn.ParameterDict(
                {
                    name: torch.nn.Parameter(torch.rand(1) + 1.0, requires_grad=True)
                    for name in seq.declared_variables
                }
            )

        # build actual sequence from parameterized one
        self.update_sequence()

        # create simulation object
        self._sim = TorchEmulator.from_sequence(self.built_seq, sampling_rate=self.sampling_rate)

    def update_sequence(self) -> None:
        self.built_seq = self._seq.build(**self.trainable_params)

    def _run(self) -> tuple[Tensor, SimulationResults]:
        self._sim = TorchEmulator.from_sequence(self.built_seq, sampling_rate=self.sampling_rate)
        results = self._sim.run(
            time_grad=self.time_grad, dist_grad=self.dist_grad, solver=self.solver, **self.options
        )
        return self._sim.evaluation_times, results

    def forward(self) -> tuple[Tensor, Tensor]:
        # run sequence
        evaluation_times, results = self._run()
        return evaluation_times, results.states

    def expectation(self, obs: Tensor | None = None) -> tuple[Tensor, Tensor]:
        # run sequence
        evaluation_times, results = self._run()

        if obs is None:
            n_qubits = len(self._seq._register._coords)  # type: ignore [union-attr]
            obs = total_magnetization(n_qubits)

        exp_val = results.expect([obs])[0]

        return evaluation_times, exp_val
