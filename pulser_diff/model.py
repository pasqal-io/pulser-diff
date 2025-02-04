from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Union, cast
from uuid import uuid4

import torch
from pulser import Pulse, Register, Sequence
from pulser.channels import Rydberg
from pulser.devices import VirtualDevice
from pulser.parametrized import ParamObj
from pulser.parametrized.variable import VariableItem
from pyqtorch.utils import SolverType
from torch import Tensor
from torch.nn import Module, ParameterDict

from pulser_diff.backend import TorchEmulator
from pulser_diff.simresults import SimulationResults
from pulser_diff.utils import basis_state, total_magnetization
from pulser_diff.waveform_funcs import constant_waveform

MockDevice = VirtualDevice(
    name="MockDevice",
    dimensions=2,
    rydberg_level=60,
    channel_objects=(Rydberg.Global(125.66370614359172, 12.566370614359172, max_duration=None),),
)


DEFAULT_CONSTRAINTS = {
    "duration": {"min": 1, "max": 1000},
    "amplitude": {"min": 0.0, "max": 100.0},
    "detuning": {"min": -100.0, "max": 100.0},
    "phase": {"min": -2 * float(torch.pi), "max": 2 * float(torch.pi)},
}


@dataclass()
class Parameter:
    name: str
    value: Union[int, float, Tensor, None] = None
    trainable: bool = False
    type: str = ""


class QuantumModel(Module):
    def __init__(
        self,
        seq: Sequence,
        trainable_param_values: dict[str, Tensor],
        constraints: dict[str, Any] = {},
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

        self.register = seq.register
        self.constraints = constraints
        self.device = seq.device
        self.sampling_rate = sampling_rate
        self.solver = solver
        self.time_grad = time_grad
        self.dist_grad = dist_grad
        self.options = options

        # get abstract representation of initial sequence
        self.seq_abs_repr, self.optimize_duration, self.params = self._get_abstract_repr(seq)

        if self.optimize_duration:
            # we need to build a new sequence to accomodate pulse duration optimization
            total_duration = self._get_total_duration(trainable_param_values)
            self._seq_opt = self._create_opt_sequence(total_duration)

            # register trainable parameters
            self.param_values = ParameterDict()
            for name, val in self.params.items():
                self.param_values[name] = (
                    torch.nn.Parameter(
                        cast(
                            Tensor,
                            val.value if val.value is not None else trainable_param_values[name],
                        ),
                        requires_grad=True,
                    )
                    if val.trainable
                    else val.value
                )
        else:
            # initial sequence can be used directly for optimization
            self._seq_opt = seq
            self.param_values = ParameterDict(
                {
                    name: torch.nn.Parameter(val, requires_grad=True)
                    for name, val in trainable_param_values.items()
                }
            )

        # build actual sequence from parameterized one
        self.update_sequence()

    def _create_opt_sequence(self, total_duration: int) -> Sequence:
        # create internal sequence and declare channels
        seq_opt = Sequence(self.register, self.device)
        seq_opt.declare_channel("rydberg_global", "rydberg_global")

        # construct envelope functions
        envelopes = self._create_envelopes(seq_opt)

        for t in range(int(total_duration)):
            # calculate amplitude value
            amp_val = sum([fn(t) for fn in envelopes["amplitude"]])

            # calculate detuning value
            det_val = sum([fn(t) for fn in envelopes["detuning"]])

            # calculate phase value
            phase_val = sum([fn(t) for fn in envelopes["phase"]])

            # create shortest possible pulse with given possibly parameterized values
            pulse = Pulse.ConstantPulse(1, amp_val, det_val, phase_val)
            seq_opt.add(pulse, "rydberg_global")

        return seq_opt

    def _get_abstract_repr(self, seq: Sequence) -> tuple[list[dict], bool, dict[str, Parameter]]:
        pulse_list = []
        all_calls = [call for call in seq._calls + seq._to_build_calls if call.name == "add"]
        for i, call in enumerate(all_calls):
            pulse = call.args[0]
            d = {
                k: v._to_abstract_repr() if hasattr(v, "_to_abstract_repr") else v
                for k, v in pulse._to_abstract_repr().items()
            }
            pulse_list.append(d)

        optimize_duration = False
        for pulse in pulse_list:
            duration = pulse["amplitude"]["duration"]

            # get the name of duration variable
            if isinstance(duration, VariableItem):
                optimize_duration = True
                break

        params = {}
        for pulse in pulse_list:
            pulse["duration"] = pulse["amplitude"]["duration"]
            pulse["amplitude"].pop("duration")
            pulse["detuning"].pop("duration")

            if optimize_duration:
                # convert duration to Parameter object
                if isinstance(pulse["duration"], VariableItem):
                    pulse["duration"] = Parameter(
                        pulse["duration"].var.name, trainable=True, type="duration"
                    )
                else:
                    pulse["duration"] = Parameter(
                        f"dur_var_{uuid4()}",
                        value=pulse["duration"] / 1000,
                        trainable=False,
                        type="duration",
                    )
                params[pulse["duration"].name] = pulse["duration"]

            # convert amplitude to Parameter object
            if pulse["amplitude"]["kind"] == "constant":
                if isinstance(pulse["amplitude"]["value"], VariableItem):
                    pulse["amplitude"]["value"] = Parameter(
                        pulse["amplitude"]["value"].var.name, trainable=True, type="amplitude"
                    )
                else:
                    pulse["amplitude"]["value"] = Parameter(
                        f"amp_var_{uuid4()}",
                        value=pulse["amplitude"]["value"],
                        trainable=False,
                        type="amplitude",
                    )
                params[pulse["amplitude"]["value"].name] = pulse["amplitude"]["value"]

            # convert detuning to Parameter object
            if pulse["detuning"]["kind"] == "constant":
                if isinstance(pulse["detuning"]["value"], VariableItem):
                    pulse["detuning"]["value"] = Parameter(
                        pulse["detuning"]["value"].var.name, trainable=True, type="detuning"
                    )
                else:
                    pulse["detuning"]["value"] = Parameter(
                        f"det_var_{uuid4()}",
                        value=pulse["detuning"]["value"],
                        trainable=False,
                        type="detuning",
                    )
                params[pulse["detuning"]["value"].name] = pulse["detuning"]["value"]

            # convert phase to Parameter object
            if isinstance(pulse["phase"], dict):
                pulse["phase"] = Parameter(pulse["phase"]["lhs"].name, trainable=True, type="phase")
            else:
                pulse["phase"] = Parameter(
                    f"phase_var_{uuid4()}", value=pulse["phase"], trainable=False, type="phase"
                )
            params[pulse["phase"].name] = pulse["phase"]

        return pulse_list, optimize_duration, params

    def _get_total_duration(self, trainable_param_values: dict[str, Tensor] | ParameterDict) -> int:
        total_duration = 0
        for pulse in self.seq_abs_repr:
            duration = pulse["duration"]

            # get the name of duration variable
            var_name = duration.name
            if duration.trainable:
                value = trainable_param_values[var_name]
            else:
                value = duration.value

            # increase total duration
            total_duration += int(value * 1000)

        return total_duration

    def _create_envelopes(self, seq_opt: Sequence) -> dict[str, list]:
        ti = 0
        envelope_funcs: dict = {"amplitude": [], "detuning": [], "phase": []}
        for pulse in self.seq_abs_repr:
            duration = pulse["duration"]

            # declare duration variable
            var_name = duration.name
            if var_name not in seq_opt.declared_variables:
                duration_var = seq_opt.declare_variable(var_name).var
            else:
                duration_var = seq_opt.declared_variables[var_name]

            # end of current pulse
            tf = ti + duration_var

            # get amplitude/detuning envelope function for current pulse
            for s in ["amplitude", "detuning"]:
                wf_type = pulse[s]["kind"]
                if wf_type == "constant":
                    # declare new variable
                    var_name = pulse[s]["value"].name
                    if var_name not in seq_opt.declared_variables:
                        var = seq_opt.declare_variable(var_name).var
                    else:
                        var = seq_opt.declared_variables[var_name]
                    fn = constant_waveform(cast(ParamObj, ti), tf, var)
                else:
                    raise NotImplementedError(
                        f"{s} waveform type {wf_type} currently not supported."
                    )
                envelope_funcs[s].append(fn)

            # get phase envelope function for current pulse
            var_name = pulse["phase"].name
            if var_name not in seq_opt.declared_variables:
                var = seq_opt.declare_variable(var_name).var
            else:
                var = seq_opt.declared_variables[var_name]
            envelope_funcs["phase"].append(constant_waveform(cast(ParamObj, ti), tf, var))

            # shift ti to end of pulse
            ti = tf  # type: ignore [assignment]

        return envelope_funcs

    def check_boundaries(self) -> None:
        for n, p in self.named_parameters():
            name = n.split(".")[-1]
            if name in self.constraints:
                p.data.clamp_(self.constraints[name]["min"], self.constraints[name]["max"])

    def update_sequence(self, reconstruct_sequence: bool = False) -> None:
        if reconstruct_sequence and self.optimize_duration:
            total_duration = self._get_total_duration(self.param_values)
            self._seq_opt = self._create_opt_sequence(total_duration)

        # construct final sequence with actual values
        self.built_seq = self._seq_opt.build(**self.param_values)

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
            n_qubits = len(self.register._coords)
            obs = total_magnetization(n_qubits)

        exp_val = results.expect([obs])[0]

        return evaluation_times, exp_val


class QGateModel(Module):
    def __init__(
        self,
        seq: Sequence,
        trainable_params: dict[str, Tensor] | None = None,
        solver: SolverType = SolverType.DP5_SE,
        sampling_rate: float = 1.0,
        time_grad: bool = False,
        dist_grad: bool = False,
    ) -> None:
        """`torch` module wrapper for a `pulser_diff` sequence. Makes sequence pulse parameters
        trainable using standard `torch` training loop code.

        Args:
            seq (Sequence): parameterized sequence
            trainable_params (dict[str, Tensor]): dict containing tensor values for
            pulse parameters
            sampling_rate (float, optional): sampling rate for creating
            amplitude/detuning/phase samples. Defaults to 1.0.
            solver (str, optional): solver to use in state vector simulation.
            Defaults to "krylov".
            time_grad (bool, optional): whether to enable differentiability of model output
            with respect to time. Defaults to False.
            dist_grad (bool, optional): whether to enable differentiability of model output
            with respect to inter-qubit distances. Defaults to False.
        """

        super().__init__()

        self._seq = seq
        self.solver = solver
        self.sampling_rate = sampling_rate
        self.time_grad = time_grad
        self.dist_grad = dist_grad
        self.t: list[int] = []

        self.n_qubits = len(self._seq.qubit_info)

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

    def update_sequence(self) -> None:
        self.built_seq = self._seq.build(**self.trainable_params)

    def forward(self) -> Tensor:
        # run sequence for different input state

        self._sim = TorchEmulator.from_sequence(
            self.built_seq,
            evaluation_times=0.05,
            sampling_rate=self.sampling_rate,
            with_modulation=False,
        )

        res_list = []
        n_dim = 2**self.n_qubits
        for k in range(n_dim):
            self._sim.set_initial_state(basis_state(n_dim, k))
            results = self._sim.run(solver=self.solver)
            res_list.append(results.states[-1])

        return torch.hstack(res_list)


class QGATEM2(Module):
    def __init__(
        self,
        register: Register,
        seq_length: int,
        trainable_params: dict[str, Tensor],
        pulse_fn: Callable,
        solver: SolverType = SolverType.DP5_SE,
    ) -> None:
        """`torch` module wrapper for a `pulser_diff` sequence. Makes sequence pulse parameters
        trainable using standard `torch` training loop code.

        Args:
            reg (Register) : the quantum register
            seq_length (int) : the length of the sequence in ns
            trainable_params (dict[str, Tensor]): dict containing tensor values for
            pulse parameters
            pulse_fn: the function that build the sequence from the primary parameters
        """

        super().__init__()

        self.solver = solver
        self.register = register
        self.n_qubits = len(register.qubits)
        self.seq_length = seq_length

        self._create_sequence()
        self.pulse_fn = pulse_fn
        self.set_params(trainable_params)

        # build actual sequence from parameterized one
        self.update_sequence()

    def set_params(self, trainable_params: dict) -> None:
        self.trainable_params = torch.nn.ParameterDict(
            {
                name: torch.nn.Parameter(val, requires_grad=True)
                for name, val in trainable_params.items()
            }
        )

    def _create_sequence(self) -> None:
        seq = Sequence(self.register, MockDevice)
        seq.declare_channel("rydberg_global", "rydberg_global")
        amp = seq.declare_variable("amp", size=self.seq_length)
        det = seq.declare_variable("det", size=self.seq_length)
        phi = seq.declare_variable("phi", size=self.seq_length)

        for k in range(self.seq_length):
            seq.add(Pulse.ConstantPulse(1, amp[k], det[k], phi[k]), "rydberg_global")

        self._seq = seq

    def update_sequence(self) -> None:
        self.pulse_values = self.pulse_fn(self.trainable_params)
        self.built_seq = self._seq.build(**self.pulse_values)

    def forward(self) -> Tensor:
        # run sequence for different input state

        self._sim = TorchEmulator.from_sequence(
            self.built_seq,
            evaluation_times=0.05,
            with_modulation=False,
        )

        self._sim.set_initial_state(torch.eye(2**self.n_qubits))
        res = self._sim.run(solver=self.solver)
        return res.states[-1]


class StatePreparationModel(Module):
    def __init__(
        self,
        register: Register,
        seq_length: int,
        trainable_params: dict[str, Tensor],
        pulse_fn: Callable,
        solver: SolverType = SolverType.DP5_SE,
    ) -> None:
        """`torch` module wrapper for a `pulser_diff` sequence. Makes sequence pulse parameters
        trainable using standard `torch` training loop code.

        Args:
            reg (Register) : the quantum register
            seq_length (int) : the length of the sequence in ns
            trainable_params (dict[str, Tensor]): dict containing tensor values for
            pulse parameters
            pulse_fn: the function that build the sequence from the primary parameters
        """

        super().__init__()

        self.solver = solver
        self.register = register
        self.n_qubits = len(register.qubits)
        self.seq_length = seq_length

        self._create_sequence()
        self.pulse_fn = pulse_fn
        self.set_params(trainable_params)

        # build actual sequence from parameterized one
        self.update_sequence()

    def set_params(self, trainable_params: dict) -> None:
        self.trainable_params = torch.nn.ParameterDict(
            {
                name: torch.nn.Parameter(val, requires_grad=True)
                for name, val in trainable_params.items()
            }
        )

    def _create_sequence(self) -> None:
        seq = Sequence(self.register, MockDevice)
        seq.declare_channel("rydberg_global", "rydberg_global")
        amp = seq.declare_variable("amp", size=self.seq_length)
        det = seq.declare_variable("det", size=self.seq_length)
        phi = seq.declare_variable("phi", size=self.seq_length)

        for k in range(self.seq_length):
            seq.add(Pulse.ConstantPulse(1, amp[k], det[k], phi[k]), "rydberg_global")

        self._seq = seq

    def update_sequence(self) -> None:
        self.pulse_values = self.pulse_fn(self.trainable_params)
        self.built_seq = self._seq.build(**self.pulse_values)

    def forward(self) -> Tensor:
        # run sequence for different input state

        self._sim = TorchEmulator.from_sequence(
            self.built_seq,
            evaluation_times=0.05,
            with_modulation=False,
        )

        res = self._sim.run(solver=self.solver)
        return res.states[-1]
