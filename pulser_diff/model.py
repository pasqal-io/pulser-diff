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


@dataclass()
class Parameter:
    name: str
    value: Union[int, float, Tensor, None] = None
    trainable: bool = False
    is_duration: bool = False


class QuantumModel(Module):
    def __init__(
        self,
        seq: Sequence,
        trainable_param_values: dict[str, Tensor],
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
        self.sampling_rate = sampling_rate
        self.solver = solver
        self.time_grad = time_grad
        self.dist_grad = dist_grad
        self.options = options

        # get abstract representation of initial sequence
        self.seq_abs_repr, self.optimize_duration = self._get_abstract_repr(seq)

        if self.optimize_duration:
            # we need to build a new sequence to accomodate pulse duration optimization
            self._seq_opt, self.params = self._create_opt_sequence(trainable_param_values)

            # register trainable parameters
            self.param_values = ParameterDict(
                {
                    name: (
                        torch.nn.Parameter(cast(Tensor, val.value), requires_grad=True)
                        if val.trainable
                        else val.value
                    )
                    for name, val in self.params.items()
                }
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

    def _create_opt_sequence(
        self, trainable_param_values: dict[str, Tensor] | ParameterDict
    ) -> tuple[Sequence, dict[str, Parameter]]:
        # create internal sequence and declare channels
        seq_opt = Sequence(self.register, MockDevice)
        seq_opt.declare_channel("rydberg_global", "rydberg_global")

        # construct nevelope functions
        total_duration, params, envelope_funcs = self._create_envelope_funcs(
            seq_opt, trainable_param_values
        )

        for t in range(int(total_duration)):
            # calculate amplitude value
            amp_val = sum([fn(t) for fn in envelope_funcs["amplitude"]])

            # calculate detuning value
            det_val = sum([fn(t) for fn in envelope_funcs["detuning"]])

            # calculate phase value
            phase_val = sum([fn(t) for fn in envelope_funcs["phase"]])

            # create shortest possible pulse with given possibly parameterized values
            pulse = Pulse.ConstantPulse(1, amp_val, det_val, phase_val)
            seq_opt.add(pulse, "rydberg_global")

        return seq_opt, params

    def _get_abstract_repr(self, seq: Sequence) -> tuple[list[dict], bool]:
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

        for pulse in pulse_list:
            pulse["duration"] = pulse["amplitude"]["duration"]
            pulse["amplitude"].pop("duration")
            pulse["detuning"].pop("duration")

            if optimize_duration:
                # convert duration to Parameter object
                if isinstance(pulse["duration"], VariableItem):
                    pulse["duration"] = Parameter(
                        pulse["duration"].var.name, trainable=True, is_duration=True
                    )
                else:
                    pulse["duration"] = Parameter(
                        f"dur_var_{uuid4()}",
                        value=pulse["duration"],
                        trainable=False,
                        is_duration=True,
                    )

            # convert amplitude to Parameter object
            if pulse["amplitude"]["kind"] == "constant":
                if isinstance(pulse["amplitude"]["value"], VariableItem):
                    pulse["amplitude"]["value"] = Parameter(
                        pulse["amplitude"]["value"].var.name, trainable=True
                    )
                else:
                    pulse["amplitude"]["value"] = Parameter(
                        f"amp_var_{uuid4()}", value=pulse["amplitude"]["value"], trainable=False
                    )

            # convert detuning to Parameter object
            if pulse["detuning"]["kind"] == "constant":
                if isinstance(pulse["detuning"]["value"], VariableItem):
                    pulse["detuning"]["value"] = Parameter(
                        pulse["detuning"]["value"].var.name, trainable=True
                    )
                else:
                    pulse["detuning"]["value"] = Parameter(
                        f"det_var_{uuid4()}", value=pulse["detuning"]["value"], trainable=False
                    )

            # convert phase to Parameter object
            if isinstance(pulse["phase"], dict):
                pulse["phase"] = Parameter(pulse["phase"]["lhs"].name, trainable=True)
            else:
                pulse["phase"] = Parameter(
                    f"phase_var_{uuid4()}", value=pulse["phase"], trainable=False
                )

        return pulse_list, optimize_duration

    def _is_duration_optimizable(self) -> bool:
        optimize_duration = False
        for pulse in self.seq_abs_repr:
            duration = pulse["amplitude"]["duration"]

            # get the name of duration variable
            if isinstance(duration, VariableItem):
                optimize_duration = True
                break
        return optimize_duration

    def _create_envelope_funcs(
        self, seq_opt: Sequence, trainable_param_values: dict[str, Tensor] | ParameterDict
    ) -> tuple[int, dict[str, Parameter], dict[str, list]]:
        ti = 0
        total_duration = 0
        params = {}
        envelope_funcs: dict = {"amplitude": [], "detuning": [], "phase": []}
        for pulse in self.seq_abs_repr:
            duration = pulse["duration"]

            # get the name of duration variable
            var_name = duration.name
            if duration.trainable:
                value = trainable_param_values.get(var_name, None)
                if value is None:
                    raise ValueError(f"Value for parameter {var_name} is not provided.")
                trainable = True
            else:
                value = duration.value / 1000
                trainable = False

            # declare new variable
            if var_name not in seq_opt.declared_variables:
                duration_var = seq_opt.declare_variable(var_name).var
                duration_param = Parameter(name=var_name, value=value, trainable=trainable)
                params[var_name] = duration_param
            else:
                duration_var = seq_opt.declared_variables[var_name]

            # increase total duration
            total_duration += int(value * 1000)

            # end of current pulse
            tf = ti + duration_var

            # get amplitude/detuning envelope function for current pulse
            for s in ["amplitude", "detuning"]:
                wf_type = pulse[s]["kind"]
                if wf_type == "constant":
                    param = pulse[s]["value"]
                    var_name = param.name
                    if param.trainable:
                        value = trainable_param_values.get(var_name, None)
                        if value is None:
                            raise ValueError(f"Value for parameter {var_name} is not provided.")
                        trainable = True
                    else:
                        value = param.value
                        trainable = False

                    # declare new variable
                    if var_name not in seq_opt.declared_variables:
                        var = seq_opt.declare_variable(var_name).var
                        param = Parameter(name=var_name, value=value, trainable=trainable)
                        params[var_name] = param
                    else:
                        var = seq_opt.declared_variables[var_name]
                    fn = constant_waveform(cast(ParamObj, ti), tf, var)
                else:
                    raise NotImplementedError(
                        f"{s} waveform type {wf_type} currently not supported."
                    )
                envelope_funcs[s].append(fn)

            # get phase envelope function for current pulse
            param = pulse["phase"]
            var_name = param.name
            if isinstance(value, dict):
                value = trainable_param_values.get(var_name, None)
                if value is None:
                    raise ValueError(f"Value for parameter {var_name} is not provided.")
                trainable = True
            else:
                value = param.value
                trainable = False

            # declare new variable
            if var_name not in seq_opt.declared_variables:
                var = seq_opt.declare_variable(var_name).var
                param = Parameter(name=var_name, value=value, trainable=trainable)
                params[var_name] = param
            else:
                var = seq_opt.declared_variables[var_name]
            envelope_funcs["phase"].append(constant_waveform(cast(ParamObj, ti), tf, var))

            # shift ti to end of pulse
            ti = tf  # type: ignore [assignment]

        # add 5 ns at the end of sequence
        total_duration += 5

        return total_duration, params, envelope_funcs

    def update_sequence(self, reconstruct_sequence: bool = False) -> None:
        if reconstruct_sequence and self.optimize_duration:
            self._seq_opt, self.params = self._create_opt_sequence(self.param_values)
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
