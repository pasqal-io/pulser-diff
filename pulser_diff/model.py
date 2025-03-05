from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Union, cast
from uuid import uuid4

import torch
from pulser import Pulse, Register, Sequence
from pulser.parametrized import ParamObj
from pulser.parametrized.variable import Variable, VariableItem
from pyqtorch.utils import SolverType
from torch import Tensor
from torch.nn import Module, ParameterDict

from pulser_diff.backend import TorchEmulator
from pulser_diff.simresults import SimulationResults
from pulser_diff.utils import total_magnetization
from pulser_diff.waveform_funcs import constant_waveform


@dataclass
class Parameter:
    name: str
    value: Union[int, float, Tensor, None] = None
    trainable: bool = False
    type: str = ""


class QuantumModel(Module):
    def __init__(
        self,
        seq: Sequence,
        trainable_param_values: dict[str, Tensor] | dict[str, tuple[tuple, Callable]] = {},
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
            constraints (dict[str, Any]): dict specifying min/max values for trainable parameters.
            Has form {"param1": {"min": min_val, "max": max_val}}.
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

        # process trainable parameter dict
        self.callable_params = {
            name: val[0] for name, val in trainable_param_values.items() if isinstance(val, tuple)
        }
        self.callables = {
            name: val[1] for name, val in trainable_param_values.items() if isinstance(val, tuple)
        }

        # remove callable parameter (that is not a true trainable parameter, more of a placeholder)
        # from trainable param dict
        for name in self.callables.keys():
            trainable_param_values.pop(name)

        # get abstract representation of initial sequence
        self.seq_abs_repr, self.optimize_duration, self.seq_params = self._get_abstract_repr(seq)

        # extract optimizable register parameters
        self.register_params = self._extract_register_params()

        # check whether any coordinates are trainable
        # this means that register has to be reconstructed after
        # each optimization iteration
        self.reconstruct_register = any([p.trainable for p in self.register_params.values()])

        # declare trainable sequence parameters
        param_names = list(
            set(self.seq_params.keys())
            .union(set(trainable_param_values.keys()))
            .union(set(seq.declared_variables))
            .difference(set(self.register_params.keys()))
        )
        self.seq_param_values = ParameterDict()
        for name in param_names:
            if (name in trainable_param_values) and (name in seq.declared_variables):
                self.seq_param_values[name] = torch.nn.Parameter(
                    cast(Tensor, trainable_param_values[name]), requires_grad=True
                )
            elif name in self.seq_params:
                if self.seq_params[name].trainable:
                    raise ValueError(f"No value for trainable sequence parameter {name} is given.")
                self.seq_param_values[name] = self.seq_params[name].value

        # declare trainable register parameters
        self.reg_param_values = ParameterDict()
        for name, param in self.register_params.items():
            if name in trainable_param_values and param.trainable:
                self.reg_param_values[name] = torch.nn.Parameter(
                    cast(Tensor, trainable_param_values[name]), requires_grad=True
                )
            else:
                self.reg_param_values[name] = param.value.tolist()  # type: ignore [union-attr]

        # declare trainable callable function params
        self.call_param_values = ParameterDict()
        for name, param in self.callable_params.items():  # type: ignore [assignment]
            for i, v in enumerate(cast(tuple, param)):
                self.call_param_values[f"{name}_{i}"] = torch.nn.Parameter(v, requires_grad=True)

        # construct register with Parameter objects for coordinates
        self.register = self._construct_register()

        if self.optimize_duration:
            # we need to build a new sequence to accommodate pulse duration optimization
            total_duration = self._get_total_duration(trainable_param_values)
            self._seq_opt = self._create_opt_sequence(total_duration)
        else:
            # initial sequence can be used directly for optimization
            seq._set_register(seq, self.register)
            self._seq_opt = seq

        # build actual sequence from parameterized one
        if self._seq_opt.is_parametrized():
            # select parameters that are needed to build a sequence
            build_params = {
                name: val
                for name, val in self.seq_param_values.items()
                if name in self._seq_opt.declared_variables
            }
            # evaluate callables and add resulting tensors to build parameters
            for name, fn in self.callables.items():
                call_param_values = [
                    v
                    for n, v in self.call_param_values.items()
                    if "_".join(n.split("_")[:-1]) == name
                ]
                build_params[name] = fn(*call_param_values)
            self.built_seq = self._seq_opt.build(**build_params)
        else:
            self.built_seq = self._seq_opt

    def _extract_register_params(self) -> dict[str, Parameter]:
        # create Parameter objects from register's qubit coordinates
        register_params = {}
        for qubit_id, coord in self.register.qubits.items():
            register_params[str(qubit_id)] = Parameter(
                str(qubit_id), coord.as_tensor(), coord.requires_grad, type="coord"
            )
        return register_params

    def _construct_register(self) -> Register:
        coord_names = set([p.name for p in self.register_params.values()])
        coords = {
            name: value for name, value in self.reg_param_values.items() if name in coord_names
        }
        return Register(coords)

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
        pulses = []
        all_calls = [call for call in seq._calls + seq._to_build_calls if call.name == "add"]
        for call in all_calls:
            pulse = call.args[0]
            d = {
                k: v._to_abstract_repr() if hasattr(v, "_to_abstract_repr") else v
                for k, v in pulse._to_abstract_repr().items()
            }
            pulses.append(d)

        optimize_duration = False
        for pulse in pulses:
            if "duration" in pulse["amplitude"]:
                duration = pulse["amplitude"]["duration"]
            else:
                samples = pulse["amplitude"]["samples"]
                duration = samples.size if isinstance(samples, Variable) else len(samples)

            # get the name of duration variable
            if isinstance(duration, VariableItem):
                optimize_duration = True
                break

        params = {}
        for pulse in pulses:
            # get duration of the pulse
            if "duration" in pulse["amplitude"]:
                pulse["duration"] = pulse["amplitude"]["duration"]
                pulse["amplitude"].pop("duration")
            else:
                samples = pulse["amplitude"]["samples"]
                pulse["duration"] = samples.size if isinstance(samples, Variable) else len(samples)

            if "duration" in pulse["detuning"]:
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
                        value=pulse["duration"] / 1000,  # convert duration in ns to us
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

        return pulses, optimize_duration, params

    def _get_total_duration(
        self,
        trainable_param_values: (
            dict[str, Tensor] | dict[str, tuple[tuple, Callable]] | ParameterDict
        ),
    ) -> int:
        total_duration = 0
        for pulse in self.seq_abs_repr:
            duration = pulse["duration"]

            if duration.trainable:
                value = trainable_param_values[duration.name]
            else:
                value = duration.value

            # increase total duration
            total_duration += int(cast(float, value) * 1000)

        # add 5 ns to improve convergence
        total_duration += 5

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

    def check_constraints(self) -> None:
        for n, p in self.named_parameters():
            name = n.split(".")[-1]
            if name in self.constraints:
                p.data.clamp_(self.constraints[name]["min"], self.constraints[name]["max"])

    def update_sequence(self) -> None:
        if self.reconstruct_register:
            self.register = self._construct_register()

        if self.optimize_duration:
            total_duration = self._get_total_duration(self.seq_param_values)
            self._seq_opt = self._create_opt_sequence(total_duration)
        else:
            self._seq_opt._set_register(self._seq_opt, self.register)

        if self._seq_opt.is_parametrized():
            # construct final sequence with actual values
            build_params = {
                name: val
                for name, val in self.seq_param_values.items()
                if name in self._seq_opt.declared_variables
            }
            # evaluate callables and add resulting tensors to build parameters
            for name, fn in self.callables.items():
                call_param_values = [
                    v
                    for n, v in self.call_param_values.items()
                    if "_".join(n.split("_")[:-1]) == name
                ]
                build_params[name] = fn(*call_param_values)
            self.built_seq = self._seq_opt.build(**build_params)
        else:
            self.built_seq = self._seq_opt

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
