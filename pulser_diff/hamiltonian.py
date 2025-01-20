from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Mapping
from functools import reduce
from math import floor
from typing import Any, Callable, Union, cast

import torch
from pulser.devices._device_datacls import BaseDevice
from pulser.math import AbstractArray
from pulser.noise_model import NoiseModel
from pulser.register.base_register import QubitId
from pulser.sampler.samples import SequenceSamples, _PulseTargetSlot
from pulser_simulation.simconfig import SUPPORTED_NOISES, doppler_sigma
from pyqtorch.matrices import XMAT, YMAT, ZMAT
from torch import Tensor

from pulser_diff.utils import basis_state, kron


class Hamiltonian:
    r"""Generates Hamiltonian from a sampled sequence and noise.

    Args:
        samples_obj: A sampled sequence whose ChannelSamples have same
            duration.
        qdict: A dictionary associating coordinates to qubit ids.
        device: The device specifications.
        sampling_rate: The fraction of samples that we wish to extract from
            the samples to simulate. Has to be a value between 0.05 and 1.0.
        config: Configuration to be used for this simulation.
    """

    def __init__(
        self,
        samples_obj: SequenceSamples,
        qdict: dict[QubitId, AbstractArray],
        device: BaseDevice,
        sampling_rate: float,
        config: NoiseModel,
    ) -> None:
        """Instantiates a Hamiltonian object."""
        self.samples_obj = samples_obj
        self._qdict = {k: v.as_tensor() for k, v in qdict.items()}
        self._device = device
        self._sampling_rate = sampling_rate

        # Type hints for attributes defined outside of __init__
        self.basis_name: str
        self._config: NoiseModel
        self.op_matrix: dict[str, Tensor]
        self.basis: dict[str, Tensor]
        self.dim: int
        self._bad_atoms: dict[Union[str, int], bool] = {}
        self._doppler_detune: dict[Union[str, int], float] = {}
        self._dist_dict: dict[str, Tensor] = {}

        # Define interaction
        self._interaction = "XY" if self.samples_obj._in_xy else "ising"

        # Initializing qubit infos
        self._size = len(self._qdict)
        self._qid_index = {qid: i for i, qid in enumerate(self._qdict)}

        # Compute sampling times
        self._duration = self.samples_obj.max_duration
        self.sampling_times = self._adapt_to_sampling_rate(
            # Include extra time step for final instruction from samples:
            torch.arange(self._duration, dtype=torch.double)
            / 1000
        )

        # Stores the qutip operators used in building the Hamiltonian
        self.operators: dict[str, defaultdict[str, dict]] = {
            addr: defaultdict(dict) for addr in ["Global", "Local"]
        }
        self._collapse_ops: list[Tensor] = []

        self.set_config(config)

    def _adapt_to_sampling_rate(self, full_array: Tensor) -> Tensor:
        """Adapt list to correspond to sampling rate."""
        indices = torch.linspace(
            0,
            len(full_array) - 1,
            int(self._sampling_rate * self._duration),
            dtype=torch.int,
        )
        return full_array[indices]

    @property
    def config(self) -> NoiseModel:
        """The current configuration, as a NoiseModel instance."""
        return self._config

    def _build_collapse_operators(self, config: NoiseModel) -> None:
        def basis_check(noise_type: str) -> None:
            """Checks if the basis allows for the use of noise."""
            if self.basis_name == "all":
                # Go back to previous config
                raise NotImplementedError(f"Cannot include {noise_type} noise in all-basis.")

        local_collapse_ops = []
        if "dephasing" in config.noise_types:
            basis_check("dephasing")
            rate = (
                config.hyperfine_dephasing_rate
                if self.basis_name == "digital"
                else config.dephasing_rate
            )
            local_collapse_ops.append(torch.sqrt(torch.as_tensor(rate) / 2) * ZMAT.to_sparse())

        if "relaxation" in config.noise_types:
            coeff = torch.sqrt(torch.as_tensor(config.relaxation_rate))
            try:
                local_collapse_ops.append(coeff * self.op_matrix["sigma_gr"])
            except KeyError:
                raise ValueError(
                    "'relaxation' noise requires addressing of the 'ground-rydberg' basis."
                )

        if "depolarizing" in config.noise_types:
            basis_check("depolarizing")
            coeff = torch.sqrt(torch.as_tensor(config.depolarizing_rate) / 4)
            local_collapse_ops.append(coeff * XMAT.to_sparse())
            local_collapse_ops.append(coeff * YMAT.to_sparse())
            local_collapse_ops.append(coeff * ZMAT.to_sparse())

        if "eff_noise" in config.noise_types:
            basis_check("effective")
            for id, rate in enumerate(config.eff_noise_rates):
                local_collapse_ops.append(
                    torch.sqrt(torch.as_tensor(rate))
                    * torch.as_tensor(config.eff_noise_opers[id]).to_sparse()
                )

        # Building collapse operators
        self._collapse_ops = []
        for operator in local_collapse_ops:
            self._collapse_ops += [
                self.build_operator([(operator, [qid])]) for qid in self._qid_index
            ]

    def set_config(self, cfg: NoiseModel) -> None:
        """Sets current config to cfg and updates simulation parameters.

        Args:
            cfg: New configuration.
        """
        if not isinstance(cfg, NoiseModel):
            raise ValueError(f"Object {cfg} is not a valid `NoiseModel`.")
        not_supported = set(cfg.noise_types) - SUPPORTED_NOISES[self._interaction]
        if not_supported:
            raise NotImplementedError(
                f"Interaction mode '{self._interaction}' does not support "
                f"simulation of noise types: {', '.join(not_supported)}."
            )
        if not hasattr(self, "basis_name"):
            self._build_basis_and_op_matrices()
        self._build_collapse_operators(cfg)
        self._config = cfg
        if not ("SPAM" in self.config.noise_types and self.config.state_prep_error > 0):
            self._bad_atoms = {qid: False for qid in self._qid_index}
        if "doppler" not in self.config.noise_types:
            self._doppler_detune = {qid: 0.0 for qid in self._qid_index}
        # Noise, samples and Hamiltonian update routine
        self._construct_hamiltonian()

    def _extract_samples(self) -> None:
        """Populates samples dictionary with every pulse in the sequence."""
        local_noises = True
        if set(self.config.noise_types).issubset(
            {"dephasing", "relaxation", "SPAM", "depolarizing", "eff_noise"}
        ):
            local_noises = "SPAM" in self.config.noise_types and self.config.state_prep_error > 0
        samples = self.samples_obj.to_nested_dict(all_local=local_noises, samples_type="tensor")

        def add_noise(
            slot: _PulseTargetSlot,
            samples_dict: Mapping[QubitId, dict[str, Tensor]],
            is_global_pulse: bool,
        ) -> None:
            """Builds hamiltonian coefficients.

            Taking into account, if necessary, noise effects, which are local
            and depend on the qubit's id qid.
            """
            noise_amp_base = max(0, float(torch.normal(torch.tensor([1.0]), self.config.amp_sigma)))
            for qid in slot.targets:
                if "doppler" in self.config.noise_types:
                    noise_det = self._doppler_detune[qid]
                    samples_dict[qid]["det"][slot.ti : slot.tf] += noise_det
                # Gaussian beam loss in amplitude for global pulses only
                # Noise is drawn at random for each pulse
                if "amplitude" in self.config.noise_types and is_global_pulse:
                    amp_fraction = torch.tensor(1.0)
                    if self.config.laser_waist is not None:
                        position = self._qdict[qid]
                        r = torch.linalg.norm(position)
                        w0 = torch.as_tensor(self.config.laser_waist)
                        amp_fraction = torch.exp(-((r / w0) ** 2))
                    noise_amp = noise_amp_base * amp_fraction
                    samples_dict[qid]["amp"][slot.ti : slot.tf] *= noise_amp

        if local_noises:
            for ch, ch_samples in self.samples_obj.channel_samples.items():
                addr = self.samples_obj._ch_objs[ch].addressing
                basis = self.samples_obj._ch_objs[ch].basis
                samples_dict = samples["Local"][basis]
                for slot in ch_samples.slots:
                    add_noise(slot, samples_dict, addr == "Global")
            # Delete samples for badly prepared atoms
            for basis in samples["Local"]:
                for qid in samples["Local"][basis]:
                    if self._bad_atoms[qid]:
                        for qty in ("amp", "det", "phase"):
                            samples["Local"][basis][qid][qty] = 0.0
        self.samples = samples

    def build_operator(self, operations: Union[list, tuple]) -> Tensor:
        """Creates an operator with non-trivial actions on some qubits.

        Takes as argument a list of tuples ``[(operator_1, qubits_1),
        (operator_2, qubits_2)...]``. Returns the operator given by the tensor
        product of {``operator_i`` applied on ``qubits_i``} and Id on the rest.
        ``(operator, 'global')`` returns the sum for all ``j`` of operator
        applied at ``qubit_j`` and identity elsewhere.

        Example for 4 qubits: ``[(Z, [1, 2]), (Y, [3])]`` returns `ZZYI`
        and ``[(X, 'global')]`` returns `XIII + IXII + IIXI + IIIX`

        Args:
            operations: List of tuples `(operator, qubits)`.
                `operator` can be a ``qutip.Quobj`` or a string key for
                ``self.op_matrix``. `qubits` is the list on which operator
                will be applied. The qubits can be passed as their
                index or their label in the register.

        Returns:
            The final operator.
        """
        op_list = [self.op_matrix["I"] for j in range(self._size)]

        if not isinstance(operations, list):
            operations = [operations]

        for operator, qubits in operations:
            if qubits == "global":
                return reduce(
                    lambda a, b: a + b,
                    [self.build_operator([(operator, [q_id])]) for q_id in self._qdict],
                )
            else:
                qubits_set = set(qubits)
                if len(qubits_set) < len(qubits):
                    raise ValueError("Duplicate atom ids in argument list.")
                if not qubits_set.issubset(self._qdict.keys()):
                    raise ValueError("Invalid qubit names: " f"{qubits_set - self._qdict.keys()}")
                if isinstance(operator, str):
                    try:
                        operator = self.op_matrix[operator]
                    except KeyError:
                        raise ValueError(f"{operator} is not a valid operator")
                for qubit in qubits:
                    k = self._qid_index[qubit]
                    op_list[k] = operator
        return kron(*op_list)

    def _update_noise(self) -> None:
        """Updates noise random parameters.

        Used at the start of each run. If SPAM isn't in chosen noises, all
        atoms are set to be correctly prepared.
        """
        if "SPAM" in self.config.noise_types and self.config.state_prep_error > 0:
            dist = torch.rand(size=len(self._qid_index)) < self.config.state_prep_error
            self._bad_atoms = dict(zip(self._qid_index, dist))
        if "doppler" in self.config.noise_types:
            temp = self.config.temperature * 1e-6
            detune = torch.normal(
                0,
                doppler_sigma(temp),
                size=(len(self._qid_index),),
            )
            self._doppler_detune = dict(zip(self._qid_index, detune))

    def _build_basis_and_op_matrices(self) -> None:
        """Determine dimension, basis and projector operators."""
        if self._interaction == "XY":
            self.basis_name = "XY"
            self.dim = 2
            basis = ["u", "d"]
            projectors = ["uu", "du", "ud", "dd"]
        else:
            if "digital" not in self.samples_obj.used_bases:
                self.basis_name = "ground-rydberg"
                self.dim = 2
                basis = ["r", "g"]
                projectors = ["gr", "rr", "gg"]
            elif "ground-rydberg" not in self.samples_obj.used_bases:
                self.basis_name = "digital"
                self.dim = 2
                basis = ["g", "h"]
                projectors = ["hg", "hh", "gg"]
            else:
                self.basis_name = "all"  # All three states
                self.dim = 3
                basis = ["r", "g", "h"]
                projectors = ["gr", "hg", "rr", "gg", "hh"]

        self.basis = {b: basis_state(self.dim, i) for i, b in enumerate(basis)}
        self.op_matrix = {"I": torch.eye(self.dim).to_sparse()}

        for proj in projectors:
            self.op_matrix["sigma_" + proj] = (
                self.basis[proj[0]] * self.basis[proj[1]].mH
            ).to_sparse()

    def _construct_hamiltonian(self, update: bool = True) -> None:
        """Constructs the hamiltonian from the sampled Sequence and noise.

        Also builds qutip.Qobjs related to the Sequence if not built already,
        and refreshes potential noise parameters by drawing new at random.

        Args:
            update: Whether to update the noise parameters.
        """
        if update:
            self._update_noise()
        self._extract_samples()

        def make_vdw_term(q1: QubitId, q2: QubitId) -> Tensor:
            """Construct the Van der Waals interaction Term.

            For each pair of qubits, calculate the distance between them,
            then assign the local operator "sigma_rr" at each pair.
            The units are given so that the coefficient includes a
            1/hbar factor.
            """
            dist = torch.linalg.norm(self._qdict[q1] - self._qdict[q2])
            self._dist_dict[f"{q1}-{q2}"] = dist
            U = (1.0 + 0.0j) * 0.5 * self._device.interaction_coeff / dist**6
            return cast(Tensor, self.build_operator([("sigma_rr", [q1, q2])]) * U)

        def make_xy_term(q1: QubitId, q2: QubitId) -> Tensor:
            """Construct the XY interaction Term.

            For each pair of qubits, calculate the distance between them,
            then assign the local operator "sigma_ud * sigma_du" at each pair.
            The units are given so that the coefficient
            includes a 1/hbar factor.
            """
            dist = torch.linalg.norm(self._qdict[q1] - self._qdict[q2])
            coords_dim = len(self._qdict[q1])
            mag_field = cast(Tensor, self.samples_obj._magnetic_field)[:coords_dim]
            mag_norm = torch.linalg.norm(mag_field)
            if mag_norm < 1e-8:
                cosine = 0.0
            else:
                cosine = torch.dot(
                    (self._qdict[q1] - self._qdict[q2]),
                    mag_field,
                ) / (dist * mag_norm)
            U = 0.5 * cast(float, self._device.interaction_coeff_xy) * (1 - 3 * cosine**2) / dist**3
            return cast(Tensor, self.build_operator([("sigma_ud", [q1]), ("sigma_du", [q2])]) * U)

        def make_interaction_term(masked: bool = False) -> Tensor:
            if masked:
                # Calculate the total number of good, unmasked qubits
                effective_size = self._size - sum(self._bad_atoms.values())
                for q in self.samples_obj._slm_mask.targets:
                    if not self._bad_atoms[q]:
                        effective_size -= 1
                if effective_size < 2:
                    return 0.0 * self.build_operator([("I", "global")])

            # make interaction term
            dipole_interaction = torch.sparse_coo_tensor(
                torch.as_tensor([[0], [0]]),
                [0],
                (2**self._size, 2**self._size),
                dtype=torch.complex128,
            )
            for q1, q2 in itertools.combinations(self._qdict.keys(), r=2):
                if (
                    self._bad_atoms[q1]
                    or self._bad_atoms[q2]
                    or (
                        masked
                        and self._interaction == "XY"
                        and (
                            q1 in self.samples_obj._slm_mask.targets
                            or q2 in self.samples_obj._slm_mask.targets
                        )
                    )
                ):
                    continue

                if self._interaction == "XY":
                    dipole_interaction += make_xy_term(q1, q2)
                else:
                    dipole_interaction += make_vdw_term(q1, q2)
            return dipole_interaction

        def build_coeffs_ops(basis: str, addr: str) -> list[list]:
            """Build coefficients and operators for the hamiltonian QobjEvo."""
            samples = self.samples[addr][basis]
            operators = self.operators[addr][basis]
            # Choose operator names according to addressing:
            if basis == "ground-rydberg":
                op_ids = ["sigma_gr", "sigma_rr"]
            elif basis == "digital":
                op_ids = ["sigma_hg", "sigma_gg"]
            elif basis == "XY":
                op_ids = ["sigma_du", "sigma_uu"]

            terms = []
            if addr == "Global":
                coeffs = [
                    0.5 * samples["amp"] * torch.exp(-1j * samples["phase"]),
                    -0.5 * samples["det"],
                ]
                for op_id, coeff in zip(op_ids, coeffs):
                    if torch.any(coeff != 0):
                        # Build once global operators as they are needed
                        if op_id not in operators:
                            operators[op_id] = self.build_operator([(op_id, "global")])
                        terms.append(
                            [
                                operators[op_id],
                                self._adapt_to_sampling_rate(coeff),
                            ]
                        )
            elif addr == "Local":
                for q_id, samples_q in samples.items():
                    if q_id not in operators:
                        operators[q_id] = {}
                    coeffs = [
                        0.5 * samples_q["amp"] * torch.exp(-1j * samples_q["phase"]),
                        -0.5 * samples_q["det"],
                    ]
                    for coeff, op_id in zip(coeffs, op_ids):
                        if torch.any(coeff != 0):
                            if op_id not in operators[q_id]:
                                operators[q_id][op_id] = self.build_operator([(op_id, [q_id])])
                            terms.append(
                                [
                                    operators[q_id][op_id],
                                    self._adapt_to_sampling_rate(coeff),
                                ]
                            )
            self.operators[addr][basis] = operators
            return terms

        qobj_list: Any = []
        # Time independent term:
        effective_size = self._size - sum(self._bad_atoms.values())
        if self.basis_name != "digital" and effective_size > 1:
            # Build time-dependent or time-independent interaction term based
            # on whether an SLM mask was defined or not
            if self.samples_obj._slm_mask.end > 0 and self._interaction == "XY":
                # Build an array of binary coefficients for the interaction
                # term of unmasked qubits
                coeff = torch.ones(self._duration - 1)
                coeff[0 : self.samples_obj._slm_mask.end] = 0
                # Build the interaction term for unmasked qubits
                qobj_list = [
                    [
                        make_interaction_term(),
                        self._adapt_to_sampling_rate(coeff),
                    ]
                ]
                # Build the interaction term for masked qubits
                qobj_list += [
                    [
                        make_interaction_term(masked=True),
                        self._adapt_to_sampling_rate(
                            torch.logical_not(coeff).to(torch.int),
                        ),
                    ]
                ]
            else:
                qobj_list = [make_interaction_term()]

        # Time dependent terms:
        for addr in self.samples:
            for basis in self.samples[addr]:
                if self.samples[addr][basis]:
                    qobj_list += cast(list, build_coeffs_ops(basis, addr))

        if not qobj_list:  # If qobj_list ends up empty
            qobj_list = cast(list[Tensor], qobj_list)
            qobj_list = [Tensor, 0 * self.build_operator([("I", "global")])]

        ham = self.build_ham_tensor(qobj_list)
        self._hamiltonian = ham

    def build_ham_tensor(self, qobj_list: list) -> Callable[[float | Tensor], Tensor]:
        # get interaction, amplitude and detuning components
        if self._size > 1:
            int_mat = qobj_list[0]
            qobj_list = qobj_list[1:]
        else:
            int_mat = torch.zeros((2, 2), dtype=torch.complex128).to_sparse()
        int_mat = cast(Tensor, int_mat)
        amp_matrices: list[Tensor] = []
        amp_values: list[Tensor] = []
        det_matrices: list[Tensor] = []
        det_values: list[Tensor] = []
        for i in range(len(qobj_list)):
            mat = cast(Tensor, qobj_list[i][0])
            if torch.equal(mat.indices()[0], mat.indices()[1]):
                # is diagonal - detuning part
                det_matrices.append(mat)
                det_values.append((1.0 + 0.0j) * qobj_list[i][1])
            else:
                # not diagonal - ampplitude part
                amp_matrices.append(mat)
                amp_values.append(qobj_list[i][1])

        # calculate time step and sample length
        dt = 0.001 / self._sampling_rate
        n_samples = len(qobj_list[-1][1])

        def H_t(t: float | Tensor) -> Tensor:
            # make sure that time is a tensor
            if not isinstance(t, Tensor):
                t = torch.tensor(t)

            # calculate time indices for interpolation
            t_idx1 = max(int(min(floor(t / dt), n_samples - 2)), 0)
            t_idx2 = min(t_idx1 + 1, n_samples - 2)

            # construct Hamiltonian
            ham = 2 * int_mat
            for det_mat, det_val in zip(det_matrices, det_values):
                det = det_val[t_idx1] + (det_val[t_idx2] - det_val[t_idx1]) * (t - t_idx1 * dt) / dt
                ham_mat = det_mat * det
                ham += ham_mat + ham_mat.adjoint()
            for amp_mat, amp_val in zip(amp_matrices, amp_values):
                amp = amp_val[t_idx1] + (amp_val[t_idx2] - amp_val[t_idx1]) * (t - t_idx1 * dt) / dt
                ham_mat = amp_mat * amp
                ham += ham_mat + ham_mat.adjoint()

            return cast(Tensor, ham)

        return H_t
