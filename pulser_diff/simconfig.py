from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import qutip
from pulser.noise_model import NoiseModel
from pulser_simulation import SimConfig as SC
from torch import Tensor


@dataclass(frozen=True)
class SimConfig(SC):
    """Specifies a simulation's configuration.

    Note:
        Being a frozen dataclass, the configuration chosen upon instantiation
        cannot be changed later on.

    Args:
        noise: Types of noises to be used in the
            simulation. You may specify just one, or a tuple of the allowed
            noise types:

            - "relaxation": Relaxation from the Rydberg to the ground state.
            - "dephasing": Random phase (Z) flip.
            - "depolarizing": Quantum noise where the state (rho) is
              turned into a mixed state I/2 at a rate gamma (in rad/µs).
            - "eff_noise": General effective noise channel defined by
              the set of collapse operators **eff_noise_opers** and the
              corresponding rates **eff_noise_rates** (in rad/µs).
            - "doppler": Local atom detuning due to finite speed of the
              atoms and Doppler effect with respect to laser frequency.
            - "amplitude": Gaussian damping due to finite laser waist
            - "SPAM": SPAM errors. Defined by **eta**, **epsilon** and
              **epsilon_prime**.

        eta: Probability of each atom to be badly prepared.
        epsilon: Probability of false positives.
        epsilon_prime: Probability of false negatives.
        runs: Number of runs needed : each run draws a new random
            noise.
        samples_per_run: Number of samples per noisy run.
            Useful for cutting down on computing time, but unrealistic.
        temperature: Temperature, set in µK, of the Rydberg array.
            Also sets the standard deviation of the speed of the atoms.
        laser_waist: Waist of the gaussian laser, set in µm, in global
            pulses.
        amp_sigma: Dictates the fluctuations in amplitude as a standard
            deviation of a normal distribution centered in 1.
        solver_options: Options for the qutip solver.
    """

    def to_pulser(self) -> SimConfig:
        """Change attribute types from tensors to ints/floats/QObjs
        Returns:
            SimConfig: SimConfig class instance with attributes suitable for Qutip backend
        """

        simconfig = deepcopy(self)
        param_list = [
            "temperature",
            "laser_waist",
            "amp_sigma",
            "eta",
            "epsilon",
            "epsilon_prime",
            "relaxation_rate",
            "dephasing_rate",
            "hyperfine_dephasing_rate",
            "depolarizing_rate",
            "eff_noise_rates",
            "eff_noise_opers",
        ]
        for param in param_list:
            val = getattr(simconfig, param)
            if param == "eff_noise_rates":
                new_val = [float(v.item()) if isinstance(v, Tensor) else v for v in val]
                simconfig._change_attribute(param, new_val)
            elif param == "eff_noise_opers":
                new_val = [
                    qutip.Qobj(v.numpy()) if isinstance(v, Tensor) else qutip.Qobj(v) for v in val
                ]
                simconfig._change_attribute(param, new_val)
            else:
                (
                    simconfig._change_attribute(param, float(val.item()))
                    if isinstance(val, Tensor)
                    else val
                )

        return simconfig

    def _check_eff_noise(self) -> None:
        # Check the validity of operators
        for operator in self.eff_noise_opers:
            # type checking
            if not isinstance(operator, (qutip.Qobj, Tensor)):
                raise TypeError(f"{str(operator)} is not a Qobj or AbstractArray.")
            if isinstance(operator, qutip.Qobj):
                if operator.type != "oper":
                    raise TypeError("Operators are supposed to be of Qutip type 'oper'.")
        NoiseModel._check_eff_noise(
            self.eff_noise_rates,
            self.eff_noise_opers,
            "eff_noise" in self.noise,
        )
