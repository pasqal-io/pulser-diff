import torch
from torch import Tensor

from pulser_diff.dq._utils import cache
from pulser_diff.dq.solvers.solver import Solver


class MESolver(Solver):
    r"""Base class for Linblad Master equation solver.

    Args:
        *args : The arguments of the Solver class.
        L : a (nL, n, n) (sparse) Tensor containing nL Linblad operators, each of dimension (n, n)
    """

    def __init__(self, *args, L: Tensor):
        super().__init__(*args)

        self.L = L  # (nL, n, n)
        self.L_tuple = torch.unbind(self.L)

        L_concat_v = torch.cat(self.L_tuple)

        self.sum_LdagL = L_concat_v.mH @ L_concat_v  # (..., n, n)

        # define identity operator
        n = self.H.size(-1)
        self.I = torch.eye(n, device=self.device, dtype=self.cdtype)  # (n, n)

        # define cached non-hermitian Hamiltonian
        self.Hnh = cache(lambda H: H - 0.5j * self.sum_LdagL)  # (..., n, n)

    def lindbladian(self, t: float, rho: Tensor) -> Tensor:
        """Compute the action of the Lindbladian on the density matrix.

        Notes:
            Hermiticity of the output is enforced to avoid numerical instability
            with Runge-Kutta solvers.
        """
        # rho: (..., n, n) -> (..., n, n)
        H = self.H(t)
        lindblad_term = sum(L @ rho @ L.mH for L in self.L_tuple)
        out = -1j * self.Hnh(H) @ rho + 0.5 * lindblad_term
        return out + out.mH

    def adjoint_lindbladian(self, t: float, phi: Tensor) -> Tensor:
        """Compute the action of the adjoint Lindbladian on an operator.

        Notes:
            Hermiticity of the output is enforced to avoid numerical instability
            with Runge-Kutta solvers.
        """
        # phi: (..., n, n) -> (..., n, n)
        H = self.H(t)
        lindblad_term = sum(L.mH @ phi @ L for L in self.L_tuple)
        out = 1j * self.Hnh(H).mH @ phi + 0.5 * lindblad_term
        return out + out.mH
