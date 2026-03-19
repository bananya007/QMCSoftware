from .abstract_true_measure import AbstractTrueMeasure
from ..util import MethodImplementationError, ParameterError
import numpy as np
import warnings
import math


class AcceptanceRejection(AbstractTrueMeasure):
    """
    Deterministic Acceptance-Rejection (DAR) sampler on the unit cube.

    Implements Algorithm 2 from Zhu & Dick (2014). A (t,m,s)-net in
    dimension s = d+1 is used as the driver, where the first d coordinates
    form the candidate point and the last coordinate is the acceptance
    threshold. This gives a star discrepancy bound of O(N^{-1/s}) on the
    accepted samples, compared to O(N^{-1/2}) for standard random
    acceptance-rejection.

    The sampler dimension must be d+1 where d is the target dimension.
    The number of driver points is always a power of 2 (required for the
    (t,m,s)-net property of Theorem 1).

    Args:
        sampler (AbstractDiscreteDistribution): A QMCPy discrete
            distribution of dimension s = target_dim + 1. Must mimic
            StdUniform. The last coordinate is used as the acceptance
            threshold.
        target_density (callable): Unnormalised target density psi(x)
            where x has shape (N, d). Must return shape (N,) and be
            non-negative on [0,1]^d.
        upper_bound (float): L = sup_{x in [0,1]^d} psi(x). Every
            evaluation of psi must be <= L.
        density_integral (float): C = integral_{[0,1]^d} psi(x) dx.
            The acceptance rate is C/L.
        max_retries (int): Number of times gen_samples will double the
            driver size if not enough points are accepted. Default 4.

    Examples:
        >>> import numpy as np
        >>> from qmcpy.discrete_distribution import DigitalNetB2
        >>> from qmcpy.true_measure import AcceptanceRejection
        >>> def psi(x): return 2 * x[:, 0]   # target density on [0,1]
        >>> sampler = DigitalNetB2(dimension=2, seed=7)
        >>> measure = AcceptanceRejection(sampler, psi, upper_bound=2., density_integral=1.)
        >>> samples = measure.gen_samples(n=8)
        >>> samples.shape
        (8, 1)
        >>> measure
        AcceptanceRejection (AbstractTrueMeasure)
            target_dim      1
            upper_bound     2.0
            density_integral 1.0
            acceptance_rate 0.5
    """

    def __init__(self, sampler, target_density, upper_bound, density_integral, max_retries=4):
        self.parameters = ['target_dim', 'upper_bound', 'density_integral', 'acceptance_rate']
        self.domain = np.array([[0, 1]])
        self._parse_sampler(sampler)
        # self.d is now the driver dimension s = target_dim + 1
        if self.d < 2:
            raise ParameterError(
                "sampler dimension must be >= 2 (driver dim s = target_dim + 1)."
            )
        self.target_dim = self.d - 1
        self.target_density = target_density
        self.upper_bound = float(upper_bound)
        self.density_integral = float(density_integral)
        if self.upper_bound <= 0:
            raise ParameterError("upper_bound must be strictly positive.")
        if self.density_integral <= 0:
            raise ParameterError("density_integral must be strictly positive.")
        if self.density_integral > self.upper_bound:
            warnings.warn(
                "density_integral C > upper_bound L. "
                "Check that C is the integral of psi and L is the supremum.",
                UserWarning
            )
        self.acceptance_rate = self.density_integral / self.upper_bound
        self.max_retries = int(max_retries)
        self.range = np.tile([0, 1], (self.target_dim, 1))
        super(AcceptanceRejection, self).__init__()

    def gen_samples(self, n=None, n_min=None, n_max=None, return_weights=False, warn=True):
        """
        Generate n accepted samples from the target density.

        Unlike other TrueMeasures, this method cannot be decomposed into
        a fixed 1-to-1 _transform because acceptance-rejection produces
        a variable number of outputs from a fixed driver batch. gen_samples
        is therefore overridden directly.

        Args:
            n (int): Number of accepted samples to return.
            n_min: Ignored. Included for interface compatibility.
            n_max: Ignored. Included for interface compatibility.
            return_weights (bool): If True, also return importance weights
                psi(x)/C for each accepted sample.
            warn (bool): If True, warn when fewer than n samples are
                returned after all retries.

        Returns:
            samples (np.ndarray): Shape (n, target_dim).
            weights (np.ndarray): Shape (n,). Only returned when
                return_weights=True.
        """
        if n is None:
            raise ParameterError("n must be supplied to AcceptanceRejection.gen_samples.")

        # choose smallest m such that 2^m >= ceil(n / acceptance_rate)
        M_min = int(math.ceil(n / max(self.acceptance_rate, 1e-12)))
        m = max(int(math.ceil(math.log2(max(M_min, 1)))), 1)

        accepted_batches = []
        n_collected = 0

        for _ in range(1 + self.max_retries):
            M = 2 ** m
            Q = self.discrete_distrib(n=M)          # (M, s)
            x = Q[:, :self.target_dim]              # (M, target_dim) candidates
            u = Q[:, -1]                            # (M,) thresholds
            psi_vals = self.target_density(x)       # (M,)
            mask = psi_vals >= self.upper_bound * u
            accepted_batches.append(x[mask])
            n_collected += mask.sum()
            if n_collected >= n:
                break
            m += 1

        samples = np.concatenate(accepted_batches, axis=0)[:n]

        if warn and len(samples) < n:
            warnings.warn(
                f"AcceptanceRejection: only {len(samples)}/{n} samples accepted. "
                "Increase max_retries or check upper_bound >= sup(psi).",
                RuntimeWarning
            )

        if return_weights:
            weights = self.target_density(samples) / self.density_integral
            return samples, weights
        return samples

    def _transform(self, x):
        # A-R is variable-output and cannot be expressed as a 1-to-1
        # transform. Use gen_samples directly.
        raise MethodImplementationError(self, '_transform')

    def _weight(self, x):
        return self.target_density(x) / self.density_integral

    def _spawn(self, sampler, dimension):
        return AcceptanceRejection(
            sampler,
            self.target_density,
            upper_bound=self.upper_bound,
            density_integral=self.density_integral,
            max_retries=self.max_retries,
        )


class AcceptanceRejectionReal(AbstractTrueMeasure):
    """
    Deterministic Acceptance-Rejection (DAR) sampler on real space R^d.

    Implements Algorithm 3 from Zhu & Dick (2014). Extends Algorithm 2
    to densities on R^d by mapping the unit-cube driver through marginal
    quantile functions (inverse Rosenblatt transform, Lemma 4) before
    applying the acceptance test.

    The driver point (u_1, ..., u_d, u_{d+1}) is transformed as:

        z_j = F_j^{-1}(u_j)   for j = 1, ..., d
        u   = u_{d+1}          threshold coordinate (unchanged)

    Acceptance condition:  psi(z) >= L * H(z) * u

    where H is the auxiliary bound function satisfying psi(z) <= L * H(z)
    for all z in R^d. This gives the same discrepancy bound O(N^{-1/s})
    as Algorithm 2.

    Note:
        inv_cdfs applies each quantile function independently per
        dimension. This is exact when H factors as a product of
        independent marginals (e.g. a product of univariate distributions).

    Args:
        sampler (AbstractDiscreteDistribution): A QMCPy discrete
            distribution of dimension s = target_dim + 1. Must mimic
            StdUniform.
        target_density (callable): Unnormalised target density psi(z)
            where z has shape (N, d). Must return shape (N,).
            Must satisfy psi(z) <= L * H(z) for all z.
        inv_cdfs (list of callable): List of d quantile functions
            [F_1^{-1}, ..., F_d^{-1}], one per dimension. Each maps
            a 1-D array of uniforms in [0,1] to R.
            Example: [scipy.stats.norm.ppf] for a 1-D standard Gaussian.
        H_func (callable): Auxiliary bound function H(z) where z has
            shape (N, d). Must return shape (N,) and satisfy
            psi(z) <= L * H(z) for all z in R^d.
        upper_bound (float): L satisfying psi(z) <= L * H(z) for all z.
        density_integral (float): C = integral_{R^d} psi(z) dz.
            The acceptance rate is C/L.
        max_retries (int): Number of times gen_samples will double the
            driver size if not enough points are accepted. Default 4.

    Examples:
        >>> import numpy as np
        >>> from scipy.stats import norm
        >>> from qmcpy.discrete_distribution import DigitalNetB2
        >>> from qmcpy.true_measure import AcceptanceRejectionReal
        >>> def psi(z): return norm.pdf(z[:, 0], loc=0, scale=1)
        >>> def H(z):   return norm.pdf(z[:, 0], loc=0, scale=2)
        >>> sampler = DigitalNetB2(dimension=2, seed=7)
        >>> measure = AcceptanceRejectionReal(
        ...     sampler, psi,
        ...     inv_cdfs=[lambda u: norm.ppf(u, loc=0, scale=2)],
        ...     H_func=H, upper_bound=2., density_integral=1.)
        >>> samples = measure.gen_samples(n=8)
        >>> samples.shape
        (8, 1)
        >>> measure
        AcceptanceRejectionReal (AbstractTrueMeasure)
            target_dim      1
            upper_bound     2.0
            density_integral 1.0
            acceptance_rate 0.5
    """

    def __init__(self, sampler, target_density, inv_cdfs, H_func,
                 upper_bound, density_integral, max_retries=4):
        self.parameters = ['target_dim', 'upper_bound', 'density_integral', 'acceptance_rate']
        self.domain = np.array([[0, 1]])
        self._parse_sampler(sampler)
        # self.d is now the driver dimension s = target_dim + 1
        if self.d < 2:
            raise ParameterError(
                "sampler dimension must be >= 2 (driver dim s = target_dim + 1)."
            )
        self.target_dim = self.d - 1
        if len(inv_cdfs) != self.target_dim:
            raise ParameterError(
                f"inv_cdfs must have one entry per target dimension. "
                f"Got {len(inv_cdfs)}, expected {self.target_dim}."
            )
        self.target_density = target_density
        self.inv_cdfs = inv_cdfs
        self.H_func = H_func
        self.upper_bound = float(upper_bound)
        self.density_integral = float(density_integral)
        if self.upper_bound <= 0:
            raise ParameterError("upper_bound must be strictly positive.")
        if self.density_integral <= 0:
            raise ParameterError("density_integral must be strictly positive.")
        self.acceptance_rate = self.density_integral / self.upper_bound
        self.max_retries = int(max_retries)
        self.range = np.tile([-np.inf, np.inf], (self.target_dim, 1))
        super(AcceptanceRejectionReal, self).__init__()

    def gen_samples(self, n=None, n_min=None, n_max=None, return_weights=False, warn=True):
        """
        Generate n accepted samples from the target density on R^d.

        Unlike other TrueMeasures, this method cannot be decomposed into
        a fixed 1-to-1 _transform because acceptance-rejection produces
        a variable number of outputs from a fixed driver batch. gen_samples
        is therefore overridden directly.

        Args:
            n (int): Number of accepted samples to return.
            n_min: Ignored. Included for interface compatibility.
            n_max: Ignored. Included for interface compatibility.
            return_weights (bool): If True, also return importance weights
                psi(z)/C for each accepted sample.
            warn (bool): If True, warn when fewer than n samples are
                returned after all retries.

        Returns:
            samples (np.ndarray): Shape (n, target_dim).
            weights (np.ndarray): Shape (n,). Only returned when
                return_weights=True.
        """
        if n is None:
            raise ParameterError("n must be supplied to AcceptanceRejectionReal.gen_samples.")

        M_min = int(math.ceil(n / max(self.acceptance_rate, 1e-12)))
        m = max(int(math.ceil(math.log2(max(M_min, 1)))), 1)

        accepted_batches = []
        n_collected = 0

        for _ in range(1 + self.max_retries):
            M = 2 ** m
            Q = self.discrete_distrib(n=M)              # (M, s)
            U = Q[:, :self.target_dim]                  # (M, target_dim) uniform coords
            u = Q[:, -1]                                # (M,) threshold

            # transform each dimension through its quantile function
            eps = 1e-8
            U = np.clip(U, eps, 1 - eps)
            Z = np.column_stack([
                self.inv_cdfs[j](U[:, j]) for j in range(self.target_dim)
            ])                                          # (M, target_dim) real-valued

            H_vals   = self.H_func(Z)                  # (M,)
            psi_vals = self.target_density(Z)           # (M,)
            mask     = psi_vals >= self.upper_bound * H_vals * u
            accepted_batches.append(Z[mask])
            n_collected += mask.sum()
            if n_collected >= n:
                break
            m += 1

        samples = np.concatenate(accepted_batches, axis=0)[:n]

        if warn and len(samples) < n:
            warnings.warn(
                f"AcceptanceRejectionReal: only {len(samples)}/{n} samples accepted. "
                "Increase max_retries or check psi(z) <= upper_bound * H(z) everywhere.",
                RuntimeWarning
            )

        if return_weights:
            weights = self.target_density(samples) / self.density_integral
            return samples, weights
        return samples

    def _transform(self, x):
        # A-R is variable-output and cannot be expressed as a 1-to-1
        # transform. Use gen_samples directly.
        raise MethodImplementationError(self, '_transform')

    def _weight(self, x):
        return self.target_density(x) / self.density_integral

    def _spawn(self, sampler, dimension):
        return AcceptanceRejectionReal(
            sampler,
            self.target_density,
            inv_cdfs=self.inv_cdfs,
            H_func=self.H_func,
            upper_bound=self.upper_bound,
            density_integral=self.density_integral,
            max_retries=self.max_retries,
        )
