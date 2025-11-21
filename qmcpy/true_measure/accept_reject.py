"""
Deterministic acceptance–rejection TrueMeasure (Zhu & Dick style).

This class implements a QMC version of the acceptance–rejection sampler.
The underlying discrete distribution (driver) lives in [0,1]^{d_driver} with
at least d+1 dimensions, where:

- The first d coordinates are transformed by a proposal TrueMeasure to give
  Y ~ g      (proposal distribution)
- The (d+1)-th coordinate v in [0,1] is used to deterministically accept/reject
  according to f(y) / (c * g(y)), where f is the target pdf and c is a
  known bound with f(x) <= c * g(x) for all x.

Mathematically, this follows the spirit of Zhu & Dick’s deterministic
acceptance–rejection sampler: we use one extra QMC dimension for the
“acceptance variable” and we never randomize or shuffle the driver points.
The class itself is agnostic to the choice of driver: IID, digital nets,
Halton, lattices, etc., as long as the driver conforms to the
AbstractDiscreteDistribution interface.
"""

from typing import Callable, Optional

import numpy as np

from .abstract_true_measure import AbstractTrueMeasure
from ..discrete_distribution.abstract_discrete_distribution import (
    AbstractDiscreteDistribution,
)
from ..util import ParameterError, MethodImplementationError


class AcceptReject(AbstractTrueMeasure):
    """QMC acceptance–rejection TrueMeasure.

    Parameters
    ----------
    proposal_measure : AbstractTrueMeasure
        A TrueMeasure that transforms the first d coordinates of some driver
        into proposal samples Y ~ g. Conceptually, we view the driver as
        providing points in [0,1]^{d_driver}; the proposal measure should
        consume at least the first d coordinates.

    target_pdf : Callable[[np.ndarray], np.ndarray]
        The target density f(x) evaluated on an array of shape (m, d).

    proposal_pdf : Callable[[np.ndarray], np.ndarray]
        The proposal density g(x) evaluated on an array of shape (m, d).

    bound_c : float
        A constant c such that  f(x) <= c * g(x)  for all x in the support.
        In Zhu & Dick's setup this gives the acceptance probability

            a(x) = f(x) / (c * g(x))  in [0,1].

    driver_discrete_distrib : Optional[AbstractDiscreteDistribution]
        Optional driver distribution in [0,1]^{d_driver}. If provided, this
        driver will be used by the acceptance–rejection sampler. If not
        provided, we fall back to ``proposal_measure.discrete_distrib``
        (which must then be an AbstractDiscreteDistribution).

    batch_size : Optional[int], default None
        Optional minimum batch size for each call to the underlying
        discrete distribution when more proposals are needed. If None,
        a heuristic based on ``bound_c`` and the remaining requested
        number of accepted samples is used.

    name : str, default "AcceptReject"
        Name used for pretty-printing.

    Notes
    -----
    - This class intentionally overrides ``gen_samples`` rather than using
      the default implementation in ``AbstractTrueMeasure``, because
      acceptance–rejection requires consuming more driver points than the
      number of accepted samples returned.
    - From the perspective of the rest of QMCPy, this is still a
      TrueMeasure: calling ``AcceptReject(n)`` returns an (n, d) array of
      samples from the target distribution.
    """

    def __init__(
        self,
        proposal_measure: AbstractTrueMeasure,
        target_pdf: Callable[[np.ndarray], np.ndarray],
        proposal_pdf: Callable[[np.ndarray], np.ndarray],
        bound_c: float,
        driver_discrete_distrib: Optional[AbstractDiscreteDistribution] = None,
        batch_size: Optional[int] = None,
        name: str = "AcceptReject",
    ) -> None:
        # Store user-facing configuration
        self.proposal_measure = proposal_measure
        self.target_pdf = target_pdf
        self.proposal_pdf = proposal_pdf
        self.bound_c = float(bound_c)
        self.batch_size = batch_size
        self.name = name

        self.d = self.proposal_measure.d

        if driver_discrete_distrib is None:
            driver = getattr(self.proposal_measure, "discrete_distrib", None)
            if not isinstance(driver, AbstractDiscreteDistribution):
                raise ParameterError(
                    "Either driver_discrete_distrib must be supplied, or "
                    "proposal_measure.discrete_distrib must be an "
                    "AbstractDiscreteDistribution."
                )
        else:
            if not isinstance(driver_discrete_distrib, AbstractDiscreteDistribution):
                raise ParameterError(
                    "driver_discrete_distrib must be an AbstractDiscreteDistribution."
                )
            driver = driver_discrete_distrib

        self.discrete_distrib = driver
        self.driver_dim = self.discrete_distrib.d

        if self.driver_dim < self.d + 1:
            raise ParameterError(
                f"AcceptReject expects driver_dim >= d+1; got driver_dim={self.driver_dim}, "
                f"target_dim={self.d}."
            )

        self.transform = self.proposal_measure
        self.sub_compatibility_error = False

        self.domain = np.tile([0.0, 1.0], (self.driver_dim, 1))

        self.range = self.proposal_measure.range

        self.parameters = ["bound_c", "name"]

        self._accepted_cache = np.empty((0, self.d))

        super().__init__()

    def gen_samples(
        self,
        n: int = None,
        n_min: int = None,
        n_max: int = None,
        return_weights: bool = False,
        warn: bool = True,
    ):
        """Generate ``n`` samples from the target distribution using QMC
        acceptance–rejection.

        The driver produces points X in [0,1]^{driver_dim}. We use

            U = X[:, :d]     # proposal coordinates
            V = X[:, d]      # acceptance coordinate in [0,1]

        and transform

            Y = proposal_measure._jacobian_transform_r(U, return_weights=False)

        giving Y ~ g. We then accept those Y for which

            V <= f(Y) / (c * g(Y)),

        with c >= sup_x f(x)/g(x).

        Parameters
        ----------
        n : int
            Number of *accepted* samples requested.
        n_min, n_max : ignored for now
            Not supported in this implementation. Passing non-None values
            will raise a ParameterError.
        return_weights : bool, default False
            If True, returns (samples, weights) where weights are identically 1
            for accepted samples from the target measure.
        warn : bool
            Passed through to the underlying discrete distribution.

        Returns
        -------
        samples : np.ndarray, shape (n, d)
            Accepted samples from the target distribution.
        (samples, weights) if return_weights=True, where weights is
        an array of ones with shape (n,).

        Raises
        ------
        ParameterError
            If arguments are invalid or if acceptance appears to be
            essentially zero (driver budget exceeded).
        """
        # --- basic argument checks -----------------------------------------
        if n is None:
            raise ParameterError(
                "AcceptReject.gen_samples requires n (number of accepted samples)."
            )

        if not isinstance(n, (int, np.integer)) or n <= 0:
            raise ParameterError(f"n must be a positive integer; got {n}.")

        if n_min is not None or n_max is not None:
            # We can extend to support these later, but for now we fail loudly.
            raise ParameterError(
                "AcceptReject currently supports only gen_samples(n=...). "
                "n_min and n_max are not supported."
            )

        # --- start with any cached accepted samples ------------------------
        accepted_chunks = []

        cache = self._accepted_cache
        if cache is not None and cache.shape[0] > 0:
            if cache.shape[0] >= n:
                # We already have enough in the cache
                samples = cache[:n]
                self._accepted_cache = cache[n:]
                if return_weights:
                    weights = np.ones(n, dtype=float)
                    return samples, weights
                return samples
            else:
                # Use entire cache and clear it
                accepted_chunks.append(cache)
                self._accepted_cache = np.empty((0, self.d))

        max_driver_points = int(max(1e5, 10.0 * n * max(self.bound_c, 1.0)))
        used_driver_points = 0

        def _accepted_so_far():
            if len(accepted_chunks) == 0:
                return 0
            return sum(chunk.shape[0] for chunk in accepted_chunks)

        # --- main AR loop --------------------------------------------------
        while _accepted_so_far() < n:
            remaining_budget = max_driver_points - used_driver_points
            if remaining_budget <= 0:
                raise ParameterError(
                    "AcceptReject: maximum driver point budget exceeded before "
                    "collecting requested samples. This suggests that the "
                    "acceptance probability is extremely small or zero. "
                    "Check target_pdf, proposal_pdf, and bound_c."
                )

            need = n - _accepted_so_far()

            if self.batch_size is not None and self.batch_size > 0:
                m = max(self.batch_size, need)
            else:
                m = int(max(need * self.bound_c, need * 1.5))

            m = min(m, remaining_budget)
            if m <= 0:
                raise ParameterError(
                    "AcceptReject: non-positive batch size encountered."
                )

            X = self.discrete_distrib(n=m, warn=warn)

            if X.shape[-1] != self.driver_dim:
                raise ParameterError(
                    f"AcceptReject: driver produced points with dimension {X.shape[-1]}, "
                    f"expected {self.driver_dim}."
                )

            used_driver_points += m

            U = X[:, : self.d]
            V = X[:, self.d]

            Y = self.proposal_measure._jacobian_transform_r(x=U, return_weights=False)

            if Y.shape[0] != m or Y.shape[-1] != self.d:
                raise ParameterError(
                    f"AcceptReject: proposal_measure returned shape {Y.shape}, "
                    f"expected (?, {self.d})."
                )

            f_vals = np.asarray(self.target_pdf(Y), dtype=float).reshape(-1)
            g_vals = np.asarray(self.proposal_pdf(Y), dtype=float).reshape(-1)

            if f_vals.shape[0] != m or g_vals.shape[0] != m:
                raise ParameterError(
                    "AcceptReject: target_pdf and proposal_pdf must return 1D arrays "
                    f"of length m={m}."
                )

            a = np.zeros_like(f_vals)
            positive_g = g_vals > 0

            a[positive_g] = f_vals[positive_g] / (self.bound_c * g_vals[positive_g])
            a = np.clip(a, 0.0, 1.0)

            accepted_mask = V <= a
            Y_acc = Y[accepted_mask, :]

            if Y_acc.shape[0] > 0:
                accepted_chunks.append(Y_acc)

        if len(accepted_chunks) == 0:
            raise ParameterError(
                "AcceptReject: no samples accepted. This suggests that the "
                "acceptance probability is effectively zero. Check "
                "target_pdf, proposal_pdf, and bound_c."
            )

        if len(accepted_chunks) == 1:
            all_accepted = accepted_chunks[0]
        else:
            all_accepted = np.vstack(accepted_chunks)

        if all_accepted.shape[0] < n:
            raise ParameterError(
                "AcceptReject: internal error, fewer accepted samples than requested."
            )

        samples = all_accepted[:n, :]
        extras = all_accepted[n:, :]

        self._accepted_cache = extras

        if return_weights:
            weights = np.ones(n, dtype=float)
            return samples, weights

        return samples

    def _transform(self, x: np.ndarray) -> np.ndarray:
        """Placeholder transform.

        In this design, ``gen_samples`` handles the full acceptance–rejection
        logic, so this method should not be called directly. It is defined
        only to satisfy the AbstractTrueMeasure API.
        """
        raise MethodImplementationError(
            self, "_transform is not used for AcceptReject."
        )

    def _weight(self, x: np.ndarray) -> np.ndarray:
        """Placeholder weight.

        For the accepted points, the target measure is already encoded in
        the AR logic, so the natural Jacobian weight is 1. This method is
        not used in the current design.
        """
        raise MethodImplementationError(self, "_weight is not used for AcceptReject.")

    def _spawn(self, sampler: AbstractDiscreteDistribution, dimension: int):
        """Spawn a new AcceptReject measure.

        Spawning for AcceptReject is non-trivial (it must preserve the
        link to the proposal measure and ensure that the driver has at
        least ``d+1`` dimensions). This is left unimplemented for now.
        """
        raise MethodImplementationError(
            self, "_spawn not yet implemented for AcceptReject."
        )
