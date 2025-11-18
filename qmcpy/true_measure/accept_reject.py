"""
Deterministic acceptance–rejection TrueMeasure (Zhu & Dick style).

This class is meant to implement a QMC version of the acceptance–rejection
sampler where:
- The underlying discrete distribution lives in [0,1]^{d+1}
- The first d coordinates are transformed by a proposal TrueMeasure to give
  Y ~ g      (proposal distribution)
- The last coordinate v in [0,1] is used to deterministically accept/reject
  according to f(y) / (c * g(y)), where f is the target pdf and c is a
  known bound with f(x) <= c * g(x) for all x.

Mathematically, this follows the spirit of Zhu & Dick’s deterministic
acceptance–rejection sampler: we use one extra QMC dimension for the
“acceptance variable” and we never randomize or shuffle the QMC points.
"""

from typing import Callable, Optional

import numpy as np

from .abstract_true_measure import AbstractTrueMeasure
from ..discrete_distribution.abstract_discrete_distribution import (
    AbstractDiscreteDistribution,
)
from ..util import ParameterError, MethodImplementationError


class AcceptReject(AbstractTrueMeasure):
    """
    QMC acceptance–rejection TrueMeasure.

    Parameters
    ----------
    proposal_measure : AbstractTrueMeasure
        A TrueMeasure that transforms the first d coordinates of the driver
        into proposal samples Y ~ g. Its `discrete_distrib` is assumed to
        have dimension d+1:
            driver_dim = proposal_measure.discrete_distrib.d
            target_dim = proposal_measure.d
            driver_dim == target_dim + 1

        The extra (last) coordinate of the driver is used as the
        acceptance variable v in [0,1].

    target_pdf : Callable[[np.ndarray], np.ndarray]
        The target density f(x) evaluated on an array of shape (..., d).

    proposal_pdf : Callable[[np.ndarray], np.ndarray]
        The proposal density g(x) evaluated on an array of shape (..., d).

    bound_c : float
        A constant c such that  f(x) <= c * g(x)  for all x in the support.
        In Zhu & Dick's setup this gives the acceptance probability
            a(x) = f(x) / (c * g(x))  in [0,1].

    batch_size : Optional[int], default None
        Optional minimum batch size for each call to the underlying
        discrete distribution when more proposals are needed.

    Notes
    -----
    - This class intentionally overrides ``gen_samples`` rather than using
      the default implementation in ``AbstractTrueMeasure``, because
      acceptance–rejection requires consuming more driver points than the
      number of accepted samples returned.
    - We still behave as a TrueMeasure from the perspective of the rest
      of QMCPy: calling ``AcceptReject(n)`` should return an (n, d) array
      of samples from the target distribution.
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

        # --- Dimensions ----------------------------------------------------
        # Target dimension (what integrands / integrals see)
        self.d = self.proposal_measure.d

        # Driver distribution (what produces QMC points in [0,1]^{driver_dim})
        if driver_discrete_distrib is None:
            # Fallback: use the proposal's own driver, but require it to
            # have at least one extra dimension for the acceptance variable.
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

        # We need at least one extra coordinate for the acceptance variable.
        if self.driver_dim < self.d + 1:
            raise ParameterError(
                f"AcceptReject expects driver_dim >= d+1; got driver_dim={self.driver_dim}, "
                f"target_dim={self.d}."
            )
        # By default we use coordinate index self.d as the acceptance coord.
        # If driver_dim > d+1, extra coordinates are simply unused.
        # (We still keep the domain as [0,1]^{driver_dim}.)

        # --- Transform chain structure ------------------------------------
        # We conceptually compose on top of the proposal_measure:
        #   (u, v) --proposal_measure--> Y = T(u)  (use first d coords)
        #   then use v for accept/reject.
        # The 'transform' attribute is the sub-transform in the chain.
        self.transform = self.proposal_measure
        self.sub_compatibility_error = False  # we manage our own compatibility

        # Domain: raw driver space. For the acceptance–rejection map this is
        # [0,1]^{d+1} (unit cube in driver_dim dimensions).
        self.domain = np.tile([0.0, 1.0], (self.driver_dim, 1))

        # Range: support of the target; same as the proposal's range.
        # (We thin proposals by rejection, but do not move them.)
        self.range = self.proposal_measure.range

        # Parameters to show in __repr__
        self.parameters = ["bound_c", "name"]

        # Cache of extra accepted points not yet returned
        self._accepted_cache = np.empty((0, self.d))

        # Let the base class validate domain/range/parameters
        super().__init__()

    # ------------------------------------------------------------------
    # Core sampling method: deterministic acceptance–rejection
    # ------------------------------------------------------------------
    def gen_samples(
        self,
        n: int = None,
        n_min: int = None,
        n_max: int = None,
        return_weights: bool = False,
        warn: bool = True,
    ):
        """
        Generate n samples from the target distribution using QMC
        acceptance–rejection.

        Implements a deterministic AR scheme à la Zhu & Dick:
        - underlying discrete distribution lives in [0,1]^{d+1}
        - first d coords drive the proposal TrueMeasure
        - last coord v ∈ [0,1] is used for acceptance test

            v <= f(Y) / (c * g(Y))

        Parameters
        ----------
        n : int
            Number of *accepted* samples requested.
        return_weights : bool
            If True, returns (samples, weights) where weights are 1.

        Returns
        -------
        samples : array, shape (n, d)
        (samples, weights) if return_weights=True
        """
        # --- Argument validation --------------------------------------------------
        if n is None:
            raise ParameterError(
                "AcceptReject.gen_samples requires n (number of accepted samples)."
            )

        if not isinstance(n, (int, np.integer)) or n <= 0:
            raise ParameterError(f"n must be a positive integer; got {n}.")

        if n_min is not None or n_max is not None:
            raise ParameterError(
                "AcceptReject supports only gen_samples(n=...). "
                "n_min and n_max are not supported."
            )

        # --- 1. Use cached accepted samples first --------------------------------
        accepted_chunks = []

        cache = self._accepted_cache
        if cache is not None and cache.shape[0] > 0:
            if cache.shape[0] >= n:
                samples = cache[:n]
                self._accepted_cache = cache[n:]
                if return_weights:
                    return samples, np.ones(n)
                return samples
            else:
                accepted_chunks.append(cache)
                n_remaining = n - cache.shape[0]
                self._accepted_cache = np.empty((0, self.d))
        else:
            n_remaining = n

        # --- 2. Safety cap to prevent infinite loops ------------------------------
        # Max number of driver points we allow ourselves to consume.
        # Theoretical acceptance ~ 1/c, so expected ~ n*c proposals needed.
        # We allow up to 10x that, with a floor of 100k.
        max_driver_points = int(max(1e5, 10.0 * n * max(self.bound_c, 1.0)))
        used_driver_points = 0

        def _accepted_so_far():
            if len(accepted_chunks) == 0:
                return 0
            return sum(chunk.shape[0] for chunk in accepted_chunks)

        # --- 3. Main acceptance–rejection loop ------------------------------------
        while _accepted_so_far() < n:
            remaining_budget = max_driver_points - used_driver_points
            if remaining_budget <= 0:
                raise ParameterError(
                    "AcceptReject: driver point budget exceeded before collecting "
                    "requested samples. Acceptance probability may be zero.\n"
                    "Check target_pdf, proposal_pdf, and bound_c."
                )

            needed = n - _accepted_so_far()

            # Choose batch size
            if self.batch_size is not None and self.batch_size > 0:
                m = max(self.batch_size, needed)
            else:
                # expected ~ needed * c
                m = int(max(needed * self.bound_c, needed * 1.5))

            m = min(m, remaining_budget)
            if m <= 0:
                raise ParameterError(
                    "AcceptReject: encountered non-positive batch size."
                )

            # --- Draw m driver points in [0,1]^{d+1} --------------------------
            X = self.discrete_distrib(n=m, warn=warn)
            if X.shape[-1] != self.driver_dim:
                raise ParameterError(
                    f"Driver produced dimension {X.shape[-1]}, expected {self.driver_dim}."
                )

            used_driver_points += m

            # Split driver coords
            U = X[:, : self.d]  # proposal coords
            V = X[:, self.d]  # acceptance uniform

            # Transform proposals
            Y = self.proposal_measure._jacobian_transform_r(x=U, return_weights=False)

            if Y.shape[0] != m or Y.shape[-1] != self.d:
                raise ParameterError(
                    f"Proposal measure returned shape {Y.shape}, expected (?, {self.d})."
                )

            # Evaluate PDFs
            f_vals = np.asarray(self.target_pdf(Y), float).reshape(-1)
            g_vals = np.asarray(self.proposal_pdf(Y), float).reshape(-1)

            if f_vals.shape[0] != m or g_vals.shape[0] != m:
                raise ParameterError(
                    f"target_pdf/proposal_pdf must return arrays of length {m}."
                )

            # Acceptance probabilities
            a = np.zeros_like(f_vals)
            positive_g = g_vals > 0
            a[positive_g] = f_vals[positive_g] / (self.bound_c * g_vals[positive_g])
            a = np.clip(a, 0.0, 1.0)

            # Deterministic acceptance
            mask = V <= a
            Y_acc = Y[mask, :]

            if Y_acc.shape[0] > 0:
                accepted_chunks.append(Y_acc)

        # --- 4. Combine and cache leftovers ------------------------------------
        all_acc = np.vstack(accepted_chunks)
        samples = all_acc[:n, :]
        extras = all_acc[n:, :]

        self._accepted_cache = extras

        if return_weights:
            return samples, np.ones(n, float)
        return samples

    # ------------------------------------------------------------------
    # Minimal implementations to satisfy AbstractTrueMeasure interface.
    # We override gen_samples, so _transform/_weight are not used in the
    # main flow, but they are required by the base class.
    # ------------------------------------------------------------------
    def _transform(self, x: np.ndarray) -> np.ndarray:
        """
        Placeholder transform.

        In this design, ``gen_samples`` handles the full acceptance–rejection
        logic, so this method should not be called directly. It is defined
        only to satisfy the AbstractTrueMeasure API.
        """
        raise MethodImplementationError(
            self, "_transform is not used for AcceptReject."
        )

    def _weight(self, x: np.ndarray) -> np.ndarray:
        """
        Placeholder weight.

        For the accepted points, the target measure is already encoded in
        the AR logic, so the natural Jacobian weight is 1. This method is
        not used in the current design.
        """
        raise MethodImplementationError(self, "_weight is not used for AcceptReject.")

    def _spawn(self, sampler: AbstractDiscreteDistribution, dimension: int):
        """
        Spawning for AcceptReject is non-trivial (it must preserve the
        driver dimension = d+1 and the link to the proposal measure).
        We will implement this once the basic sampler is working.
        """
        raise MethodImplementationError(
            self, "_spawn not yet implemented for AcceptReject."
        )
