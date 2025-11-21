import numpy as np

# from qmcpy.discrete_distribution import DigitalNetB2 received an error because the batch size had to be 0 or powers of 2
# (DigitalNetB2 in natural order requires n_min and n_max be 0 or powers of 2)
from qmcpy.true_measure import Uniform, AcceptReject
from qmcpy.discrete_distribution import IIDStdUniform


# 1D target: Beta(2,5) on [0,1]
def beta_2_5_pdf(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    x1 = x[:, 0]
    pdf = 30.0 * x1 * (1.0 - x1) ** 4
    pdf[(x1 < 0.0) | (x1 > 1.0)] = 0.0
    return pdf


# Proposal: Uniform(0,1)
def uniform_01_pdf(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    x1 = x[:, 0]
    g = np.ones_like(x1, dtype=float)
    g[(x1 < 0.0) | (x1 > 1.0)] = 0.0
    return g


def main():
    # Target/proposal dimension
    d = 1

    # 1D QMC driver for the proposal measure
    dd_proposal = IIDStdUniform(d)

    # Proposal TrueMeasure: Uniform(0,1) in 1D
    proposal = Uniform(dd_proposal, lower_bound=0.0, upper_bound=1.0)

    # 2D QMC driver for acceptance–rejection: [0,1]^{d+1}
    dd_driver = IIDStdUniform(d + 1)

    # Bound c >= sup_x f(x)/g(x); for Beta(2,5) max f(x) ≈ 2.4576, so 3 is safe
    c_bound = 3.0

    # Our acceptance–rejection TrueMeasure, using separate driver
    ar_measure = AcceptReject(
        proposal_measure=proposal,
        target_pdf=beta_2_5_pdf,
        proposal_pdf=uniform_01_pdf,
        bound_c=c_bound,
        driver_discrete_distrib=dd_driver,
    )

    n = 5000
    samples = ar_measure(n=n)

    print("samples.shape =", samples.shape)
    sample_mean = samples.mean()
    true_mean = 2.0 / 7.0
    print(f"sample mean ≈ {sample_mean:.6f}")
    print(f"true   mean  = {true_mean:.6f}")
    print(f"abs error    = {abs(sample_mean - true_mean):.6f}")


if __name__ == "__main__":
    main()
