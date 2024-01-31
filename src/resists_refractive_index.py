import numpy as np


def nCSAR62(λ):
    λ_nm = λ * 1e9
    n = 1.543 + 71.4 / λ_nm**2 + 0 / λ_nm**4
    return n

def nMEDUSA82(λ):
    λ_nm = λ * 1e9
    n = 1.461 + 72 / λ_nm**2 + 0 / λ_nm**4
    return n

def nMAN2400(λ):
    λ_nm = λ * 1e9
    n = 1601e-3 + 123e2 / λ_nm**2 + 0e7 / λ_nm**4
    return n

def nPMMA950(λ):
    λ_µm = λ * 1e6
    n = 1.488 + 2.898e-3 / λ_µm**2 + 1.579e-4 / λ_µm**4
    return n


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    λ = np.linspace(400, 1100, 1001) * 1e-9
    n_ = [nCSAR62, nMEDUSA82, nMAN2400, nPMMA950]
    lb_ = ["CSAR 62", "Medusa 82", "ma-N2400", "PMMA (950)"]

    fig, ax = plt.subplots(tight_layout=True)

    for n, lb in zip(n_, lb_):
        ax.plot(1e9 * λ, n(λ), label=lb)

    ax.legend()
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Refractive index")

    plt.show()
