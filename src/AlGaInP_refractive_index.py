import numpy as np
from scipy.constants import h, c, elementary_charge as q

def wavelength_to_eV(λ):
    E = h * c / q / λ
    return E

def H(x, ε):  # Heviside function
    return 0.5 * (np.sign(x) + 1)

def E0(x):
    return 1.89 + 0.64 * x  # [eV]

def A(x):
    return 4.27 + 6.45 * x  # [eV**1.5]

def Γ(x):
    return 0.059 - 0.031 * x  # [eV]

def E1(x):
    return 3.28 + 0.33 * x + 0.13 * x**2  # [eV]

def B1(x):
    return 5.02 + 0.88 * x - 1.95 * x**2

def B1x(x):
    return 2.39 + 0.15 * x + 0.50 * x**2  # [eV]

def Γ1(x):
    return 0.30 + 0.12 * x - 0.06 * x**2  # [eV]

def E2(x):
    return 4.78 - 0.05 * x + 0.06 * x**2  # [eV]

def C(x):
    return 2.03 - 1.89 * x + 1.37 * x**2

def γ(x):
    return 0.178 - 0.105 * x + 0.085 * x**2

def ε1_inf(x):
    return 0.56 - 0.68 * x + 0.33 * x**2


def dielectric_function(E, x):
    Δ0 = 0
    fso = 0

    # E0 transitions
    χ0 = (E + 1j * Γ(x)) / E0(x)
    χso = (E + 1j * Γ(x)) / (E0(x) + Δ0)
    # f0 = (2 - np.sqrt(1 + χ0) - np.sqrt(1 - χ0) * H(1 - χ0)) / χ0**2
    f0 = (2 - np.sqrt(1 + χ0) - np.sqrt(1 - χ0)) / χ0**2
    fso = (2 - np.sqrt(1 + χso) - np.sqrt(1 - χso)) / χso**2
    ε = A(x) * E0(x)**(-1.5) * (f0 + 0.5 * np.sqrt(E0(x) / (E0(x) + Δ0)) * fso)

    # E1 transitions
    χ1d = (E + 1j * Γ1(x)) / E1(x)
    ε += -B1(x) / χ1d**2 * np.log(1 - χ1d**2)

    # Wannier-type 2D excitons
    n = 1
    G2D_1 = 0
    E2D_x1 = E1(x) - G2D_1 / (n - 0.5)**2
    ε += B1x(x) / (E2D_x1 - E - 1j * Γ1(x))

    # E2 transitions
    χ2 = E / E2(x)
    ε += C(x) / (1 - χ2**2 - 1j * χ2 * γ(x))

    ε += ε1_inf(x)
    ε = ε.real + 1j * ε.imag * H(0.02, E / E0(x) - 1)
    return ε


def refractive_index(E, x):
    ε = dielectric_function(E, x)
    ε1 = ε.real
    ε2 = ε.imag

    n_ = np.sqrt(0.5 * (np.sqrt(ε1**2 + ε2**2) + ε1))
    k_ = np.sqrt(0.5 * (np.sqrt(ε1**2 + ε2**2) - ε1))
    n = n_ + 1j * k_
    return n


def nAlGaInP_Kato(λ, x):
    """[Al(x)Ga(1-x)](0.5)In(0.5)P refractive index model according to [1].

    Parameters
    ----------
    x : float
        Al molar fraction (0 <= x <= 1)
    λ : float
        Wavelength in [m] (λ > 0)

    Returns
    -------
    n + ik : complex
        Complex refractive index

    References
    ----------
    .. [1] H. Kato, S. Adachi, H. Nakanishi, and K. Ohtsuka,
       "Optical Properties of (AlxGa1-x)0.5In0.5P Quaternary Alloys",
       Jpn. J. Appl. Phys., 33, 1R, 186 (1994).

    """
    E = wavelength_to_eV(λ)  # [eV]
    return refractive_index(E, x)


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    E = np.linspace(1, 6, 101)  # [eV], photon energy
    x = np.arange(0, 1.2, 0.2)  # [-], Al molar fraction

    fig, axs = plt.subplots(ncols=2, nrows=2, sharex=True)
    for X in x:
        lb = f"x = {X:.1f}"
        ε = dielectric_function(E, X)
        n = refractive_index(E, X)

        axs[0, 0].plot(E, ε.real, label=lb)
        axs[1, 0].plot(E, ε.imag, label=lb)
        axs[0, 1].plot(E, n.real, label=lb)
        axs[1, 1].plot(E, n.imag, label=lb)

    y_labels = ["ε₁", "ε₂", "n", "k"]
    y_limits = [(-10, 20), (0, 25), (0, 5), (0, 4)]
    for ax, ylb, ylims in zip(axs.T.flatten(), y_labels, y_limits):
        ax.set_ylabel(ylb)
        ax.set_xlim(0, 6)
        ax.set_ylim(ylims)
        ax.legend(frameon=False)

    for ax in axs[-1, :]:
        ax.set_xlabel("Photon energy (eV)")

    fig.suptitle("(Al$_x$Ga$_{1-x}$)$_{0.5}$In$_{0.5}$P/GaAs")
    fig.tight_layout()

    λ = np.linspace(h * c / (q * E.max()), h * c / (q * E.min()), 1001)

    fig, axs = plt.subplots(ncols=2, nrows=2, sharex=not True)
    for X in x:
        lb = f"x = {X:.1f}"

        n = nAlGaInP_Kato(λ, X)
        axs[0, 0].plot(λ * 1e6, n.real, label=lb)
        axs[1, 0].plot(λ * 1e6, n.imag, label=lb)

        E_ = wavelength_to_eV(λ)
        axs[0, 1].plot(E_, n.real, label=lb)
        axs[1, 1].plot(E_, n.imag, label=lb)

    axs[0, 0].set_xlabel("Wavelength (µm)")
    axs[0, 0].set_ylabel("n")
    axs[1, 0].set_xlabel("Wavelength (µm)")
    axs[1, 0].set_ylabel("k")
    axs[0, 1].set_xlabel("Photon energy (eV)")
    axs[0, 1].set_ylabel("n")
    axs[1, 1].set_xlabel("Photon energy (eV)")
    axs[1, 1].set_ylabel("k")

    plt.show()
