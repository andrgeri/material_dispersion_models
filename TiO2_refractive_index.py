import numpy as np
from scipy.constants import h, c, e as q


def nTiO2_Devlin(λ):
    """Amorphous TiO2 refractive index model according to [1].

    Parameters
    ----------
    λ : float
        Wavelength in [m] (λ > 0)

    Returns
    -------
    n + ik : complex
        Complex refractive index

    References
    ----------
    .. [1] R. C. Devlin, M. Khorasaninejad, W. T. Chen, J. Oh, and F. Capasso, ‘Broadband high-efficiency dielectric metasurfaces for the visible spectrum’, PNAS (2016), doi: 10.1073/pnas.1611740113.;

    """

    ε1, ε2 = dielectric_constant(λ)
    n2 = 0.5 * (np.sqrt(np.abs(ε1)**2 + np.abs(ε2)**2) + ε1)
    κ2 = 0.5 * (np.sqrt(np.abs(ε1)**2 + np.abs(ε2)**2) - ε1)
    n = np.sqrt(n2)
    κ = np.sqrt(κ2)
    n_ = n + 1j * κ
    return n_


def dielectric_constant(λ):
    """
    Parameters
    ----------
    λ : float
        Wavelenth in [m] (λ > 0)

    Returns
    -------
    ε1, ε2 : complex
             Complex dielectric constant

    References
    ----------
    .. [2] G. E. Jellison Jr. and F. A. Modine, ‘Parameterization of the optical functions of amorphous materials in the interband region’, APL (1996), doi: 10.1063/1.118064;
    .. [3] G. E. Jellison Jr. and F. A. Modine, ‘Erratum: “‘Parameterization of the optical functions of amorphous materials in the interband region’”, APL (1996), doi: 10.1063/1.118155.

    """

    E = h * c / λ / q  # [eV], energy

    # Fitting values from Devlin_2016 [1]
    A = 422.4  # [eV]
    C = 1.434  # [eV]
    E0 = 3.819  # [eV]
    Eg = 3.456  # [eV]
    ε_inf = 2.13  # [-], not present in Devlin_2016

    # Imaginary part of dielectric constant for amorphous materials
    # From Jellison_1996 [2]
    T1 = A * C * E0 * (E - Eg)**2 / ((E**2 - E0**2)**2 + C**2 * E**2) / E
    T2 = 0
    ε2 = T1 * (E > Eg) + T2 * (E <= Eg)

    # Real part of dielectric constant for amorphous materials
    # From Jellison_1996 [2, 3]
    γ = np.sqrt(E0**2 - 0.5 * C**2)
    α = np.sqrt(4 * E0**2 - C**2)
    ξ4 = (E**2 - γ**2)**2 + 0.25 * α**2 * C**2
    a_atan = (E**2 - E0**2) * (E0**2 + Eg**2) + Eg**2 * C**2
    a_ln = (Eg**2 - E0**2) * E**2 + Eg**2 * C**2 - E0**2 * (E0**2 + 3 * Eg**2)

    A1 = (A * C / np.pi / ξ4) * (a_ln / 2 / α / E0) * np.log((E0**2 + Eg**2 + α * Eg) / (E0**2 + Eg**2 - α * Eg))
    A2 = -(A / np.pi / ξ4) * a_atan / E0 * (np.pi - np.arctan((2 * Eg + α) / C) + np.arctan((-2 * Eg + α) / C))
    A3 = 2 * (A * E0 / np.pi / ξ4 / α) * Eg * (E**2 - γ**2) * (np.pi + 2 * np.arctan(2 * (γ**2 - Eg**2) / α / C))
    A4 = -(A * E0 * C / np.pi / ξ4) * (E**2 + Eg**2) / E * np.log(np.abs(E - Eg) / (E + Eg))
    A5 = 2 * (A * E0 * C / np.pi / ξ4) * Eg * np.log((np.abs(E - Eg) * (E + Eg)) / np.sqrt((E0**2 - Eg**2)**2 + Eg**2 * C**2))

    ε1 = ε_inf + A1 + A2 + A3 + A4 + A5
    return ε1, ε2


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    λ = np.linspace(0.25, 1.0, 1001) * 1e-6  # [m], wavelength
    n = nTiO2_Devlin(λ)

    fig, ax_n = plt.subplots(tight_layout=True)
    ax_k = ax_n.twinx()

    ax_n.plot(λ * 1e9, n.real, color="black")
    ax_k.plot(λ * 1e9, n.imag, color="red")

    ax_n.set_title("Amorphous TiO₂ from Devlin_2016")
    ax_n.set_xlabel("Wavelength (nm)")
    ax_n.set_xlim(λ.min() * 1e9, λ.max() * 1e9)
    ax_n.set_ylim(2.0, 3.5)
    ax_k.set_ylim(0, 1.5)
    ax_n.set_ylabel(r"$n$", color="black")
    ax_k.set_ylabel(r"$\kappa$", color="red")

    try:
        data_n = np.loadtxt("n_delvin.txt", skiprows=1, delimiter=";")
        data_k = np.loadtxt("k_delvin.txt", skiprows=1, delimiter=";")

        fig, axs = plt.subplots(ncols=2)
        axs[0].plot(data_n[:, 0], data_n[:, 1] / 10, linestyle="None", marker=".", markersize=5)
        axs[0].plot(λ * 1e9, n.real)
        axs[1].plot(data_k[:, 0], data_k[:, 1] / 10, linestyle="None", marker=".", markersize=5)
        axs[1].plot(λ * 1e9, n.imag)

    except FileNotFoundError:
        pass


    plt.show()

# from scipy.signal import hilbert
# fig, axs = plt.subplots(ncols=3)
# ε2 = nTiO2_Devlin(λ)  # real
# ε1 = jellison(λ)
# #ε1 = 1 + 2 / np.pi * hilbert(ε2.real)

# n2 = 0.5 * (np.sqrt(np.abs(ε1)**2 + np.abs(ε2)**2) + ε1)
# κ2 = 0.5 * (np.sqrt(np.abs(ε1)**2 + np.abs(ε2)**2) - ε1)
# n = np.sqrt(n2)
# κ = np.sqrt(κ2)

# axs[1].plot(λ * 1e9, ε1.real, label="ε1.real")
# axs[1].plot(λ * 1e9, ε1.imag, label="ε1.imag")
# axs[0].plot(λ * 1e9, ε2.real, label="ε2.real")
# axs[0].plot(λ * 1e9, ε2.imag, label="ε2.imag")
# axs[2].plot(λ * 1e9, n, label="n")
# axs[2].plot(λ * 1e9, κ, label="κ")


# for ax in axs:
#     ax.legend()
