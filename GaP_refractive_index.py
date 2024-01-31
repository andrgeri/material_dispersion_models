import numpy as np


def nGaP_Wei(λ, Temperature=300):
    """GaP refractive index model according to [1].

    Parameters
    ----------
    λ : float
        Wavelength in [µm] (λ > 0)
    Temperature : float
        Temperature in [K]

    Returns
    -------
    n + ik : complex
        Complex refractive index

    References
    ----------
    .. [1] J. Wei, J. M. Murray, J. O. Barnes, D. M. Krein, P. G. Schunemann,
       and S. Guha, "Temperature dependent Sellmeier equation for the
       refractive index of GaP", Optical Materials Express 8, 485-490 (2018).

    """
    # T = Temperature + 273.15
    T = Temperature

    A = 10.926 + 7.0787e-4 * T + 1.8594e-7 * T**2
    B = 0.53718 + 5.8035e-5 * T + 1.9819e-7 * T**2
    C = 0.0911014
    D = 1504 + 0.25935 * T - 0.00023326 * T**2
    E = 758.048
    n2 = A + B / (λ**2 - C) + D / (λ**2 - E)
    return np.sqrt(n2, dtype=complex)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    λ = np.linspace(0.7, 12.5, 1001)  # [µm]
    Temperatures = [78, 150, 200, 250, 300, 350, 400, 450]  # [K]

    fig, ax = plt.subplots()
    for T in Temperatures:
        n = nGaP_Wei(λ, T)
        ax.plot(λ, n.real, label=f"T = {T}K")
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("GaP refractive index")
    ax.legend()
    ax.set_xlim(0.7, 12.5)
    ax.set_ylim(2.90, 3.25)

    plt.show()
