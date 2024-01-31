import numpy as np


def nGaN_Zheng(λ_):
    """GaN refractive index model according to [1].

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
    .. [1] Y. Zheng et al., "Integrated Gallium Nitride Nonlinear Photonics", LPR, 2100071, (2021).


    """
    λ = λ_ * 1e6
    n2 = 3.6 + 1.75 * λ**2 / (λ**2 - 0.256**2) + 4.1 * λ**2 / (λ**2 - 17.86**2)
    return np.sqrt(n2)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    λ = np.linspace(0.5, 3, 1001) * 1e-6
    n = nGaN_Zheng(λ)

    fig, ax = plt.subplots()
    ax.plot(λ * 1e6, n)
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("GaN refractive index")

    plt.show()
