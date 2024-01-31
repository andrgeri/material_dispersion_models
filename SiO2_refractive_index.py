import numpy as np
from scipy.constants import c


def nSiO2_Malitson(λ_):
    """SiO2 refractive index model according to [1].

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
    .. [1] I. H. Malitson, "Interspecimen Comparison of the Refractive Index
           of Fused Silica*,†", JOSA, 55, 10, 1205–1209, (1965).

    """

    λ = λ_ * 1e6
    ω = 2 * np.pi * c / λ

    B1 = 0.6961663
    B2 = 0.4079426
    B3 = 0.8974794
    λ1 = 0.0684043
    λ2 = 0.1162414
    λ3 = 9.8961610

    ω1 = 2 * np.pi * c / λ1
    ω2 = 2 * np.pi * c / λ2
    ω3 = 2 * np.pi * c / λ3

    n2 = 1 + B1 * ω1**2 / (ω1**2 - ω**2) + \
             B2 * ω2**2 / (ω2**2 - ω**2) + \
             B3 * ω3**2 / (ω3**2 - ω**2)

    return np.sqrt(n2)


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    λ = np.arange(0.6, 5 + 0.005, 0.005) * 1e-6
    n = nSiO2_Malitson(λ)

    fig, ax = plt.subplots()
    ax.plot(λ * 1e6, n)

    plt.show()
