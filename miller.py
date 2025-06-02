import matplotlib.pyplot as plt
import numpy as np


def flux_surface(R0, A, kappa, delta, theta):
    """
    Creates a Miller flux surface

    Parameters
    ----------
    R0 : float
        Major radius of the magnetic axis
    A : float
        Aspect ratio
    kappa : float
        Elongation
    delta : float
        Triangularity
    theta : 1D np.ndarray
        Geometric poloidal angles

    Returns
    -------
    R_s : 1D np.ndarray
        Major radius of the flux surface
    Z_s : 1D np.ndarray
        Vertical coordinate of the flux surface
    """

    r = R0 / A

    return R0 + r * np.cos(
        theta + (np.arcsin(delta) * np.sin(theta))
    ), kappa * r * np.sin(theta)


def plot_surface(R_s, Z_s, savefig=True):
    """
    Creates a Miller flux surface

    Parameters
    ----------
    R_s : 1D np.ndarray
        Major radius of the flux surface
    Z_s : 1D np.ndarray
        Vertical coordinate of the flux surface
    savefig : bool, default=True

    Returns
    -------
    None
    """

    plt.plot(R_s, Z_s)
    plt.axis("equal")
    plt.xlabel("R [m]")
    plt.ylabel("Z [m]")

    if savefig:
        plt.savefig("./miller.png")


def main():
    R_s, Z_s = flux_surface(
        R0=2.5, A=2.2, kappa=1.5, delta=0.3, theta=np.linspace(0, 2 * np.pi)
    )

    plot_surface(R_s, Z_s)


if __name__ == "__main__":
    main()
