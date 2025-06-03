import matplotlib.pyplot as plt
import numpy as np


def flux_surface(R0: float = 2.5,
                 A:float = 2.2,
                 kappa: float = 1.5,
                 delta:float = 0.3,
                 theta: np.ndarray =np.linspace(0, 2 * np.pi)):
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


def plot_surface(x: np.ndarray, y: np.ndarray,
                 xlabel: str ="x", ylabel: str ="y",
                 figname: str ="miller.png",
                 savefig: bool =True):
    """
    Creates a plot of provided data

    Parameters
    ----------
    x : 1D np.ndarray
    y : 1D np.ndarray
    xlabel : str
    ylabel : str
    figname : str
    savefig : bool, default=True

    Returns
    -------
    None
    """

    plt.plot(x, y)
    plt.axis("equal")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if savefig:
        plt.savefig(figname)

    plt.close()


def area(r, z):
    return np.abs(np.trapezoid(z, r))


def main():
    deltas = np.linspace(-1.0, 1.0, 1000)

    R_s, Z_s = flux_surface()

    R_s_vals, Z_s_vals = zip(*[flux_surface(delta=delta) for delta in deltas])

    areas = area(R_s_vals, Z_s_vals)

    plot_surface(R_s, Z_s)
    plot_surface(deltas, areas, figname="deltas.png")


if __name__ == "__main__":
    main()
