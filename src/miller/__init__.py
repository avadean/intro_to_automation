import argparse
import matplotlib.pyplot as plt
import numpy as np
import setuptools_scm as stscm
import tomli


def flux_surface(R0: float, A: float, kappa: float, delta: float, thetas: np.ndarray):
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
    thetas : 1D np.ndarray
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
        thetas + (np.arcsin(delta) * np.sin(thetas))
    ), kappa * r * np.sin(thetas)


def plot_surface(x: np.ndarray, y: np.ndarray,
                 xlabel: str = "x", ylabel: str = "y",
                 figname: str = "miller.png",
                 savefig: bool = True):
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
    parser = argparse.ArgumentParser(
        prog="Miller",
        description="Creates a Miller plot"
    )

    parser.add_argument("filein", nargs="?", default=None, help="filename of input")
    parser.add_argument("-f", "--fileout", type=str, default="miller.png", help="filename of output")
    parser.add_argument("-v", "--version", default=False, action="store_true", help="print version")
    parser.add_argument("-A", "--A", type=float, default=2.2, help="aspect ratio")
    parser.add_argument("-k", "--kappa", type=float, default=1.5, help="elongation")
    parser.add_argument("-d", "--delta", type=float, default=0.3, help="triangularity")
    parser.add_argument("-R", "--R0", type=float, default=2.5, help="major radius of magnetic axis")

    args = parser.parse_args()

    if args.version:
        print(f"Version: {stscm.get_version()}")
        return

    thetas = np.linspace(0.0, 2.0 * np.pi)
    deltas = np.linspace(-1.0, 1.0, 100)

    R_s, Z_s = flux_surface(R0=args.R0,
                            A=args.A,
                            kappa=args.kappa,
                            delta=args.delta,
                            thetas=thetas)

    R_s_vals, Z_s_vals = zip(*[flux_surface(R0=args.R0,
                                            A=args.A,
                                            kappa=args.kappa,
                                            delta=delta,
                                            thetas=thetas) for delta in deltas])

    areas = area(R_s_vals, Z_s_vals)

    plot_surface(R_s, Z_s, figname=args.fileout)
    plot_surface(deltas, areas, figname="deltas.png")


if __name__ == "__main__":
    main()
