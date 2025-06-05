import numpy as np
import pytest

from hypothesis import assume, given, strategies as st

from miller import flux_surface, plot_surface


@given(
    R0=st.floats(min_value=-100.0, max_value=100.0),
    A=st.floats(min_value=-100.0, max_value=100.0).filter(lambda a: not np.isclose(a, 0.0)),
    kappa=st.floats(min_value=-100.0, max_value=100.0),
    delta=st.floats(min_value=-1.0, max_value=1.0)
)
def test_flux_surface(R0: float, A: float, kappa:float, delta: float):
    thetas = np.linspace(0.0, 2.0 * np.pi, 100)

    R_s, Z_s = flux_surface(R0=R0,
                            A=A,
                            kappa=kappa,
                            delta=delta,
                            thetas=thetas)

def test_flux_surface_circle():
    thetas = np.linspace(0.0, 2.0 * np.pi, 100)

    R_s, Z_s = flux_surface(R0=1.0,
                            A=1.0,
                            kappa=1.0,
                            delta=0.0,
                            thetas=thetas)

    # (x-x0)**2 + (y-y0)**2 = 1.0 ; x0=1, y0=0
    # radius should be 1
    assert np.testing.assert_allclose((R_s - 1.0) ** 2.0 + Z_s ** 2.0, 1.0) is None

def test_flux_surface_A_is_0():
    thetas = np.linspace(0.0, 2.0 * np.pi, 100)

    with pytest.raises(ValueError, match="Aspect ratio should not be 0."):
        R_s, Z_s = flux_surface(R0=1.0,
                                A=0.0,
                                kappa=1.0,
                                delta=0.5,
                                thetas=thetas)

def test_flux_surface_delta_1():
    thetas = np.linspace(0.0, 2.0 * np.pi, 100)

    with pytest.raises(ValueError, match="Magnitude of triangularity must not be greater than 1."):
        R_s, Z_s = flux_surface(R0=1.0,
                                A=1.0,
                                kappa=1.0,
                                delta=-1.1,
                                thetas=thetas)
