import numpy as np

from lcmish import trapezoid


def test_trapezoid_helper():
    x = np.linspace(0.0, 1.0, 1001)
    y = x**2
    assert np.isclose(trapezoid(y, x), 1 / 3, rtol=1e-5)
