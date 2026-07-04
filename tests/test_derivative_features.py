import numpy as np

from seed_hsi_optical_passport.features import derivative_features


def test_derivative_feature_mapping():
    x = np.array([[1.0, 3.0, 6.0], [2.0, 4.0, 8.0]])
    wavelengths = np.array([400.0, 405.0, 415.0])
    derivative, names, mapping = derivative_features(x, wavelengths)
    assert derivative.shape == (2, 2)
    assert names[0].startswith("d001")
    assert mapping.loc[0, "midpoint_nm"] == 402.5
    assert np.isclose(derivative[0, 0], 0.4)
    assert np.isclose(derivative[0, 1], 0.3)
