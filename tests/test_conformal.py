from __future__ import annotations

import pytest

from functime.conformal import _validate_alphas


def test_validate_alphas():
    assert _validate_alphas(None) == (0.1, 0.9)
    assert _validate_alphas([0.1, 0.9]) == [0.1, 0.9]
    assert _validate_alphas([0.1, 0.5]) == [0.1, 0.5]
    assert _validate_alphas([0.5, 0.9]) == [0.5, 0.9]

    with pytest.raises(ValueError):
        _validate_alphas([0.1, 0.5, 0.9, 0.2])
    with pytest.raises(ValueError):
        _validate_alphas([0.1, 0.5, 0.9, 0.2, 0.3])
    with pytest.raises(ValueError):
        _validate_alphas([0.1, -0.5])
    with pytest.raises(ValueError):
        _validate_alphas([0.1, 1.5])
    with pytest.raises(ValueError):
        _validate_alphas([-0.1, 0.5])
    with pytest.raises(ValueError):
        _validate_alphas([1.1, 0.5])
