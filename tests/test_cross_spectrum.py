"""Tests for cross_spectrum_ratios.crystallinity_index.

The whole point of the index is that it is a WITHIN-spectrum ratio (FTO peak
height / TO-band integral), so it must be invariant to the overall intensity
scale (laser coupling) and must fall as a sharp folded-TO collapses into a broad
disordered band. Both properties are checked on synthetic spectra.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cross_spectrum_ratios import crystallinity_index  # noqa: E402
from models import lorentzian  # noqa: E402


def _spectrum(sharp_amp, broad_amp, sharp_wid=4.0, broad_wid=60.0):
    """A sharp folded-TO at 788 on top of a broad disordered band at 770."""
    x = np.linspace(700, 850, 600)
    y = (lorentzian(x, sharp_amp, 788.0, sharp_wid)
         + lorentzian(x, broad_amp, 770.0, broad_wid))
    return x, y


def test_coupling_invariant():
    """Scaling the whole spectrum (coupling change) must not change C."""
    x, y = _spectrum(sharp_amp=1.0, broad_amp=0.5)
    c1 = crystallinity_index(x, y)
    c2 = crystallinity_index(x, 37.5 * y)   # arbitrary coupling factor
    assert np.isfinite(c1)
    assert abs(c1 - c2) / c1 < 1e-9


def test_monotonic_with_disorder():
    """More broad (disordered) band relative to the sharp FTO -> lower C."""
    x_cryst, y_cryst = _spectrum(sharp_amp=1.0, broad_amp=0.3)   # crystalline
    x_dam, y_dam = _spectrum(sharp_amp=0.2, broad_amp=1.0)       # damaged
    assert crystallinity_index(x_cryst, y_cryst) > crystallinity_index(x_dam, y_dam)


def test_nan_when_windows_empty():
    x = np.linspace(900, 1000, 100)   # no coverage of FTO/TO windows
    y = np.ones_like(x)
    assert np.isnan(crystallinity_index(x, y))
