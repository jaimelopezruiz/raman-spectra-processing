"""Spectrum-file parsing regression tests for preprocessing._read_spectrum_table.

The quoted-line case is not hypothetical: the Au 2.5e15 RT-before spectrum
(`03 SiC Au 2,5E15 RT before anneal--Spec--003--Spec.Data 1.csv`) wraps every
line in double quotes, and it is the source of the published RT-before chi
(1.55, Figure 12a). Every delimiter-based pandas attempt reads such a line as a
single non-numeric field, so before the loose fallback was added the file did
not load at all and that number was not reproducible from the deposited code.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing import _read_spectrum_table  # noqa: E402

TAB = "\t"


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_plain_tab_separated_no_header(tmp_path):
    path = _write(tmp_path, "plain.csv",
                  f" 5.00000E+02{TAB} 1.00000E+03\r\n 5.01000E+02{TAB} 1.10000E+03\r\n")
    x, y = _read_spectrum_table(path)
    assert np.allclose(x, [500.0, 501.0])
    assert np.allclose(y, [1000.0, 1100.0])


def test_whole_line_double_quoted(tmp_path):
    """The Au RT-before export format: leading space, then a quoted line."""
    path = _write(tmp_path, "quoted.csv",
                  f' " 5.00000E+02{TAB} 1.00000E+03"\r\n'
                  f' " 5.01000E+02{TAB} 1.10000E+03"\r\n'
                  f' " 5.02000E+02{TAB} 1.20000E+03"\r\n')
    x, y = _read_spectrum_table(path)
    assert np.allclose(x, [500.0, 501.0, 502.0])
    assert np.allclose(y, [1000.0, 1100.0, 1200.0])


def test_witec_data_section_still_wins(tmp_path):
    """A [Data] section must be parsed by the dedicated path, header text skipped."""
    path = _write(tmp_path, "witec.txt",
                  "//Exported ASCII-File\n[Header]\nXAxisUnit = rel. 1/cm\n\n"
                  f"[Data]\nX-Axis{TAB}Spec.Data 1\n"
                  f" 5.00000E+02{TAB} 1.00000E+03\n 5.01000E+02{TAB} 1.10000E+03\n")
    x, y = _read_spectrum_table(path)
    assert np.allclose(x, [500.0, 501.0])
    assert np.allclose(y, [1000.0, 1100.0])


def test_unparseable_file_still_raises(tmp_path):
    path = _write(tmp_path, "junk.csv", "not a spectrum\nnor is this\n")
    try:
        _read_spectrum_table(path)
    except ValueError:
        return
    raise AssertionError("expected ValueError for a file with no numeric pairs")
