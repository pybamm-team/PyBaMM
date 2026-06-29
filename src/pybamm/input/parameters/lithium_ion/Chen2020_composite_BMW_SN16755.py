"""
Chen2020_composite parameter set with a data-driven cathode OCP correction for BMW SN16755.

The base cathode OCP (nmc_LGM50_ocp_Chen2020) is supplemented with an empirical delta
correction fitted to the BMW SN16755 cell at 25 °C. The delta was computed from the
residual between the measured GITT-derived pOCV and the best DFN simulation, expressed
as a function of positive electrode stoichiometry (x_p).

The delta CSV must exist at:
  pybamm/input/parameters/lithium_ion/data/cathode_ocp_delta_sn16755_25dgc.csv

Generate it with:
  uv run scripts/build_ocp_delta.py --sim <...>/best_simulation.csv --params <...>/optimized_parameters.json
"""

from pathlib import Path

import pybamm
from pybamm.input.parameters.lithium_ion.Chen2020_composite import (
    get_parameter_values as _base_get_parameter_values,
    nmc_LGM50_ocp_Chen2020,
)

_data_path = str(Path(__file__).parent / "data")

# Loaded at import time — CSV must exist before first import of this module.
_delta_data = pybamm.parameters.process_1D_data(
    "cathode_ocp_delta_sn16755_25dgc.csv", path=_data_path
)


def nmc_LGM50_ocp_BMW_SN16755(sto):
    """NMC LGM50 OCP + empirical delta correction fitted to BMW SN16755 at 25 °C."""
    u_base = nmc_LGM50_ocp_Chen2020(sto)
    name, (x, y) = _delta_data
    delta = pybamm.Interpolant(x, y, sto, name=name, interpolator="linear")
    return u_base + delta


def get_parameter_values():
    """Chen2020_composite with corrected cathode OCP for BMW SN16755."""
    params = _base_get_parameter_values()
    params["Positive electrode OCP [V]"] = nmc_LGM50_ocp_BMW_SN16755
    return params
