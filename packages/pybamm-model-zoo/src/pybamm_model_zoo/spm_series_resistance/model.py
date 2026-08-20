"""Single Particle Model with a lumped ohmic series resistance."""

from __future__ import annotations

import pybamm
import pybamm_model_zoo
from pybamm_model_zoo import _compat

SLUG = "spm_series_resistance"
DEFAULT_SERIES_RESISTANCE = 0.01


class SPMSeriesResistance(pybamm.lithium_ion.SPM):
    """SPM whose terminal voltage carries a lumped ohmic drop, ``V - I R``.

    The resistance ``R`` stands in for everything outside the electrochemistry:
    tabs, welds, busbars, and cabling. It enters as a new parameter,
    ``"Series resistance [Ohm]"``, and is applied to the terminal voltage before
    the base class derives the cut-off events, power, and battery voltage from
    it, so those all see the drop too.

    Parameters
    ----------
    options : dict, optional
        Model options, as for :class:`pybamm.lithium_ion.SPM`. The
        ``"voltage as a state"`` option is not supported.
    name : str, optional
        The model name.
    build : bool, optional
        Whether to build the model on instantiation.

    Examples
    --------
    >>> import pybamm_model_zoo as zoo
    >>> model = zoo.load("SPMSeriesResistance")()
    >>> "Series resistance overpotential [V]" in model.variables
    True
    """

    def __init__(
        self,
        options: dict | None = None,
        name: str = "Single Particle Model with series resistance",
        build: bool = True,
    ) -> None:
        super().__init__(
            options=_compat.spm_default_options(options), name=name, build=build
        )
        pybamm_model_zoo.register_citation(SLUG)

    def set_voltage_variables(self) -> None:
        if self.options["voltage as a state"] == "true":
            raise pybamm.OptionError(
                "SPMSeriesResistance does not support 'voltage as a state': the "
                "algebraic constraint would pin the state to the voltage before "
                "the series drop is applied."
            )
        resistance = pybamm.Parameter("Series resistance [Ohm]")
        overpotential = -self.variables["Current [A]"] * resistance
        for key in ("Voltage [V]", "Terminal voltage [V]"):
            self.variables[key] = self.variables[key] + overpotential
        self.variables["Series resistance overpotential [V]"] = overpotential
        super().set_voltage_variables()

    @property
    def default_parameter_values(self) -> pybamm.ParameterValues:
        values = super().default_parameter_values
        values.update({"Series resistance [Ohm]": DEFAULT_SERIES_RESISTANCE})
        return values
