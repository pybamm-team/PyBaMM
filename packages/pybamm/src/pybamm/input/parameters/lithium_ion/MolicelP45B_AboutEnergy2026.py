"""
Conversion of the About:Energy "User-defined" BPX fields is adapted from AEPyBaMM
v0.2.3 (https://github.com/About-Energy-OpenSource/AEPyBaMM/tree/v0.2.3),
``src/aepybamm/func.py`` and ``src/aepybamm/params.py``.
"""

from __future__ import annotations

import warnings
from functools import partial
from pathlib import Path

import pybamm

BPX_PATH = Path(__file__).parent / "data" / "MolicelP45B_AboutEnergy2026_bpx.json"

# BPX particle names, in the order the BPX importer assigns them to phases
PHASE_PREFIXES = {"Graphite: ": "Primary: ", "Silicon Additive: ": "Secondary: "}


def cross_polynomial(c_e, T, terms):
    """
    Polynomial in electrolyte concentration and temperature, sum(a * c_e^m * T^n).

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]
    terms : list of tuple
        Coefficient ``a`` and exponents ``m`` and ``n`` of each term

    Returns
    -------
    :class:`pybamm.Symbol`
        Value of the polynomial
    """
    return sum(a * c_e**m * T**n for a, m, n in terms)


def electrolyte_conductivity_Landesfeind2019(c_e, T, p1, p2, p3, p4, p5, p6):
    """
    Electrolyte conductivity in the functional form of :footcite:t:`Landesfeind2019`,
    with ``p1`` scaled to give SI units.

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]
    p1, p2, p3, p4, p5, p6 : float
        Fitting coefficients

    Returns
    -------
    :class:`pybamm.Symbol`
        Electrolyte conductivity [S.m-1]
    """
    c = c_e / 1000  # mol.m-3 -> mol.l-1
    exp_T = pybamm.exp(1000 / T)
    A = p1 * (1 + T - p2) * c
    B = 1 + p3 * pybamm.sqrt(c) + p4 * c * (1 + p5 * exp_T)
    C = 1 + c**4 * p6 * exp_T
    return A * B / C


def electrolyte_diffusivity_Landesfeind2019(c_e, T, p1, p2, p3, p4):
    """
    Electrolyte diffusivity in the functional form of :footcite:t:`Landesfeind2019`,
    with ``p1`` scaled to give SI units.

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]
    p1, p2, p3, p4 : float
        Fitting coefficients

    Returns
    -------
    :class:`pybamm.Symbol`
        Electrolyte diffusivity [m2.s-1]
    """
    c = c_e / 1000  # mol.m-3 -> mol.l-1
    return p1 * pybamm.exp(p2 * c) * pybamm.exp(p3 / T) * pybamm.exp(p4 * c / T)


def exchange_current_density(
    c_e,
    c_s_surf,
    c_s_max,
    T,
    reference_value,
    activation_energy,
    reference_temperature,
    pre_multiplier,
):
    """
    Exchange-current density with Arrhenius temperature dependence, scaled by a
    function of the surface stoichiometry.

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_s_surf : :class:`pybamm.Symbol`
        Particle concentration at the surface [mol.m-3]
    c_s_max : :class:`pybamm.Symbol`
        Maximum particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]
    reference_value : float
        Exchange-current density at the reference temperature [A.m-2]
    activation_energy : float
        Activation energy [J.mol-1]
    reference_temperature : float
        Reference temperature [K]
    pre_multiplier : callable
        Function of the surface stoichiometry scaling the exchange-current density

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """
    arrhenius = pybamm.exp(
        activation_energy / pybamm.constants.R * (1 / reference_temperature - 1 / T)
    )
    return pre_multiplier(c_s_surf / c_s_max) * reference_value * arrhenius


def _build_electrolyte_functions(values):
    """Replace "<name> func_type <type>" coefficient groups with functions."""
    func_types = dict(
        key.split(" func_type ") for key in list(values) if " func_type " in key
    )
    for name, func_type in func_types.items():
        coefficients = {
            key.removeprefix(f"{name} "): values.pop(key)
            for key in list(values)
            if key.startswith(f"{name} ")
        }
        if func_type == "crosspoly":
            terms = [
                (
                    coefficients[f"poly.poly{i}.a"],
                    coefficients[f"poly.poly{i}.m"],
                    coefficients[f"poly.poly{i}.n"],
                )
                for i in range(sum(key.endswith(".a") for key in coefficients))
            ]
            values[name] = partial(cross_polynomial, terms=terms)
        elif func_type == "Landesfeind2019_cond":
            values[name] = partial(
                electrolyte_conductivity_Landesfeind2019,
                **{f"p{i}": coefficients[f"p{i}"] for i in range(1, 7)},
            )
        elif func_type == "Landesfeind2019_diff":
            values[name] = partial(
                electrolyte_diffusivity_Landesfeind2019,
                **{f"p{i}": coefficients[f"p{i}"] for i in range(1, 5)},
            )


def _build_exchange_current_densities(values):
    """Replace each phase's BPX exchange-current density with one scaled by the
    "exchange-current density pre-multiplier" function of the surface stoichiometry.
    """
    suffix = "exchange-current density pre-multiplier"
    for key in [key for key in values if key.endswith(suffix)]:
        material, _, name = key.partition(": ")
        prefix = PHASE_PREFIXES[f"{material}: "] + name.removesuffix(suffix)
        values[f"{prefix}exchange-current density [A.m-2]"] = partial(
            exchange_current_density,
            reference_value=pybamm.constants.F.value
            * values.pop(f"{prefix}reaction rate constant [mol.m-2.s-1]"),
            activation_energy=values[
                f"{prefix}reaction rate constant activation energy [J.mol-1]"
            ],
            reference_temperature=values["Reference temperature [K]"],
            pre_multiplier=values.pop(key),
        )


def get_parameter_values():
    """
    Parameters for a Molicel INR21700-P45B cell (NCA / graphite-SiOx), from the BPX
    file published by About:Energy :footcite:p:`Dickinson2026`, with electrolyte
    functions of the form in :footcite:t:`Landesfeind2019`. Intended for the DFN with
    options ``{"particle phases": ("2", "1"), "open-circuit potential": (("single",
    "one-state hysteresis"), "single"), "contact resistance": "true"}``. The BPX
    "User-defined" fields are converted as in
    `AEPyBaMM v0.2.3 <https://github.com/About-Energy-OpenSource/AEPyBaMM/tree/v0.2.3>`_.

    .. note::
        Kinetic and transport parameters are effective values fitted to full-cell data.
        The negative electrode exchange-current densities depend only on the surface
        stoichiometry (through a pre-multiplier) and temperature, not on the
        electrolyte concentration. The initial hysteresis state is 0 (average branch).
    """
    pybamm.import_optional_dependency("bpx")
    with warnings.catch_warnings():
        # The BPX file gives no open-circuit voltages at 0% and 100% SOC, so the
        # voltage cut-offs are used, as intended
        warnings.filterwarnings(
            "ignore", message=r"'Open-circuit voltage at (0|100)% SOC \[V\]'"
        )
        values = dict(pybamm.ParameterValues.create_from_bpx(BPX_PATH).items())

    _build_electrolyte_functions(values)
    _build_exchange_current_densities(values)
    # Only used by the BPX importer to compute active material volume fractions
    for key in [key for key in values if "surface area per unit volume" in key]:
        del values[key]

    values["chemistry"] = "lithium_ion"
    values["citations"] = ["Dickinson2026", "Landesfeind2019"]
    return values
