"""Parameter sets and data of Paten et al. (2026).

:func:`paten2026_dc1` and :func:`paten2026_dfn` are Table IV (LG M50 cell) on
top of :footcite:t:`Chen2020`, with tortuosities given both as tortuosity
factors and as Bruggeman coefficients. :func:`paten2026_3d_case` and
:func:`load_3d_case_data` are the tomographic NMC half cell, with its
transient closure variable and a DNS of a discharge and rest.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import pybamm
import pybamm_model_zoo as zoo

#: Folder of the 3D case data.
DATA_3D_CASE = Path(__file__).parent / "data" / "3d_case_data_Paten_2026"

#: Table IV, DFN column.
TABLE_IV_DFN = {
    "Initial concentration in negative electrode [mol.m-3]": 29541.5,
    "Initial concentration in positive electrode [mol.m-3]": 16842.2,
    "Maximum concentration in negative electrode [mol.m-3]": 32694.2,
    "Maximum concentration in positive electrode [mol.m-3]": 62266.0,
    "negative rate constant": 4.908e-7,
    "positive rate constant": 4.414e-6,
    "negative tortuosity (electrolyte)": 2.738,
    "positive tortuosity (electrolyte)": 2.802,
    "Negative particle diffusivity [m2.s-1]": 1.332e-13,
    "Positive particle diffusivity [m2.s-1]": 5.126e-15,
    "Negative particle radius [m]": 6.401e-6,
    "Positive particle radius [m]": 5.266e-6,
}

#: Table IV, DC1 column.
TABLE_IV_DC1 = {
    "Initial concentration in negative electrode [mol.m-3]": 29901.3,
    "Initial concentration in positive electrode [mol.m-3]": 18051.3,
    "Maximum concentration in negative electrode [mol.m-3]": 33304.8,
    "Maximum concentration in positive electrode [mol.m-3]": 57399.3,
    "negative rate constant": 9.480e-7,
    "positive rate constant": 2.053e-6,
    "negative tortuosity (electrolyte)": 2.735,
    "positive tortuosity (electrolyte)": 1.944,
    "negative specific surface area": 441818.3,
    "positive specific surface area": 412729.1,
    "Negative electrode s0 surface average": -335.06,
    "Positive electrode s0 surface average": -500.38,
}

#: Activation energies of the Chen2020 exchange-current densities [J.mol-1].
ACTIVATION_ENERGY = {"negative": 35000, "positive": 17800}


def paten2026_dc1() -> dict:
    """Table IV, DC1 column, on top of Chen2020.

    Table IV's specific surface area is used as the AM-electrolyte area, with
    no AM-CBD or AM-separator contribution.
    """
    zoo.register_citation("dual_continuum", "Paten2026")
    table = TABLE_IV_DC1
    values = _with_common_entries(table)
    values["citations"].append("Paten2026")
    values.update(
        {
            "Negative electrode s0 surface average": table[
                "Negative electrode s0 surface average"
            ],
            "Positive electrode s0 surface average": table[
                "Positive electrode s0 surface average"
            ],
            "CBD surface porosity": 0.5,
            "Separator surface porosity": values["Separator porosity"],
        },
    )
    for domain in ["negative", "positive"]:
        prefix = f"{domain.capitalize()} electrode specific surface area from image"
        values.update(
            {
                f"{prefix} (AM-electrolyte) [m-1]": table[
                    f"{domain} specific surface area"
                ],
                f"{prefix} (AM-CBD) [m-1]": 0.0,
                f"{prefix} (AM-separator) [m-1]": 0.0,
            },
        )
    return values


def paten2026_dfn() -> pybamm.ParameterValues:
    """Table IV, DFN column, on top of Chen2020, for comparison with DC1."""
    zoo.register_citation("dual_continuum", "Paten2026_DFN")
    table = TABLE_IV_DFN
    values = _with_common_entries(table)
    values["citations"].append("Paten2026_DFN")
    for key in [
        "Negative particle diffusivity [m2.s-1]",
        "Positive particle diffusivity [m2.s-1]",
        "Negative particle radius [m]",
        "Positive particle radius [m]",
    ]:
        values[key] = table[key]
    return values


def _with_common_entries(table):
    """Chen2020 with the Table IV entries shared by both columns."""
    values = pybamm.parameter_sets["Chen2020"]
    for key in [
        "Initial concentration in negative electrode [mol.m-3]",
        "Initial concentration in positive electrode [mol.m-3]",
        "Maximum concentration in negative electrode [mol.m-3]",
        "Maximum concentration in positive electrode [mol.m-3]",
    ]:
        values[key] = table[key]

    for domain in ["negative", "positive"]:
        Domain = domain.capitalize()
        values[f"{Domain} electrode exchange-current density [A.m-2]"] = (
            _exchange_current_density(
                table[f"{domain} rate constant"], ACTIVATION_ENERGY[domain]
            )
        )
        porosity = values[f"{Domain} electrode porosity"]
        tortuosity = table[f"{domain} tortuosity (electrolyte)"]
        brugg_solid = values[f"{Domain} electrode Bruggeman coefficient (electrode)"]
        values.update(
            {
                # eps**b = eps / tau
                f"{Domain} electrode Bruggeman coefficient (electrolyte)": 1
                - math.log(tortuosity) / math.log(porosity),
                f"{Domain} electrode tortuosity factor (electrolyte)": tortuosity,
                f"{Domain} electrode tortuosity factor (electrode)": values[
                    f"{Domain} electrode active material volume fraction"
                ]
                ** (1 - brugg_solid),
            },
        )

    separator_porosity = values["Separator porosity"]
    values.update(
        {
            "Separator tortuosity factor (electrolyte)": separator_porosity
            ** (1 - values["Separator Bruggeman coefficient (electrolyte)"])
        },
    )
    return values


def _exchange_current_density(rate_constant, activation_energy):
    """Chen2020's exchange-current density with Table IV's rate constant."""

    def exchange_current_density(c_e, c_s_surf, c_s_max, T):
        arrhenius = pybamm.exp(
            activation_energy / pybamm.constants.R * (1 / 298.15 - 1 / T)
        )
        return (
            rate_constant
            * arrhenius
            * c_e**0.5
            * c_s_surf**0.5
            * (c_s_max - c_s_surf) ** 0.5
        )

    return exchange_current_density


def paten2026_3d_case() -> pybamm.ParameterValues:
    """NMC half cell reconstructed from a tomographic image, without CBD.

    The DNS comparison of Paten et al. (2026), on top of "Xu2019" for values
    the models do not use. "Current function [A]" is the DNS current (also
    1C). Fit the transient closure modes to :func:`load_3d_case_data` with
    :func:`~pybamm_model_zoo.dual_continuum.fit_transient_closure`.

    Returns
    -------
    pybamm.ParameterValues
        Parameters for the DFN and the dual-continuum model, for
        ``{"working electrode": "positive"}``.
    """
    eps_s, eps_e = 0.492777, 0.505414
    area = 75.62e-6 * 75.62e-6
    current = 50.3357753 * area
    values = pybamm.parameter_sets["Xu2019"]
    values["citations"].append("Paten2026_3D_NMC")
    values.update(
        {
            "Electrode height [m]": 75.62e-6,
            "Electrode width [m]": 75.62e-6,
            "Positive electrode thickness [m]": 75.62e-6,
            "Separator thickness [m]": 20e-6,
            "Positive electrode active material volume fraction": eps_s,
            "Positive electrode porosity": eps_e,
            "Separator porosity": 0.4,
            # eps**b = eps / tau, with the tortuosities from the image
            "Positive electrode Bruggeman coefficient (electrode)": 1
            - math.log(3.285) / math.log(eps_s),
            "Positive electrode Bruggeman coefficient (electrolyte)": 1
            - math.log(1.739) / math.log(eps_e),
            "Separator Bruggeman coefficient (electrolyte)": 1.9,
            "Positive electrode conductivity [S.m-1]": 2.8,
            "Positive particle diffusivity [m2.s-1]": 4e-14,
            "Positive particle radius [m]": 3.4054911011155747e-06,
            "Maximum concentration in positive electrode [mol.m-3]": 50400.0,
            "Initial concentration in positive electrode [mol.m-3]": 25000.0,
            "Positive electrode OCP [V]": _nmc_ocp,
            "Positive electrode exchange-current density [A.m-2]": (
                _nmc_exchange_current_density
            ),
            "Positive electrode OCP entropic change [V.K-1]": 0.0,
            "Exchange-current density for lithium metal electrode [A.m-2]": 10.0,
            "Initial concentration in electrolyte [mol.m-3]": 1000.0,
            "Electrolyte diffusivity [m2.s-1]": 1e-10,
            "Electrolyte conductivity [S.m-1]": 1.0,
            "Cation transference number": 0.4,
            "Thermodynamic factor": 1.0,
            "Reference temperature [K]": 298.15,
            "Ambient temperature [K]": 298.15,
            "Initial temperature [K]": 298.15,
            "Current function [A]": current,
            "Nominal cell capacity [A.h]": current,
            "Number of electrodes connected in parallel to make a cell": 1.0,
            "Lower voltage cut-off [V]": 3.0,
            "Upper voltage cut-off [V]": 4.5,
        },
    )
    values.update(
        {
            "Positive electrode specific surface area from image "
            "(AM-electrolyte) [m-1]": 391289.8,
            "Positive electrode specific surface area from image "
            "(AM-separator) [m-1]": 5735.779,
            "Positive electrode specific surface area from image (AM-CBD) [m-1]": 0.0,
            "Separator surface porosity": 1.0,
            # No CBD in this electrode, so this value is not used
            "CBD surface porosity": 0.5,
            "Positive electrode s0 surface average": -416.7084594772896,
            "Positive electrode Yang fitting parameter": 1.0,
        },
    )
    return values


def load_3d_case_data() -> dict[str, dict[str, np.ndarray]]:
    """Closure and DNS data of the 3D NMC half cell of :func:`paten2026_3d_case`.

    Returns
    -------
    dict
        ``"closure"``: ``"Time [s]"`` and ``"Closure variable [mol.m-1.A-1]"``,
        the surface-averaged transient closure variable solved on the
        microstructure. ``"dns"``: ``"Time [s]"``, ``"Voltage [V]"`` and
        ``"Current density [A.m-2]"`` (negative for discharge) of the DNS.
    """
    return {
        "closure": _read_csv(DATA_3D_CASE / "closure_transient.csv"),
        "dns": _read_csv(DATA_3D_CASE / "dns_discharge_relaxation.csv"),
    }


def _read_csv(path):
    """Columns of a CSV file whose last comment line holds the column names."""
    with open(path, encoding="utf-8") as file:
        names = [line for line in file if line.startswith("#")][-1]
    names = names.lstrip("# ").strip().split(",")
    columns = np.loadtxt(path, delimiter=",", ndmin=2).T
    return dict(zip(names, columns, strict=True))


def _nmc_ocp(sto):
    """NMC open-circuit potential [V] of the 3D case (Kremer et al., 2020)."""
    return (
        13.49050
        - 10.96038 * sto
        + 8.203617 * sto**1.358699
        - 3.10758e-6 * pybamm.exp(127.1216 * sto - 114.2593)
        - 7.033556 * sto ** (-3.362749e-2)
    )


def _nmc_exchange_current_density(c_e, c_s_surf, c_s_max, T):
    """NMC exchange-current density [A.m-2] of the 3D case, k F = 4.824e-6."""
    return 4.824e-6 * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5
