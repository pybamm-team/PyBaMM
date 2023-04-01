from bpx import BPX, Function, InterpolatedTable
import pybamm
import math
from dataclasses import dataclass
import numpy as np
from pybamm import constants
from pybamm import exp
import copy


import types
import functools


def _copy_func(f):
    """Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)"""
    g = types.FunctionType(
        f.__code__,
        f.__globals__,
        name=f.__name__,
        argdefs=f.__defaults__,
        closure=f.__closure__,
    )
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g


@dataclass
class Domain:
    name: str
    pre_name: str
    short_pre_name: str


cell = Domain(name="cell", pre_name="", short_pre_name="")
negative_electrode = Domain(
    name="negative electrode",
    pre_name="Negative electrode ",
    short_pre_name="Negative ",
)
positive_electrode = Domain(
    name="positive electrode",
    pre_name="Positive electrode ",
    short_pre_name="Positive ",
)
electrolyte = Domain(name="electrolyte", pre_name="Electrolyte ", short_pre_name="")
separator = Domain(name="separator", pre_name="Separator ", short_pre_name="")
experiment = Domain(name="experiment", pre_name="", short_pre_name="")


def _bpx_to_param_dict(bpx: BPX) -> dict:
    pybamm_dict = {}
    pybamm_dict = _bpx_to_domain_param_dict(
        bpx.parameterisation.cell, pybamm_dict, cell
    )
    pybamm_dict = _bpx_to_domain_param_dict(
        bpx.parameterisation.negative_electrode, pybamm_dict, negative_electrode
    )
    pybamm_dict = _bpx_to_domain_param_dict(
        bpx.parameterisation.positive_electrode, pybamm_dict, positive_electrode
    )
    pybamm_dict = _bpx_to_domain_param_dict(
        bpx.parameterisation.electrolyte, pybamm_dict, electrolyte
    )
    pybamm_dict = _bpx_to_domain_param_dict(
        bpx.parameterisation.separator, pybamm_dict, separator
    )
    pybamm_dict = _bpx_to_domain_param_dict(
        bpx.parameterisation.separator, pybamm_dict, experiment
    )

    # set a default current function and typical current based on the nominal capacity
    # i.e. a default C-rate of 1
    pybamm_dict["Current function [A]"] = pybamm_dict["Nominal cell capacity [A.h]"]

    # activity
    pybamm_dict["Thermodynamic factor"] = 1.0

    # assume Bruggeman relation for effection electrolyte properties
    for domain in [negative_electrode, separator, positive_electrode]:
        pybamm_dict[domain.pre_name + "Bruggeman coefficient (electrolyte)"] = 1.5

    # solid phase properties reported in BPX are already "effective",
    # so no correction is applied
    for domain in [negative_electrode, positive_electrode]:
        pybamm_dict[domain.pre_name + "Bruggeman coefficient (electrode)"] = 0

    # BPX is for single cell in series, user can change this later
    pybamm_dict["Number of cells connected in series to make a battery"] = 1
    pybamm_dict[
        "Number of electrodes connected in parallel to make a cell"
    ] = pybamm_dict["Number of electrode pairs connected in parallel to make a cell"]

    # electrode area
    equal_len_width = math.sqrt(pybamm_dict["Electrode area [m2]"])
    pybamm_dict["Electrode width [m]"] = equal_len_width
    pybamm_dict["Electrode height [m]"] = equal_len_width

    # surface area
    pybamm_dict["Cell cooling surface area [m2]"] = pybamm_dict[
        "External surface area [m2]"
    ]

    # volume
    pybamm_dict["Cell volume [m3]"] = pybamm_dict["Volume [m3]"]

    # reference temperature
    T_ref = pybamm_dict["Reference temperature [K]"]

    # lumped parameters
    for name in [
        "Specific heat capacity [J.K-1.kg-1]",
        "Density [kg.m-3]",
        "Thermal conductivity [W.m-1.K-1]",
    ]:
        for domain in [negative_electrode, positive_electrode, separator]:
            pybamm_name = domain.pre_name + name[:1].lower() + name[1:]
            if name in pybamm_dict:
                pybamm_dict[pybamm_name] = pybamm_dict[name]

    # BET surface area
    for domain in [negative_electrode, positive_electrode]:
        pybamm_dict[domain.pre_name + "active material volume fraction"] = (
            pybamm_dict[domain.pre_name + "surface area per unit volume [m-1]"]
            * pybamm_dict[domain.short_pre_name + "particle radius [m]"]
        ) / 3.0

    # transport efficiency
    for domain in [negative_electrode, separator, positive_electrode]:
        pybamm_dict[domain.pre_name + "porosity"] = pybamm_dict[
            domain.pre_name + "transport efficiency"
        ] ** (1.0 / 1.5)

    # TODO: allow setting function parameters in a loop over domains

    # entropic change

    # negative electrode
    U_n = pybamm_dict[
        negative_electrode.pre_name + "entropic change coefficient [V.K-1]"
    ]
    if callable(U_n):

        def _negative_electrode_entropic_change(sto, c_s_max):
            return U_n(sto)

    else:

        def _negative_electrode_entropic_change(sto, c_s_max):
            return U_n

    pybamm_dict[
        negative_electrode.pre_name + "OCP entropic change [V.K-1]"
    ] = _negative_electrode_entropic_change

    # positive electrode
    U_p = pybamm_dict[
        positive_electrode.pre_name + "entropic change coefficient [V.K-1]"
    ]
    if callable(U_p):

        def _positive_electrode_entropic_change(sto, c_s_max):
            return U_p(sto)

    else:

        def _positive_electrode_entropic_change(sto, c_s_max):
            return U_p

    pybamm_dict[
        positive_electrode.pre_name + "OCP entropic change [V.K-1]"
    ] = _positive_electrode_entropic_change

    # reaction rates in pybamm exchange current is defined j0 = k * sqrt(ce * cs *
    # (cs-cs_max)) in BPX exchange current is defined j0 = F * k_norm * sqrt((ce/ce0) *
    # (cs/cs_max) * (1-cs/cs_max))
    c_e = pybamm_dict["Initial concentration in electrolyte [mol.m-3]"]
    F = 96485

    # negative electrode
    c_n_max = pybamm_dict[
        "Maximum concentration in " + negative_electrode.pre_name.lower() + "[mol.m-3]"
    ]
    k_n_norm = pybamm_dict[
        negative_electrode.pre_name + "reaction rate constant [mol.m-2.s-1]"
    ]
    E_a_n = pybamm_dict.get(
        negative_electrode.pre_name + "reaction rate activation energy [J.mol-1]", 0.0
    )
    # Note that in BPX j = 2*F*k_norm*sqrt((ce/ce0)*(c/c_max)*(1-c/c_max))*sinh(...),
    # and in PyBaMM j = 2*k*sqrt(ce*c*(c_max - c))*sinh(...)
    k_n = k_n_norm * F / (c_n_max * c_e**0.5)

    def _negative_electrode_exchange_current_density(c_e, c_s_surf, c_s_max, T):
        k_ref = k_n  # (A/m2)(m3/mol)**1.5 - includes ref concentrations

        arrhenius = exp(E_a_n / constants.R * (1 / T_ref - 1 / T))
        return (
            k_ref
            * arrhenius
            * c_e**0.5
            * c_s_surf**0.5
            * (c_s_max - c_s_surf) ** 0.5
        )

    pybamm_dict[
        negative_electrode.pre_name + "exchange-current density [A.m-2]"
    ] = _copy_func(_negative_electrode_exchange_current_density)

    # positive electrode
    c_p_max = pybamm_dict[
        "Maximum concentration in " + positive_electrode.pre_name.lower() + "[mol.m-3]"
    ]
    k_p_norm = pybamm_dict[
        positive_electrode.pre_name + "reaction rate constant [mol.m-2.s-1]"
    ]
    E_a_p = pybamm_dict.get(
        positive_electrode.pre_name + "reaction rate activation energy [J.mol-1]", 0.0
    )
    # Note that in BPX j = 2*F*k_norm*sqrt((ce/ce0)*(c/c_max)*(1-c/c_max))*sinh(...),
    # and in PyBaMM j = 2*k*sqrt(ce*c*(c_max - c))*sinh(...)
    k_p = k_p_norm * F / (c_p_max * c_e**0.5)

    def _positive_electrode_exchange_current_density(c_e, c_s_surf, c_s_max, T):
        k_ref = k_p  # (A/m2)(m3/mol)**1.5 - includes ref concentrations

        arrhenius = exp(E_a_p / constants.R * (1 / T_ref - 1 / T))
        return (
            k_ref
            * arrhenius
            * c_e**0.5
            * c_s_surf**0.5
            * (c_s_max - c_s_surf) ** 0.5
        )

    pybamm_dict[domain.pre_name + "exchange-current density [A.m-2]"] = _copy_func(
        _positive_electrode_exchange_current_density
    )

    # diffusivity

    # negative electrode
    E_a = pybamm_dict.get(
        negative_electrode.pre_name + "diffusivity activation energy [J.mol-1]", 0.0
    )
    D_n_ref = pybamm_dict[negative_electrode.pre_name + "diffusivity [m2.s-1]"]

    if callable(D_n_ref):

        def _negative_electrode_diffusivity(sto, T):
            arrhenius = exp(E_a / constants.R * (1 / T_ref - 1 / T))
            return arrhenius * D_n_ref(sto)

    else:

        def _negative_electrode_diffusivity(sto, T):
            arrhenius = exp(E_a / constants.R * (1 / T_ref - 1 / T))
            return arrhenius * D_n_ref

    pybamm_dict[negative_electrode.pre_name + "diffusivity [m2.s-1]"] = _copy_func(
        _negative_electrode_diffusivity
    )

    # positive electrode
    E_a = pybamm_dict.get(
        positive_electrode.pre_name + "diffusivity activation energy [J.mol-1]", 0.0
    )
    D_p_ref = pybamm_dict[positive_electrode.pre_name + "diffusivity [m2.s-1]"]

    if callable(D_p_ref):

        def _positive_electrode_diffusivity(sto, T):
            arrhenius = exp(E_a / constants.R * (1 / T_ref - 1 / T))
            return arrhenius * D_p_ref(sto)

    else:

        def _positive_electrode_diffusivity(sto, T):
            arrhenius = exp(E_a / constants.R * (1 / T_ref - 1 / T))
            return arrhenius * D_p_ref

    pybamm_dict[positive_electrode.pre_name + "diffusivity [m2.s-1]"] = _copy_func(
        _positive_electrode_diffusivity
    )

    # electrolyte
    E_a = pybamm_dict.get(
        electrolyte.pre_name + "diffusivity activation energy [J.mol-1]", 0.0
    )
    D_e_ref = pybamm_dict[electrolyte.pre_name + "diffusivity [m2.s-1]"]

    if callable(D_e_ref):

        def _electrolyte_diffusivity(sto, T):
            arrhenius = exp(E_a / constants.R * (1 / T_ref - 1 / T))
            return arrhenius * D_e_ref(sto)

    else:

        def _electrolyte_diffusivity(sto, T):
            arrhenius = exp(E_a / constants.R * (1 / T_ref - 1 / T))
            return arrhenius * D_e_ref

    pybamm_dict[electrolyte.pre_name + "diffusivity [m2.s-1]"] = _copy_func(
        _electrolyte_diffusivity
    )

    # conductivity
    E_a = pybamm_dict.get(
        electrolyte.pre_name + "conductivity activation energy [J.mol-1]", 0.0
    )
    C_ref_value = pybamm_dict[electrolyte.pre_name + "conductivity [S.m-1]"]

    if callable(C_ref_value):
        C_ref_fun = copy.copy(C_ref_value)

        def _conductivity(c_e, T):
            C_ref = C_ref_fun(c_e)
            arrhenius = exp(E_a / constants.R * (1 / T_ref - 1 / T))
            return arrhenius * C_ref

    else:
        C_ref_number = C_ref_value

        def _conductivity(c_e, T):
            C_ref = C_ref_number
            arrhenius = exp(E_a / constants.R * (1 / T_ref - 1 / T))
            return arrhenius * C_ref

    pybamm_dict[electrolyte.pre_name + "conductivity [S.m-1]"] = _copy_func(
        _conductivity
    )

    return pybamm_dict


preamble = "from pybamm import exp, tanh, cosh\n\n"


def _bpx_to_domain_param_dict(instance: BPX, pybamm_dict: dict, domain: Domain) -> dict:
    for name, field in instance.__fields__.items():
        value = getattr(instance, name)
        if value is None:
            continue
        elif isinstance(value, Function):
            value = value.to_python_function(preamble=preamble)
        elif isinstance(value, InterpolatedTable):
            x = np.array(value.x)
            y = np.array(value.y)
            interpolator = "linear"
            value = pybamm.Interpolant(
                [x], y, pybamm.t, name=name, interpolator=interpolator
            )

        pybamm_name = field.field_info.alias
        pybamm_name_lower = pybamm_name[:1].lower() + pybamm_name[1:]
        if pybamm_name.startswith("Initial concentration") or pybamm_name.startswith(
            "Maximum concentration"
        ):
            init_len = len("Initial concentration ")
            pybamm_name = (
                pybamm_name[:init_len]
                + "in "
                + domain.pre_name.lower()
                + pybamm_name[init_len:]
            )
        elif pybamm_name.startswith("Particle radius"):
            pybamm_name = domain.short_pre_name + pybamm_name_lower
        elif pybamm_name.startswith("OCP"):
            pybamm_name = domain.pre_name + pybamm_name
        elif pybamm_name.startswith("Cation transference number"):
            pybamm_name = pybamm_name
        elif domain.pre_name != "":
            pybamm_name = domain.pre_name + pybamm_name_lower

        pybamm_dict[pybamm_name] = value
    return pybamm_dict
