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

def copy_func(f):
    """Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)"""
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
                           argdefs=f.__defaults__,
                           closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g


@dataclass
class Domain:
    name: str
    pre_name: str
    short_pre_name: str


cell = Domain(name='cell', pre_name='', short_pre_name='')
anode = Domain(name='anode', pre_name='Negative electrode ', short_pre_name='Negative ')
cathode = Domain(name='cathode', pre_name='Positive electrode ',
                 short_pre_name='Positive ')
electrolyte = Domain(name='electrolyte', pre_name='Electrolyte ', short_pre_name='')
separator = Domain(name='separator', pre_name='Separator ', short_pre_name='')
experiment = Domain(name='experiment', pre_name='', short_pre_name='')


def bpx_to_param_dict(bpx: BPX) -> dict:
    pybamm_dict = {}
    pybamm_dict = _bpx_to_param_dict(
        bpx.parameterisation.cell, pybamm_dict, cell
    )
    pybamm_dict = _bpx_to_param_dict(
        bpx.parameterisation.anode, pybamm_dict, anode
    )
    pybamm_dict = _bpx_to_param_dict(
        bpx.parameterisation.cathode, pybamm_dict, cathode
    )
    pybamm_dict = _bpx_to_param_dict(
        bpx.parameterisation.electrolyte, pybamm_dict, electrolyte
    )
    print(pybamm_dict, bpx)
    pybamm_dict = _bpx_to_param_dict(
        bpx.parameterisation.separator, pybamm_dict, separator
    )
    pybamm_dict = _bpx_to_param_dict(
        bpx.parameterisation.separator, pybamm_dict, experiment
    )

    # set a default current function and typical current
    pybamm_dict['Current function [A]'] = pybamm_dict['Nominal cell capacity [A.h]']
    pybamm_dict['Typical current [A]'] = pybamm_dict['Nominal cell capacity [A.h]']

    # Ambient temp
    pybamm_dict['Ambient temperature [K]'] = pybamm_dict['Initial temperature [K]']

    for domain in [anode, cathode]:
        pybamm_dict[domain.pre_name + 'electrons in reaction'] = 1.0

    # typical electrolyte concentration
    pybamm_dict['Typical electrolyte concentration [mol.m-3]'] = pybamm_dict[
        'Initial concentration in electrolyte [mol.m-3]'
    ]

    for domain in [anode, cathode]:
        pybamm_dict[domain.pre_name + 'OCP entropic change [V.K-1]'] = 0.0

    for domain in [anode, separator, cathode]:
        pybamm_dict[domain.pre_name + 'Bruggeman coefficient (electrolyte)'] = 1.5
        pybamm_dict[domain.pre_name + 'Bruggeman coefficient (electrode)'] = 1.5

    pybamm_dict['Number of cells connected in series to make a battery'] = 1

    # electrode area
    equal_len_width = math.sqrt(pybamm_dict['Electrode area [m2]'])
    pybamm_dict['Electrode width [m]'] = equal_len_width
    pybamm_dict['Electrode height [m]'] = equal_len_width

    # cell geometry
    pybamm_dict['Cell volume [m3]'] = (
        pybamm_dict['Cell width [m]']
        * pybamm_dict['Cell height [m]']
        * pybamm_dict['Cell thickness [m]']
    )
    pybamm_dict['Cell cooling surface area [m2]'] = (
        2 * pybamm_dict['Cell width [m]'] * pybamm_dict['Cell thickness [m]']
        + 2 * pybamm_dict['Cell width [m]'] * pybamm_dict['Cell height [m]']
        + 2 * pybamm_dict['Cell thickness [m]'] * pybamm_dict['Cell height [m]']
    )

    pybamm_dict['1 + dlnf/dlnc'] = 1.0

    # lumped parameters
    for name in [
            'Specific heat capacity [J.K-1.kg-1]',
            'Density [kg.m-3]',
            'Thermal conductivity [W.m-1.K-1]',
    ]:
        for domain in [anode, cathode, separator]:
            pybamm_name = domain.pre_name + name[:1].lower() + name[1:]
            if name in pybamm_dict:
                pybamm_dict[pybamm_name] = pybamm_dict[name]

    # BET surface area
    for domain in [anode, cathode]:
        pybamm_dict[domain.pre_name + 'active material volume fraction'] = (
            (pybamm_dict[domain.pre_name + 'surface area per unit volume'] *
             pybamm_dict[domain.short_pre_name + 'particle radius [m]']) / 3.0
        )

    # transport efficiency
    for domain in [anode, separator, cathode]:
        pybamm_dict[domain.pre_name + 'porosity'] = (
            pybamm_dict[domain.pre_name + 'transport efficiency'] ** (1.0 / 1.5)
        )

    # reaction rates in pybamm exchange current is defined j0 = k * sqrt(ce * cs *
    # (cs-cs_max)) in BPX exchange current is defined j0 = F * k_norm * sqrt((ce/ce0) *
    # (cs/cs_max) * (1-cs/cs_max))
    for domain in [anode, cathode]:
        c_max = pybamm_dict[
            'Maximum concentration in ' +
            domain.pre_name.lower() +
            '[mol.m-3]'
        ]
        c_e = pybamm_dict[
            'Initial concentration in ' +
            domain.pre_name.lower() +
            '[mol.m-3]'
        ]
        k_norm = pybamm_dict[
            domain.pre_name + 'reaction rate [mol.m-2.s-1]'
        ]
        E_a = pybamm_dict.get(
            domain.pre_name + "reaction rate activation energy [J.mol-1]",
            0.0
        )
        T_ref = pybamm_dict["Reference temperature [K]"]
        F = 96485
        k = k_norm * F / (c_max * c_e ** 0.5)

        def exchange_current_density(c_e, c_s_surf, c_s_max, T):
            k_ref = k  # (A/m2)(m3/mol)**1.5 - includes ref concentrations
            arrhenius = exp(E_a / constants.R * (1 / T_ref - 1 / T))
            return (
                k_ref
                * arrhenius
                * c_e**0.5
                * c_s_surf**0.5
                * (c_s_max - c_s_surf) ** 0.5
            )
        pybamm_dict[domain.pre_name + 'exchange-current density [A.m-2]'] = (
            copy_func(exchange_current_density)
        )

    # diffusivity
    for domain in [anode, electrolyte, cathode]:
        T_ref = pybamm_dict["Reference temperature [K]"]
        E_a = pybamm_dict.get(
            domain.pre_name + 'diffusivity activation energy [J.mol-1]',
            0.0
        )
        D_ref_value = pybamm_dict[domain.pre_name + 'diffusivity [m2.s-1]']

        if callable(D_ref_value):
            D_ref_fun = copy.copy(D_ref_value)

            def diffusivity(sto, T):
                D_ref = D_ref_fun(sto)
                arrhenius = exp(E_a / constants.R * (1 / T_ref - 1 / T))
                return arrhenius * D_ref
        else:
            D_ref_number = D_ref_value

            def diffusivity(sto, T):
                D_ref = D_ref_number
                arrhenius = exp(E_a / constants.R * (1 / T_ref - 1 / T))
                return arrhenius * D_ref

        pybamm_dict[domain.pre_name + 'diffusivity [m2.s-1]'] = copy_func(diffusivity)

    # conductivity
    for domain in [electrolyte]:
        T_ref = pybamm_dict["Reference temperature [K]"]
        E_a = pybamm_dict.get(
            domain.pre_name + 'conductivity activation energy [J.mol-1]',
            0.0
        )
        C_ref_value = pybamm_dict[domain.pre_name + 'conductivity [S.m-1]']

        if callable(C_ref_value):
            C_ref_fun = copy.copy(C_ref_value)

            def conductivity(c_e, T):
                C_ref = C_ref_fun(c_e)
                arrhenius = exp(E_a / constants.R * (1 / T_ref - 1 / T))
                return arrhenius * C_ref
        else:
            C_ref_number = C_ref_value

            def conductivity(c_e, T):
                C_ref = C_ref_number
                arrhenius = exp(E_a / constants.R * (1 / T_ref - 1 / T))
                return arrhenius * C_ref

        pybamm_dict[domain.pre_name + 'conductivity [S.m-1]'] = copy_func(conductivity)

    import pprint
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(pybamm_dict)
    return pybamm_dict


preamble = (
    'from pybamm import exp, tanh, cosh\n\n'
)


def _bpx_to_param_dict(instance: BPX, pybamm_dict: dict, domain: Domain) -> dict:
    for name, field in instance.__fields__.items():
        value = getattr(instance, name)
        if value is None:
            continue
        elif isinstance(value, Function):
            value = value.to_python_function(preamble=preamble)
        elif isinstance(value, InterpolatedTable):
            timescale = 1
            x = np.array(value.x)
            y = np.array(value.y)
            interpolator = 'linear'
            value = pybamm.Interpolant(
                [x], y, pybamm.t * timescale,
                name=name, interpolator=interpolator
            )

        pybamm_name = field.field_info.alias
        pybamm_name_lower = pybamm_name[:1].lower() + pybamm_name[1:]
        if (
            pybamm_name.startswith("Initial concentration") or
            pybamm_name.startswith("Maximum concentration")
        ):
            init_len = len("Initial concentration ")
            pybamm_name = (
                pybamm_name[:init_len] +
                'in ' +
                domain.pre_name.lower() +
                pybamm_name[init_len:]
            )
        elif pybamm_name.startswith("Particle radius"):
            pybamm_name = domain.short_pre_name + pybamm_name_lower
        elif pybamm_name.startswith('OCP'):
            pybamm_name = domain.pre_name + pybamm_name
        elif pybamm_name.startswith('Cation transference number'):
            pybamm_name = pybamm_name
        elif domain.pre_name != '':
            pybamm_name = domain.pre_name + pybamm_name_lower

        pybamm_dict[pybamm_name] = value
    return pybamm_dict
