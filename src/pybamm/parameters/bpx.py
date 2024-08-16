from bpx import BPX, Function, InterpolatedTable
from bpx.schema import ElectrodeBlended, ElectrodeBlendedSPM
import pybamm
import math
from dataclasses import dataclass
import numpy as np
from pybamm import constants
from pybamm import exp


from functools import partial


def _callable_func(var, fun):
    return fun(var)


def _interpolant_func(var, name, x, y):
    return pybamm.Interpolant(x, y, var, name=name, interpolator="linear")


preamble = "from pybamm import exp, tanh, cosh\n\n"


def process_float_function_table(value, name):
    """
    Process BPX FloatFunctionTable to a float, python function or data for a pybamm
    Interpolant.
    """
    if isinstance(value, Function):
        value = value.to_python_function(preamble=preamble)
    elif isinstance(value, InterpolatedTable):
        # return (name, (x, y)) to match the output of
        # `pybamm.parameters.process_1D_data` we will create an interpolant on a
        # case-by-case basis to get the correct argument for each parameter
        x = np.array(value.x)
        y = np.array(value.y)
        # sort the arrays as CasADi requires x to be in ascending order
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = y[sort_idx]
        value = (name, (x, y))
    return value


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
negative_particle = Domain(
    name="negative particle",
    pre_name="Negative particle ",
    short_pre_name="Negative ",
)
positive_particle = Domain(
    name="positive particle",
    pre_name="Positive particle ",
    short_pre_name="Positive ",
)
positive_current_collector = Domain(
    name="positive current collector",
    pre_name="Positive current collector ",
    short_pre_name="",
)
negative_current_collector = Domain(
    name="negative current collector",
    pre_name="Negative current collector ",
    short_pre_name="",
)

electrolyte = Domain(name="electrolyte", pre_name="Electrolyte ", short_pre_name="")
separator = Domain(name="separator", pre_name="Separator ", short_pre_name="")
experiment = Domain(name="experiment", pre_name="", short_pre_name="")

PHASE_NAMES = ["Primary: ", "Secondary: "]


def _get_phase_names(domain):
    """
    Return a list of the phase names in a given domain
    """
    if isinstance(domain, (ElectrodeBlended, ElectrodeBlendedSPM)):
        phases = len(domain.particle.keys())
    else:
        phases = 1
    if phases == 1:
        return [""]
    elif phases == 2:
        return ["Primary: ", "Secondary: "]
    else:
        raise NotImplementedError(
            "PyBaMM does not support more than two "
            "particle phases in blended electrodes"
        )


def _bpx_to_param_dict(bpx: BPX) -> dict:
    """
    Turns a BPX object in to a dictionary of parameters for PyBaMM
    """
    domain_phases = {
        "negative electrode": _get_phase_names(bpx.parameterisation.negative_electrode),
        "positive electrode": _get_phase_names(bpx.parameterisation.positive_electrode),
    }

    # Loop over each component of BPX and add to pybamm dictionary
    pybamm_dict: dict = {}
    pybamm_dict = _bpx_to_domain_param_dict(
        bpx.parameterisation.positive_electrode, pybamm_dict, positive_electrode
    )
    pybamm_dict = _bpx_to_domain_param_dict(
        bpx.parameterisation.cell, pybamm_dict, cell
    )
    pybamm_dict = _bpx_to_domain_param_dict(
        bpx.parameterisation.negative_electrode, pybamm_dict, negative_electrode
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

    # assume Bruggeman relation for effective electrolyte properties
    for domain in [negative_electrode, separator, positive_electrode]:
        pybamm_dict[domain.pre_name + "Bruggeman coefficient (electrolyte)"] = 1.5

    # solid phase properties reported in BPX are already "effective",
    # so no correction is applied
    for domain in [negative_electrode, positive_electrode]:
        pybamm_dict[domain.pre_name + "Bruggeman coefficient (electrode)"] = 0

    # BPX is for single cell in series, user can change this later
    pybamm_dict["Number of cells connected in series to make a battery"] = 1
    pybamm_dict["Number of electrodes connected in parallel to make a cell"] = (
        pybamm_dict["Number of electrode pairs connected in parallel to make a cell"]
    )

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
        for domain in [
            negative_electrode,
            positive_electrode,
            separator,
            negative_current_collector,
            positive_current_collector,
        ]:
            pybamm_name = domain.pre_name + name[:1].lower() + name[1:]
            if name in pybamm_dict:
                pybamm_dict[pybamm_name] = pybamm_dict[name]

    # correct BPX specific heat capacity units to be consistent with pybamm
    for domain in [
        negative_electrode,
        positive_electrode,
        separator,
        negative_current_collector,
        positive_current_collector,
    ]:
        incorrect_name = domain.pre_name + "specific heat capacity [J.K-1.kg-1]"
        new_name = domain.pre_name + "specific heat capacity [J.kg-1.K-1]"
        if incorrect_name in pybamm_dict:
            pybamm_dict[new_name] = pybamm_dict[incorrect_name]
            del pybamm_dict[incorrect_name]

    # lumped thermal model requires current collector parameters. Arbitrarily assign
    for domain in [negative_current_collector, positive_current_collector]:
        pybamm_dict[domain.pre_name + "thickness [m]"] = 0
        pybamm_dict[domain.pre_name + "conductivity [S.m-1]"] = 4e7

    # add a default heat transfer coefficient
    pybamm_dict.update(
        {"Total heat transfer coefficient [W.m-2.K-1]": 0}, check_already_exists=False
    )

    # transport efficiency
    for domain in [negative_electrode, separator, positive_electrode]:
        pybamm_dict[domain.pre_name + "porosity"] = pybamm_dict[
            domain.pre_name + "transport efficiency"
        ] ** (1.0 / 1.5)

    # define functional forms for pybamm parameters that depend on more than one
    # variable

    def _arrhenius(Ea, T):
        return exp(Ea / constants.R * (1 / T_ref - 1 / T))

    def _entropic_change(sto, c_s_max, dUdT, constant=False):
        if constant:
            return dUdT
        else:
            return dUdT(sto)

    # reaction rates in pybamm exchange current is defined j0 = k * sqrt(ce * cs *
    # (cs-cs_max)) in BPX exchange current is defined j0 = F * k_norm * sqrt((ce/ce0) *
    # (cs/cs_max) * (1-cs/cs_max))
    c_e = pybamm_dict["Initial concentration in electrolyte [mol.m-3]"]
    F = pybamm.constants.F.value

    def _exchange_current_density(c_e, c_s_surf, c_s_max, T, k_ref, Ea):
        return (
            k_ref
            * _arrhenius(Ea, T)
            * c_e**0.5
            * c_s_surf**0.5
            * (c_s_max - c_s_surf) ** 0.5
        )

    def _diffusivity(sto, T, D_ref, Ea, constant=False):
        if constant:
            return _arrhenius(Ea, T) * D_ref
        else:
            return _arrhenius(Ea, T) * D_ref(sto)

    def _conductivity(c_e, T, Ea, sigma_ref, constant=False):
        if constant:
            return _arrhenius(Ea, T) * sigma_ref
        else:
            return _arrhenius(Ea, T) * sigma_ref(c_e)

    # Loop over electrodes and construct derived parameters
    for domain in [negative_electrode, positive_electrode]:
        for phase_pre_name in domain_phases[domain.name]:
            phase_domain_pre_name = phase_pre_name + domain.pre_name

            # BET surface area
            pybamm_dict[phase_domain_pre_name + "active material volume fraction"] = (
                pybamm_dict[
                    phase_domain_pre_name + "surface area per unit volume [m-1]"
                ]
                * pybamm_dict[
                    phase_pre_name + domain.short_pre_name + "particle radius [m]"
                ]
            ) / 3.0

            # ocp
            U = pybamm_dict[phase_domain_pre_name + "OCP [V]"]
            if isinstance(U, tuple):
                pybamm_dict[phase_domain_pre_name + "OCP [V]"] = partial(
                    _interpolant_func, name=U[0], x=U[1][0], y=U[1][1]
                )

            # entropic change
            dUdT = pybamm_dict[
                phase_domain_pre_name + "entropic change coefficient [V.K-1]"
            ]
            if callable(dUdT):
                pybamm_dict[phase_domain_pre_name + "OCP entropic change [V.K-1]"] = (
                    partial(_entropic_change, dUdT=dUdT)
                )
            elif isinstance(dUdT, tuple):
                pybamm_dict[phase_domain_pre_name + "OCP entropic change [V.K-1]"] = (
                    partial(
                        _entropic_change,
                        dUdT=partial(
                            _interpolant_func, name=dUdT[0], x=dUdT[1][0], y=dUdT[1][1]
                        ),
                    )
                )
            else:
                pybamm_dict[phase_domain_pre_name + "OCP entropic change [V.K-1]"] = (
                    partial(_entropic_change, dUdT=dUdT, constant=True)
                )

            # reaction rate
            c_max = pybamm_dict[
                phase_pre_name
                + "Maximum concentration in "
                + domain.pre_name.lower()
                + "[mol.m-3]"
            ]
            k_norm = pybamm_dict[
                phase_domain_pre_name + "reaction rate constant [mol.m-2.s-1]"
            ]
            Ea_k = pybamm_dict.get(
                phase_domain_pre_name
                + "reaction rate constant activation energy [J.mol-1]",
                0.0,
            )
            # Note that in BPX j = 2*F*k_norm*sqrt((ce/ce0)*(c/c_max)*(1-c/c_max))...
            # *sinh(),
            # and in PyBaMM j = 2*k*sqrt(ce*c*(c_max - c))*sinh()
            k = k_norm * F / (c_max * c_e**0.5)
            pybamm_dict[phase_domain_pre_name + "exchange-current density [A.m-2]"] = (
                partial(_exchange_current_density, k_ref=k, Ea=Ea_k)
            )

            # diffusivity
            Ea_D = pybamm_dict.get(
                phase_domain_pre_name + "diffusivity activation energy [J.mol-1]",
                0.0,
            )
            pybamm_dict[
                phase_domain_pre_name + "diffusivity activation energy [J.mol-1]"
            ] = Ea_D
            D_ref = pybamm_dict[phase_domain_pre_name + "diffusivity [m2.s-1]"]

            if callable(D_ref):
                pybamm_dict[phase_domain_pre_name + "diffusivity [m2.s-1]"] = partial(
                    _diffusivity, D_ref=D_ref, Ea=Ea_D
                )
            elif isinstance(D_ref, tuple):
                pybamm_dict[phase_domain_pre_name + "diffusivity [m2.s-1]"] = partial(
                    _diffusivity,
                    D_ref=partial(
                        _interpolant_func, name=D_ref[0], x=D_ref[1][0], y=D_ref[1][1]
                    ),
                    Ea=Ea_D,
                )
            else:
                pybamm_dict[phase_domain_pre_name + "diffusivity [m2.s-1]"] = partial(
                    _diffusivity, D_ref=D_ref, Ea=Ea_D, constant=True
                )

    # electrolyte
    Ea_D_e = pybamm_dict.get(
        electrolyte.pre_name + "diffusivity activation energy [J.mol-1]", 0.0
    )
    D_e_ref = pybamm_dict[electrolyte.pre_name + "diffusivity [m2.s-1]"]

    if callable(D_e_ref):
        pybamm_dict[electrolyte.pre_name + "diffusivity [m2.s-1]"] = partial(
            _diffusivity, D_ref=D_e_ref, Ea=Ea_D_e
        )
    elif isinstance(D_e_ref, tuple):
        pybamm_dict[electrolyte.pre_name + "diffusivity [m2.s-1]"] = partial(
            _diffusivity,
            D_ref=partial(
                _interpolant_func, name=D_e_ref[0], x=D_e_ref[1][0], y=D_e_ref[1][1]
            ),
            Ea=Ea_D_e,
        )
    else:
        pybamm_dict[electrolyte.pre_name + "diffusivity [m2.s-1]"] = partial(
            _diffusivity, D_ref=D_e_ref, Ea=Ea_D_e, constant=True
        )

    # conductivity
    Ea_sigma_e = pybamm_dict.get(
        electrolyte.pre_name + "conductivity activation energy [J.mol-1]", 0.0
    )
    sigma_e_ref = pybamm_dict[electrolyte.pre_name + "conductivity [S.m-1]"]

    if callable(sigma_e_ref):
        pybamm_dict[electrolyte.pre_name + "conductivity [S.m-1]"] = partial(
            _conductivity, sigma_ref=sigma_e_ref, Ea=Ea_sigma_e
        )
    elif isinstance(sigma_e_ref, tuple):
        pybamm_dict[electrolyte.pre_name + "conductivity [S.m-1]"] = partial(
            _conductivity,
            sigma_ref=partial(
                _interpolant_func,
                name=sigma_e_ref[0],
                x=sigma_e_ref[1][0],
                y=sigma_e_ref[1][1],
            ),
            Ea=Ea_sigma_e,
        )
    else:
        pybamm_dict[electrolyte.pre_name + "conductivity [S.m-1]"] = partial(
            _conductivity, sigma_ref=sigma_e_ref, Ea=Ea_sigma_e, constant=True
        )

    # Add user-defined parameters, if any
    user_defined = bpx.parameterisation.user_defined
    if user_defined:
        for name in user_defined.__dict__.keys():
            value = getattr(user_defined, name)
            value = process_float_function_table(value, name)
            if callable(value):
                pybamm_dict[name] = partial(_callable_func, fun=value)
            elif isinstance(value, tuple):
                pybamm_dict[name] = partial(
                    _interpolant_func, name=value[0], x=value[1][0], y=value[1][1]
                )
            else:
                pybamm_dict[name] = value
    return pybamm_dict


def _get_pybamm_name(pybamm_name, domain):
    """
    Process pybamm name to include domain name and handle special cases
    """
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
    return pybamm_name


def _bpx_to_domain_param_dict(instance: BPX, pybamm_dict: dict, domain: Domain) -> dict:
    """
    Turns a BPX instance in to a dictionary of parameters for PyBaMM for a given domain
    """
    # Loop over fields in BPX instance and add to pybamm dictionary
    for name, field in instance.__fields__.items():
        value = getattr(instance, name)
        # Handle blended electrodes, where the field is now an instance of
        # ElectrodeBlended or ElectrodeBlendedSPM
        if (
            isinstance(instance, (ElectrodeBlended, ElectrodeBlendedSPM))
            and name == "particle"
        ):
            particle_instance = instance.particle
            # Loop over phases
            for i, phase_name in enumerate(particle_instance.keys()):
                phase_instance = particle_instance[phase_name]
                # Loop over fields in phase instance and add to pybamm dictionary
                for name, field in phase_instance.__fields__.items():
                    value = getattr(phase_instance, name)
                    pybamm_name = PHASE_NAMES[i] + _get_pybamm_name(
                        field.field_info.alias, domain
                    )
                    value = process_float_function_table(value, name)
                    pybamm_dict[pybamm_name] = value
        # Handle other fields, which correspond directly to parameters
        else:
            pybamm_name = _get_pybamm_name(field.field_info.alias, domain)
            value = process_float_function_table(value, name)
            pybamm_dict[pybamm_name] = value
    return pybamm_dict
