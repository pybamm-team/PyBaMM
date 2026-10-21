#
# Battery model options: resolution and validation
#
from functools import cached_property

import pybamm


def represents_positive_integer(s):
    """Check if a string represents a positive integer"""
    try:
        val = int(s)
    except ValueError:
        return False
    else:
        return val > 0


_ELECTRODES = ("negative", "positive")
_PHASES = ("primary", "secondary")

_PER_ELECTRODE_OPTIONS = frozenset(
    {
        "diffusivity",
        "exchange-current density",
        "intercalation kinetics",
        "interface utilisation",
        "lithium plating",
        "loss of active material",
        "number of MSMR reactions",
        "open-circuit potential",
        "particle",
        "particle mechanics",
        "particle phases",
        "particle size",
        "SEI",
        "SEI on cracks",
        "stress-induced diffusion",
    }
)
_PER_PHASE_OPTIONS = frozenset(
    {
        "diffusivity",
        "exchange-current density",
        "lithium plating",
        "open-circuit potential",
        "particle mechanics",
        "SEI",
        "stress-induced diffusion",
    }
)

# In a full cell a scalar non-default value applies to the negative electrode only;
# the positive electrode takes the value given here.
_NEGATIVE_ONLY_SHORTHAND = {
    "SEI": "none",
    "SEI on cracks": "false",
    "lithium plating": "none",
}


def _option_location(path):
    """Describe an electrode/phase path for an error, e.g. ``" at negative.primary"``.

    Parameters
    ----------
    path : tuple of str
        Electrode and optionally phase names; empty for whole-cell options.

    Returns
    -------
    str
        The location suffix, or ``""`` for an empty path.
    """
    return f" at {'.'.join(path)}" if path else ""


def iter_option_leaves(option, value):
    """Validate the tuple shape of an option value and return its leaves.

    Parameters
    ----------
    option : str
        The option name.
    value : object
        A scalar, a ``(negative, positive)`` tuple, or a tuple whose electrode
        entries may be ``(primary, secondary)`` tuples.

    Returns
    -------
    list of tuple
        ``(path, leaf)`` pairs, where ``path`` is ``()``, ``(electrode,)`` or
        ``(electrode, phase)``.

    Raises
    ------
    pybamm.OptionError
        If the value is not a scalar or a supported tuple shape for the option.
    """
    if not isinstance(value, (tuple, list)):
        return [((), value)]
    if (
        isinstance(value, list)
        or option not in _PER_ELECTRODE_OPTIONS
        or len(value) != 2
    ):
        raise pybamm.OptionError(
            f"\n'{value}' is not recognized in option '{option}'. Values must be "
            "strings or (in some cases) 2-tuples of strings"
        )
    leaves = []
    for domain, electrode_value in zip(_ELECTRODES, value, strict=True):
        if not isinstance(electrode_value, (tuple, list)):
            leaves.append(((domain,), electrode_value))
            continue
        if option not in _PER_PHASE_OPTIONS:
            raise pybamm.OptionError(
                f"\n'{electrode_value}' at {domain} is not recognized in option "
                f"'{option}'. This option cannot vary by particle phase, so each "
                "electrode takes a single string"
            )
        if (
            isinstance(electrode_value, list)
            or len(electrode_value) != 2
            or any(isinstance(leaf, (tuple, list)) for leaf in electrode_value)
        ):
            raise pybamm.OptionError(
                f"\n'{electrode_value}' at {domain} is not recognized in option "
                f"'{option}'. Per-phase values must be 2-tuples of strings"
            )
        leaves.extend(
            ((domain, phase), leaf)
            for phase, leaf in zip(_PHASES, electrode_value, strict=True)
        )
    return leaves


def resolve_option(option, value, domain, phase=None, working_electrode="both"):
    """Resolve an option value for one electrode and, optionally, one phase.

    Parameters
    ----------
    option : str
        The option name.
    value : object
        The stored option value.
    domain : str
        ``"negative"`` or ``"positive"``.
    phase : str, optional
        ``"primary"`` or ``"secondary"``. If not given, an electrode's
        per-phase tuple is returned unresolved.
    working_electrode : str, optional
        The ``"working electrode"`` option, which controls the negative-only
        shorthand for side reactions. Default is ``"both"``.

    Returns
    -------
    object
        The resolved value.
    """
    positive_default = _NEGATIVE_ONLY_SHORTHAND.get(option)
    if (
        positive_default is not None
        and working_electrode == "both"
        and not isinstance(value, tuple)
        and value != positive_default
    ):
        value = (value, positive_default)
    if isinstance(value, tuple):
        value = value[_ELECTRODES.index(domain)]
    if phase is not None and isinstance(value, tuple):
        value = value[_PHASES.index(phase)]
    return value


def validate_option_value(option, value, possible_values, path=()):
    """Check that one option leaf is an allowed value.

    Parameters
    ----------
    option : str
        The option name.
    value : object
        A single (non-tuple) option value.
    possible_values : list
        The allowed values for the option.
    path : tuple of str, optional
        The electrode/phase path of the leaf, used in the error message.

    Raises
    ------
    pybamm.OptionError
        If the value is not allowed.
    """
    if value in possible_values:
        return
    if option == "operating mode" and callable(value):
        return
    if option == "number of MSMR reactions" and represents_positive_integer(value):
        return
    raise pybamm.OptionError(
        f"\n'{value}' is not recognized in option '{option}'"
        f"{_option_location(path)}. "
        f"Possible values are {possible_values}"
    )


def replace_option_leaf(value, old, new):
    """Replace every leaf equal to ``old`` with ``new``, keeping tuple structure.

    Parameters
    ----------
    value : object
        A scalar or (nested) tuple option value.
    old, new : object
        The leaf to replace and its replacement.

    Returns
    -------
    object
        The value with replacements made.
    """
    if isinstance(value, tuple):
        return tuple(replace_option_leaf(leaf, old, new) for leaf in value)
    return new if value == old else value


def join_electrode_values(option, negative, positive, working_electrode="both"):
    """Store per-electrode values in the shortest form that resolves to them.

    Parameters
    ----------
    option : str
        The option name.
    negative, positive : object
        The values for each electrode.
    working_electrode : str, optional
        The ``"working electrode"`` option. Default is ``"both"``.

    Returns
    -------
    object
        ``negative`` if it resolves to both values on its own, otherwise the
        ``(negative, positive)`` tuple.
    """
    resolves_to_both = all(
        resolve_option(option, negative, domain, working_electrode=working_electrode)
        == expected
        for domain, expected in zip(_ELECTRODES, (negative, positive), strict=True)
    )
    return negative if resolves_to_both else (negative, positive)


def active_electrodes(working_electrode):
    """Return the electrodes present in a cell.

    Parameters
    ----------
    working_electrode : str
        The ``"working electrode"`` option.

    Returns
    -------
    tuple of str
        Both electrodes for a full cell, only ``"positive"`` for a half cell.
    """
    return _ELECTRODES if working_electrode == "both" else ("positive",)


def option_values_match(option, first, second, working_electrode="both"):
    """Check whether two values of an option resolve the same everywhere.

    Parameters
    ----------
    option : str
        The option name.
    first, second : object
        The option values to compare, in any supported shape.
    working_electrode : str, optional
        The ``"working electrode"`` option. Default is ``"both"``.

    Returns
    -------
    bool
        Whether both values give the same leaf for every electrode and phase.

    Raises
    ------
    pybamm.OptionError
        If either value is not a supported shape for the option.
    """
    iter_option_leaves(option, first)
    iter_option_leaves(option, second)
    return all(
        resolve_option(option, first, domain, phase, working_electrode)
        == resolve_option(option, second, domain, phase, working_electrode)
        for domain in _ELECTRODES
        for phase in _PHASES
    )


def dependency_error(option, value, companion, requirement, path=()):
    """Build the error for an option whose companion option is incompatible.

    Parameters
    ----------
    option : str
        The option that imposes the requirement.
    value : object
        Its value.
    companion : str
        The option that must satisfy the requirement.
    requirement : str
        What the companion must be, e.g. ``"'pouch'"``.
    path : tuple of str, optional
        The electrode/phase path, omitted for whole-cell options.

    Returns
    -------
    pybamm.OptionError
        The error, for the caller to raise.
    """
    return pybamm.OptionError(
        f"Option '{option}'{_option_location(path)} is '{value}', which requires "
        f"'{companion}' to be {requirement}."
    )


def _apply_legacy_defaults(options, supplied):
    """Fill options whose default depends on other options.

    Parameters
    ----------
    options : pybamm.FuzzyDict
        The merged options, updated in place.
    supplied : set of str
        Option names given by the caller; these are never changed.
    """
    working_electrode = options["working electrode"]

    def electrode_leaves(option):
        # per-phase entries are flattened, so a check matches if any phase does
        leaves = []
        for domain in _ELECTRODES:
            value = resolve_option(
                option, options[option], domain, working_electrode=working_electrode
            )
            leaves.append(value if isinstance(value, tuple) else (value,))
        return leaves

    def set_default(option, value):
        if option not in supplied:
            options[option] = value

    def set_per_electrode_default(option, values):
        set_default(option, join_electrode_values(option, *values, working_electrode))

    multi_phase = any(
        phases != ("1",) for phases in electrode_leaves("particle phases")
    )

    if options["dimensionality"] in (1, 2) or options["thermal"] == "x-full":
        set_default("cell geometry", "pouch")
    # derived from the SEI supplied, before plating adds its own "constant" SEI
    if any(sei != "none" for leaves in electrode_leaves("SEI") for sei in leaves):
        set_default("SEI film resistance", "distributed")
    set_per_electrode_default(
        "SEI",
        [
            "constant" if "partially reversible" in leaves else "none"
            for leaves in electrode_leaves("lithium plating")
        ],
    )
    mechanics = []
    for cracks, lam in zip(
        electrode_leaves("SEI on cracks"),
        electrode_leaves("loss of active material"),
        strict=True,
    ):
        if "true" in cracks:
            mechanics.append("swelling and cracking")
        elif any("stress" in leaf for leaf in lam):
            mechanics.append("swelling only")
        else:
            mechanics.append("none")
    set_per_electrode_default("particle mechanics", mechanics)
    stress = []
    for leaves in electrode_leaves("particle mechanics"):
        # per phase, so a phase without mechanics never gets stress diffusion
        flags = tuple("true" if leaf != "none" else "false" for leaf in leaves)
        stress.append(flags[0] if len(set(flags)) == 1 else flags)
    set_per_electrode_default("stress-induced diffusion", stress)
    if multi_phase:
        set_default("surface form", "algebraic")
    film_resistance = options["SEI film resistance"]
    needs_current_state = film_resistance == "distributed" or (
        film_resistance != "none" and multi_phase
    )
    if needs_current_state:
        set_default("total interfacial current density as a state", "true")
    if options["operating mode"] in ("explicit power", "explicit resistance"):
        set_default("voltage as a state", "true")

    current_state = options["total interfacial current density as a state"]
    if needs_current_state and current_state == "false":
        raise dependency_error(
            "SEI film resistance",
            film_resistance,
            "total interfacial current density as a state",
            "'true'"
            if film_resistance == "distributed"
            else "'true' when an electrode has multiple particle phases",
        )


def _check_electrode_compatibility(options):
    """Check options that depend on each other within one electrode or phase.

    Parameters
    ----------
    options : pybamm.FuzzyDict
        The merged options, after dependent defaults are applied.

    Raises
    ------
    pybamm.OptionError
        If a per-electrode or per-phase combination is incompatible.
    """
    working_electrode = options["working electrode"]
    # every electrode, so a half cell's unused entry is still checked for shape
    for domain in _ELECTRODES:
        if resolve_option("particle phases", options["particle phases"], domain) != "1":
            continue
        for option in sorted(_PER_PHASE_OPTIONS):
            electrode_value = resolve_option(
                option, options[option], domain, working_electrode=working_electrode
            )
            if isinstance(electrode_value, tuple):
                raise pybamm.OptionError(
                    f"Option '{option}' at {domain} is {electrode_value}, a "
                    "(primary, secondary) tuple, but 'particle phases' at "
                    f"{domain} is '1'. Give one value for that electrode or "
                    "set its 'particle phases' to '2'."
                )
    for domain in active_electrodes(working_electrode):
        number_of_phases = int(
            resolve_option("particle phases", options["particle phases"], domain)
        )
        phases = _PHASES[:number_of_phases]
        if number_of_phases > 1 and (
            options["surface form"] == "false"
            or any(
                resolve_option("particle", options["particle"], domain, phase)
                != "Fickian diffusion"
                for phase in phases
            )
        ):
            raise pybamm.OptionError(
                f"Electrode at {domain} has multiple particle phases, which "
                "requires 'surface form' to be 'differential' or 'algebraic' and "
                "'particle' to be 'Fickian diffusion'."
            )
        for phase in phases:
            path = (domain, phase) if number_of_phases > 1 else (domain,)
            value = {
                option: resolve_option(
                    option, options[option], domain, phase, working_electrode
                )
                for option in (
                    "lithium plating",
                    "SEI",
                    "loss of active material",
                    "particle mechanics",
                    "SEI on cracks",
                    "stress-induced diffusion",
                )
            }
            if (
                value["lithium plating"] == "partially reversible"
                and value["SEI"] == "none"
            ):
                raise dependency_error(
                    "lithium plating",
                    "partially reversible",
                    "SEI",
                    "a model other than 'none' (e.g. 'constant')",
                    path,
                )
            # stress-driven LAM acts on the particle stress from a mechanics model
            lam = value["loss of active material"]
            if "stress" in lam and value["particle mechanics"] == "none":
                raise dependency_error(
                    "loss of active material",
                    lam,
                    "particle mechanics",
                    "a model other than 'none'",
                    path,
                )
            if value["SEI on cracks"] == "true" and (
                value["particle mechanics"] != "swelling and cracking"
            ):
                raise dependency_error(
                    "SEI on cracks",
                    "true",
                    "particle mechanics",
                    "'swelling and cracking'",
                    path,
                )
            if value["stress-induced diffusion"] == "true" and (
                value["particle mechanics"] == "none"
            ):
                raise dependency_error(
                    "stress-induced diffusion",
                    "true",
                    "particle mechanics",
                    "a model other than 'none'",
                    path,
                )


def _rename_option(options_dict, option_name, old_name, new_name):
    """Rename every leaf equal to ``old_name`` in ``options_dict[option_name]``.

    Parameters
    ----------
    options_dict : dict
        The options dict to update in place (a caller-owned copy).
    option_name : str
        The option key to rename leaves within.
    old_name, new_name : str
        The leaf value to replace and its replacement.
    """
    value = options_dict.get(option_name)
    if value is None:
        return
    renamed = replace_option_leaf(value, old_name, new_name)
    if renamed != value:
        pybamm.logger.warning(
            f"The '{old_name}' {option_name} model has been renamed to '{new_name}'"
        )
        options_dict[option_name] = renamed


# Options saved before this release skipped the per-electrode checks; bump it
# if the release that ships them is not 26.11.
_PER_ELECTRODE_OPTIONS_RELEASE = (26, 11)


class LegacyOptions(dict):
    """Options saved by a PyBaMM version that validated them as whole values.

    Battery models process these like a plain dict but skip the per-electrode
    and per-phase compatibility checks, so a saved model loads with the options
    it was built with.
    """


def restore_saved_options(options, version):
    """Mark options loaded from a file with the rules they were saved under.

    Parameters
    ----------
    options : dict
        The saved options.
    version : str or None
        The PyBaMM version that saved them, or None if it was not recorded.

    Returns
    -------
    dict
        ``options`` as a :class:`LegacyOptions` if ``version`` is unknown or
        predates per-electrode option checks, otherwise ``options`` unchanged.
    """
    try:
        release = tuple(int(part) for part in str(version).split(".")[:2])
    except ValueError:
        release = ()
    if release < _PER_ELECTRODE_OPTIONS_RELEASE:
        return LegacyOptions(options or {})
    return options


class BatteryModelOptions(pybamm.FuzzyDict):
    """
    Attributes
    ----------

    options: dict
        A dictionary of options to be passed to the model. The options that can
        be set are listed below. Note that not all of the options are compatible with
        each other and with all of the models implemented in PyBaMM. Each option is
        optional and takes a default value if not provided.
        In general, the option provided must be a string, but there are some cases
        where a 2-tuple of strings can be provided instead to indicate a different
        option for the negative and positive electrodes.

        An option that varies by electrode takes a ``(negative, positive)`` 2-tuple.
        An option that also varies by particle phase takes a ``(primary, secondary)``
        2-tuple inside the entry for that electrode. A scalar value applies to
        everywhere below it (both electrodes, or both phases of an electrode).

            * "calculate discharge energy": str
                Whether to calculate the discharge energy, throughput energy and
                throughput capacity in addition to discharge capacity. Must be one of
                "true" or "false". "false" is the default, since calculating discharge
                energy can be computationally expensive for simple models like SPM.
            * "cell geometry" : str
                Sets the geometry of the cell. Can be "arbitrary" (default) or
                "pouch". The arbitrary geometry option solves a 1D electrochemical
                model with prescribed cell volume and cross-sectional area, and
                (if thermal effects are included) solves a lumped thermal model
                with prescribed surface area for cooling.
            * "calculate heat source for isothermal models" : str
                Whether to calculate the heat source terms during isothermal operation.
                Can be "true" or "false". If "false", the heat source terms are set
                to zero. Default is "false" since this option may require additional
                parameters not needed by the electrochemical model.
            * "contact resistance" : str
                Whether to include an additional series resistance, added to the
                terminal voltage and contributing Ohmic (I^2 R) heating. Can be
                "false" (default) or "true". Set via the parameter "Contact
                resistance [Ohm]", which may be a constant or a function of the
                volume-averaged cell temperature.
            * "convection" : str
                Whether to include the effects of convection in the model. Can be
                "none" (default), "uniform transverse" or "full transverse".
                Must be "none" for lithium-ion models.
            * "current collector" : str
                Sets the current collector model to use. Can be "uniform" (default),
                "potential pair" or "potential pair quite conductive".
            * "diffusivity" : str
                Sets the model for the diffusivity. Can be "single"
                (default) or "current sigmoid". A 2-tuple can be provided for different
                behaviour in negative and positive electrodes.
            * "dimensionality" : int
                Sets the dimension of the current collector problem. Can be 0
                (default), 1 or 2.
            * "electrolyte conductivity" : str
                Can be "default" (default), "full", "leading order", "composite" or
                "integrated".
            * "exchange-current density" : str
                Sets the model for the exchange-current density. Can be "single"
                (default) or "current sigmoid". A 2-tuple can be provided for different
                behaviour in negative and positive electrodes.
            * "hydrolysis" : str
                Whether to include hydrolysis in the model. Only implemented for
                lead-acid models. Can be "false" (default) or "true". If "true", then
                "surface form" cannot be 'false'.
            * "intercalation kinetics" : str
                Model for intercalation kinetics. Can be "symmetric Butler-Volmer"
                (default), "asymmetric Butler-Volmer", "linear", "Marcus",
                "Marcus-Hush-Chidsey" (which uses the asymptotic form from Zeng 2014),
                or "MSMR" (which uses the form from Baker 2018). A 2-tuple can be
                provided for different behaviour in negative and positive electrodes.
            * "interface utilisation": str
                Can be "full" (default), "constant", or "current-driven".
            * "lithium plating" : str
                Sets the model for lithium plating. Can be "none" (default),
                "reversible", "partially reversible", or "irreversible". In a full
                cell, a scalar (non-default) value applies to the negative electrode
                only; use a 2-tuple to also set the positive electrode.
                ``options["lithium plating"]`` stores the value as given.
            * "lithium plating porosity change" : str
                Whether to include porosity change due to lithium plating, can be
                "false" (default) or "true".
            * "loss of active material" : str
                Sets the model for loss of active material. Can be "none" (default),
                "stress-driven", "asymmetric stress-driven", "reaction-driven",
                "current-driven", "stress and reaction-driven", or
                "asymmetric stress and reaction-driven".
                A 2-tuple can be provided for different behaviour in negative and
                positive electrodes.
            * "number of MSMR reactions" : str
                Sets the number of reactions to use in the MSMR model in each electrode.
                A 2-tuple can be provided to give a different number of reactions in
                the negative and positive electrodes. Default is "none". Can be any
                2-tuple of strings of integers. For example, set to ("6", "4") for a
                negative electrode with 6 reactions and a positive electrode with 4
                reactions.
            * "open-circuit potential" : str
                Sets the model for the open circuit potential. Can be "single"
                (default), "current sigmoid", "one-state hysteresis", "one-state differential capacity hysteresis", or "MSMR".
                If "MSMR" then the "particle" option must also be "MSMR".
                A 2-tuple can be provided for different behaviour in negative
                and positive electrodes.
            * "operating mode" : str
                Sets the operating mode for the model. This determines how the current
                is set. Can be:

                - "current" (default) : the current is explicity supplied
                - "voltage"/"power"/"resistance" : solve an algebraic equation for \
                    current such that voltage/power/resistance is correct
                - "differential power"/"differential resistance" : solve a \
                    differential equation for the power or resistance
                - "explicit power"/"explicit resistance" : current is defined in terms \
                    of the voltage such that power/resistance is correct
                - "CCCV": a special implementation of the common constant-current \
                    constant-voltage charging protocol, via an ODE for the current
                - callable : if a callable is given as this option, the function \
                    defines the residual of an algebraic equation. The applied current \
                    will be solved for such that the algebraic constraint is satisfied.
            * "particle" : str
                Sets the submodel to use to describe behaviour within the particle.
                Can be "Fickian diffusion" (default), "uniform profile",
                "quadratic profile", "quartic profile", or "MSMR". If "MSMR" then the
                "open-circuit potential" option must also be "MSMR". A 2-tuple can be
                provided for different behaviour in negative and positive electrodes.
            * "particle mechanics" : str
                Sets the model to account for mechanical effects such as particle
                swelling and cracking. Can be "none", "swelling only",
                or "swelling and cracking". A 2-tuple can be provided for different
                behaviour in negative and positive electrodes. The default is
                derived per electrode: "swelling and cracking" if "SEI on cracks"
                is "true" there, else "swelling only" if "loss of active material"
                is stress-driven there, else "none".
            * "particle phases": str
                Number of phases present in the electrode. A 2-tuple can be provided for
                different behaviour in negative and positive electrodes.
                For example, set to ("2", "1") for a negative electrode with 2 phases,
                e.g. graphite and silicon.
            * "particle shape" : str
                Sets the model shape of the electrode particles. This is used to
                calculate the surface area to volume ratio. Can be "spherical"
                (default), or "no particles".
            * "particle size" : str
                Sets the model to include a single active particle size or a
                distribution of sizes at any macroscale location. Can be "single"
                (default) or "distribution". Option applies to both electrodes.
            * "SEI" : str
                Set the SEI submodel to be used. Options are:

                - "none": :class:`pybamm.sei.NoSEI` (no SEI growth)
                - "constant": :class:`pybamm.sei.Constant` (constant SEI thickness)
                - "reaction limited", "reaction limited (asymmetric)", \
                    "solvent-diffusion limited", "electron-migration limited", \
                    "interstitial-diffusion limited", "ec reaction limited" ,   \
                    "VonKolzenberg2020", "tunnelling limited",\
                    or "ec reaction limited (asymmetric)": :class:`pybamm.sei.SEIGrowth`

                In a full cell, a scalar (non-default) value applies to the negative
                electrode only; use a 2-tuple to also set the positive electrode.
                ``options["SEI"]`` stores the value as given. The default is
                "constant" on an electrode where "lithium plating" is "partially
                reversible" there and "SEI" was not supplied.
            * "SEI film resistance" : str
                Set the submodel for additional term in the overpotential due to SEI.
                The default value is "none" if the "SEI" option is "none", and
                "distributed" otherwise. This is because the "distributed" model is more
                complex than the model with no additional resistance, which adds
                unnecessary complexity if there is no SEI in the first place

                - "none": no additional resistance\

                    .. math::
                        \\eta_r = \\frac{F}{RT} * (\\phi_s - \\phi_e - U)

                - "distributed": properly included additional resistance term\

                    .. math::
                        \\eta_r = \\frac{F}{RT}
                        * (\\phi_s - \\phi_e - U - R_{sei} * L_{sei} * j)

                - "average": constant additional resistance term (approximation to the \
                    true model). This model can give similar results to the \
                    "distributed" case without needing to make j an algebraic state\

                    .. math::
                        \\eta_r = \\frac{F}{RT}
                        * (\\phi_s - \\phi_e - U - R_{sei} * L_{sei} * \\frac{I}{aL})
            * "SEI on cracks" : str
                Whether to include SEI growth on particle cracks, can be "false"
                (default) or "true". In a full cell, a scalar (non-default) value
                applies to the negative electrode only; use a 2-tuple to also set
                the positive electrode. ``options["SEI on cracks"]`` stores the
                value as given.
            * "SEI porosity change" : str
                Whether to include porosity change due to SEI formation, can be "false"
                (default) or "true".
            * "stress-induced diffusion" : str
                Whether to include stress-induced diffusion, can be "false" or "true".
                The default is derived per electrode (and phase): "false" if
                "particle mechanics" is "none" there and "true" otherwise. A 2-tuple
                can be provided for different behaviour in negative and positive
                electrodes.
            * "surface form" : str
                Whether to use the surface formulation of the problem. Can be "false"
                (default), "differential" or "algebraic".
            * "surface temperature" : str
                Sets the surface temperature model to use. Can be "ambient" (default),
                which sets the surface temperature equal to the ambient temperature, or
                "lumped", which adds an ODE for the surface temperature (e.g. to model
                internal heating of a thermal chamber).
            * "thermal" : str
                Sets the thermal model to use. Can be "isothermal" (default), "lumped",
                "x-lumped", or "x-full". The 'cell geometry' option must be set to
                'pouch' for 'x-lumped' or 'x-full' to be valid. Using the 'x-lumped'
                option with 'dimensionality' set to 0 is equivalent to using the
                'lumped' option.
            * "total interfacial current density as a state" : str
                Whether to make a state for the total interfacial current density and
                solve an algebraic equation for it. Default is "false", unless "SEI film
                resistance" is distributed in which case it is automatically set to
                "true".
            * "voltage as a state" : str
                Whether to promote voltage to an algebraic state variable.
                Can be "false" (default) or "true", unless the operating mode
                is "explicit power" or "explicit resistance", in which case it
                is automatically set to "true". When "true", the model is a DAE
                and requires a DAE-capable solver (e.g. the default
                IDAKLUSolver). When "false", voltage is computed as an
                expression. Note that setting this to "false" only removes the
                voltage algebraic equation; SPM/SPMe with ``surface
                form="false"`` become pure ODE models, but DFN retains other
                algebraic states (electrode/electrolyte potentials) regardless
                of this option.
            * "working electrode" : str
                Can be "both" (default) for a standard battery or "positive" for a
                half-cell where the negative electrode is replaced with a lithium metal
                counter electrode.
            * "x-average side reactions": str
                Whether to average the side reactions (SEI growth, lithium plating and
                the respective porosity change) over the x-axis in Single Particle
                Models, can be "false" or "true". Default is "false" for SPMe and
                "true" for SPM.
            * "use lumped thermal capacity" : str
                Whether to use a lumped capacity model for the thermal model. Can be
                "false" (default) or "true". This is only available for the lumped
                thermal model.

    legacy : bool, optional
        Whether the options were saved by a PyBaMM version that validated them as
        whole values. If True, the per-electrode and per-phase compatibility
        checks are skipped. Default is False.
    """

    def __init__(self, extra_options, legacy=False):
        self.possible_options = {
            "calculate discharge energy": ["false", "true"],
            "calculate heat source for isothermal models": ["false", "true"],
            "cell geometry": ["arbitrary", "pouch", "cylindrical"],
            "contact resistance": ["false", "true"],
            "convection": ["none", "uniform transverse", "full transverse"],
            "current collector": [
                "uniform",
                "potential pair",
                "potential pair quite conductive",
            ],
            "diffusivity": ["single", "current sigmoid"],
            "dimensionality": [0, 1, 2, 3],
            "electrolyte conductivity": [
                "default",
                "full",
                "leading order",
                "composite",
                "integrated",
            ],
            "exchange-current density": ["single", "current sigmoid"],
            "heat of mixing": ["false", "true"],
            "hydrolysis": ["false", "true"],
            "intercalation kinetics": [
                "symmetric Butler-Volmer",
                "asymmetric Butler-Volmer",
                "linear",
                "Marcus",
                "Marcus-Hush-Chidsey",
                "MSMR",
            ],
            "interface utilisation": ["full", "constant", "current-driven"],
            "lithium plating": [
                "none",
                "reversible",
                "partially reversible",
                "irreversible",
            ],
            "lithium plating porosity change": ["false", "true"],
            "loss of active material": [
                "none",
                "stress-driven",
                "asymmetric stress-driven",
                "reaction-driven",
                "current-driven",
                "stress and reaction-driven",
                "asymmetric stress and reaction-driven",
            ],
            "number of MSMR reactions": ["none"],
            "open-circuit potential": [
                "single",
                "current sigmoid",
                "MSMR",
                "one-state hysteresis",
                "one-state differential capacity hysteresis",
            ],
            "operating mode": [
                "current",
                "voltage",
                "power",
                "differential power",
                "explicit power",
                "resistance",
                "differential resistance",
                "explicit resistance",
                "CCCV",
            ],
            "particle": [
                "Fickian diffusion",
                "uniform profile",
                "quadratic profile",
                "quartic profile",
                "MSMR",
            ],
            "particle mechanics": ["none", "swelling only", "swelling and cracking"],
            "particle phases": ["1", "2"],
            "particle shape": ["spherical", "no particles"],
            "particle size": ["single", "distribution"],
            "SEI": [
                "none",
                "constant",
                "reaction limited",
                "reaction limited (asymmetric)",
                "solvent-diffusion limited",
                "electron-migration limited",
                "interstitial-diffusion limited",
                "ec reaction limited",
                "ec reaction limited (asymmetric)",
                "VonKolzenberg2020",
                "tunnelling limited",
            ],
            "SEI film resistance": ["none", "distributed", "average"],
            "SEI on cracks": ["false", "true"],
            "SEI porosity change": ["false", "true"],
            "stress-induced diffusion": ["false", "true"],
            "surface form": ["false", "differential", "algebraic"],
            "surface temperature": ["ambient", "lumped"],
            "thermal": ["isothermal", "lumped", "x-lumped", "x-full"],
            "total interfacial current density as a state": ["false", "true"],
            "transport efficiency": [
                "Bruggeman",
                "ordered packing",
                "hyperbola of revolution",
                "overlapping spheres",
                "tortuosity factor",
                "random overlapping cylinders",
                "heterogeneous catalyst",
                "cation-exchange membrane",
            ],
            "voltage as a state": ["false", "true"],
            "working electrode": ["both", "positive"],
            "x-average side reactions": ["false", "true"],
            "use lumped thermal capacity": ["false", "true"],
        }

        default_options = {
            "calculate discharge energy": "false",
            "calculate heat source for isothermal models": "false",
            "cell geometry": "arbitrary",
            "contact resistance": "false",
            "convection": "none",
            "current collector": "uniform",
            "diffusivity": "single",
            "dimensionality": 0,
            "electrolyte conductivity": "default",
            "exchange-current density": "single",
            "heat of mixing": "false",
            "hydrolysis": "false",
            "intercalation kinetics": "symmetric Butler-Volmer",
            "interface utilisation": "full",
            "lithium plating": "none",
            "lithium plating porosity change": "false",
            "loss of active material": "none",
            "number of MSMR reactions": "none",
            "open-circuit potential": "single",
            "operating mode": "current",
            "particle": "Fickian diffusion",
            "particle mechanics": "none",
            "particle phases": "1",
            "particle shape": "spherical",
            "particle size": "single",
            "SEI": "none",
            "SEI film resistance": "none",
            "SEI on cracks": "false",
            "SEI porosity change": "false",
            "stress-induced diffusion": "false",
            "surface form": "false",
            "surface temperature": "ambient",
            "thermal": "isothermal",
            "total interfacial current density as a state": "false",
            "transport efficiency": "Bruggeman",
            "voltage as a state": "false",
            "working electrode": "both",
            "x-average side reactions": "false",
            "use lumped thermal capacity": "false",
        }
        extra_options = dict(extra_options or {})

        # Handle OCP option renaming
        _rename_option(
            extra_options,
            "open-circuit potential",
            "Wycisk",
            "one-state differential capacity hysteresis",
        )

        _rename_option(
            extra_options,
            "open-circuit potential",
            "Axen",
            "one-state hysteresis",
        )

        options = pybamm.FuzzyDict(default_options)
        # any extra options overwrite the default options
        for name, opt in extra_options.items():
            if name in default_options:
                options[name] = opt
            else:
                if name == "particle cracking":
                    raise pybamm.OptionError(
                        "The 'particle cracking' option has been renamed. "
                        "Use 'particle mechanics' instead."
                    )
                else:
                    raise pybamm.OptionError(
                        f"Option '{name}' not recognised. Best matches are {options.get_best_matches(name)}"
                    )

        # Must run before generic value validation so users see migration
        # guidance, not just "not recognized", for this removed option.
        if options["working electrode"] == "negative":
            raise pybamm.OptionError(
                "The 'negative' working electrode option has been removed because "
                "the voltage - and therefore the energy stored - would be negative. "
                "Use the 'positive' working electrode option instead and set whatever "
                "would normally be the negative electrode as the positive electrode."
            )

        for option, value in options.items():
            for path, leaf in iter_option_leaves(option, value):
                validate_option_value(option, leaf, self.possible_options[option], path)

        _apply_legacy_defaults(options, set(extra_options))
        if not legacy:
            _check_electrode_compatibility(options)

        # MSMR OCP, particle and kinetics only work together, so check them per
        # electrode and phase; one electrode may be MSMR and the other not
        msmr_options = ["open-circuit potential", "particle", "intercalation kinetics"]
        for domain in active_electrodes(options["working electrode"]):
            phases = _PHASES[
                : int(
                    resolve_option(
                        "particle phases", options["particle phases"], domain
                    )
                )
            ]
            uses_msmr = [
                [
                    resolve_option(opt, options[opt], domain, phase) == "MSMR"
                    for opt in msmr_options
                ]
                for phase in phases
            ]
            for phase, flags in zip(phases, uses_msmr, strict=True):
                if any(flags) and not all(flags):
                    path = (domain, phase) if len(phases) > 1 else (domain,)
                    raise pybamm.OptionError(
                        "If any of 'open-circuit potential', 'particle' or "
                        "'intercalation kinetics' is 'MSMR' then all of them must "
                        f"be 'MSMR'{_option_location(path)}"
                    )
            domain_uses_msmr = any(any(flags) for flags in uses_msmr)
            domain_count = resolve_option(
                "number of MSMR reactions", options["number of MSMR reactions"], domain
            )
            if domain_uses_msmr and not represents_positive_integer(domain_count):
                raise pybamm.OptionError(
                    "'number of MSMR reactions' must be a positive integer for "
                    f"the {domain} electrode when it uses 'MSMR' "
                    f"(got {domain_count!r})"
                )

        # Options not yet compatible with contact resistance
        if options["contact resistance"] == "true":
            if options["operating mode"] == "explicit power":
                raise NotImplementedError(
                    "Contact resistance not yet supported for explicit power."
                )
            if options["operating mode"] == "explicit resistance":
                raise NotImplementedError(
                    "Contact resistance not yet supported for explicit resistance."
                )

        # I = P/V and I = V/R are circular unless the voltage is a state
        if options["voltage as a state"] == "false" and options["operating mode"] in (
            "explicit power",
            "explicit resistance",
        ):
            raise pybamm.OptionError(
                f"Cannot use '{options['operating mode']}' operating mode with "
                "'voltage as a state' set to 'false'. Explicit power and "
                "resistance control require voltage as an algebraic state."
            )

        # Options not yet compatible with particle-size distributions
        distribution_electrodes = [
            domain
            for domain in active_electrodes(options["working electrode"])
            if resolve_option("particle size", options["particle size"], domain)
            == "distribution"
        ]
        if distribution_electrodes:
            if options["lithium plating porosity change"] != "false":
                raise pybamm.OptionError(
                    "Lithium plating porosity change not yet supported for particle-size"
                    " distributions."
                )
            if options["SEI porosity change"] == "true":
                raise NotImplementedError(
                    "SEI porosity change submodels do not yet support particle-size "
                    "distributions."
                )
            if options["heat of mixing"] != "false":
                raise NotImplementedError(
                    "Heat of mixing submodels do not yet support particle-size "
                    "distributions."
                )
            if any(
                resolve_option("particle", options["particle"], domain)
                in ["quadratic profile", "quartic profile"]
                for domain in distribution_electrodes
            ):
                raise NotImplementedError(
                    "'quadratic' and 'quartic' concentration profiles have not yet "
                    "been implemented for particle-size ditributions"
                )
            if options["particle shape"] != "spherical":
                raise NotImplementedError(
                    "Particle shape must be 'spherical' for particle-size distribution"
                    " submodels."
                )
            if options["thermal"] == "x-full":
                raise NotImplementedError(
                    "X-full thermal submodels do not yet support particle-size"
                    " distributions."
                )

        # Some standard checks to make sure options are compatible
        if options["dimensionality"] == 0:
            if options["current collector"] not in ["uniform"]:
                raise pybamm.OptionError(
                    "current collector model must be uniform in 0D model"
                )
            if options["convection"] == "full transverse":
                raise pybamm.OptionError(
                    "cannot have transverse convection in 0D model"
                )
        if options["dimensionality"] == 3 and options["cell geometry"] not in [
            "pouch",
            "cylindrical",
        ]:
            raise pybamm.OptionError(
                "'cell geometry' must be 'pouch' or 'cylindrical' if 'dimensionality' is '3'"
            )

        if options["cell geometry"] == "cylindrical" and options["dimensionality"] != 3:
            raise pybamm.OptionError(
                "'dimensionality' must be '3' if 'cell geometry' is 'cylindrical'"
            )

        if (
            options["thermal"] in ["x-lumped", "x-full"]
            and options["cell geometry"] != "pouch"
        ):
            raise pybamm.OptionError(
                options["thermal"] + " model must have pouch cell geometry."
            )
        if options["thermal"] == "x-full" and options["dimensionality"] != 0:
            n = options["dimensionality"]
            raise pybamm.OptionError(
                f"X-full thermal submodels do not yet support {n}D current collectors"
            )

        if (
            options["use lumped thermal capacity"] == "true"
            and "lumped" not in options["thermal"]
        ):
            raise pybamm.OptionError(
                "Lumped thermal capacity model only compatible with lumped thermal "
                "models"
            )

        if options["working electrode"] != "both":
            if options["thermal"] == "x-full":
                raise pybamm.OptionError(
                    "X-full thermal submodel is not compatible with half-cell models"
                )
            elif options["thermal"] == "x-lumped" and options["dimensionality"] != 0:
                n = options["dimensionality"]
                raise pybamm.OptionError(
                    f"X-lumped thermal submodels do not yet support {n}D "
                    "current collectors in a half-cell configuration"
                )

        if options["surface temperature"] == "lumped" and options["thermal"] not in [
            "isothermal",
            "lumped",
        ]:
            raise pybamm.OptionError(
                "lumped surface temperature model only compatible with isothermal "
                "or lumped thermal model"
            )
        super().__init__(options.items())

    @property
    def phases(self):
        try:
            return self._phases
        except AttributeError:
            self._phases = {}
            for domain in ["negative", "positive"]:
                number = int(getattr(self, domain)["particle phases"])
                phases = ["primary"]
                if number >= 2:
                    phases.append("secondary")
                self._phases[domain] = phases
            return self._phases

    @cached_property
    def whole_cell_domains(self):
        if self["working electrode"] == "positive":
            return ["separator", "positive electrode"]
        elif self["working electrode"] == "both":
            return ["negative electrode", "separator", "positive electrode"]
        else:
            raise NotImplementedError  # future proofing

    @property
    def electrode_types(self):
        try:
            return self._electrode_types
        except AttributeError:
            self._electrode_types = {}
            for domain in ["negative", "positive"]:
                if f"{domain} electrode" in self.whole_cell_domains:
                    self._electrode_types[domain] = "porous"
                else:
                    self._electrode_types[domain] = "planar"
            return self._electrode_types

    def print_options(self):
        """
        Print the possible options with the ones currently selected
        """
        for key, value in self.items():
            print(rf"{key!r}: {value!r} (possible: {self.possible_options[key]!r})")

    def print_detailed_options(self):
        """
        Print the docstring for Options
        """
        print(self.__doc__)

    @property
    def negative(self):
        "Returns the options for the negative electrode"
        # index 0 in a 2-tuple for the negative electrode
        return BatteryModelDomainOptions(self.items(), 0)

    @property
    def positive(self):
        "Returns the options for the positive electrode"
        # index 1 in a 2-tuple for the positive electrode
        return BatteryModelDomainOptions(self.items(), 1)


class BatteryModelDomainOptions(dict):
    def __init__(self, dict_items, index):
        super().__init__(dict_items)
        self.index = index

    def __getitem__(self, key):
        return resolve_option(
            key,
            super().__getitem__(key),
            _ELECTRODES[self.index],
            working_electrode=self.get("working electrode", "both"),
        )

    @property
    def primary(self):
        return BatteryModelPhaseOptions(self, 0)

    @property
    def secondary(self):
        return BatteryModelPhaseOptions(self, 1)


class BatteryModelPhaseOptions(dict):
    def __init__(self, domain_options, index):
        super().__init__(domain_options.items())
        self.domain_options = domain_options
        self.index = index

    def __getitem__(self, key):
        value = self.domain_options[key]
        return value[self.index] if isinstance(value, tuple) else value
