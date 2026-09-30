#
# Base battery model class
#

import os
from functools import cached_property

import pybamm
from pybamm.expression_tree.operations.serialise import Serialise

# set by PyBaMM's test suites; read at import so every process agrees
_FORBID_LEGACY_OPTION_DEFAULTS = (
    os.environ.get("PYBAMM_TEST_FORBID_LEGACY_OPTION_DEFAULTS") == "1"
)


def _legacy_default_message(fired):
    """Describe options that were set from other options.

    Parameters
    ----------
    fired : dict
        Option names and the values they were given.

    Returns
    -------
    str
        The deprecation message.
    """
    return (
        f"Options were set from other options because they were not given: "
        f"{fired!r}. Relying on these defaults is deprecated and a future release "
        "will raise an OptionError instead. Pass these options explicitly to keep "
        "the current behaviour."
    )


def _warn_legacy_defaults(fired):
    """Warn that options were set from other options.

    Parameters
    ----------
    fired : dict
        Option names and the values they were given; nothing happens if empty.

    Raises
    ------
    pybamm.OptionError
        If legacy defaults are forbidden (in PyBaMM's own tests).
    """
    if not fired:
        return
    message = _legacy_default_message(fired)
    if _FORBID_LEGACY_OPTION_DEFAULTS:
        raise pybamm.OptionError(message)
    pybamm.util.warn_outside_pybamm(message, pybamm.OptionDefaultDeprecationWarning)


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


def format_option_path(path):
    """Format an electrode/phase path, e.g. ``("negative", "primary")``.

    Parameters
    ----------
    path : tuple of str
        Electrode and optionally phase names.

    Returns
    -------
    str
        The dotted path, e.g. ``"negative.primary"``.
    """
    return ".".join(path)


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
        if (
            isinstance(electrode_value, list)
            or option not in _PER_PHASE_OPTIONS
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
    location = f" at {format_option_path(path)}" if path else ""
    raise pybamm.OptionError(
        f"\n'{value}' is not recognized in option '{option}'{location}. "
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
    """
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
    location = f" at {format_option_path(path)}" if path else ""
    return pybamm.OptionError(
        f"Option '{option}'{location} is '{value}', which requires "
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

    Returns
    -------
    dict
        Option names and the values they were given, for options that were
        not supplied and whose default differed from the value already set.
    """
    working_electrode = options["working electrode"]
    fired = {}

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
            if value != options[option]:
                fired[option] = value
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
    if "SEI" in fired and "SEI film resistance" not in supplied:
        # pin it explicitly so migrating "SEI" alone can't let this fire later
        fired["SEI film resistance"] = options["SEI film resistance"]
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
    if film_resistance == "distributed" or (film_resistance != "none" and multi_phase):
        set_default("total interfacial current density as a state", "true")
    if options["operating mode"] in ("explicit power", "explicit resistance"):
        set_default("voltage as a state", "true")

    current_state = options["total interfacial current density as a state"]
    if film_resistance == "distributed" and current_state == "false":
        raise dependency_error(
            "SEI film resistance",
            "distributed",
            "total interfacial current density as a state",
            "'true'",
        )
    if film_resistance != "none" and multi_phase and current_state == "false":
        raise dependency_error(
            "SEI film resistance",
            film_resistance,
            "total interfacial current density as a state",
            "'true' when an electrode has multiple particle phases",
        )

    # Stress-driven LAM needs a mechanical model on the same electrode/phase to
    # supply the particle stress it acts on.
    for domain in active_electrodes(working_electrode):
        num_phases = int(
            resolve_option(
                "particle phases",
                options["particle phases"],
                domain,
                working_electrode=working_electrode,
            )
        )
        for phase in _PHASES[:num_phases]:
            lam_leaf = resolve_option(
                "loss of active material",
                options["loss of active material"],
                domain,
                phase,
                working_electrode=working_electrode,
            )
            mechanics_leaf = resolve_option(
                "particle mechanics",
                options["particle mechanics"],
                domain,
                phase,
                working_electrode=working_electrode,
            )
            if "stress" in lam_leaf and mechanics_leaf == "none":
                path = (domain,) if num_phases == 1 else (domain, phase)
                raise dependency_error(
                    "loss of active material",
                    lam_leaf,
                    "particle mechanics",
                    "a model other than 'none'",
                    path,
                )

    return fired


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

            def value(option, domain=domain, phase=phase):
                return resolve_option(
                    option, options[option], domain, phase, working_electrode
                )

            if (
                value("lithium plating") == "partially reversible"
                and value("SEI") == "none"
            ):
                raise dependency_error(
                    "lithium plating",
                    "partially reversible",
                    "SEI",
                    "a model other than 'none' (e.g. 'constant')",
                    path,
                )
            if value("SEI on cracks") == "true" and (
                value("particle mechanics") != "swelling and cracking"
            ):
                raise dependency_error(
                    "SEI on cracks",
                    "true",
                    "particle mechanics",
                    "'swelling and cracking'",
                    path,
                )
            if value("stress-induced diffusion") == "true" and (
                value("particle mechanics") == "none"
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

        Some options below "should be given" when another option is set. If
        they are not, they still take a legacy default derived from the other
        option, with a :class:`pybamm.OptionDefaultDeprecationWarning` naming the
        options to pass; a future release will raise an ``OptionError`` instead.

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
                with prescribed surface area for cooling. Should be given as
                "pouch" when "dimensionality" is 1 or 2, or "thermal" is
                "x-full" (legacy default: "pouch").
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
                behaviour in negative and positive electrodes. Should be given
                explicitly on an electrode where "SEI on cracks" is "true"
                (then "swelling and cracking" is intended) or "loss of active
                material" is stress-driven (then "swelling only"); the legacy
                default is set per electrode in the same way, else "none".
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
                ``options["SEI"]`` stores the value as given. Should be given
                explicitly as "constant" on an electrode where "lithium plating"
                is "partially reversible" (legacy default: "constant" there,
                which leaves the "SEI film resistance" default at "none").
            * "SEI film resistance" : str
                Set the submodel for additional term in the overpotential due to SEI.
                Should be given explicitly as "distributed" on any electrode where
                the "SEI" option is not "none" (legacy default: "distributed"
                then, else "none"). This is because
                the "distributed" model is more complex than the model with no
                additional resistance, which adds unnecessary complexity if
                there is no SEI in the first place

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
                Should be given explicitly on an electrode (and phase) where
                "particle mechanics" is not "none" (legacy default, per
                electrode and phase: "true" there, else "false"). A 2-tuple
                can be provided for different behaviour in negative and positive
                electrodes.
            * "surface form" : str
                Whether to use the surface formulation of the problem. Can be "false"
                (default), "differential" or "algebraic". Should be given
                explicitly as "algebraic" when an electrode has multiple
                particle phases, or (for SPM and SPMe, but not MPM) when
                "intercalation kinetics" is given or a "distribution"
                "particle size" is set (legacy default: "algebraic"). MPM always
                defaults "surface form" to "algebraic" as part of its own
                model identity, which is not deprecated.
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
                solve an algebraic equation for it. Should be given explicitly as
                "true" when "SEI film resistance" is "distributed", or when it
                is not "none" and an electrode has multiple particle phases;
                (legacy default: "true" then, else "false").
            * "voltage as a state" : str
                Whether to promote voltage to an algebraic state variable.
                Can be "false" (default) or "true". Should be given explicitly
                as "true" when "operating mode" is "explicit power" or
                "explicit resistance" (legacy default: "true").
                When "true", the model is a DAE
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
    """

    def __init__(self, extra_options, warn_legacy_defaults=True):
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

        fired = _apply_legacy_defaults(options, set(extra_options))
        _check_electrode_compatibility(options)

        # All-or-nothing on full cells: if any of OCP/particle/intercalation
        # kinetics requests MSMR (incl. inside a per-electrode tuple), all must.
        # Half-cells are loosened -- a tuple sets MSMR in the working electrode.
        msmr_check_list = [
            any(leaf == "MSMR" for _, leaf in iter_option_leaves(opt, options[opt]))
            for opt in ["open-circuit potential", "particle", "intercalation kinetics"]
        ]
        if (
            options["working electrode"] == "both"
            and any(msmr_check_list)
            and not all(msmr_check_list)
        ):
            raise pybamm.OptionError(
                "If any of 'open-circuit potential', 'particle' or "
                "'intercalation kinetics' is 'MSMR' then all of them must be 'MSMR'"
            )

        # Validate per electrode so a mixed full cell (MSMR in one electrode,
        # conventional in the other) is accepted.
        for domain in active_electrodes(options["working electrode"]):
            domain_uses_msmr = any(
                resolve_option(opt, options[opt], domain) == "MSMR"
                for opt in [
                    "open-circuit potential",
                    "particle",
                    "intercalation kinetics",
                ]
            )
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

        # Explicit power/resistance need voltage as a state variable because
        # I = P/V (or I = V/R) creates a circular dependency when V is an
        # expression that itself depends on I.
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
        if options["particle size"] == "distribution":
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
            if options["particle"] in ["quadratic profile", "quartic profile"]:
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
        self._legacy_defaults = fired
        if warn_legacy_defaults:
            _warn_legacy_defaults(fired)

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
        return resolve_option(
            key,
            dict.__getitem__(self.domain_options, key),
            _ELECTRODES[self.domain_options.index],
            _PHASES[self.index],
            working_electrode=self.domain_options.get("working electrode", "both"),
        )


class BaseBatteryModel(pybamm.BaseModel):
    """
    Base model class with some default settings and required variables

    Parameters
    ----------
    options : dict-like, optional
        A dictionary of options to be passed to the model. If this is a dict (and not
        a subtype of dict), it will be processed by :class:`pybamm.BatteryModelOptions`
        to ensure that the options are valid. If this is a subtype of dict, it is
        assumed that the options have already been processed and are valid. This allows
        for the use of custom options classes. The default options are given by
        :class:`pybamm.BatteryModelOptions`.
    name : str, optional
        The name of the model. The default is "Unnamed battery model".
    """

    def __init__(self, options=None, name="Unnamed battery model"):
        super().__init__(name)
        self.options = options

    @classmethod
    def deserialise(cls, properties: dict):
        """
        Create a model instance from a serialised object.
        """

        # append the model name with _saved to differentiate
        instance = cls(
            options=properties["options"], name=properties["name"] + "_saved"
        )

        return cls.generic_deserialise(instance, properties)

    @property
    def default_geometry(self):
        if self.options["cell geometry"] == "cylindrical":
            return pybamm.battery_geometry(
                options=self.options, form_factor="cylindrical"
            )
        else:
            return pybamm.battery_geometry(options=self.options)

    @property
    def default_var_pts(self):
        base_var_pts = {
            "x_n": 20,
            "x_s": 20,
            "x_p": 20,
            "r_n": 20,
            "r_p": 20,
            "r_n_prim": 20,
            "r_p_prim": 20,
            "r_n_sec": 20,
            "r_p_sec": 20,
            "y": 10,
            "z": 10,
            "R_n": 30,
            "R_p": 30,
            "R_n_prim": 30,
            "R_p_prim": 30,
            "R_n_sec": 30,
            "R_p_sec": 30,
        }
        # Reduce the default points for 2D current collectors
        if self.options["dimensionality"] == 2:
            base_var_pts.update({"x_n": 10, "x_s": 10, "x_p": 10})
        return base_var_pts

    @property
    def default_submesh_types(self):
        base_submeshes = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
            "negative particle": pybamm.Uniform1DSubMesh,
            "positive particle": pybamm.Uniform1DSubMesh,
            "negative primary particle": pybamm.Uniform1DSubMesh,
            "positive primary particle": pybamm.Uniform1DSubMesh,
            "negative secondary particle": pybamm.Uniform1DSubMesh,
            "positive secondary particle": pybamm.Uniform1DSubMesh,
            "negative particle size": pybamm.Uniform1DSubMesh,
            "positive particle size": pybamm.Uniform1DSubMesh,
            "negative primary particle size": pybamm.Uniform1DSubMesh,
            "positive primary particle size": pybamm.Uniform1DSubMesh,
            "negative secondary particle size": pybamm.Uniform1DSubMesh,
            "positive secondary particle size": pybamm.Uniform1DSubMesh,
        }
        if self.options["dimensionality"] == 0:
            base_submeshes["current collector"] = pybamm.SubMesh0D
        elif self.options["dimensionality"] == 1:
            base_submeshes["current collector"] = pybamm.Uniform1DSubMesh

        elif self.options["dimensionality"] == 2:
            base_submeshes["current collector"] = pybamm.ScikitUniform2DSubMesh
        elif self.options["dimensionality"] == 3:
            base_submeshes["current collector"] = pybamm.SubMesh0D
            geom_type = self.options.get("cell geometry", "pouch")
            if geom_type == "pouch":
                base_submeshes["cell"] = pybamm.ScikitFemGenerator3D(
                    geom_type="pouch", h="0.1"
                )
            elif geom_type == "cylindrical":
                base_submeshes["cell"] = pybamm.ScikitFemGenerator3D(
                    geom_type="cylinder", h="0.1"
                )
        return base_submeshes

    @property
    def default_spatial_methods(self):
        base_spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "positive particle": pybamm.FiniteVolume(),
            "negative primary particle": pybamm.FiniteVolume(),
            "positive primary particle": pybamm.FiniteVolume(),
            "negative secondary particle": pybamm.FiniteVolume(),
            "positive secondary particle": pybamm.FiniteVolume(),
            "negative particle size": pybamm.FiniteVolume(),
            "positive particle size": pybamm.FiniteVolume(),
            "negative primary particle size": pybamm.FiniteVolume(),
            "positive primary particle size": pybamm.FiniteVolume(),
            "negative secondary particle size": pybamm.FiniteVolume(),
            "positive secondary particle size": pybamm.FiniteVolume(),
        }
        if self.options["dimensionality"] == 0:
            # 0D submesh - use base spatial method
            base_spatial_methods["current collector"] = (
                pybamm.ZeroDimensionalSpatialMethod()
            )
        elif self.options["dimensionality"] == 1:
            base_spatial_methods["current collector"] = pybamm.FiniteVolume()
        elif self.options["dimensionality"] == 2:
            base_spatial_methods["current collector"] = pybamm.ScikitFiniteElement()
        elif self.options["dimensionality"] == 3:
            base_spatial_methods["current collector"] = (
                pybamm.ZeroDimensionalSpatialMethod()
            )
            base_spatial_methods["cell"] = pybamm.ScikitFiniteElement3D()
        return base_spatial_methods

    def _model_default_options(self, supplied):
        """Return the options that define this model, merged under the caller's.

        Parameters
        ----------
        supplied : dict
            The options given by the caller.

        Returns
        -------
        dict
            The model's default options.

        Raises
        ------
        pybamm.OptionError
            If a supplied option is incompatible with the model.
        """
        return {}

    def _model_legacy_defaults(self, supplied):
        """Return legacy defaults introduced by this model."""
        return {}

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, extra_options):
        # if extra_options is a dict then process it into a BatteryModelOptions
        # this does not catch cases that subclass the dict type
        # so other submodels can pass in their own options class if needed
        if extra_options is None or type(extra_options) == dict:
            supplied = dict(extra_options or {})
            options = BatteryModelOptions(
                {**self._model_default_options(supplied), **supplied},
                warn_legacy_defaults=False,
            )
            legacy_defaults = options._legacy_defaults
            model_legacy_defaults = self._model_legacy_defaults(supplied)
        else:
            options = extra_options
            legacy_defaults = {}
            model_legacy_defaults = {}
            # processed options carry every key, so only the model checks matter
            self._model_default_options(dict(options))

        # Options that are incompatible with models
        if (
            isinstance(self, pybamm.lithium_ion.BaseModel)
            and options["convection"] != "none"
        ):
            raise pybamm.OptionError(
                "convection not implemented for lithium-ion models"
            )
        if isinstance(self, pybamm.lithium_ion.SPMe) and options[
            "electrolyte conductivity"
        ] not in [
            "default",
            "composite",
            "integrated",
        ]:
            raise pybamm.OptionError(
                "electrolyte conductivity '{}' not suitable for SPMe".format(
                    options["electrolyte conductivity"]
                )
            )
        if (
            isinstance(self, pybamm.lithium_ion.SPM)
            and not isinstance(self, pybamm.lithium_ion.SPMe)
            and options["x-average side reactions"] == "false"
        ):
            raise pybamm.OptionError(
                "x-average side reactions cannot be 'false' for SPM models"
            )
        if isinstance(self, pybamm.lithium_ion.SPM) and (
            "distribution" in options["particle size"]
            and options["surface form"] == "false"
        ):
            raise pybamm.OptionError(
                "surface form must be 'algebraic' or 'differential' if "
                " 'particle size' contains a 'distribution'"
            )
        if isinstance(self, pybamm.lead_acid.BaseModel):
            if options["thermal"] != "isothermal" and options["dimensionality"] != 0:
                raise pybamm.OptionError(
                    "Lead-acid models can only have thermal "
                    "effects if dimensionality is 0."
                )
            if options["SEI"] != "none" or options["SEI film resistance"] != "none":
                raise pybamm.OptionError("Lead-acid models cannot have SEI formation")
            if options["lithium plating"] != "none":
                raise pybamm.OptionError("Lead-acid models cannot have lithium plating")
            if options["open-circuit potential"] == "MSMR":
                raise pybamm.OptionError(
                    "Lead-acid models cannot use the MSMR open-circuit potential model"
                )

        if (
            isinstance(self, pybamm.lead_acid.LOQS)
            and options["surface form"] == "false"
            and options["hydrolysis"] == "true"
        ):
            raise pybamm.OptionError(
                f"must use surface formulation to solve {self!s} with hydrolysis"
            )
        _warn_legacy_defaults(legacy_defaults)
        _warn_legacy_defaults(model_legacy_defaults)
        self._options = options
        # rebuild whenever options are (re)assigned.
        # No-op unless the subclass overrides ``_rebuild_param``.
        self._rebuild_param()

    def set_standard_output_variables(self):
        # Time
        self.variables.update(
            {
                "Time [s]": pybamm.t,
                "Time [min]": pybamm.t / 60,
                "Time [h]": pybamm.t / 3600,
            }
        )

        # Spatial
        var = pybamm.standard_spatial_vars
        self.variables.update(
            {"x [m]": var.x, "x_n [m]": var.x_n, "x_s [m]": var.x_s, "x_p [m]": var.x_p}
        )
        if self.options["dimensionality"] == 1:
            self.variables.update({"z [m]": var.z})
        elif self.options["dimensionality"] == 2:
            self.variables.update({"y [m]": var.y, "z [m]": var.z})

    def build_model_equations(self):
        # Set model equations
        for submodel_name, submodel in self.submodels.items():
            pybamm.logger.verbose(
                f"Setting rhs for {submodel_name} submodel ({self.name})"
            )

            submodel.set_rhs(self.variables)
            pybamm.logger.verbose(
                f"Setting algebraic for {submodel_name} submodel ({self.name})"
            )

            submodel.set_algebraic(self.variables)
            pybamm.logger.verbose(
                f"Setting boundary conditions for {submodel_name} submodel ({self.name})"
            )

            submodel.set_boundary_conditions(self.variables)
            pybamm.logger.verbose(
                f"Setting initial conditions for {submodel_name} submodel ({self.name})"
            )
            submodel.set_initial_conditions(self.variables)
            submodel.add_events_from(self.variables)
            pybamm.logger.verbose(f"Updating {submodel_name} submodel ({self.name})")
            self.update(submodel)
            self.check_no_repeated_keys()

    def _build_model(self):
        if self._built:
            raise pybamm.ModelError(
                """Model already built. If you are adding a new submodel, try using
                `model.update` instead."""
            )

        pybamm.logger.info(f"Start building {self.name}")

        if self._built_fundamental is False:
            self.build_fundamental()

        # Register the voltage state Variable before coupled variables,
        # so submodels that need V in get_coupled_variables can find it.
        if self.options["voltage as a state"] == "true":
            self._register_voltage_variable()

        self.build_coupled_variables()

        # Now that the expression is available, add the algebraic constraint
        # before equations are built.
        if self.options["voltage as a state"] == "true":
            self._constrain_voltage_to_expression()

        self.build_model_equations()

    def build_model(self):
        # Build model variables and equations
        self._build_model()

        # Set battery specific variables
        pybamm.logger.debug(f"Setting voltage variables ({self.name})")
        self.set_voltage_variables()

        pybamm.logger.debug(f"Setting SoC variables ({self.name})")
        self.set_soc_variables()

        pybamm.logger.debug(f"Setting degradation variables ({self.name})")
        self.set_degradation_variables()
        self.set_summary_variables()

        self._built = True
        pybamm.logger.info(f"Finish building {self.name}")

    @property
    def summary_variables(self):
        return self._summary_variables

    @summary_variables.setter
    def summary_variables(self, value):
        """
        Set summary variables

        Parameters
        ----------
        value : list of strings
            Names of the summary variables. Must all be in self.variables.
        """
        for var in value:
            if var not in self.variables:
                raise KeyError(
                    f"No cycling variable defined for summary variable '{var}'"
                )
        self._summary_variables = value

    def set_summary_variables(self):
        self._summary_variables = []

    def get_intercalation_kinetics(self, domain):
        options = getattr(self.options, domain)
        if options["intercalation kinetics"] == "symmetric Butler-Volmer":
            return pybamm.kinetics.SymmetricButlerVolmer
        elif options["intercalation kinetics"] == "asymmetric Butler-Volmer":
            return pybamm.kinetics.AsymmetricButlerVolmer
        elif options["intercalation kinetics"] == "linear":
            return pybamm.kinetics.Linear
        elif options["intercalation kinetics"] == "Marcus":
            return pybamm.kinetics.Marcus
        elif options["intercalation kinetics"] == "Marcus-Hush-Chidsey":
            return pybamm.kinetics.MarcusHushChidsey
        elif options["intercalation kinetics"] == "MSMR":
            return pybamm.kinetics.MSMRButlerVolmer

    def get_inverse_intercalation_kinetics(self, domain):
        options = getattr(self.options, domain)
        if options["intercalation kinetics"] == "symmetric Butler-Volmer":
            return pybamm.kinetics.InverseButlerVolmer
        elif options["intercalation kinetics"] == "linear":
            return pybamm.kinetics.InverseLinear
        else:
            raise pybamm.OptionError(
                "Inverse kinetics are only implemented for symmetric Butler-Volmer. "
                "Use option {'surface form': 'algebraic'} to use forward kinetics "
                "instead."
            )

    def set_external_circuit_submodel(self):
        """
        Define how the external circuit defines the boundary conditions for the model,
        e.g. (not necessarily constant-) current, voltage, etc
        """
        if self.options["operating mode"] == "current":
            model = pybamm.external_circuit.ExplicitCurrentControl(
                self.param, self.options
            )
        elif self.options["operating mode"] == "voltage":
            model = pybamm.external_circuit.VoltageFunctionControl(
                self.param, self.options
            )
        elif self.options["operating mode"] == "power":
            model = pybamm.external_circuit.PowerFunctionControl(
                self.param, self.options, "algebraic"
            )
        elif self.options["operating mode"] == "differential power":
            model = pybamm.external_circuit.PowerFunctionControl(
                self.param, self.options, "differential"
            )
        elif self.options["operating mode"] == "explicit power":
            model = pybamm.external_circuit.ExplicitPowerControl(
                self.param, self.options
            )
        elif self.options["operating mode"] == "resistance":
            model = pybamm.external_circuit.ResistanceFunctionControl(
                self.param, self.options, "algebraic"
            )
        elif self.options["operating mode"] == "differential resistance":
            model = pybamm.external_circuit.ResistanceFunctionControl(
                self.param, self.options, "differential"
            )
        elif self.options["operating mode"] == "explicit resistance":
            model = pybamm.external_circuit.ExplicitResistanceControl(
                self.param, self.options
            )
        elif self.options["operating mode"] == "CCCV":
            model = pybamm.external_circuit.CCCVFunctionControl(
                self.param, self.options
            )
        elif callable(self.options["operating mode"]):
            model = pybamm.external_circuit.FunctionControl(
                self.param, self.options["operating mode"], self.options
            )
        self.submodels["external circuit"] = model
        self.submodels["discharge and throughput variables"] = (
            pybamm.external_circuit.DischargeThroughput(self.param, self.options)
        )

    def set_transport_efficiency_submodels(self):
        if self.options["transport efficiency"] == "Bruggeman":
            self.submodels["electrolyte transport efficiency"] = (
                pybamm.transport_efficiency.Bruggeman(
                    self.param, "Electrolyte", self.options
                )
            )
            self.submodels["electrode transport efficiency"] = (
                pybamm.transport_efficiency.Bruggeman(
                    self.param, "Electrode", self.options
                )
            )
        elif self.options["transport efficiency"] == "tortuosity factor":
            self.submodels["electrolyte transport efficiency"] = (
                pybamm.transport_efficiency.TortuosityFactor(
                    self.param, "Electrolyte", self.options
                )
            )
            self.submodels["electrode transport efficiency"] = (
                pybamm.transport_efficiency.TortuosityFactor(
                    self.param, "Electrode", self.options
                )
            )
        elif self.options["transport efficiency"] == "ordered packing":
            self.submodels["electrolyte transport efficiency"] = (
                pybamm.transport_efficiency.OrderedPacking(
                    self.param, "Electrolyte", self.options
                )
            )
            self.submodels["electrode transport efficiency"] = (
                pybamm.transport_efficiency.OrderedPacking(
                    self.param, "Electrode", self.options
                )
            )
        elif self.options["transport efficiency"] == "hyperbola of revolution":
            self.submodels["electrolyte transport efficiency"] = (
                pybamm.transport_efficiency.HyperbolaOfRevolution(
                    self.param, "Electrolyte", self.options
                )
            )
            self.submodels["electrode transport efficiency"] = (
                pybamm.transport_efficiency.HyperbolaOfRevolution(
                    self.param, "Electrode", self.options
                )
            )
        elif self.options["transport efficiency"] == "overlapping spheres":
            self.submodels["electrolyte transport efficiency"] = (
                pybamm.transport_efficiency.OverlappingSpheres(
                    self.param, "Electrolyte", self.options
                )
            )
            self.submodels["electrode transport efficiency"] = (
                pybamm.transport_efficiency.OverlappingSpheres(
                    self.param, "Electrode", self.options
                )
            )
        elif self.options["transport efficiency"] == "random overlapping cylinders":
            self.submodels["electrolyte transport efficiency"] = (
                pybamm.transport_efficiency.RandomOverlappingCylinders(
                    self.param, "Electrolyte", self.options
                )
            )
            self.submodels["electrode transport efficiency"] = (
                pybamm.transport_efficiency.RandomOverlappingCylinders(
                    self.param, "Electrode", self.options
                )
            )
        elif self.options["transport efficiency"] == "heterogeneous catalyst":
            self.submodels["electrolyte transport efficiency"] = (
                pybamm.transport_efficiency.HeterogeneousCatalyst(
                    self.param, "Electrolyte", self.options
                )
            )
            self.submodels["electrode transport efficiency"] = (
                pybamm.transport_efficiency.HeterogeneousCatalyst(
                    self.param, "Electrode", self.options
                )
            )
        elif self.options["transport efficiency"] == "cation-exchange membrane":
            self.submodels["electrolyte transport efficiency"] = (
                pybamm.transport_efficiency.CationExchangeMembrane(
                    self.param, "Electrolyte", self.options
                )
            )
            self.submodels["electrode transport efficiency"] = (
                pybamm.transport_efficiency.CationExchangeMembrane(
                    self.param, "Electrode", self.options
                )
            )

    def set_thermal_submodel(self):
        if self.options["thermal"] == "isothermal":
            thermal_submodel = pybamm.thermal.isothermal.Isothermal
        elif self.options["thermal"] == "lumped":
            thermal_submodel = pybamm.thermal.Lumped
        elif self.options["thermal"] == "x-lumped":
            if self.options["dimensionality"] == 0:
                thermal_submodel = pybamm.thermal.Lumped
            elif self.options["dimensionality"] == 1:
                thermal_submodel = pybamm.thermal.pouch_cell.CurrentCollector1D
            elif self.options["dimensionality"] == 2:
                thermal_submodel = pybamm.thermal.pouch_cell.CurrentCollector2D
        elif (
            self.options["thermal"] == "x-full" and self.options["dimensionality"] == 0
        ):
            thermal_submodel = pybamm.thermal.pouch_cell.OneDimensionalX

        x_average = getattr(self, "x_average", False)
        self.submodels["thermal"] = thermal_submodel(
            self.param, self.options, x_average
        )

    def set_surface_temperature_submodel(self):
        if self.options["surface temperature"] == "ambient":
            submodel = pybamm.thermal.surface.Ambient
        elif self.options["surface temperature"] == "lumped":
            submodel = pybamm.thermal.surface.Lumped
        self.submodels["surface temperature"] = submodel(self.param, self.options)

    def set_current_collector_submodel(self):
        if self.options["current collector"] in ["uniform"]:
            submodel = pybamm.current_collector.Uniform(self.param)
        elif self.options["current collector"] == "potential pair":
            if self.options["dimensionality"] == 1:
                submodel = pybamm.current_collector.PotentialPair1plus1D(self.param)
            elif self.options["dimensionality"] == 2:
                submodel = pybamm.current_collector.PotentialPair2plus1D(self.param)
        self.submodels["current collector"] = submodel

    def set_interface_utilisation_submodel(self):
        for domain in ["negative", "positive"]:
            Domain = domain.capitalize()
            util = getattr(self.options, domain)["interface utilisation"]
            if util == "full":
                submodel = pybamm.interface_utilisation.Full(
                    self.param, domain, self.options
                )
            elif util == "constant":
                submodel = pybamm.interface_utilisation.Constant(
                    self.param, domain, self.options
                )
            elif util == "current-driven":
                if self.options.electrode_types[domain] == "planar":
                    reaction_loc = "interface"
                elif self.x_average:
                    reaction_loc = "x-average"
                else:
                    reaction_loc = "full electrode"
                submodel = pybamm.interface_utilisation.CurrentDriven(
                    self.param, domain, self.options, reaction_loc
                )
            self.submodels[f"{Domain} interface utilisation"] = submodel

    def _register_voltage_variable(self):
        V = pybamm.Variable("Voltage [V]")
        self.variables["Voltage [V]"] = V
        self.initial_conditions[V] = self.param.ocv_init

    def _constrain_voltage_to_expression(self):
        V = self.variables["Voltage [V]"]
        V_expr = self.variables.get("Voltage expression [V]")
        if V_expr is None:
            raise pybamm.ModelError(
                "'voltage as a state' requires the model to define "
                "'Voltage expression [V]'"
            )
        self.algebraic[V] = V - V_expr

    def set_voltage_variables(self):
        if self.options.negative["particle phases"] == "1":
            # Only one phase, no need to distinguish between
            # "primary" and "secondary"
            phase_n = ""
        else:
            # add a space so that we can use "" or (e.g.) "primary " interchangeably
            # when naming variables
            phase_n = "primary "
        if self.options.positive["particle phases"] == "1":
            phase_p = ""
        else:
            phase_p = "primary "

        ocp_surf_n_av = self.variables[
            f"X-averaged negative electrode {phase_n}open-circuit potential [V]"
        ]
        ocp_surf_p_av = self.variables[
            f"X-averaged positive electrode {phase_p}open-circuit potential [V]"
        ]
        ocp_n_bulk = self.variables[
            f"Negative electrode {phase_n}bulk open-circuit potential [V]"
        ]
        ocp_p_bulk = self.variables[
            f"Positive electrode {phase_p}bulk open-circuit potential [V]"
        ]
        eta_particle_n = self.variables[
            f"Negative {phase_n}particle concentration overpotential [V]"
        ]
        eta_particle_p = self.variables[
            f"Positive {phase_p}particle concentration overpotential [V]"
        ]

        ocv_surf = ocp_surf_p_av - ocp_surf_n_av
        ocv_bulk = ocp_p_bulk - ocp_n_bulk

        eta_particle = eta_particle_p - eta_particle_n

        # overpotentials
        if self.options.electrode_types["negative"] == "planar":
            eta_r_n_av = self.variables[
                "Lithium metal interface reaction overpotential [V]"
            ]
        else:
            eta_r_n_av = self.variables[
                f"X-averaged negative electrode {phase_n}reaction overpotential [V]"
            ]
        eta_r_p_av = self.variables[
            f"X-averaged positive electrode {phase_p}reaction overpotential [V]"
        ]
        eta_r_av = eta_r_p_av - eta_r_n_av

        delta_phi_s_n_av = self.variables[
            "X-averaged negative electrode ohmic losses [V]"
        ]
        delta_phi_s_p_av = self.variables[
            "X-averaged positive electrode ohmic losses [V]"
        ]
        delta_phi_s_av = delta_phi_s_p_av - delta_phi_s_n_av

        # SEI film overpotential
        if self.options.electrode_types["negative"] == "planar":
            eta_sei_n_av = self.variables[
                "Negative electrode SEI film overpotential [V]"
            ]
        else:
            eta_sei_n_av = self.variables[
                f"X-averaged negative electrode {phase_n}SEI film overpotential [V]"
            ]
        eta_sei_p_av = self.variables[
            f"X-averaged positive electrode {phase_p}SEI film overpotential [V]"
        ]
        eta_sei_av = eta_sei_n_av + eta_sei_p_av

        # TODO: add current collector losses to the voltage in 3D

        self.variables.update(
            {
                "Surface open-circuit voltage [V]": ocv_surf,
                "Bulk open-circuit voltage [V]": ocv_bulk,
                "Particle concentration overpotential [V]": eta_particle,
                "X-averaged reaction overpotential [V]": eta_r_av,
                "X-averaged SEI film overpotential [V]": eta_sei_av,
                "X-averaged solid phase ohmic losses [V]": delta_phi_s_av,
            }
        )

        # Battery-wide variables
        V = self.variables["Voltage [V]"]
        eta_e_av = self.variables["X-averaged electrolyte ohmic losses [V]"]
        eta_c_av = self.variables["X-averaged concentration overpotential [V]"]
        num_cells = pybamm.Parameter(
            "Number of cells connected in series to make a battery"
        )
        self.variables.update(
            {
                "Battery open-circuit voltage [V]": ocv_bulk * num_cells,
                "Battery negative electrode bulk open-circuit potential [V]": ocp_n_bulk
                * num_cells,
                "Battery positive electrode bulk open-circuit potential [V]": ocp_p_bulk
                * num_cells,
                "Battery particle concentration overpotential [V]": eta_particle
                * num_cells,
                "Battery negative particle concentration overpotential [V]"
                "": eta_particle_n * num_cells,
                "Battery positive particle concentration overpotential [V]"
                "": eta_particle_p * num_cells,
                "X-averaged battery reaction overpotential [V]": eta_r_av * num_cells,
                "X-averaged battery negative reaction overpotential [V]": eta_r_n_av
                * num_cells,
                "X-averaged battery positive reaction overpotential [V]": eta_r_p_av
                * num_cells,
                "X-averaged battery solid phase ohmic losses [V]": delta_phi_s_av
                * num_cells,
                "X-averaged battery negative solid phase ohmic losses [V]"
                "": delta_phi_s_n_av * num_cells,
                "X-averaged battery positive solid phase ohmic losses [V]"
                "": delta_phi_s_p_av * num_cells,
                "X-averaged battery electrolyte ohmic losses [V]": eta_e_av * num_cells,
                "X-averaged battery concentration overpotential [V]": eta_c_av
                * num_cells,
                "Battery voltage [V]": V * num_cells,
            }
        )

        # Calculate equivalent resistance of an OCV-R Equivalent Circuit Model
        # ECM overvoltage is OCV minus voltage
        v_ecm = ocv_bulk - V

        # Hack to avoid division by zero if i_cc is exactly zero
        # If i_cc is zero, i_cc_not_zero becomes 1. But multiplying by sign(i_cc) makes
        # the local resistance 'zero' (really, it's not defined when i_cc is zero)
        def x_not_zero(x):
            return ((x > 0) + (x < 0)) * x + (x >= 0) * (x <= 0)

        i_cc = self.variables["Current collector current density [A.m-2]"]
        i_cc_not_zero = x_not_zero(i_cc)
        A_cc = self.param.A_cc

        self.variables.update(
            {
                "Local ECM resistance [Ohm]": pybamm.sign(i_cc)
                * v_ecm
                / (i_cc_not_zero * A_cc),
            }
        )

        # Cut-off voltage
        self.events.append(
            pybamm.Event(
                "Minimum voltage [V]",
                V - self.param.voltage_low_cut,
                pybamm.EventType.TERMINATION,
            )
        )
        self.events.append(
            pybamm.Event(
                "Maximum voltage [V]",
                self.param.voltage_high_cut - V,
                pybamm.EventType.TERMINATION,
            )
        )

        # Cut-off open-circuit voltage (for event switch with casadi 'fast with events'
        # mode)
        tol = 0.1
        self.events.append(
            pybamm.Event(
                "Minimum voltage switch [V]",
                V - (self.param.voltage_low_cut - tol),
                pybamm.EventType.SWITCH,
            )
        )
        self.events.append(
            pybamm.Event(
                "Maximum voltage switch [V]",
                V - (self.param.voltage_high_cut + tol),
                pybamm.EventType.SWITCH,
            )
        )

        # Power and resistance
        I = self.variables["Current [A]"]
        I_not_zero = x_not_zero(I)
        self.variables.update(
            {
                "Terminal power [W]": I * V,
                "Power [W]": I * V,
                "Resistance [Ohm]": pybamm.sign(I) * V / I_not_zero,
            }
        )

    def set_degradation_variables(self):
        """
        Set variables that quantify degradation.
        This function is overriden by the base battery models
        """

    def set_soc_variables(self):
        """
        Set variables relating to the state of charge.
        This function is overriden by the base battery models
        """

    def save_model(self, filename=None, mesh=None, variables=None):
        """
        Write out a discretised model to a JSON file

        Parameters
        ----------
        filename: str, optional
        The desired name of the JSON file. If no name is provided, one will be created
        based on the model name, and the current datetime.
        """
        if variables and not mesh:
            raise ValueError(
                "Serialisation: Please provide the mesh if variables are required"
            )

        Serialise().save_model(self, filename=filename, mesh=mesh, variables=variables)
