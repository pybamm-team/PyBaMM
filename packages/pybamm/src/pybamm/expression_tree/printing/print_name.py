#
# Prettify print_name
#
import re

PRINT_NAME_OVERRIDES = {
    "current_with_time": "I",
    "current_density_with_time": r"i_{\mathrm{cell}}",
    "thermodynamic_factor": r"\left(1+\frac{dlnf}{dlnc}\right)",
    "t_plus": r"t_{\mathrm{+}}",
}

GREEK_LETTERS = [
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "pi",
    "rho",
    "sigma",
    "tau",
    "upsilon",
    "phi",
    "chi",
    "psi",
    "omega",
]


def _has_tag(name, tag):
    """Whether ``name`` contains ``_tag`` as a whole part, e.g. ``_p`` but not ``_plus``"""
    return re.search(rf"_{re.escape(tag)}(?=_|$)", name) is not None


def _remove_tag(name, tag):
    """Remove each ``_tag`` that is a whole part of ``name``"""
    return re.sub(rf"_{re.escape(tag)}(?=_|$)", "", name)


def prettify_print_name(name):
    """Prettify print_name using regex"""

    # Skip prettify_print_name() for cases like `new_copy()`
    if name is None or "{" in name or "\\" in name:
        return name

    # Find subscripts, superscripts, and averaging
    # Remove them from the name one by one and add them later in processed form
    subscripts = []
    superscripts = []
    average = False

    processing = True
    while processing:
        # Set processing to False. If any of the following conditions are met,
        # it will be set to True again
        processing = False
        for superscript in ["init", "ref", "typ", "max", "0", "surf"]:
            if _has_tag(name, superscript):
                superscripts.append(superscript)
                name = _remove_tag(name, superscript)
                processing = True
                break
        # "0" might also appear without a preceding underscore
        for superscript in ["0"]:
            if superscript in name:
                superscripts.append(superscript)
                name = name.replace(superscript, "")
                processing = True
                break
        for subscript in ["cc", "dl", "R", "e", "s", "n", "p", "amb"]:
            if _has_tag(name, subscript):
                subscripts.append(subscript)
                name = _remove_tag(name, subscript)
                processing = True
                break
        for av in ["av", "xav"]:
            if _has_tag(name, av):
                average = True
                name = _remove_tag(name, av)
                processing = True
                break

    # Process name
    # Override print_name if name exists in the dictionary
    if name in PRINT_NAME_OVERRIDES:
        name = PRINT_NAME_OVERRIDES[name]

    # Replace eps with epsilon (e.g. eps_n --> epsilon_n)
    if name == "eps":
        name = r"\epsilon"
    if name == "eps_c":
        name = r"(\epsilon c)"

    # Split off the parts of the name after an underscore (e.g. j in X_j), so that they
    # join the other subscripts, since e.g. X_j_{n} is invalid LaTeX
    name_subscripts = []
    if re.fullmatch(r"\w+", name) and "_" in name:
        parts = [part for part in name.split("_") if part]
        if parts:
            name, name_subscripts = parts[0], parts[1:]

    # Greek letters (delta --> \delta)
    if name.lower() in GREEK_LETTERS:
        name = "\\" + name

    # Add subscripts and superscripts
    if average:
        name = r"\overline{" + name + "}"
    if len(name_subscripts) == 1 and not subscripts:
        # A single subscript of the name's own, e.g. U_j, is kept as it is
        name += "_" + name_subscripts[0]
    elif name_subscripts or subscripts:
        name += r"_{\mathrm{" + ",".join(name_subscripts + subscripts) + "}}"
    if superscripts:
        name += r"^{\mathrm{" + ",".join(superscripts) + "}}"

    return name
