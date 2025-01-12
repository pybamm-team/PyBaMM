#
# Prettify print_name
#
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
            if f"_{superscript}_" in name or name.endswith(f"_{superscript}"):
                superscripts.append(superscript)
                name = name.replace(f"_{superscript}", "")
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
            if f"_{subscript}_" in name or name.endswith(f"_{subscript}"):
                subscripts.append(subscript)
                name = name.replace(f"_{subscript}", "")
                processing = True
                break
        for av in ["av", "xav"]:
            if f"_{av}_" in name or name.endswith(f"_{av}"):
                average = True
                name = name.replace(f"_{av}", "")
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

    # Greek letters (delta --> \delta)
    if name.lower() in GREEK_LETTERS:
        name = "\\" + name

    # Add subscripts and superscripts
    if average:
        name = r"\overline{" + name + "}"
    if subscripts:
        name += r"_{\mathrm{" + ",".join(subscripts) + "}}"
    if superscripts:
        name += r"^{\mathrm{" + ",".join(superscripts) + "}}"

    return name
