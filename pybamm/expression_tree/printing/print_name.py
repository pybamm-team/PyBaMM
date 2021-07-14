#
# Prettify print_name
#
import re

PRINT_NAME_OVERRIDES = {
    "potential_scale": r"\frac{RT^{ref}}{F}",
    "Theta": r"\frac{1}{\hat{T}^{ref}}",
    "current_with_time": "I",
    "timescale": r"\tau",
    "dimensional_current_with_time": r"\hat{I}",
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
    if "{" in name:
        return name

    # Return print_name if name exists in the dictionary
    if name in PRINT_NAME_OVERRIDES:
        return PRINT_NAME_OVERRIDES[name]

    # Superscripts with comma separated
    # e.g. U_n_ref --> U^{n\,ref}
    if any(elem in name for elem in ["_init", "_ref", "_typ", "_max", "_0"]):
        superscript_re = re.findall(
            r"^[0-9a-zA-Z]+_(\w*_*(?:init|ref|typ|max|0))", name
        )[0]
        superscript_str = r"^{" + superscript_re.replace("_", "\,") + r"}"
        name = name.replace(superscript_re, superscript_str).replace("_", "")

    # Subscripts with comma separated
    # e.g. a_R_p --> a_{R\,p}
    if name.startswith("a_"):
        subscript_re = re.findall(r"^a_+(\w+)", name)[0]
        subscript_str = r"{" + subscript_re.replace("_", "\,") + r"}"
        name = name.replace(subscript_re, subscript_str)

    # Dim and Dimensional with comma separated
    # e.g. j0_n_ref_dimensional --> \hat{j0}^{n\,ref}
    if name.endswith("dim") or name.endswith("dimensional"):
        dim_re1, dim_re2 = re.findall(
            r"([\da-zA-Z]+)_?(.*?)_?(?:dim|dimensional)", name
        )[0]

        if "^" in name:
            name = r"\hat{" + dim_re1 + r"}" + dim_re2.replace("_", "\,")
        else:
            name = r"\hat{" + dim_re1 + r"}_{" + dim_re2.replace("_", "\,") + r"}"

    # Bar with comma separated
    # e.g. c_s_n_xav --> \bar{c}_{s\,n}
    if any(elem in name for elem in ["_av", "_xav"]):
        bar_re1, bar_re2 = re.findall(r"^([a-zA-Z]+)_*(\w*?)_(?:av|xav)", name)[0]
        name = r"\bar{" + bar_re1 + r"}_{" + bar_re2.replace("_", "\,") + r"}"

    # Greek letters
    # e.g. delta_phi_n --> \delta_\phi_n
    greek_re = f"({'|'.join(GREEK_LETTERS)})"
    name = re.sub(greek_re, r"\\\1", name, flags=re.IGNORECASE)

    return name
