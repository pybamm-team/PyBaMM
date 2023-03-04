#
# Prettify print_name
#
import re

PRINT_NAME_OVERRIDES = {
    "current_with_time": "I",
    "eps_c_e": r"\epsilon{c_e}",
    "one_plus_dlnf_dlnc": r"1+\frac{dlnf}{dlnc}",
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

    # Return print_name if name exists in the dictionary
    if name in PRINT_NAME_OVERRIDES:
        return PRINT_NAME_OVERRIDES[name]

    # Superscripts with comma separated (U_ref_n --> U_{n}^{ref})
    sup_re1 = re.search(r"^[\da-zA-Z]+_?((?:init|ref|typ|max|0))_?(.*)", name)
    if sup_re1:
        sup_str = (
            r"{"
            + sup_re1.group(2).replace("_", "\,")
            + r"}^{"
            + sup_re1.group(1)
            + r"}"
        )
        sup_var = sup_re1.group(1) + "_" + sup_re1.group(2)
        name = name.replace(sup_var, sup_str)

    # Superscripts with comma separated (U_n_ref --> U_{n}^{ref})
    sup_re2 = re.search(r"^[\da-zA-Z]+_?(.*?)_?((?:init|ref|typ|max|0))", name)
    if sup_re2:
        sup_str = (
            r"{"
            + sup_re2.group(1).replace("_", "\,")
            + r"}^{"
            + sup_re2.group(2)
            + r"}"
        )
        sup_var = sup_re2.group(1) + "_" + sup_re2.group(2)
        name = name.replace(sup_var, sup_str)

    # Subscripts with comma separated (a_R_p --> a_{R\,p})
    sub_re = re.search(r"^a_+(\w+)", name)
    if sub_re:
        sub_str = r"{" + sub_re.group(1).replace("_", "\,") + r"}"
        name = name.replace(sub_re.group(1), sub_str)

    # Bar with comma separated (c_s_n_xav --> \bar{c}_{s\,n})
    bar_re = re.search(r"^([a-zA-Z]+)_*(\w*?)_(?:av|xav)", name)
    if bar_re:
        name = (
            r"\bar{"
            + bar_re.group(1)
            + r"}_{"
            + bar_re.group(2).replace("_", "\,")
            + r"}"
        )

    # Replace eps with epsilon (eps_n --> epsilon_n)
    name = re.sub(r"(eps)(?![0-9a-zA-Z])", "epsilon", name)

    # Greek letters (delta --> \delta)
    greek_re = r"(?<!\\)(" + "|".join(GREEK_LETTERS) + r")(?![0-9a-zA-Z])"
    name = re.sub(greek_re, r"\\\1", name, flags=re.IGNORECASE)

    return name
