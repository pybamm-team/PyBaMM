import pybamm


def get_sei_params(param_set_name):
    params = pybamm.ParameterValues(param_set_name)

    # List of keywords to search for
    keywords = [
        "SEI",
        "Initial inner SEI thickness [m]",
        "Initial outer SEI thickness [m]",
        "Inner SEI partial molar volume [m3.mol-1]",
        "Outer SEI partial molar volume [m3.mol-1]",
        "SEI reaction exchange current density [A.m-2]",
        "SEI kinetic rate constant [m.s-1]",
        "SEI resistivity [Ohm.m]",
        "SEI open-circuit potential [V]",
        "Negative electrode active material volume fraction",
        "Positive electrode active material volume fraction",
    ]

    data = {}
    for key in params.keys():
        for kw in keywords:
            if kw.lower() in key.lower():
                data[key] = params[key]
                break

    # Try to find chemistry info loosely
    try:
        # Some versions expose it, others don't.
        # Or check for 'chemistry' in the metadata if available, but params is a dict-like
        pass
    except:
        pass

    return data, params


print("Loading Mohtat2020...")
mohtat_data, mohtat_params = get_sei_params("Mohtat2020")

print("Loading Chen2020...")
chen_data, chen_params = get_sei_params("Chen2020")

print("Loading OKane2022...")
okane_data, okane_params = get_sei_params("OKane2022")

# Compare
all_keys = sorted(
    list(set(mohtat_data.keys()) | set(chen_data.keys()) | set(okane_data.keys()))
)

print(
    f"\n{'Parameter':<60} | {'Mohtat2020':<15} | {'Chen2020':<15} | {'OKane2022':<15}"
)
print("-" * 135)

for key in all_keys:
    val_m = mohtat_data.get(key, "N/A")
    val_c = chen_data.get(key, "N/A")
    val_o = okane_data.get(key, "N/A")

    # Simple check for equality to highlight diffs or N/As
    s_m, s_c, s_o = str(val_m), str(val_c), str(val_o)
    if not (s_m == s_c == s_o):
        print(f"{key:<60} | {s_m:<15} | {s_c:<15} | {s_o:<15}")
    else:
        # Uncomment to show identicals too
        # print(f"{key:<60} | {str(val_m):<25} | {str(val_c):<25}")
        pass
