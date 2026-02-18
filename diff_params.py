import pybamm


def compare_sets(set1, set2):
    p1 = pybamm.ParameterValues(set1)
    p2 = pybamm.ParameterValues(set2)

    keys1 = set(p1.keys())
    keys2 = set(p2.keys())

    common_keys = keys1.intersection(keys2)

    diffs = []

    # Interested categories
    keywords = ["solvent diffusivity", "exchange-current"]

    print(f"--- Differences between {set1} and {set2} ---")
    print(f"{'Parameter':<60} | {set1:<15} | {set2:<15}")
    print("-" * 100)

    # Check specific params even if keys don't perfectly overlap (sometimes they do)
    specific_keys = [
        "Negative electrode OCP [V]",
        "Negative particle diffusivity [m2.s-1]",
        "Negative electrode exchange-current density [A.m-2]",
    ]

    for k in specific_keys:
        v1 = p1.get(k, "Missing")
        v2 = p2.get(k, "Missing")
        v1_str = str(v1)
        v2_str = str(v2)
        if len(v1_str) > 15:
            v1_str = "Func/Data"
        if len(v2_str) > 15:
            v2_str = "Func/Data"
        print(f"{k:<60} | {v1_str:<15} | {v2_str:<15}")

    print("\n--- General Scan ---")

    for k in sorted(list(common_keys)):
        # Filter for relevant physics
        if not any(kw in k.lower() for kw in keywords):
            continue

        val1 = p1[k]
        val2 = p2[k]

        # Check if values are different
        is_diff = False
        try:
            # Handle scalars
            if isinstance(val1, (float, int)) and isinstance(val2, (float, int)):
                if abs(val1 - val2) > 1e-12:
                    is_diff = True
            # Handle strings/functions (naive check)
            elif str(val1) != str(val2):
                is_diff = True
        except:
            is_diff = True

        if is_diff:
            # Format for printing
            v1_str = str(val1)
            v2_str = str(val2)
            if len(v1_str) > 15:
                v1_str = "Func/Data"
            if len(v2_str) > 15:
                v2_str = "Func/Data"

            print(f"{k:<60} | {v1_str:<15} | {v2_str:<15}")


compare_sets("Chen2020", "OKane2022")
