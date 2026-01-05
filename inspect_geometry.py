import pybamm

def get_geo_params(param_set_name):
    try:
        params = pybamm.ParameterValues(param_set_name)
        
        # Extract key geometric params for Anode
        eps_s = params["Negative electrode active material volume fraction"]
        try:
            r_p = params["Negative particle radius [m]"]
        except KeyError:
             # Some sets might use a function or different naming, but standard sets usually have this.
             # Chen uses distribution? 
             # Let's try "Negative electrode particle radius [m]" or check if it's a function
             r_p = params.get("Negative particle radius [m]", "N/A")

        L_neg = params["Negative electrode thickness [m]"]
        
        # Specific Surface Area (a) [m2/m3]
        if isinstance(r_p, (float, int)):
            a_surf = 3 * eps_s / r_p
        else:
            a_surf = "N/A"
        
        # Additional checks for identical geometry cases
        D_s = params.get("Negative electrode particle diffusivity [m2.s-1]", "N/A")
        # j0 usually requires a function evaluation, might be hard to print directly.
        # Let's check Reference OCP or Initial Concentration
        c_init = params.get("Initial concentration in negative electrode [mol.m-3]", "N/A")
        c_max = params.get("Maximum concentration in negative electrode [mol.m-3]", "N/A")
            
        return {
            "eps_s": eps_s,
            "r_p": r_p,
            "L_neg": L_neg,
            "a_surf": a_surf,
            "c_max": c_max
        }
    except Exception as e:
        return {"error": str(e)}

sets = ["Mohtat2020", "Chen2020", "OKane2022"]
results = {}

print(f"{'Parameter Set':<15} | {'Vol. Frac':<10} | {'Radius [um]':<12} | {'Thick [um]':<10} | {'C_max':<12}")
print("-" * 75)

for p in sets:
    data = get_geo_params(p)
    if "error" in data:
        print(f"{p:<15} | ERROR: {data['error']}")
    else:
        # Format
        eps = data['eps_s']
        r_p = data['r_p']
        l_neg = data['L_neg']
        c_max = data['c_max']
        
        r_str = f"{r_p*1e6:.2f}" if isinstance(r_p, (float, int)) else str(r_p)
        l_str = f"{l_neg*1e6:.1f}" if isinstance(l_neg, (float, int)) else str(l_neg)
        c_str = f"{c_max:.2e}" if isinstance(c_max, (float, int)) else str(c_max)
        
        print(f"{p:<15} | {eps:<10.3f} | {r_str:<12} | {l_str:<10} | {c_str:<12}")
