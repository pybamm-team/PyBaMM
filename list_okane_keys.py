import pybamm
p = pybamm.ParameterValues("OKane2022")
for k in sorted(p.keys()):
    if "diffusivity" in k.lower() or "sei" in k.lower():
        print(f"{k}: {p[k]}")
