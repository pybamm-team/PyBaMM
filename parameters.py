import pybamm
import subprocess

all_parameter_sets = [x for x in dir(pybamm.parameter_sets) if not x.startswith("__")]

for name in all_parameter_sets:
    if name == "Sulzer2019":
        relpath = "pybamm/input/parameters/lead_acid/"
    else:
        relpath = "pybamm/input/parameters/lithium_ion/"
    parameter_values = pybamm.ParameterValues(getattr(pybamm.parameter_sets, name))
    parameter_values.export_python_script(name, path=relpath)

    # test that loading the parameter set works
    pybamm.ParameterValues(name)

    print(f"{name}: ok")

subprocess.run(["black", "."])
