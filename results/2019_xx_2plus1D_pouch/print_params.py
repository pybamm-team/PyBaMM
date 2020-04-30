import pybamm
import os

os.chdir(pybamm.root_dir())

parameters = pybamm.standard_parameters_lithium_ion
parameter_values = pybamm.lithium_ion.DFN().default_parameter_values
val = 1e6 / 4.758
parameter_values.update(
    {
        "Negative current collector conductivity [S.m-1]": val,
        "Positive current collector conductivity [S.m-1]": val,
    }
)
output_file = "results/2019_xx_2plus1D_pouch/parameters.txt"
pybamm.print_parameters(parameters, parameter_values, output_file)
