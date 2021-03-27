#
# Example for printing the (dimensional and dimensionless) parameters of a parameter set
#
import pybamm

parameters = pybamm.LithiumIonParameters()
parameter_values = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Yang2017)
output_file = "lithium_ion_parameters.txt"
parameter_values.print_parameters(parameters, output_file)
