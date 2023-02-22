#
# Example for printing the parameters of a parameter set
#
import pybamm

parameters = pybamm.LithiumIonParameters()
parameter_values = pybamm.ParameterValues("Marquis2019")
output_file = "lithium_ion_parameters.txt"
parameter_values.print_parameters(parameters, output_file)
