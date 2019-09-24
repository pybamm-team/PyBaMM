#
# Print parameters for lead-acid models
#
import pybamm

parameters = pybamm.standard_parameters_lead_acid
parameter_values = pybamm.lead_acid.BaseModel().default_parameter_values
output_file = "results/2019_09_sulzer_thesis/parameters.txt"

pybamm.print_parameters(parameters, parameter_values, output_file)
