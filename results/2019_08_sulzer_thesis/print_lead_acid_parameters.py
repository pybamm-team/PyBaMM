#
# Print parameters for lead-acid models
#
import pybamm

parameters = pybamm.standard_parameters_lead_acid
parameter_values = pybamm.LeadAcidBaseModel().default_parameter_values
output_file = "results/sulzer_thesis/parameters.txt"

pybamm.print_parameters(parameters, parameter_values, output_file)
