#
# Print parameters (hacky)
#
import pybamm
from collections import defaultdict

param = pybamm.standard_parameters_lead_acid
parameter_values = pybamm.LeadAcidBaseModel().default_parameter_values

ignore = [
    "__name__",
    "__doc__",
    "__package__",
    "__loader__",
    "__spec__",
    "__file__",
    "__cached__",
    "__builtins__",
    "absolute_import",
    "division",
    "print_function",
    "unicode_literals",
    "pybamm",
    "constants",
]

param_eval = defaultdict(list)
for Ityp in [17, 1.7]:
    # Update Crate
    parameter_values.update({"Typical current density": Ityp})
    for name, symbol in param.__dict__.items():
        if not (name in ignore):
            proc_symbol = parameter_values.process_symbol(symbol)
            if not (
                callable(proc_symbol) or isinstance(proc_symbol, pybamm.Concatenation)
            ):
                param_eval[name].append(proc_symbol.evaluate(t=0))

with open("results/out/parameters.txt", "w") as file:
    for name, values in sorted(param_eval.items()):
        if 0.001 < abs(values[0]) < 1000:
            if abs(values[0] - values[1]) < 1e-13:
                file.write("{:34}: {:10.4g}\n".format(name, values[0]))
            elif abs(values[0] - 10 * values[1]) < 1e-13:
                file.write("{:34}: {:10.4g} * C_rate\n".format(name, values[0]))
            elif abs(10 * values[0] - values[1]) < 1e-13:
                file.write("{:34}: {:10.4g} / C_rate\n".format(name, values[0]))
        else:
            if abs(values[0] - values[1]) < 1e-13:
                file.write("{:34}: {:10.3E}\n".format(name, values[0]))
            elif abs(values[0] - 10 * values[1]) < 1e-13:
                file.write("{:34}: {:10.3E} * C_rate\n".format(name, values[0]))
            elif abs(10 * values[0] - values[1]) < 1e-13:
                file.write("{:34}: {:10.3E} / C_rate\n".format(name, values[0]))
