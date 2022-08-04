#
# Loop through all combinations of models and parameters and print whether that
# model can be parameterized by those parameters
#
import pybamm

all_options = pybamm.BatteryModelOptions({}).possible_options
all_parameter_sets = [
    x
    for x in dir(pybamm.parameter_sets)
    if not x.startswith("__") and x not in ["Sulzer2019"]
]

for option_name, option_list in all_options.items():
    for parameter_set in all_parameter_sets:
        parameter_values = pybamm.ParameterValues(parameter_set)
        for option_value in option_list[
            1:
        ]:  # skip the first one since that's the default
            options = {option_name: option_value}
            try:
                model = pybamm.lithium_ion.SPM(options.copy())
            except pybamm.OptionError as e:
                print(f"Cannot create model with {options}. (OptionError: {str(e)})")
            except pybamm.ModelError as e:
                # todo: properly resolve the cases that raise these errors
                print(f"Cannot create model with {options}. (ModelError: {str(e)})")
            except AttributeError as e:
                # todo: properly resolve the cases that raise these errors
                print(f"Cannot create model with {options}. (AttributeError: {str(e)})")
            else:
                output = f"{options} with {parameter_set} parameters: "
                try:
                    parameter_values.process_model(model)
                    output += "success"
                except KeyError:
                    output += "failure"
                print(output)
