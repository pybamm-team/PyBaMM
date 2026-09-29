#
# Loop through all combinations of models and parameters and print whether that
# model can be parameterized by those parameters
#
import warnings

import pybamm
from pybamm.models.full_battery_models import base_battery_model

all_options = pybamm.BatteryModelOptions({}).possible_options
all_parameter_sets = [
    k for k, v in pybamm.parameter_sets.items() if v["chemistry"] == "lithium_ion"
]


def make_model(options):
    previous_forbid = base_battery_model._FORBID_LEGACY_OPTION_DEFAULTS
    base_battery_model._FORBID_LEGACY_OPTION_DEFAULTS = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pybamm.OptionDefaultDeprecationWarning)
            return pybamm.lithium_ion.SPM(options)
    finally:
        base_battery_model._FORBID_LEGACY_OPTION_DEFAULTS = previous_forbid


for option_name, option_list in all_options.items():
    for parameter_set in all_parameter_sets:
        parameter_values = pybamm.ParameterValues(parameter_set)
        for option_value in option_list[
            1:
        ]:  # skip the first one since that's the default
            options = {option_name: option_value}
            try:
                model = make_model(options.copy())
            except pybamm.OptionError as e:
                print(f"Cannot create model with {options}. (OptionError: {e!s})")
            except pybamm.ModelError as e:
                # todo: properly resolve the cases that raise these errors
                print(f"Cannot create model with {options}. (ModelError: {e!s})")
            except AttributeError as e:
                # todo: properly resolve the cases that raise these errors
                print(f"Cannot create model with {options}. (AttributeError: {e!s})")
            else:
                output = f"{options} with {parameter_set} parameters: "
                try:
                    parameter_values.process_model(model)
                    output += "success"
                except KeyError:
                    output += "failure"
                print(output)
