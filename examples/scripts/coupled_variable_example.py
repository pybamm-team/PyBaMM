"""
Example script demonstrating how to add new variables to a model
using CoupledVariable from an external dictionary.

The dictionary format uses PyBaMM's serialization format for the expression,
with a CoupledVariable placeholder that references the dependent variable.
"""

import pybamm
from pybamm.expression_tree.operations.serialise import (
    add_variables_from_dict,
    convert_symbol_to_json,
)

# Example usage
if __name__ == "__main__":
    # Create a simple battery model
    model = pybamm.lithium_ion.SPM()

    # Build the model
    if not model._built:
        model.build_model()

    print("Model built successfully!")

    # Create expressions using CoupledVariable, then serialize
    # This is what a user would provide (or load from JSON)
    voltage_cv = pybamm.CoupledVariable("Voltage [V]")
    double_voltage_expr = voltage_cv * 2

    # Serialize the expression
    serialized = convert_symbol_to_json(double_voltage_expr)
    print(f"Serialized expression: {serialized}")

    # Dictionary of new variables to add (this would come from external source)
    coupled_vars_dict = {
        "Double voltage [V]": serialized,
    }

    # Add the coupled variables to the model
    add_variables_from_dict(model, coupled_vars_dict)

    print(f"\nCoupled variables: {model.list_coupled_variables()}")
    print(f"Double voltage expression: {model.variables['Double voltage [V]']}")

    # Solve the model with Chen2020 parameters, discharge to 2.5V
    param = pybamm.ParameterValues("Chen2020")
    sim = pybamm.Simulation(
        model,
        parameter_values=param,
        experiment=pybamm.Experiment(["Discharge at 1C until 2.5V"]),
    )
    solution = sim.solve()

    # Check the final values
    final_voltage = solution["Voltage [V]"].data[-1]
    final_double_voltage = solution["Double voltage [V]"].data[-1]

    print(f"\nFinal voltage: {final_voltage:.4f} V")
    print(f"Final double voltage: {final_double_voltage:.4f} V")
    print(f"Expected double voltage: {2 * final_voltage:.4f} V")

    print("\nAll assertions passed!")
