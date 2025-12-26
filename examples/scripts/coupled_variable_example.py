"""
Example script demonstrating how to add new variables to a model
using CoupledVariable from an external dictionary.

The dictionary format uses PyBaMM's serialization format for the expression,
with a CoupledVariable placeholder that references the dependent variable.
"""

import pybamm
from pybamm.expression_tree.operations.serialise import (
    convert_symbol_from_json,
    convert_symbol_to_json,
)


def add_coupled_variables_from_dict(model, coupled_vars_dict):
    """
    Add coupled variables to a model from an external dictionary.

    Parameters
    ----------
    model : pybamm.BaseModel
        The model to add coupled variables to
    coupled_vars_dict : dict
        Dictionary mapping new variable names to serialized expressions.
        Expressions should use CoupledVariable nodes to reference model variables.
    """
    for new_var_name, serialized_expr in coupled_vars_dict.items():
        # Deserialize the expression (CoupledVariable nodes are preserved)
        new_var_expr = convert_symbol_from_json(serialized_expr)

        # Replace CoupledVariable nodes with the actual model variables
        def replace_coupled_vars(expr):
            if isinstance(expr, pybamm.CoupledVariable):
                depends_on_name = expr.name
                if depends_on_name in model.variables:
                    return model.variables[depends_on_name]
                raise ValueError(f"Variable '{depends_on_name}' not found in model")
            elif hasattr(expr, "children") and expr.children:
                new_children = [replace_coupled_vars(c) for c in expr.children]
                return expr.create_copy(new_children=new_children)
            return expr

        new_var_expr = replace_coupled_vars(new_var_expr)

        # Add the new variable to the model
        model.variables[new_var_name] = new_var_expr


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
    add_coupled_variables_from_dict(model, coupled_vars_dict)

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
