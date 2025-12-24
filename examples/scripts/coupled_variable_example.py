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

        # Add the new variable to the model
        model.variables[new_var_name] = new_var_expr

        # Find all CoupledVariables in the expression and set them
        coupled_vars = new_var_expr.pre_order()
        for node in coupled_vars:
            if isinstance(node, pybamm.CoupledVariable):
                depends_on_name = node.name
                if depends_on_name in model.variables:
                    actual_var = model.variables[depends_on_name]
                    node.set_coupled_variable(new_var_expr, actual_var)
                    model.coupled_variables[depends_on_name] = node


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
    print(f"Double voltage: {model.variables['Double voltage [V]']}")
