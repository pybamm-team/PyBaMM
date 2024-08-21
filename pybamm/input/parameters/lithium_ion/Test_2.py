from Ecker2015 import get_parameter_values

def create_nested_parameter_dict(base_dict):
    """
    This function takes a flat dictionary and a list of categories
    and returns a new dictionary with nested structures based on the categories.

    Args:
        base_dict (dict): The flat dictionary containing parameters.
        categories (list): A list of categories representing the nesting structure.

    Returns:
        dict: A new dictionary with nested structures based on the categories.
    """

    # Experiment parameters (as a list of keys)
    experiment_params = [
        "Reference temperature [K]",
        "Negative current collector surface heat transfer coefficient [W.m-2.K-1]",
        "Positive current collector surface heat transfer coefficient [W.m-2.K-1]",
        "Negative tab heat transfer coefficient [W.m-2.K-1]",
        "Positive tab heat transfer coefficient [W.m-2.K-1]",
        "Edge heat transfer coefficient [W.m-2.K-1]",
        "Total heat transfer coefficient [W.m-2.K-1]",
        "Ambient temperature [K]",
        "Number of electrodes connected in parallel to make a cell",
        "Number of cells connected in series to make a battery",
        "Lower voltage cut-off [V]",
        "Upper voltage cut-off [V]",
        "Open-circuit voltage at 0% SOC [V]",
        "Open-circuit voltage at 100% SOC [V]",
        "Initial concentration in negative electrode [mol.m-3]",
        "Initial concentration in positive electrode [mol.m-3]",
        "Initial temperature [K]",
    ]
        # Define the category list based on your comment separators
    categories = [
        "lithium_plating",
        "sei",
        "cell",
        "negative_electrode",
        "positive_electrode",
        "negative current collector",
        "positive current collector",
        "separator",
        "electrolyte",
        "experiment",
        "citations",
    ]


    nested_dict = {category: {} for category in categories}

    for key, value in base_dict.items():
        added = False
        for category in categories:
            # Check if the key belongs to the experiment parameters
            if category == "experiment" and key in experiment_params:
                nested_dict["experiment"][key] = value
                added = True
                break
            # Check if the key starts with the category name
            elif key.lower().startswith(category.replace("_", " ")):
                nested_dict[category][key] = value
                added = True
                break
        
        # For keys that don't match any category
        if not added:
            nested_dict.setdefault("misc", {})[key] = value
    
    return nested_dict

# Example usage
if __name__ == "__main__":
    # Your original parameter dictionary
    parameter_dict = get_parameter_values()


    # Call the function to create the nested dictionary
    nested_parameter_dict = create_nested_parameter_dict(parameter_dict)

    # Print the nested dictionary (optional for verification)
    from pprint import pprint
    pprint(nested_parameter_dict)