from Ecker2015 import get_parameter_values

def create_nested_parameter_dict(base_dict):
    """
    This function takes a flat dictionary and returns a new dictionary with nested 
    structures categorized into components like electrode, separator, cell, experiment, 
    and a user-defined section for uncategorized parameters.

    Args:
        base_dict (dict): The flat dictionary containing parameters.

    Returns:
        dict: A new dictionary with nested structures categorized by component.
    """

    # Define the categories and their associated components
    categories = {
        "experiment": [
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
        ],
        "electrode": [
            "negative_electrode",
            "positive_electrode",
            "lithium_plating",
            "sei",
        ],
        "separator": ["separator"],
        "cell": ["cell", "negative current collector", "positive current collector"],
    }

    # Initialize the nested dictionary
    nested_dict = {category: {} for category in categories}
    nested_dict["user-defined"] = {}  # For parameters not falling into predefined categories

    # Process each key-value pair in the base dictionary
    for key, value in base_dict.items():
        added = False
        # Loop through the categories to find where the key fits
        for category, params in categories.items():
            # Check if key belongs to the "experiment" parameters
            if category == "experiment" and key in params:
                nested_dict[category][key] = value
                added = True
                break
            # Check if key starts with any of the category-specific keywords
            elif any(key.lower().startswith(param.replace("_", " ")) for param in params):
                nested_dict[category][key] = value
                added = True
                break
        
        # If the key doesn't fit into any predefined category, add it to "user-defined"
        if not added:
            nested_dict["user-defined"][key] = value
    
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
