#
# Units class
#
import re
import pybamm

KNOWN_UNITS = ["m", "kg", "s", "A", "K", "mol", "cd"]


class Units:
    """A node containing information about units. This is usually an attribute of a node
    in the expression tree, and is automatically created by the expression tree

    Parameters
    ----------
    units : str
        units of the node
    """

    def __init__(self, units):
        if isinstance(units, str):
            self.units_str = units
            self.units = self.str_to_dict(units)
        else:
            self.units_str = self.dict_to_str(units)
            self.units = units

        # Check all units are recognized
        for name in self.units.keys():
            if name not in KNOWN_UNITS:
                raise pybamm.UnitsError(
                    "Unit '{}' not recognized".format(name)
                    + "\nKNOWN_UNITS: {}".format(KNOWN_UNITS)
                )

    def __str__(self):
        return self.units_str

    def __repr__(self):
        return "Units({!s})".format(self)

    def str_to_dict(self, units_str):
        "Convert string representation of units to a dictionary"
        # Extract from square brackets
        if units_str[0] != "[" or units_str[-1] != "]":
            raise pybamm.UnitsError(
                "Units should start with '[' and end with ']' (found '{}')".format(
                    units_str
                )
            )
        units_str = units_str[1:-1]
        # Find all the units and add to the dictionary
        units = units_str.split(".")
        units_dict = {}
        for unit in units:
            # Look for negative
            if "-" in unit:
                # Split by the location of the negative
                name = unit[: unit.index("-")]
                amount = unit[unit.index("-") :]
                # amount automatically includes the negative by the way it is extracted
            else:
                # Split by character and number
                match = re.match(r"([a-z]+)([0-9]+)", unit, re.I)
                if match:
                    name, amount = match.groups()
                else:
                    # If no number was found, it must be a '1', e.g. 'm' in 'm.s-1'
                    name = unit
                    amount = 1
            # Add the unit to the dictionary
            units_dict[name] = int(amount)

        return units_dict

    def dict_to_str(self, units_dict):
        "Convert a dictionary of units to a string representation"
        # O(n2) but the dictionary is small so it doesn't matter
        # First loop through the positives
        units_str = ""
        for name, amount in units_dict.items():
            if amount == 1:
                # Don't record the amount if there's only 1, e.g. 'm.s-1' instead of
                # 'm1.s-1'
                units_str += name + "."
            elif amount > 1:
                units_str += name + str(amount) + "."
        # Then loop through the negatives
        for name, amount in units_dict.items():
            if amount < 0:
                # The '-' is already in the amount
                units_str += name + str(amount) + "."

        # Remove the final '.'
        units_str = units_str[:-1]

        return "[" + units_str + "]"

    def __add__(self, other):
        if self.units == other.units:
            return Units(self.units)
        else:
            raise pybamm.UnitsError(
                "Cannot add different units {!s} and {!s}".format(self, other)
            )

    def __sub__(self, other):
        if self.units == other.units:
            return Units(self.units)
        else:
            raise pybamm.UnitsError(
                "Cannot subtract different units {!s} and {!s}".format(self, other)
            )
