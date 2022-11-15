#
# Units class
#
import re
import pybamm
from collections import defaultdict
import numbers

KNOWN_UNITS = [
    "m",
    "kg",
    "s",
    "A",
    "K",
    "mol",
    "cd",
    "h",
    "V",
    "eV",
    "J",
    "W",
    "S",
    "F",
    "C",
    "Ohm",
    "Pa",
]


class Units:
    """A node containing information about units. This is usually an attribute of a node
    in the expression tree, and is automatically created by the expression tree

    Parameters
    ----------
    units : str
        units of the node
    """

    def __init__(self, units):
        # encode empty units
        if units is None or units == {}:
            self.units_str = "-"
            self.units_dict = defaultdict(int)
        elif isinstance(units, str):
            self.units_str = units
            self.units_dict = self.str_to_dict(units)
        else:
            units = defaultdict(int, units)
            self.units_str = self.dict_to_str(units)
            self.units_dict = units

        # Check all units are recognized
        for name in self.units_dict.keys():
            if name not in KNOWN_UNITS:
                pybamm.units_error(
                    "Unit '{}' not recognized.".format(name)
                    + "\nKNOWN_UNITS: {}".format(KNOWN_UNITS)
                )

    def __str__(self):
        return self.units_str

    def __repr__(self):
        return "Units({!s})".format(self)

    def str_to_dict(self, units_str):
        "Convert string representation of units to a dictionary"
        # Find all the units and add to the dictionary
        units = units_str.split(".")
        units_dict = defaultdict(int)
        amount = None
        for i, unit in enumerate(units):
            # Account for cases like [m1.5.s-1] by looking for points after the decimal
            if unit.isdigit():
                # There can't be a digit in the first entry
                if i == 0:
                    pybamm.units_error(
                        "Units cannot start with a digit but found '{}'.".format(
                            units_str
                        )
                    )
                else:
                    # Add the digit to the previous entry
                    # Don't change the name, just add to the amount
                    amount += "." + unit
            # Look for negative
            elif "-" in unit:
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
            float_amount = float(amount)
            if abs(round(float_amount) - float_amount) < 1e-12:
                units_dict[name] = int(float_amount)
            else:
                units_dict[name] = float_amount

        # Update units dictionary for special parameters
        units_dict = self.reformat_dict(units_dict)

        return units_dict

    def dict_to_str(self, units_dict):
        "Convert a dictionary of units to a string representation"
        # O(n2) but the dictionary is small so it doesn't matter
        # First loop through the positives
        units_str = ""

        # Update units dictionary for special parameters
        units_dict = self.reformat_dict(units_dict)

        for name, amount in sorted(units_dict.items()):
            if amount == 0:
                pybamm.units_error("Zero units should not be in dictionary.")
            elif amount == 1:
                # Don't record the amount if there's only 1, e.g. 'm.s-1' instead of
                # 'm1.s-1'
                units_str += name + "."
            elif amount > 0:
                units_str += name + str(amount) + "."
        # Then loop through the negatives
        for name, amount in sorted(units_dict.items()):
            if amount < 0:
                # The '-' is already in the amount
                units_str += name + str(amount) + "."

        # Remove the final '.'
        units_str = units_str[:-1]

        return units_str

    def reformat_dict(self, units_dict):
        "Reformat units dictionary"
        if "J" in units_dict:
            num_J = units_dict.pop("J")
            units_dict["V"] += num_J
            units_dict["A"] += num_J
            units_dict["s"] += num_J
        if "C" in units_dict:
            num_C = units_dict.pop("C")
            units_dict["A"] += num_C
            units_dict["s"] += num_C
        if "W" in units_dict:
            num_W = units_dict.pop("W")
            units_dict["V"] += num_W
            units_dict["A"] += num_W
        if "S" in units_dict:
            num_S = units_dict.pop("S")
            units_dict["A"] += num_S
            units_dict["V"] -= num_S
        if "Ohm" in units_dict:
            num_Ohm = units_dict.pop("Ohm")
            units_dict["V"] += num_Ohm
            units_dict["A"] -= num_Ohm
        if "Pa" in units_dict:
            num_Pa = units_dict.pop("Pa")
            units_dict["V"] += num_Pa
            units_dict["A"] += num_Pa
            units_dict["s"] += num_Pa
            units_dict["m"] -= 3 * num_Pa
        return units_dict

    def __add__(self, other):
        if self.units_dict == other.units_dict:
            return Units(self.units_dict)
        else:
            pybamm.units_error(
                "Cannot add different units {!s} and {!s}.".format(self, other)
            )

    def __sub__(self, other):
        # subtracting units is the same as adding
        return self + other

    def __mul__(self, other):
        # Add common elements and keep distinct elements
        # remove from units dict if equal to zero
        mul_units = {
            k: self.units_dict.get(k, 0) + other.units_dict.get(k, 0)
            for k in set(self.units_dict) | set(other.units_dict)
            if self.units_dict.get(k, 0) + other.units_dict.get(k, 0) != 0
        }
        return Units(mul_units)

    def __rmul__(self, other):
        """
        Allows setting units via multiplication, e.g. 2 * Units("m") returns
        Scalar(2, "m")
        """
        if isinstance(other, numbers.Number):
            return pybamm.Scalar(other, units=self)
        elif isinstance(other, pybamm.Symbol):
            return other * pybamm.Scalar(1, units=self)
        else:
            raise TypeError()

    def __truediv__(self, other):
        # Subtract common elements and keep distinct elements
        # remove from units dict if equal to zero
        div_units = {
            k: self.units_dict.get(k, 0) - other.units_dict.get(k, 0)
            for k in set(self.units_dict) | set(other.units_dict)
            if self.units_dict.get(k, 0) - other.units_dict.get(k, 0) != 0
        }
        return Units(div_units)

    def __rtruediv__(self, other):
        """
        Allows setting units via division
        """
        if isinstance(other, pybamm.Symbol):
            return other / pybamm.Scalar(1, units=self)
        else:
            raise TypeError()

    def __pow__(self, power):
        # Multiply units by the power
        # This is different from the other operations in that "power" has to be an
        # integer
        pow_units = {k: power * v for k, v in self.units_dict.items()}
        return Units(pow_units)

    def __eq__(self, other):
        "Two units objects are defined to be equal if their unit_dicts are equal"
        return self.units_dict == other.units_dict


def units_error(message):
    if pybamm.settings.check_units:
        raise pybamm.UnitsError(
            message
            + " Set `pybamm.settings.check_units=False` to turn off unit checking."
        )
