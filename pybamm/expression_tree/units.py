#
# Units class
#
import re
import pybamm
from functools import cached_property
import numpy as np

_KNOWN_BASE_UNITS = [
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


class _UnitsDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__({}, **kwargs)
        self.update(*args)

    def update(self, items):
        processed_items = {}
        for k, v in items.items():
            if isinstance(v, float) and abs(np.round(v) - v) < 1e-12:
                v = int(np.round(v))
            if v != 0:
                processed_items[k] = v
        super().update(processed_items)

    def __setitem__(self, k, v):
        super().update({k: v})

    def __missing__(self, key):
        return 0


_KNOWN_COMPOSED_UNITS = {
    **{k: _UnitsDict({k: 1}) for k in _KNOWN_BASE_UNITS},
    # override composed units
    "J": _UnitsDict({"V": 1, "A": 1, "s": 1}),
    "W": _UnitsDict({"V": 1, "A": 1}),
    "S": _UnitsDict({"V": -1, "A": 1}),
    "Ohm": _UnitsDict({"V": 1, "A": -1}),
    "Pa": _UnitsDict({"V": 1, "A": 1, "s": 1, "m": -3}),
    # other standard composed units
    "m.s-1": _UnitsDict({"m": 1, "s": -1}),
    "m2.s-1": _UnitsDict({"m": 2, "s": -1}),
    "m.s-2": _UnitsDict({"m": 1, "s": -2}),
    "mol.m-3": _UnitsDict({"mol": 1, "m": -3}),
    "m3.mol-1": _UnitsDict({"m": 3, "mol": -1}),
    "A.m-2": _UnitsDict({"A": 1, "m": -2}),
    "S.m-1": _UnitsDict({"V": -1, "A": 1, "m": -1}),
    "V.K-1": _UnitsDict({"V": 1, "K": -1}),
    "J.mol-1": _UnitsDict({"V": 1, "A": 1, "s": 1, "mol": -1}),
}


class Units:
    """A node containing information about units. This is usually an attribute of a node
    in the expression tree, and is automatically created by the expression tree

    Parameters
    ----------
    units : str
        units of the node
    check_units : bool, optional
        Whether to check that the units are valid. Default is True. If the units are
        one of a known number of standard cases, then they are not checked.
    """

    def __init__(self, units, check_units=True):
        # encode empty units
        if units is None or units == {}:
            self._units_str = "-"
            self.units_dict = _UnitsDict({})
            check_units = False
        elif isinstance(units, str):
            self._units_str = units
            try:
                self.units_dict = _KNOWN_COMPOSED_UNITS[units]
                check_units = False
            except KeyError:
                self.units_dict = self._str_to_dict(units)
        else:
            self.units_dict = _UnitsDict(units)
            # _dict_to_str will be called when the units_str attribute is accessed

        # Check all units are recognized
        if check_units:
            for name in self.units_dict.keys():
                if name not in _KNOWN_BASE_UNITS:
                    pybamm.units_error(
                        f"Unit '{name}' not recognized.\n"
                        f"KNOWN_BASE_UNITS: {_KNOWN_BASE_UNITS}"
                    )

    def __str__(self):
        return self.units_str

    @cached_property
    def units_str(self):
        try:
            return self._units_str
        except AttributeError:
            return self._dict_to_str(self.units_dict)

    @cached_property
    def is_dimensionless(self):
        return self.units_dict == {}

    @cached_property
    def units_tuple(self):
        return tuple(self.units_dict.items())

    def __repr__(self):
        return "Units('{!s}')".format(self)

    def _str_to_dict(self, units_str):
        "Convert string representation of units to a dictionary"
        # Find all the units and add to the dictionary
        units = units_str.split(".")
        units_dict = {}
        amount = None
        for unit in units:
            # Look for negative
            if "-" in unit:
                # Split by the location of the negative
                name, amount = unit.split("-")
                amount = "-" + amount
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

        # Update units dictionary for special parameters
        units_dict = self._reformat_dict(units_dict)

        return units_dict

    def _dict_to_str(self, units_dict):
        "Convert a dictionary of units to a string representation"
        # O(n2) but the dictionary is small so it doesn't matter
        # First loop through the positives
        units_str = ""

        sorted_items = sorted(units_dict.items())

        for name, amount in sorted_items:
            if amount == 1:
                # Don't record the amount if there's only 1, e.g. 'm.s-1' instead of
                # 'm1.s-1'
                units_str += name + "."
            elif amount > 0:
                units_str += name + str(amount) + "."
        # Then loop through the negatives so that they are at the end of the dict
        for name, amount in sorted_items:
            if amount < 0:
                # The '-' is already in the amount
                units_str += name + str(amount) + "."

        # Remove the final '.'
        units_str = units_str[:-1]

        return units_str

    def _reformat_dict(self, units_dict):
        "Reformat units dictionary"
        if any(x in units_dict for x in ["J", "C", "W", "S", "Ohm", "Pa"]):
            units_dict = _UnitsDict(units_dict)
        else:
            return units_dict

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
            return self
        else:
            pybamm.units_error(
                "Cannot add different units {!s} and {!s}.".format(self, other)
            )

    def __sub__(self, other):
        # subtracting units is the same as adding
        return self + other

    def __mul__(self, other):
        mul_units = {}
        for k in set(self.units_dict) | set(other.units_dict):
            # Add common elements and keep distinct elements
            sum_units_k = self.units_dict.get(k, 0) + other.units_dict.get(k, 0)
            # only add if not equal to zero
            if sum_units_k != 0:
                mul_units[k] = sum_units_k

        return Units(mul_units, check_units=False)

    def __rmul__(self, other):
        """
        Allows setting units via multiplication, e.g. 2 * Units("m") returns
        Scalar(2, "m")
        """
        return other * pybamm.Scalar(1, units=self)

    def __truediv__(self, other):
        div_units = {}
        for k in set(self.units_dict) | set(other.units_dict):
            # Substract common elements and keep distinct elements
            diff_units_k = self.units_dict.get(k, 0) - other.units_dict.get(k, 0)
            # only add if not equal to zero
            if diff_units_k != 0:
                div_units[k] = diff_units_k
        return Units(div_units, check_units=False)

    def __rtruediv__(self, other):
        """
        Allows setting units via division
        """
        return other / pybamm.Scalar(1, units=self)

    def __pow__(self, power):
        try:
            # Multiply units by the power
            pow_units = {k: power.value * v for k, v in self.units_dict.items()}
        except AttributeError:
            try:
                power = pybamm.Scalar(power)
            except TypeError:
                raise units_error(
                    "power must be a pybamm.Scalar or number if object has units."
                )
            return self**power
        return Units(pow_units, check_units=False)

    def __eq__(self, other):
        "Two units objects are defined to be equal if their unit_dicts are equal"
        try:
            return self.units_dict == other.units_dict
        except AttributeError:
            return self.units_dict == Units(other).units_dict


def units_error(message):
    if pybamm.settings.check_units:
        raise pybamm.UnitsError(
            message
            + " Set `pybamm.settings.check_units=False` to disable unit checking."
        )
