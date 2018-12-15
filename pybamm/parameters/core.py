#
# Dimensional and dimensionless parameter values, and scales
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import pandas as pd
import os


class BaseParameterValues(object):
    """
    The parameter values for a simulation.
    """

    def __init__(self, current=None, base_parameters={}, optional_parameters={}):
        # Input current
        # Set default
        if current is None:
            current = {"Ibar": 1, "type": "constant"}
        self.current = current

        # Default parameters
        # If base_parameters is a filename, load from that filename
        if isinstance(base_parameters, str):
            base_parameters = self.read_parameters_csv(base_parameters)
        self.raw = base_parameters

        # Optional parameters
        # If optional_parameters is a filename, load from that filename
        if isinstance(optional_parameters, str):
            optional_parameters = self.read_parameters_csv(optional_parameters)

        # Overwrite raw parameters with optional values where given
        # This avoids having to read a base parameter file each time, for example when
        # doing parameter studies
        self.raw = optional_parameters

    def read_parameters_csv(filename):
        """Reads parameters from csv file into dict.

        Parameters
        ----------
        filename : string
            The name of the csv file containing the parameters.
            Must be a file (or path/to/file) in `input/parameters/`

        Returns
        -------
        dict
            {name: value} pairs for the parameters.

        """
        # Hack to access input/parameters from any working directory
        filename = os.path.join(pybamm.ABSOLUTE_PATH, "input", "parameters", filename)

        #
        df = pd.read_csv(filename, comment="#", skip_blank_lines=True)
        # Drop rows that are all NaN (seems to not work with skip_blank_lines)
        df.dropna(how="all", inplace=True)
        return {k: v for (k, v) in zip(df.Name, df.Value)}

    @property
    def raw(self):
        return self._raw

    @raw.setter
    def raw(self, new_parameters):
        """
        Update raw parameter values with dict.

        Parameters
        ----------
        new_parameters : dict
            dict of optional parameters to overwrite some of the default parameters

        """
        # Update _raw dict
        self._raw.update(new_parameters)

    def process_model(self, model):
        """Assign parameter values to a model.
        Currently inplace, could be changed to return a new model.

        Parameters
        ----------
        model : :class:`pybamm.models.core.BaseModel` (or subclass) instance
            Model to assign parameter values for

        """
        for variable, equation in model.rhs.items():
            model.rhs[variable] = self.process_symbol(equation)

        for variable, equation in model.initial_conditions.items():
            model.initial_conditions[variable] = self.process_symbol(equation)

        for variable, equation in model.boundary_conditions.items():
            model.boundary_conditions[variable] = self.process_symbol(equation)

    def process_symbol(self, symbol):
        """Walk through the symbol and replace any Parameter with a Value.

        Parameters
        ----------
        symbol : :class:`pybamm.expression_tree.symbol.Symbol` (or subclass) instance
            Symbol or Expression tree to set parameters for

        Returns
        -------
        symbol : :class:`pybamm.expression_tree.symbol.Symbol` (or subclass) instance
            Symbol with Parameter instances replaced by Value

        """
        if isinstance(symbol, pybamm.Parameter):
            value = self.get_parameter_value(symbol)
            return pybamm.Scalar(symbol.name, value)

        elif isinstance(symbol, pybamm.BinaryOperator):
            new_symbol = copy.copy(symbol)
            new_symbol.left = self.process_symbol(symbol.left)
            new_symbol.right = self.process_symbol(symbol.right)
            return new_symbol

        elif isinstance(symbol, pybamm.UnaryOperator):
            new_symbol = copy.copy(symbol)
            new_symbol.child = self.process_symbol(symbol.child)
            return new_symbol

        elif isinstance(symbol, pybamm.Variable):
            return copy.copy(symbol)

        else:
            raise TypeError(
                """{} (of symbol {!r}) is not a recognised type for setting parameters""".format(
                    type(symbol), symbol
                )
            )

    def get_parameter_value(self, parameter):
        """
        Get the value of a Parameter.
        Different ParameterValues classes may implement this differently.

        Parameters
        ----------
        parameter : :class:`pybamm.expression_tree.parameter.Parameter` instance
            The parameter whose value to obtain

        Returns
        -------
        value : int or float
            The value of the parameter
        """
        return self.raw[parameter.name]
