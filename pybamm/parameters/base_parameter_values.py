#
# Dimensional and dimensionless parameter values, and scales
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import pandas as pd
import os
import copy


class BaseParameterValues(object):
    """
    The parameter values for a simulation.

    Parameters
    ----------
    base_parameters : dict or string
        The base parameters
        If string, gets passed to read_parameters_csv to read a file.

    optional_parameters : dict or string
        Optional parameters, overwrites base_parameters if there is a conflict
        If string, gets passed to read_parameters_csv to read a file.

    """

    def __init__(self, base_parameters={}, optional_parameters={}):
        # Default parameters
        # If base_parameters is a filename, load from that filename
        if isinstance(base_parameters, str):
            base_parameters = self.read_parameters_csv(base_parameters)
        self.update_raw(base_parameters)

        # Optional parameters
        # If optional_parameters is a filename, load from that filename
        if isinstance(optional_parameters, str):
            optional_parameters = self.read_parameters_csv(optional_parameters)

        # Overwrite raw parameters with optional values where given
        # This avoids having to read a base parameter file each time, for example when
        # doing parameter studies
        self.update_raw(optional_parameters)

    def read_parameters_csv(self, filename):
        """Reads parameters from csv file into dict.

        Parameters
        ----------
        filename : string
            The name of the csv file containing the parameters.

        Returns
        -------
        dict
            {name: value} pairs for the parameters.

        """

        #
        df = pd.read_csv(filename, comment="#", skip_blank_lines=True)
        # Drop rows that are all NaN (seems to not work with skip_blank_lines)
        df.dropna(how="all", inplace=True)
        return {k: v for (k, v) in zip(df.Name, df.Value)}

    @property
    def raw(self):
        return self._raw

    def update_raw(self, new_parameters):
        """
        Update raw parameter values with dict.

        Parameters
        ----------
        new_parameters : dict
            dict of optional parameters to overwrite some of the default parameters

        """
        if not hasattr(self, "_raw"):
            # Create raw dict if it doesn't exist
            self._raw = new_parameters
        else:
            # Update _raw dict if it already exists
            self._raw.update(new_parameters)

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
            return pybamm.Scalar(value)

        elif isinstance(symbol, pybamm.BinaryOperator):
            new_left = self.process_symbol(symbol.children[0])
            new_right = self.process_symbol(symbol.children[1])
            return symbol.__class__(new_left, new_right)

        elif isinstance(symbol, pybamm.UnaryOperator):
            new_child = self.process_symbol(symbol.children[0])
            return symbol.__class__(new_child)

        else:
            return copy.copy(symbol)
