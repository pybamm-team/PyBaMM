#
# Dimensional and dimensionless parameter values, and scales
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import pandas as pd


class ParameterValues(dict):
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
        self.update(base_parameters)

        # Optional parameters
        # If optional_parameters is a filename, load from that filename
        if isinstance(optional_parameters, str):
            optional_parameters = self.read_parameters_csv(optional_parameters)

        # Overwrite raw parameters with optional values where given
        # This avoids having to read a base parameter file each time, for example when
        # doing parameter studies
        self.update(optional_parameters)

    def read_parameters_csv(self, filename):
        """Reads parameters from csv file into dict.

        Parameters
        ----------
        filename : str
            The name of the csv file containing the parameters.

        Returns
        -------
        dict
            {name: value} pairs for the parameters.

        """
        df = pd.read_csv(filename, comment="#", skip_blank_lines=True)
        # Drop rows that are all NaN (seems to not work with skip_blank_lines)
        df.dropna(how="all", inplace=True)
        return {k: v for (k, v) in zip(df.Name, df.Value)}

    def get_parameter_value(self, parameter):
        """
        Get the value of a Parameter.
        Different ParameterValues classes may implement this differently.

        Parameters
        ----------
        parameter : :class:`pybamm.Parameter` instance
            The parameter whose value to obtain

        Returns
        -------
        value : int or float
            The value of the parameter
        """
        return self[parameter.name]

    def process_model(self, model, processing="process"):
        """Assign parameter values to a model.
        Currently inplace, could be changed to return a new model.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            Model to assign parameter values for
        processing : str, optional
            Flag to indicate how to process model (default 'process')

            * 'process': Calls :meth:`process_symbol()` (walk through the symbol \
            and replace any Parameter with a Value)
            * 'update': Calls :meth:`update_scalars()` for use on already-processed \
            model (update the value of any Scalars in the expression tree.)

        """
        if processing == "process":
            processing_function = self.process_symbol
        elif processing == "update":
            processing_function = self.update_scalars

        for variable, equation in model.rhs.items():
            model.rhs[variable] = processing_function(equation)

        for variable, equation in model.algebraic.items():
            model.algebraic[variable] = processing_function(equation)

        for variable, equation in model.initial_conditions.items():
            model.initial_conditions[variable] = processing_function(equation)

        # Boundary conditions are dictionaries {"left": left bc, "right": right bc}
        new_boundary_conditions = {}
        for variable, bcs in model.boundary_conditions.items():
            processed_variable = processing_function(variable)
            new_boundary_conditions[processed_variable] = {}
            if "left" in bcs.keys():
                new_boundary_conditions[processed_variable][
                    "left"
                ] = processing_function(bcs["left"])
            if "right" in bcs.keys():
                new_boundary_conditions[processed_variable][
                    "right"
                ] = processing_function(bcs["right"])
        model.boundary_conditions = new_boundary_conditions

        for variable, equation in model.variables.items():
            model.variables[variable] = processing_function(equation)

        for idx, equation in enumerate(model.events):
            model.events[idx] = processing_function(equation)

    def update_model(self, model, disc):
        """Process a discretised model.
        Currently inplace, could be changed to return a new model.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            Model to assign parameter values for
        disc : :class:`pybamm.Discretisation`
            The class that was used to discretise

        """
        # process parameter values for the model
        self.process_model(model, processing="update")

        # update discretised quantities using disc
        model.concatenated_rhs = disc._concatenate_in_order(model.rhs)
        model.concatenated_algebraic = disc._concatenate_in_order(model.algebraic)
        model.concatenated_initial_conditions = disc._concatenate_in_order(
            model.initial_conditions
        ).evaluate(0, None)

    def process_geometry(self, geometry):
        """
        Assign parameter values to a geometry.
        Currently inplace, could be changed to return a new model.

        Parameters
        ----------
        geometry : :class:`pybamm.Geometry`
                Geometry specs to assign parameter values to
        """

        for domain in geometry:
            for prim_sec, variables in geometry[domain].items():
                for spatial_variable, spatial_limits in variables.items():
                    for lim, sym in spatial_limits.items():
                        geometry[domain][prim_sec][spatial_variable][
                            lim
                        ] = self.process_symbol(sym).evaluate()

    def process_symbol(self, symbol):
        """Walk through the symbol and replace any Parameter with a Value.

        Parameters
        ----------
        symbol : :class:`pybamm.Symbol`
            Symbol or Expression tree to set parameters for

        Returns
        -------
        symbol : :class:`pybamm.Symbol`
            Symbol with Parameter instances replaced by Value

        """
        if isinstance(symbol, pybamm.Parameter):
            value = self.get_parameter_value(symbol)
            # Scalar inherits name (for updating parameters) and domain (for Broadcast)
            return pybamm.Scalar(value, name=symbol.name, domain=symbol.domain)

        elif isinstance(symbol, pybamm.FunctionParameter):
            new_child = self.process_symbol(symbol.children[0])
            function_name = self.get_parameter_value(symbol)
            function = pybamm.Function(pybamm.load_function(function_name), new_child)
            if symbol.diff_variable is None:
                return function
            else:
                # return differentiated function
                new_diff_variable = self.process_symbol(symbol.children[0])
                return function.diff(new_diff_variable)

        elif isinstance(symbol, pybamm.BinaryOperator):
            left, right = symbol.children
            # process children
            new_left = self.process_symbol(left)
            new_right = self.process_symbol(right)
            # make new symbol, ensure domain remains the same
            new_symbol = symbol.__class__(new_left, new_right)
            new_symbol.domain = symbol.domain
            return new_symbol

        elif isinstance(symbol, pybamm.UnaryOperator):
            new_child = self.process_symbol(symbol.children[0])
            if isinstance(symbol, pybamm.NumpyBroadcast):
                new_symbol = pybamm.NumpyBroadcast(
                    new_child, symbol.domain, symbol.mesh
                )
            elif isinstance(symbol, pybamm.Broadcast):
                new_symbol = pybamm.Broadcast(new_child, symbol.domain)
            elif isinstance(symbol, pybamm.Function):
                new_symbol = pybamm.Function(symbol.func, new_child)
            elif isinstance(symbol, pybamm.Integral):
                new_symbol = symbol.__class__(new_child, symbol.integration_variable)
            elif isinstance(symbol, pybamm.BoundaryValue):
                new_symbol = pybamm.BoundaryValue(new_child, symbol.side)
            else:
                new_symbol = symbol.__class__(new_child)
            # ensure domain remains the same
            new_symbol.domain = symbol.domain
            return new_symbol

        # Concatenations
        elif isinstance(symbol, pybamm.Concatenation):
            new_children = []
            for child in symbol.children:
                new_child = self.process_symbol(child)
                new_children.append(new_child)
            if isinstance(symbol, pybamm.DomainConcatenation):
                return pybamm.DomainConcatenation(new_children, symbol.mesh)
            else:
                # Concatenation or NumpyConcatenation
                return symbol.__class__(*new_children)

        # Other cases: return new variable to avoid tree internal corruption
        elif isinstance(symbol, pybamm.Variable):
            return pybamm.Variable(symbol.name, symbol.domain)

        elif isinstance(symbol, pybamm.Scalar):
            return pybamm.Scalar(symbol.value, symbol.name, symbol.domain)

        elif isinstance(symbol, pybamm.Array):
            return symbol.__class__(
                symbol.entries, symbol.name, symbol.domain, symbol.entries_string
            )

        elif isinstance(symbol, pybamm.SpatialVariable):
            return pybamm.SpatialVariable(symbol.name, symbol.domain, symbol.coord_sys)

        elif isinstance(symbol, pybamm.StateVector):
            return symbol.__class__(symbol.y_slice, symbol.name, symbol.domain)

        elif isinstance(symbol, pybamm.Time):
            return pybamm.Time()

        else:
            raise NotImplementedError(
                "Cannot process parameters for symbol of type '{}'".format(type(symbol))
            )

    def update_scalars(self, symbol):
        """Update the value of any Scalars in the expression tree.

        Parameters
        ----------
        symbol : :class:`pybamm.Symbol`
            Symbol or Expression tree to update

        Returns
        -------
        symbol : :class:`pybamm.Symbol`
            Symbol with Scalars updated

        """
        for x in symbol.pre_order():
            if isinstance(x, pybamm.Scalar):
                # update any Scalar nodes if their name is in the parameter dict
                try:
                    x.value = self.get_parameter_value(x)
                    # update id
                    x.set_id()
                except KeyError:
                    # KeyError -> name not in parameter dict, don't update
                    continue
        return symbol
