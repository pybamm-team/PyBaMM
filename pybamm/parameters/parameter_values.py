#
# Dimensional and dimensionless parameter values, and scales
#
import pybamm
import pandas as pd
import os


class ParameterValues(dict):
    """
    The parameter values for a simulation.

    Parameters
    ----------
    values : dict or string
        Explicit set of parameters, or reference to a file of parameters
        If string, gets passed to read_parameters_csv to read a file.
    chemistry : dict
        Dict of strings for default chemistries. Must be of the form:
        {"base chemistry": base_chemistry
         "anode": anode_chemistry_authorYear,
         "cathode": cathode_chemistry_authorYear,
         "electrolyte": electrolyte_chemistry_authorYear}.
        Then the anode chemistry is loaded from the file
        base_chemistry/anodes/anode_chemistry_authorYear, etc.

    Examples
    --------
    >>> values = {"some parameter": 1, "another parameter": 2}
    >>> param = pybamm.ParameterValues(values)
    >>> param["some parameter"]
    1

    """

    def __init__(self, values=None, chemistry=None):
        if values is None and chemistry is None:
            raise ValueError("values and chemistry cannot all be None")
        # First load chemistry
        if chemistry is not None:
            base_chemistry = chemistry["chemistry"]
            # Load each component name
            for component_name in ["electrolyte", "anode", "cathode"]:
                try:
                    component = chemistry[component_name]
                except KeyError:
                    raise KeyError(
                        "must provide {} for {} chemistry".format(
                            component_name, chemistry
                        )
                    )
                self.update(
                    self.read_parameters_csv(
                        os.path.join(
                            pybamm.root_dir(),
                            "input",
                            "parameters",
                            base_chemistry,
                            component_name + "s",
                            component,
                            "parameters.csv",
                        )
                    )
                )
        # Then update with values dictionary or file
        if values is not None:
            if isinstance(values, str):
                values = self.read_parameters_csv(values)
            # If base_parameters is a filename, load from that filename
            self.update(values)

        # Initialise empty _processed_symbols dict (for caching)
        self._processed_symbols = {}

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
        return {k: v for (k, v) in zip(df["Name [units]"], df["Value"])}

    def update(self, values):
        # check parameter values
        self.check_parameter_values(values)
        # update
        for k, v in values.items():
            self[k] = v
        # reset processed symbols
        self._processed_symbols = {}

    def check_parameter_values(self, values):
        if "Typical current [A]" in values and values["Typical current [A]"] == 0:
            raise ValueError(
                """
                "Typical current [A]" cannot be zero. A possible alternative is to set
                "Current function" to `pybamm.GetConstantCurrent(current=0)` instead
                """
            )

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

        Raises
        ------
        :class:`pybamm.ModelError`
            If an empty model is passed (`model.rhs = {}` and `model.algebraic={}`)

        """
        pybamm.logger.info("Start setting parameters for {}".format(model.name))

        if len(model.rhs) == 0 and len(model.algebraic) == 0:
            raise pybamm.ModelError("Cannot process parameters for empty model")

        if processing == "process":
            processing_function = self.process_symbol
        elif processing == "update":
            processing_function = self.update_scalars

        for variable, equation in model.rhs.items():
            pybamm.logger.debug(
                "{} parameters for {!r} (rhs)".format(processing.capitalize(), variable)
            )
            model.rhs[variable] = processing_function(equation)

        for variable, equation in model.algebraic.items():
            pybamm.logger.debug(
                "{} parameters for {!r} (algebraic)".format(
                    processing.capitalize(), variable
                )
            )
            model.algebraic[variable] = processing_function(equation)

        for variable, equation in model.initial_conditions.items():
            pybamm.logger.debug(
                "{} parameters for {!r} (initial conditions)".format(
                    processing.capitalize(), variable
                )
            )
            model.initial_conditions[variable] = processing_function(equation)

        # Boundary conditions are dictionaries {"left": left bc, "right": right bc}
        # in general, but may be imposed on the tabs (or *not* on the tab) for a
        # small number of variables, e.g. {"negative tab": neg. tab bc,
        # "positive tab": pos. tab bc "no tab": no tab bc}.
        new_boundary_conditions = {}
        for variable, bcs in model.boundary_conditions.items():
            processed_variable = processing_function(variable)
            new_boundary_conditions[processed_variable] = {}
            for side in ["left", "right", "negative tab", "positive tab", "no tab"]:
                try:
                    bc, typ = bcs[side]
                    pybamm.logger.debug(
                        "{} parameters for {!r} ({} bc)".format(
                            processing.capitalize(), variable, side
                        )
                    )
                    processed_bc = (processing_function(bc), typ)
                    new_boundary_conditions[processed_variable][side] = processed_bc
                except KeyError:
                    pass

        model.boundary_conditions = new_boundary_conditions

        for variable, equation in model.variables.items():
            pybamm.logger.debug(
                "{} parameters for {!r} (variables)".format(
                    processing.capitalize(), variable
                )
            )
            model.variables[variable] = processing_function(equation)
        for event, equation in model.events.items():
            pybamm.logger.debug(
                "{} parameters for event '{}''".format(processing.capitalize(), event)
            )
            model.events[event] = processing_function(equation)

        pybamm.logger.info("Finish setting parameters for {}".format(model.name))

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
        Assign parameter values to a geometry (inplace).

        Parameters
        ----------
        geometry : :class:`pybamm.Geometry`
                Geometry specs to assign parameter values to
        """
        for domain in geometry:
            for prim_sec_tabs, variables in geometry[domain].items():
                # process tab information if using 1 or 2D current collectors
                if prim_sec_tabs == "tabs":
                    for tab, position_size in variables.items():
                        for position_size, sym in position_size.items():
                            geometry[domain][prim_sec_tabs][tab][
                                position_size
                            ] = self.process_symbol(sym)
                else:
                    for spatial_variable, spatial_limits in variables.items():
                        for lim, sym in spatial_limits.items():
                            geometry[domain][prim_sec_tabs][spatial_variable][
                                lim
                            ] = self.process_symbol(sym)

    def process_symbol(self, symbol):
        """Walk through the symbol and replace any Parameter with a Value.
        If a symbol has already been processed, the stored value is returned.

        Parameters
        ----------
        symbol : :class:`pybamm.Symbol`
            Symbol or Expression tree to set parameters for

        Returns
        -------
        symbol : :class:`pybamm.Symbol`
            Symbol with Parameter instances replaced by Value

        """

        try:
            return self._processed_symbols[symbol.id]
        except KeyError:
            processed_symbol = self._process_symbol(symbol)

            self._processed_symbols[symbol.id] = processed_symbol
            return processed_symbol

    def _process_symbol(self, symbol):
        """ See :meth:`ParameterValues.process_symbol()`. """

        if isinstance(symbol, pybamm.Parameter):
            value = self[symbol.name]
            # Scalar inherits name (for updating parameters) and domain (for Broadcast)
            return pybamm.Scalar(value, name=symbol.name, domain=symbol.domain)

        elif isinstance(symbol, pybamm.FunctionParameter):
            new_children = [self.process_symbol(child) for child in symbol.children]
            function_name = self[symbol.name]

            # if current setter, process any parameters that are symbols and
            # store the evaluated symbol in the parameters_eval dict
            if isinstance(function_name, pybamm.GetCurrent):
                for param, sym in function_name.parameters.items():
                    if isinstance(sym, pybamm.Symbol):
                        new_sym = self.process_symbol(sym)
                        function_name.parameters[param] = new_sym
                        function_name.parameters_eval[param] = new_sym.evaluate()
                # If loading data, need to update interpolant with
                # evaluated parameters
                if isinstance(function_name, pybamm.GetCurrentData):
                    function_name.interpolate()

            if callable(function_name):
                function = pybamm.Function(function_name, *new_children)
            else:
                function = pybamm.Function(
                    pybamm.load_function(function_name), *new_children
                )

            if symbol.diff_variable is None:
                return function
            else:
                # return differentiated function
                new_diff_variable = self.process_symbol(symbol.diff_variable)
                return function.diff(new_diff_variable)

        elif isinstance(symbol, pybamm.BinaryOperator):
            # process children
            new_left = self.process_symbol(symbol.left)
            new_right = self.process_symbol(symbol.right)
            # make new symbol, ensure domain remains the same
            new_symbol = symbol.__class__(new_left, new_right)
            new_symbol.domain = symbol.domain
            return new_symbol

        # Unary operators
        elif isinstance(symbol, pybamm.UnaryOperator):
            new_child = self.process_symbol(symbol.child)
            new_symbol = symbol._unary_new_copy(new_child)
            # ensure domain remains the same
            new_symbol.domain = symbol.domain
            return new_symbol

        # Functions
        elif isinstance(symbol, pybamm.Function):
            new_children = [self.process_symbol(child) for child in symbol.children]
            return symbol._function_new_copy(new_children)

        # Concatenations
        elif isinstance(symbol, pybamm.Concatenation):
            new_children = [self.process_symbol(child) for child in symbol.children]
            return symbol._concatenation_new_copy(new_children)

        else:
            # Backup option: return new copy of the object
            try:
                return symbol.new_copy()
            except NotImplementedError:
                raise NotImplementedError(
                    "Cannot process parameters for symbol of type '{}'".format(
                        type(symbol)
                    )
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
                    x.value = self[x.name]
                    # update id
                    x.set_id()
                except KeyError:
                    # KeyError -> name not in parameter dict, don't update
                    continue
            elif isinstance(x, pybamm.Function):
                if isinstance(x.function, pybamm.GetCurrent):
                    # Need to update parameters dict to be that of the new current
                    # function and make new parameters_eval dict to be processed
                    x.function.parameters = self["Current function"].parameters
                    x.function.parameters_eval = x.function.parameters.copy()
                    for param, sym in x.function.parameters.items():
                        # Need to process again as new symbols may be passed
                        # e.g. may explicitly pass pybamm.Scalar(1) instead of
                        # pybamm.electrical_parameters.I_typ
                        if isinstance(sym, pybamm.Symbol):
                            new_sym = self.process_symbol(sym)
                            x.function.parameters[param] = new_sym
                            try:
                                x.function.parameters_eval[param] = self[new_sym.name]
                            except KeyError:
                                # KeyError -> name not in parameter dict, evaluate
                                # unnamed Scalar
                                x.function.parameters_eval[param] = new_sym.evaluate()
                    if isinstance(x.function, pybamm.GetCurrentData):
                        # update interpolant
                        x.function.interpolate()

        return symbol
