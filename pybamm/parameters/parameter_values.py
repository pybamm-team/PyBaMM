#
# Dimensional and dimensionless parameter values, and scales
#
import pybamm
import pandas as pd
import os
import numbers


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
        {"base chemistry": base_chemistry,
        "cell": cell_properties_authorYear,
        "anode": anode_chemistry_authorYear,
        "separator": separator_chemistry_authorYear,
        "cathode": cathode_chemistry_authorYear,
        "electrolyte": electrolyte_chemistry_authorYear,
        "experiment": experimental_conditions_authorYear}.
        Then the anode chemistry is loaded from the file
        inputs/parameters/base_chemistry/anodes/anode_chemistry_authorYear, etc.
        Parameters in "cell" should include geometry and current collector properties.
        Parameters in "experiment" should include parameters relating to experimental
        conditions, such as initial conditions and currents.

    Examples
    --------
    >>> import pybamm
    >>> values = {"some parameter": 1, "another parameter": 2}
    >>> param = pybamm.ParameterValues(values)
    >>> param["some parameter"]
    1
    >>> file = "/input/parameters/lithium-ion/cells/kokam_Marquis2019/parameters.csv"
    >>> param = pybamm.ParameterValues(values=pybamm.root_dir() + file)
    >>> param["Negative current collector thickness [m]"]
    2.5e-05
    >>> param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Marquis2019)
    >>> param["Reference temperature [K]"]
    298.15

    """

    def __init__(self, values=None, chemistry=None):
        # Must provide either values or chemistry, not both (nor neither)
        if values is not None and chemistry is not None:
            raise ValueError(
                """
                Only one of values and chemistry can be provided. To change parameters
                slightly from a chemistry, first load parameters with the chemistry
                (param = pybamm.ParameterValues(chemistry=...)) and then update with
                param.update({dict of values}).
                """
            )
        if values is None and chemistry is None:
            raise ValueError("values and chemistry cannot both be None")
        # First load chemistry
        if chemistry is not None:
            self.update_from_chemistry(chemistry)
        # Then update with values dictionary or file
        if values is not None:
            if isinstance(values, str):
                values = self.read_parameters_csv(values)
            # If base_parameters is a filename, load from that filename
            self.update(values)

        # Initialise empty _processed_symbols dict (for caching)
        self._processed_symbols = {}

    def update_from_chemistry(self, chemistry):
        """
        Load standard set of components from a 'chemistry' dictionary
        """
        base_chemistry = chemistry["chemistry"]
        # Create path to file
        path = os.path.join(pybamm.root_dir(), "input", "parameters", base_chemistry)
        # Load each component name
        for component_group in [
            "cell",
            "anode",
            "cathode",
            "separator",
            "electrolyte",
            "experiment",
        ]:
            # Make sure component is provided
            try:
                component = chemistry[component_group]
            except KeyError:
                raise KeyError(
                    "must provide '{}' parameters for {} chemistry".format(
                        component_group, base_chemistry
                    )
                )
            # Create path to component and load values
            component_path = os.path.join(path, component_group + "s", component)
            component_params = self.read_parameters_csv(
                os.path.join(component_path, "parameters.csv")
            )
            # Update parameters, making sure to check any conflicts
            self.update(component_params, check_conflict=True, path=component_path)

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

    def __setitem__(self, key, value):
        "Call the update functionality when doing a setitem"
        self.update({key: value})

    def update(self, values, check_conflict=False, path=""):
        # check parameter values
        values = self.check_and_update_parameter_values(values)
        # update
        for name, value in values.items():
            # check for conflicts
            if (
                check_conflict is True
                and name in self.keys()
                and not (self[name] == float(value) or self[name] == value)
            ):
                raise ValueError(
                    "parameter '{}' already defined with value '{}'".format(
                        name, self[name]
                    )
                )
            # if no conflicts, update, loading functions and data if they are specified
            else:
                # Functions are flagged with the string "[function]"
                if isinstance(value, str):
                    if value.startswith("[function]"):
                        self[name] = pybamm.load_function(
                            os.path.join(path, value[10:] + ".py")
                        )
                    # Data is flagged with the string "[data]" or "[current data]"
                    elif value.startswith("[current data]") or value.startswith(
                        "[data]"
                    ):
                        if value.startswith("[current data]"):
                            data_path = os.path.join(
                                pybamm.root_dir(), "input", "drive_cycles"
                            )
                            filename = os.path.join(data_path, value[14:] + ".csv")
                            function_name = value[14:]
                        else:
                            filename = os.path.join(path, value[6:] + ".csv")
                            function_name = value[6:]
                        data = pd.read_csv(
                            filename, comment="#", skip_blank_lines=True
                        ).to_numpy()
                        # Save name and data
                        super().__setitem__(name, (function_name, data))
                    # Special case (hacky) for zero current
                    elif value == "[zero]":
                        super().__setitem__(name, 0)
                    # Anything else should be a converted to a float
                    else:
                        super().__setitem__(name, float(value))
                else:
                    super().__setitem__(name, value)
        # reset processed symbols
        self._processed_symbols = {}

    def check_and_update_parameter_values(self, values):
        # Make sure "C-rate" and current are both non-zero
        if "C-rate" in values and values["C-rate"] == 0:
            raise ValueError(
                """
                "C-rate" cannot be zero. A possible alternative is to set
                "Current function" to `0` instead.
                """
            )
        if "Typical current [A]" in values and values["Typical current [A]"] == 0:
            raise ValueError(
                """
                "Typical current [A]" cannot be zero. A possible alternative is to set
                "Current function" to `0` instead.
                """
            )
        # If the capacity of the cell has been provided, make sure "C-rate" and current
        # match with the stated capacity
        if "Cell capacity [A.h]" in values or "Cell capacity [A.h]" in self:
            # Capacity from values takes precedence
            if "Cell capacity [A.h]" in values:
                capacity = values["Cell capacity [A.h]"]
            else:
                capacity = self["Cell capacity [A.h]"]
            # Make sure they match if both provided
            if "C-rate" in values and "Typical current [A]" in values:
                if values["C-rate"] * capacity != values["Typical current [A]"]:
                    raise ValueError(
                        """
                        "C-rate" ({}C) and Typical current ({} A) provided do not match
                        given capacity ({} Ah). These can be updated individually
                        instead.
                        """.format(
                            values["C-rate"], values["Typical current [A]"], capacity
                        )
                    )
            # Update the other if only one provided
            elif "C-rate" in values:
                values["Typical current [A]"] = float(values["C-rate"]) * capacity
            elif "Typical current [A]" in values:
                values["C-rate"] = float(values["Typical current [A]"]) / capacity

        # Update the current function if it is constant
        self_and_values = {**self, **values}
        if "Current function" in self_and_values and (
            self_and_values["Current function"] == "[constant]"
            or isinstance(self_and_values["Current function"], numbers.Number)
        ):
            values["Current function"] = {**self, **values}["Typical current [A]"]

        return values

    def process_model(self, unprocessed_model, processing="process", inplace=True):
        """Assign parameter values to a model.
        Currently inplace, could be changed to return a new model.

        Parameters
        ----------
        unprocessed_model : :class:`pybamm.BaseModel`
            Model to assign parameter values for
        processing : str, optional
            Flag to indicate how to process model (default 'process')

            * 'process': Calls :meth:`process_symbol()` (walk through the symbol \
            and replace any Parameter with a Value)
            * 'update': Calls :meth:`update_scalars()` for use on already-processed \
            model (update the value of any Scalars in the expression tree.)
        inplace: bool, optional
            If True, replace the parameters in the model in place. Otherwise, return a
            new model with parameter values set. Default is True.

        Raises
        ------
        :class:`pybamm.ModelError`
            If an empty model is passed (`model.rhs = {}` and `model.algebraic={}`)

        """
        pybamm.logger.info(
            "Start setting parameters for {}".format(unprocessed_model.name)
        )

        # set up inplace vs not inplace
        if inplace:
            # any changes to model_disc attributes will change model attributes
            # since they point to the same object
            model = unprocessed_model
        else:
            # create a blank model of the same class
            model = unprocessed_model.__class__(unprocessed_model.options)
            model.name = unprocessed_model.name
            model.options = unprocessed_model.options
            model.use_jacobian = unprocessed_model.use_jacobian
            model.use_simplify = unprocessed_model.use_simplify
            model.convert_to_format = unprocessed_model.convert_to_format

        if len(unprocessed_model.rhs) == 0 and len(unprocessed_model.algebraic) == 0:
            raise pybamm.ModelError("Cannot process parameters for empty model")

        if processing == "process":
            processing_function = self.process_symbol
        elif processing == "update":
            processing_function = self.update_scalars

        for variable, equation in unprocessed_model.rhs.items():
            pybamm.logger.debug(
                "{} parameters for {!r} (rhs)".format(processing.capitalize(), variable)
            )
            model.rhs[variable] = processing_function(equation)

        for variable, equation in unprocessed_model.algebraic.items():
            pybamm.logger.debug(
                "{} parameters for {!r} (algebraic)".format(
                    processing.capitalize(), variable
                )
            )
            model.algebraic[variable] = processing_function(equation)

        for variable, equation in unprocessed_model.initial_conditions.items():
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
        for variable, bcs in unprocessed_model.boundary_conditions.items():
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

        for variable, equation in unprocessed_model.variables.items():
            pybamm.logger.debug(
                "{} parameters for {!r} (variables)".format(
                    processing.capitalize(), variable
                )
            )
            model.variables[variable] = processing_function(equation)
        for event, equation in unprocessed_model.events.items():
            pybamm.logger.debug(
                "{} parameters for event '{}''".format(processing.capitalize(), event)
            )
            model.events[event] = processing_function(equation)

        pybamm.logger.info("Finish setting parameters for {}".format(model.name))

        return model

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

            # Create Function or Interpolant or Scalar objec
            if isinstance(function_name, tuple):
                # If function_name is a tuple then it should be (name, data) and we need
                # to create an Interpolant
                name, data = function_name
                function = pybamm.Interpolant(data, *new_children, name=name)
            elif isinstance(function_name, numbers.Number):
                # If the "function" is provided is actually a scalar, return a Scalar
                # object instead of throwing an error
                function = pybamm.Scalar(function_name, name=symbol.name)
            else:
                # otherwise evaluate the function to create a new PyBaMM object
                function = function_name(*new_children)
                # this might return a scalar, in which case convert to a pybamm scalar
                if isinstance(function, numbers.Number):
                    function = pybamm.Scalar(function, name=symbol.name)
            # Differentiate if necessary
            if symbol.diff_variable is None:
                function_out = function
            else:
                # return differentiated function
                new_diff_variable = self.process_symbol(symbol.diff_variable)
                function_out = function.diff(new_diff_variable)
            # Process again just to be sure
            return self.process_symbol(function_out)

        elif isinstance(symbol, pybamm.BinaryOperator):
            # process children
            new_left = self.process_symbol(symbol.left)
            new_right = self.process_symbol(symbol.right)
            # make new symbol, ensure domain remains the same
            new_symbol = symbol._binary_new_copy(new_left, new_right)
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

        return symbol

    def evaluate(self, symbol):
        """
        Process and evaluate a symbol.

        Parameters
        ----------
        symbol : :class:`pybamm.Symbol`
            Symbol or Expression tree to evaluate

        Returns
        -------
        number of array
            The evaluated symbol
        """
        processed_symbol = self.process_symbol(symbol)
        if processed_symbol.is_constant() and processed_symbol.evaluates_to_number():
            return processed_symbol.evaluate()
        else:
            raise ValueError("symbol must evaluate to a constant scalar")
