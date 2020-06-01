#
# Dimensional and dimensionless parameter values, and scales
#
import pybamm
import pandas as pd
import os
import numbers
from pprint import pformat
from collections import defaultdict


class ParameterValues:
    """
    The parameter values for a simulation.

    Note that this class does not inherit directly from the python dictionary class as
    this causes issues with saving and loading simulations.

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
    >>> file = "input/parameters/lithium-ion/cells/kokam_Marquis2019/parameters.csv"
    >>> values_path = pybamm.get_parameters_filepath(file)
    >>> param = pybamm.ParameterValues(values=values_path)
    >>> param["Negative current collector thickness [m]"]
    2.5e-05
    >>> param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Marquis2019)
    >>> param["Reference temperature [K]"]
    298.15

    """

    def __init__(self, values=None, chemistry=None):
        self._dict_items = pybamm.FuzzyDict()
        # Must provide either values or chemistry, not both (nor neither)
        if values is not None and chemistry is not None:
            raise ValueError(
                "Only one of values and chemistry can be provided. To change parameters"
                " slightly from a chemistry, first load parameters with the chemistry"
                " (param = pybamm.ParameterValues(chemistry=...)) and then update with"
                " param.update({dict of values})."
            )
        if values is None and chemistry is None:
            raise ValueError("values and chemistry cannot both be None")
        # First load chemistry
        if chemistry is not None:
            self.update_from_chemistry(chemistry)
        # Then update with values dictionary or file
        if values is not None:
            # If base_parameters is a filename, load from that filename
            if isinstance(values, str):
                file_path = self.find_parameter(values)
                path = os.path.split(file_path)[0]
                values = self.read_parameters_csv(file_path)
            else:
                path = None
            # Don't check parameter already exists when first creating it
            self.update(values, check_already_exists=False, path=path)

        # Initialise empty _processed_symbols dict (for caching)
        self._processed_symbols = {}

    def __getitem__(self, key):
        return self._dict_items[key]

    def get(self, key, default=None):
        "Return item correspoonding to key if it exists, otherwise return default"
        try:
            return self._dict_items[key]
        except KeyError:
            return default

    def __setitem__(self, key, value):
        "Call the update functionality when doing a setitem"
        self.update({key: value})

    def __delitem__(self, key):
        del self._dict_items[key]

    def __repr__(self):
        return pformat(self._dict_items, width=1)

    def keys(self):
        "Get the keys of the dictionary"
        return self._dict_items.keys()

    def values(self):
        "Get the values of the dictionary"
        return self._dict_items.values()

    def items(self):
        "Get the items of the dictionary"
        return self._dict_items.items()

    def copy(self):
        """Returns a copy of the parameter values. Makes sure to copy the internal
        dictionary."""
        return ParameterValues(values=self._dict_items.copy())

    def search(self, key, print_values=True):
        """
        Search dictionary for keys containing 'key'.

        See :meth:`pybamm.FuzzyDict.search()`.
        """
        return self._dict_items.search(key, print_values)

    def update_from_chemistry(self, chemistry):
        """
        Load standard set of components from a 'chemistry' dictionary
        """
        base_chemistry = chemistry["chemistry"]

        # Load each component name

        component_groups = [
            "cell",
            "anode",
            "cathode",
            "separator",
            "electrolyte",
            "experiment",
        ]

        # add sei parameters if provided
        if "sei" in chemistry:
            component_groups += ["sei"]

        for component_group in component_groups:
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
            component_path = os.path.join(
                base_chemistry, component_group + "s", component
            )
            file_path = self.find_parameter(
                os.path.join(component_path, "parameters.csv")
            )
            component_params = self.read_parameters_csv(file_path)

            # Update parameters, making sure to check any conflicts
            self.update(
                component_params,
                check_conflict=True,
                check_already_exists=False,
                path=os.path.dirname(file_path),
            )

        # register (list of) citations
        if "citation" in chemistry:
            citations = chemistry["citation"]
            if not isinstance(citations, list):
                citations = [citations]
            for citation in citations:
                pybamm.citations.register(citation)

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

    def update(self, values, check_conflict=False, check_already_exists=True, path=""):
        """
        Update parameter dictionary, while also performing some basic checks.

        Parameters
        ----------
        values : dict
            Dictionary of parameter values to update parameter dictionary with
        check_conflict : bool, optional
            Whether to check that a parameter in `values` has not already been defined
            in the parameter class when updating it, and if so that its value does not
            change. This is set to True during initialisation, when parameters are
            combined from different sources, and is False by default otherwise
        check_already_exists : bool, optional
            Whether to check that a parameter in `values` already exists when trying to
            update it. This is to avoid cases where an intended change in the parameters
            is ignored due a typo in the parameter name, and is True by default but can
            be manually overridden.
        path : string, optional
            Path from which to load functions
        """
        # check parameter values
        self.check_parameter_values(values)
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
            # check parameter already exists (for updating parameters)
            if check_already_exists is True:
                try:
                    self._dict_items[name]
                except KeyError as err:
                    raise KeyError(
                        "Cannot update parameter '{}' as it does not ".format(name)
                        + "have a default value. ({}). If you are ".format(err.args[0])
                        + "sure you want to update this parameter, use "
                        + "param.update({{name: value}}, check_already_exists=False)"
                    )
            # if no conflicts, update, loading functions and data if they are specified
            # Functions are flagged with the string "[function]"
            if isinstance(value, str):
                if value.startswith("[function]"):
                    loaded_value = pybamm.load_function(
                        os.path.join(path, value[10:] + ".py")
                    )
                    self._dict_items[name] = loaded_value
                    values[name] = loaded_value
                # Data is flagged with the string "[data]" or "[current data]"
                elif value.startswith("[current data]") or value.startswith("[data]"):
                    if value.startswith("[current data]"):
                        data_path = os.path.join(
                            pybamm.root_dir(), "pybamm", "input", "drive_cycles"
                        )
                        filename = os.path.join(data_path, value[14:] + ".csv")
                        function_name = value[14:]
                    else:
                        filename = os.path.join(path, value[6:] + ".csv")
                        function_name = value[6:]
                    filename = pybamm.get_parameters_filepath(filename)
                    data = pd.read_csv(
                        filename, comment="#", skip_blank_lines=True, header=None
                    ).to_numpy()
                    # Save name and data
                    self._dict_items[name] = (function_name, data)
                    values[name] = (function_name, data)
                elif value == "[input]":
                    self._dict_items[name] = pybamm.InputParameter(name)
                # Anything else should be a converted to a float
                else:
                    self._dict_items[name] = float(value)
                    values[name] = float(value)
            else:
                self._dict_items[name] = value
        # reset processed symbols
        self._processed_symbols = {}

    def check_parameter_values(self, values):
        # Make sure typical current is non-zero
        if "Typical current [A]" in values and values["Typical current [A]"] == 0:
            raise ValueError(
                "'Typical current [A]' cannot be zero. A possible alternative is to "
                "set 'Current function [A]' to `0` instead."
            )
        if "C-rate" in values:
            raise ValueError(
                "The 'C-rate' parameter has been deprecated, "
                "use 'Current function [A]' instead. The cell capacity can be accessed "
                "as 'Cell capacity [A.h]', and used to calculate current from C-rate."
            )
        for param in values:
            if "surface area density" in param:
                raise ValueError(
                    "Parameters involving 'surface area density' have been renamed to "
                    "'surface area to volume ratio' ('{}' found)".format(param)
                )
            if "reaction rate" in param:
                raise ValueError(
                    "Parameters involving 'reaction rate' have been replaced with "
                    "'exchange-current density' ('{}' found)".format(param)
                )

    def process_model(self, unprocessed_model, inplace=True):
        """Assign parameter values to a model.
        Currently inplace, could be changed to return a new model.

        Parameters
        ----------
        unprocessed_model : :class:`pybamm.BaseModel`
            Model to assign parameter values for
        inplace: bool, optional
            If True, replace the parameters in the model in place. Otherwise, return a
            new model with parameter values set. Default is True.

        Raises
        ------
        :class:`pybamm.ModelError`
            If an empty model is passed (`model.rhs = {}` and `model.algebraic = {}` and
            `model.variables = {}`)

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
            model = unprocessed_model.new_copy()

        if (
            len(unprocessed_model.rhs) == 0
            and len(unprocessed_model.algebraic) == 0
            and len(unprocessed_model.variables) == 0
        ):
            raise pybamm.ModelError("Cannot process parameters for empty model")

        for variable, equation in model.rhs.items():
            pybamm.logger.debug("Processing parameters for {!r} (rhs)".format(variable))
            model.rhs[variable] = self.process_symbol(equation)

        for variable, equation in model.algebraic.items():
            pybamm.logger.debug(
                "Processing parameters for {!r} (algebraic)".format(variable)
            )
            model.algebraic[variable] = self.process_symbol(equation)

        for variable, equation in model.initial_conditions.items():
            pybamm.logger.debug(
                "Processing parameters for {!r} (initial conditions)".format(variable)
            )
            model.initial_conditions[variable] = self.process_symbol(equation)

        model.boundary_conditions = self.process_boundary_conditions(model)

        for variable, equation in model.variables.items():
            pybamm.logger.debug(
                "Processing parameters for {!r} (variables)".format(variable)
            )
            model.variables[variable] = self.process_symbol(equation)

        for event in model.events:
            pybamm.logger.debug(
                "Processing parameters for event'{}''".format(event.name)
            )
            event.expression = self.process_symbol(event.expression)

        # Process timescale
        model.timescale = self.process_symbol(model.timescale)

        pybamm.logger.info("Finish setting parameters for {}".format(model.name))

        return model

    def process_boundary_conditions(self, model):
        """
        Process boundary conditions for a model
        Boundary conditions are dictionaries {"left": left bc, "right": right bc}
        in general, but may be imposed on the tabs (or *not* on the tab) for a
        small number of variables, e.g. {"negative tab": neg. tab bc,
        "positive tab": pos. tab bc "no tab": no tab bc}.
        """
        new_boundary_conditions = {}
        sides = ["left", "right", "negative tab", "positive tab", "no tab"]
        for variable, bcs in model.boundary_conditions.items():
            processed_variable = self.process_symbol(variable)
            new_boundary_conditions[processed_variable] = {}
            for side in sides:
                try:
                    bc, typ = bcs[side]
                    pybamm.logger.debug(
                        "Processing parameters for {!r} ({} bc)".format(variable, side)
                    )
                    processed_bc = (self.process_symbol(bc), typ)
                    new_boundary_conditions[processed_variable][side] = processed_bc
                except KeyError as err:
                    # don't raise error if the key error comes from the side not being
                    # found
                    if err.args[0] in side:
                        pass
                    # do raise error otherwise (e.g. can't process symbol)
                    else:
                        raise KeyError(err)

        return new_boundary_conditions

    def update_model(self, model, disc):
        raise NotImplementedError(
            """
            update_model functionality has been deprecated.
            Use pybamm.InputParameter to quickly change a parameter value instead
            """
        )

    def process_geometry(self, geometry):
        """
        Assign parameter values to a geometry (inplace).

        Parameters
        ----------
        geometry : dict
            Geometry specs to assign parameter values to
        """
        for domain in geometry:
            for spatial_variable, spatial_limits in geometry[domain].items():
                # process tab information if using 1 or 2D current collectors
                if spatial_variable == "tabs":
                    for tab, position_size in spatial_limits.items():
                        for position_size, sym in position_size.items():
                            geometry[domain]["tabs"][tab][
                                position_size
                            ] = self.process_symbol(sym)
                else:
                    for lim, sym in spatial_limits.items():
                        if isinstance(sym, pybamm.Symbol):
                            geometry[domain][spatial_variable][
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
            if isinstance(value, numbers.Number):
                # Scalar inherits name (for updating parameters) and domain (for
                # Broadcast)
                return pybamm.Scalar(value, name=symbol.name, domain=symbol.domain)
            elif isinstance(value, pybamm.Symbol):
                new_value = self.process_symbol(value)
                new_value.domain = symbol.domain
                return new_value
            else:
                raise TypeError("Cannot process parameter '{}'".format(value))

        elif isinstance(symbol, pybamm.FunctionParameter):
            new_children = [self.process_symbol(child) for child in symbol.children]
            function_name = self[symbol.name]

            # Create Function or Interpolant or Scalar object
            if isinstance(function_name, tuple):
                # If function_name is a tuple then it should be (name, data) and we need
                # to create an Interpolant
                name, data = function_name
                function = pybamm.Interpolant(data, *new_children, name=name)
            elif isinstance(function_name, numbers.Number):
                # If the "function" is provided is actually a scalar, return a Scalar
                # object instead of throwing an error.
                # Also use ones_like so that we get the right shapes
                function = pybamm.Scalar(
                    function_name, name=symbol.name
                ) * pybamm.ones_like(*new_children)
            elif isinstance(function_name, pybamm.InputParameter):
                # Replace the function with an input parameter
                function = function_name
            elif (
                isinstance(function_name, pybamm.Symbol)
                and function_name.evaluates_to_number()
            ):
                # If the "function" provided is a pybamm scalar-like, use ones_like to
                # get the right shape
                function = function_name * pybamm.ones_like(*new_children)
            elif callable(function_name):
                # otherwise evaluate the function to create a new PyBaMM object
                function = function_name(*new_children)
            else:
                raise TypeError(
                    "Parameter provided for '{}' ".format(symbol.name)
                    + "is of the wrong type (should either be scalar-like or callable)"
                )
            # Differentiate if necessary
            if symbol.diff_variable is None:
                function_out = function
            else:
                # return differentiated function
                new_diff_variable = self.process_symbol(symbol.diff_variable)
                function_out = function.diff(new_diff_variable)
            # Convert possible float output to a pybamm scalar
            if isinstance(function_out, numbers.Number):
                return pybamm.Scalar(function_out)
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

    def _ipython_key_completions_(self):
        return list(self._dict_items.keys())

    def export_csv(self, filename):

        # process functions and data to output
        # like they appear in inputs csv files
        parameter_output = {}
        for key, val in self.items():
            if callable(val):
                val = "[function]" + val.__name__
            elif isinstance(val, tuple):
                val = "[data]" + val[0]
            parameter_output[key] = [val]

        df = pd.DataFrame(parameter_output)
        df = df.transpose()
        df.to_csv(filename, header=None)

    def print_parameters(self, parameters, output_file=None):
        """
        Return dictionary of evaluated parameters, and optionally print these evaluated
        parameters to an output file.
        For dimensionless parameters that depend on the C-rate, the value is given as a
        function of the C-rate (either x * Crate or x / Crate depending on the
        dependence)

        Parameters
        ----------
        parameters : class or dict containing :class:`pybamm.Parameter` objects
            Class or dictionary containing all the parameters to be evaluated
        output_file : string, optional
            The file to print parameters to. If None, the parameters are not printed,
            and this function simply acts as a test that all the parameters can be
            evaluated, and returns the dictionary of evaluated parameters.

        Returns
        -------
        evaluated_parameters : defaultdict
            The evaluated parameters, for further processing if needed

        Notes
        -----
        A C-rate of 1 C is the current required to fully discharge the battery in 1
        hour, 2 C is current to discharge the battery in 0.5 hours, etc
        """
        # Set list of attributes to ignore, for when we are evaluating parameters from
        # a class of parameters
        ignore = [
            "__name__",
            "__doc__",
            "__package__",
            "__loader__",
            "__spec__",
            "__file__",
            "__cached__",
            "__builtins__",
            "absolute_import",
            "division",
            "print_function",
            "unicode_literals",
            "pybamm",
            "constants",
            "np",
        ]

        # If 'parameters' is a class, extract the dict
        if not isinstance(parameters, dict):
            parameters = {
                k: v for k, v in parameters.__dict__.items() if k not in ignore
            }

        evaluated_parameters = defaultdict(list)
        # Calculate parameters for each C-rate
        for Crate in [1, 10]:
            # Update Crate
            capacity = self.get("Cell capacity [A.h]")
            if capacity is not None:
                self.update(
                    {"Current function [A]": Crate * capacity},
                    check_already_exists=False,
                )
            for name, symbol in parameters.items():
                if not callable(symbol):
                    proc_symbol = self.process_symbol(symbol)
                    if not (
                        callable(proc_symbol)
                        or proc_symbol.has_symbol_of_classes(
                            (pybamm.Concatenation, pybamm.Broadcast)
                        )
                    ):
                        evaluated_parameters[name].append(proc_symbol.evaluate(t=0))

        # Calculate C-dependence of the parameters based on the difference between the
        # value at 1C and the value at C / 10
        for name, values in evaluated_parameters.items():
            if values[1] == 0 or abs(values[0] / values[1] - 1) < 1e-10:
                C_dependence = ""
            elif abs(values[0] / values[1] - 10) < 1e-10:
                C_dependence = " * Crate"
            elif abs(values[0] / values[1] - 0.1) < 1e-10:
                C_dependence = " / Crate"
            evaluated_parameters[name] = (values[0], C_dependence)
        # Print the evaluated_parameters dict to output_file
        if output_file:
            self.print_evaluated_parameters(evaluated_parameters, output_file)

        return evaluated_parameters

    def print_evaluated_parameters(self, evaluated_parameters, output_file):
        """
        Print a dictionary of evaluated parameters to an output file

        Parameters
        ----------
        evaluated_parameters : defaultdict
            The evaluated parameters, for further processing if needed
        output_file : string, optional
            The file to print parameters to. If None, the parameters are not printed,
            and this function simply acts as a test that all the parameters can be
            evaluated

        """
        # Get column width for pretty printing
        column_width = max(len(name) for name in evaluated_parameters.keys())
        s = "{{:>{}}}".format(column_width)
        with open(output_file, "w") as file:
            for name, (value, C_dependence) in sorted(evaluated_parameters.items()):
                if 0.001 < abs(value) < 1000:
                    file.write(
                        (s + " : {:10.4g}{!s}\n").format(name, value, C_dependence)
                    )
                else:
                    file.write(
                        (s + " : {:10.3E}{!s}\n").format(name, value, C_dependence)
                    )

    @staticmethod
    def find_parameter(path):
        """Look for parameter file in the different locations
        in PARAMETER_PATH
        """
        for location in pybamm.PARAMETER_PATH:
            trial_path = os.path.join(location, path)
            if os.path.isfile(trial_path):
                return trial_path
        raise FileNotFoundError("Could not find parameter {}".format(path))
