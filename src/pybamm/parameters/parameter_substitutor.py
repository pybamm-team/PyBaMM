from __future__ import annotations

import numbers
from typing import TYPE_CHECKING, Any

import numpy as np

import pybamm
from pybamm.models.base_model import ModelSolutionObservability

if TYPE_CHECKING:
    from .parameter_store import ParameterStore


class ParameterSubstitutor:
    """
    Handles symbol and model parameterization.

    This class is responsible for walking through expression trees and replacing
    Parameter nodes with their corresponding values from a ParameterStore.

    Parameters
    ----------
    store : ParameterStore
        The parameter store to read values from.

    Examples
    --------
    >>> from pybamm.parameters import ParameterStore, ParameterSubstitutor
    >>> store = ParameterStore({"Temperature [K]": 298.15})
    >>> processor = ParameterSubstitutor(store)
    >>> param = pybamm.Parameter("Temperature [K]")
    >>> processed = processor.process_symbol(param)
    >>> float(processed.evaluate())
    298.15
    """

    def __init__(self, store: ParameterStore) -> None:
        self._store = store
        self._cache: dict[pybamm.Symbol, pybamm.Symbol] = {}

    def clear_cache(self) -> None:
        """
        Invalidate the processed symbol cache.

        Example
        -------
        >>> from pybamm.parameters import ParameterStore, ParameterSubstitutor
        >>> store = ParameterStore({"Temperature [K]": 298.15})
        >>> processor = ParameterSubstitutor(store)
        >>> processor.clear_cache()  # Force re-processing of symbols
        """
        self._cache = {}

    @property
    def cache(self) -> dict[pybamm.Symbol, pybamm.Symbol]:
        """Return the processed symbols cache (read-only access)."""
        return self._cache

    def process_symbol(self, symbol: pybamm.Symbol) -> pybamm.Symbol:
        """
        Walk through the symbol and replace any Parameter with a Value.

        If a symbol has already been processed, the cached value is returned.

        Parameters
        ----------
        symbol : pybamm.Symbol
            Symbol or Expression tree to set parameters for.

        Returns
        -------
        pybamm.Symbol
            Symbol with Parameter instances replaced by values.

        Example
        -------
        >>> from pybamm.parameters import ParameterStore, ParameterSubstitutor
        >>> store = ParameterStore({"Current [A]": 5.0})
        >>> processor = ParameterSubstitutor(store)
        >>> param = pybamm.Parameter("Current [A]")
        >>> processed = processor.process_symbol(param)
        >>> result = processed.evaluate()  # Returns evaluated value
        """
        try:
            return self._cache[symbol]
        except KeyError:
            if not isinstance(symbol, pybamm.FunctionParameter):
                processed_symbol = self._process_symbol(symbol)
            else:
                processed_symbol = self._process_function_parameter(symbol)
            self._cache[symbol] = processed_symbol
            return processed_symbol

    def _process_symbol(self, symbol: pybamm.Symbol) -> pybamm.Symbol:
        """Internal symbol processing implementation."""
        if isinstance(symbol, pybamm.Parameter):
            try:
                value = self._store[symbol.name]
            except KeyError as err:
                # Handle renamed parameter with helpful error message
                if (
                    "Exchange-current density for lithium metal electrode [A.m-2]"
                    in symbol.name
                    and "Exchange-current density for plating [A.m-2]" in self._store
                ):
                    raise KeyError(
                        "'Exchange-current density for plating [A.m-2]' has been renamed "
                        "to 'Exchange-current density for lithium metal electrode [A.m-2]' "
                        "when referring to the reaction at the surface of a lithium metal "
                        "electrode. This is to avoid confusion with the exchange-current "
                        "density for the lithium plating reaction in a porous negative "
                        "electrode. To avoid this error, change your parameter file to use "
                        "the new name."
                    ) from err
                raise
            if isinstance(value, numbers.Number):
                # Check not NaN (parameter in csv file but no value given)
                if np.isnan(value):
                    raise ValueError(f"Parameter '{symbol.name}' not found")
                # Scalar inherits name
                return pybamm.Scalar(value, name=symbol.name)
            elif isinstance(value, pybamm.Symbol):
                new_value = self.process_symbol(value)
                new_value.copy_domains(symbol)
                return new_value
            else:
                raise TypeError(f"Cannot process parameter '{value}'")

        elif isinstance(symbol, pybamm.FunctionParameter):
            function_name = self._store[symbol.name]
            if isinstance(
                function_name,
                numbers.Number | pybamm.Interpolant | pybamm.InputParameter,
            ) or (
                isinstance(function_name, pybamm.Symbol)
                and function_name.size_for_testing == 1
            ):
                # no need to process children, they will only be used for shape
                new_children = symbol.children
            else:
                # process children
                new_children = []
                for child in symbol.children:
                    if symbol.diff_variable is not None and any(
                        x == symbol.diff_variable for x in child.pre_order()
                    ):
                        # Wrap with NotConstant to avoid simplification,
                        # which would stop symbolic diff from working properly
                        new_child = pybamm.NotConstant(child)
                        new_children.append(self.process_symbol(new_child))
                    else:
                        new_children.append(self.process_symbol(child))

            # Create Function or Interpolant or Scalar object
            if isinstance(function_name, tuple):
                if len(function_name) == 2:  # CSV or JSON parsed data
                    # to create an Interpolant
                    name, data = function_name

                    if len(data[0]) == 1:
                        input_data = data[0][0], data[1]
                    else:
                        input_data = data

                    # For parameters provided as data we use a cubic interpolant
                    # Note: the cubic interpolant can be differentiated
                    function = pybamm.Interpolant(
                        input_data[0],
                        input_data[-1],
                        new_children,
                        name=name,
                    )
                else:  # pragma: no cover
                    raise ValueError(
                        f"Invalid function name length: {len(function_name)}"
                    )

            elif isinstance(function_name, numbers.Number):
                # Check not NaN (parameter in csv file but no value given)
                if np.isnan(function_name):
                    raise ValueError(
                        f"Parameter '{symbol.name}' (possibly a function) not found"
                    )
                # If the "function" is provided is actually a scalar, return a Scalar
                # object instead of throwing an error.
                function = pybamm.Scalar(function_name, name=symbol.name)
            elif callable(function_name):
                # otherwise evaluate the function to create a new PyBaMM object
                function = function_name(*new_children)
            elif isinstance(
                function_name, pybamm.Interpolant | pybamm.InputParameter
            ) or (
                isinstance(function_name, pybamm.Symbol)
                and function_name.size_for_testing == 1
            ):
                function = function_name
            else:
                raise TypeError(
                    f"Parameter provided for '{symbol.name}' "
                    + "is of the wrong type (should either be scalar-like or callable)"
                )

            # Differentiate if necessary
            if symbol.diff_variable is None:
                # Use ones_like so that we get the right shapes
                function_out = function * pybamm.ones_like(*new_children)
            else:
                # return differentiated function
                new_diff_variable = self.process_symbol(symbol.diff_variable)
                function_out = function.diff(new_diff_variable)

            # Process again just to be sure
            return self.process_symbol(function_out)

        # Unary operators
        elif isinstance(symbol, pybamm.UnaryOperator):
            new_child = self.process_symbol(symbol.child)
            new_symbol = symbol.create_copy(new_children=[new_child])
            # x_average can sometimes create a new symbol with electrode thickness
            # parameters, so we process again to make sure these parameters are set
            if isinstance(symbol, pybamm.XAverage) and not isinstance(
                new_symbol, pybamm.XAverage
            ):
                new_symbol = self.process_symbol(new_symbol)
            # f_a_dist in the size average needs to be processed
            if isinstance(new_symbol, pybamm.SizeAverage):
                new_symbol.f_a_dist = self.process_symbol(new_symbol.f_a_dist)
            # position in evaluate at needs to be processed, and should be a Scalar
            if isinstance(new_symbol, pybamm.EvaluateAt):
                new_symbol_position = self.process_symbol(new_symbol.position)
                if not isinstance(new_symbol_position, pybamm.Scalar):
                    raise ValueError(
                        "'position' in 'EvaluateAt' must evaluate to a scalar"
                    )
                else:
                    new_symbol.position = new_symbol_position
            return new_symbol

        # Functions, BinaryOperators & Concatenations
        elif (
            isinstance(symbol, pybamm.Function)
            or isinstance(symbol, pybamm.Concatenation)
            or isinstance(symbol, pybamm.BinaryOperator)
        ):
            new_children = [self.process_symbol(child) for child in symbol.children]
            return symbol.create_copy(new_children)

        elif isinstance(symbol, pybamm.VectorField):
            left_symbol = self.process_symbol(symbol.lr_field)
            right_symbol = self.process_symbol(symbol.tb_field)
            return symbol.create_copy(new_children=[left_symbol, right_symbol])

        # Variables: update scale
        elif isinstance(symbol, pybamm.Variable):
            new_symbol = symbol.create_copy()
            new_symbol.scale = self.process_symbol(symbol.scale)
            reference = self.process_symbol(symbol.reference)
            if isinstance(reference, pybamm.Vector):
                # address numpy 1.25 deprecation warning: array should have ndim=0
                # before conversion
                reference = pybamm.Scalar((reference.evaluate()).item())
            new_symbol.reference = reference
            new_symbol.bounds = tuple([self.process_symbol(b) for b in symbol.bounds])
            return new_symbol

        elif isinstance(symbol, numbers.Number):
            return pybamm.Scalar(symbol)

        else:
            # Backup option: return the object
            return symbol

    def _process_function_parameter(
        self, symbol: pybamm.FunctionParameter
    ) -> pybamm.Symbol:
        """Process ExpressionFunctionParameter symbols."""
        function_parameter = self._store[symbol.name]

        # Handle symbolic function parameter case
        if isinstance(function_parameter, pybamm.ExpressionFunctionParameter):
            # Process children
            new_children = []
            for child in symbol.children:
                if symbol.diff_variable is not None and any(
                    x == symbol.diff_variable for x in child.pre_order()
                ):
                    # Wrap with NotConstant to avoid simplification,
                    # which would stop symbolic diff from working properly
                    new_child = pybamm.NotConstant(child)
                    new_children.append(self.process_symbol(new_child))
                else:
                    new_children.append(self.process_symbol(child))

            # Get the expression and inputs for the function.
            expression = function_parameter.child
            inputs = {
                arg: child
                for arg, child in zip(
                    function_parameter.func_args, new_children, strict=False
                )
            }

            # Set domains for function inputs in post-order traversal
            for node in expression.post_order():
                if node.name in inputs:
                    node.domains = inputs[node.name].domains
                else:
                    node.domains = node.get_children_domains(node.children)

            # Create a combined processor with inputs as additional parameters
            # We need to import here to avoid circular imports
            from .parameter_store import ParameterStore

            combined_store = ParameterStore(dict(self._store._data))
            combined_store.update(inputs)
            combined_processor = ParameterSubstitutor(combined_store)

            # Process any FunctionParameter children first to avoid recursion
            for child in expression.pre_order():
                if isinstance(child, pybamm.FunctionParameter):
                    # Build new child with parent inputs
                    new_child_children = [
                        inputs[child_child.name]
                        if isinstance(child_child, pybamm.Parameter)
                        and child_child.name in inputs
                        else child_child
                        for child_child in child.children
                    ]
                    new_child = pybamm.FunctionParameter(
                        child.name,
                        dict(zip(child.input_names, new_child_children, strict=False)),
                        diff_variable=child.diff_variable,
                        print_name=child.print_name,
                    )

                    # For this local combined processor, process the new child
                    # and store the result as the processed symbol for this child
                    combined_processor._cache[child] = (
                        combined_processor.process_symbol(new_child)
                    )

            # Process function with combined processor to get a symbolic expression
            function = combined_processor.process_symbol(expression)

            # Differentiate if necessary
            if symbol.diff_variable is None:
                # Use ones_like so that we get the right shapes
                function_out = function * pybamm.ones_like(*new_children)
            else:
                # return differentiated function
                new_diff_variable = self.process_symbol(symbol.diff_variable)
                function_out = function.diff(new_diff_variable)

            return function_out

        # Handle non-symbolic function_name case
        else:
            return self._process_symbol(symbol)

    def process_model(
        self,
        unprocessed_model: pybamm.BaseModel,
        *,
        inplace: bool = True,
        delayed_variable_processing: bool | None = None,
    ) -> pybamm.BaseModel:
        """
        Assign parameter values to a model.

        Example
        -------
        >>> from pybamm.parameters import ParameterStore, ParameterSubstitutor
        >>> store = pybamm.ParameterValues("Chen2020").store
        >>> processor = ParameterSubstitutor(store)
        >>> model = pybamm.lithium_ion.SPM()
        >>> processed_model = processor.process_model(model)

        Parameters
        ----------
        unprocessed_model : pybamm.BaseModel
            Model to assign parameter values for.
        inplace : bool, optional
            If True (default), replace parameters in place.
            If False, return a new model with parameter values set.
        delayed_variable_processing : bool, optional
            If True, make variable processing a post-processing step.
            Default is False.

        Returns
        -------
        pybamm.BaseModel
            The parameterized model.

        Raises
        ------
        pybamm.ModelError
            If an empty model is passed.
        """
        pybamm.logger.info(f"Start setting parameters for {unprocessed_model.name}")

        if delayed_variable_processing is None:
            delayed_variable_processing = False

        # set up inplace vs not inplace
        if inplace:
            model = unprocessed_model
        else:
            model = unprocessed_model.new_copy()

        if (
            len(unprocessed_model.rhs) == 0
            and len(unprocessed_model.algebraic) == 0
            and len(unprocessed_model.variables) == 0
        ):
            raise pybamm.ModelError("Cannot process parameters for empty model")

        # Find all InputParameters in the parameter values
        unpacker = pybamm.SymbolUnpacker(pybamm.InputParameter)
        model.fixed_input_parameters = unpacker.unpack_list_of_symbols(
            v for v in self._store.values() if isinstance(v, pybamm.Symbol)
        )

        new_rhs = {}
        for variable, equation in unprocessed_model.rhs.items():
            pybamm.logger.verbose(f"Processing parameters for {variable!r} (rhs)")
            new_variable = self.process_symbol(variable)
            new_rhs[new_variable] = self.process_symbol(equation)
        model.rhs = new_rhs

        new_algebraic = {}
        for variable, equation in unprocessed_model.algebraic.items():
            pybamm.logger.verbose(f"Processing parameters for {variable!r} (algebraic)")
            new_variable = self.process_symbol(variable)
            new_algebraic[new_variable] = self.process_symbol(equation)
        model.algebraic = new_algebraic

        new_initial_conditions = {}
        for variable, equation in unprocessed_model.initial_conditions.items():
            pybamm.logger.verbose(
                f"Processing parameters for {variable!r} (initial conditions)"
            )
            new_variable = self.process_symbol(variable)
            new_initial_conditions[new_variable] = self.process_symbol(equation)
        model.initial_conditions = new_initial_conditions

        model.boundary_conditions = self.process_boundary_conditions(unprocessed_model)

        if not delayed_variable_processing:
            # Process variables and store in _variables_processed, NOT in variables
            variables_to_process = (
                unprocessed_model.variables
                | unprocessed_model.get_processed_variables_dict()
            )
            processed_variables = {}
            for variable, equation in variables_to_process.items():
                pybamm.logger.verbose(
                    f"Processing parameters for {variable!r} (variables)"
                )
                processed_variables[variable] = self.process_symbol(equation)
            model.update_processed_variables(processed_variables)

        new_events = []
        for event in unprocessed_model.events:
            pybamm.logger.verbose(f"Processing parameters for event '{event.name}''")
            new_events.append(
                pybamm.Event(
                    event.name, self.process_symbol(event.expression), event.event_type
                )
            )

        interpolant_events = self._get_interpolant_events(model)
        for event in interpolant_events:
            pybamm.logger.verbose(f"Processing parameters for event '{event.name}''")
            new_events.append(
                pybamm.Event(
                    event.name, self.process_symbol(event.expression), event.event_type
                )
            )

        model.events = new_events

        pybamm.logger.info(f"Finish setting parameters for {model.name}")

        if unprocessed_model.is_parameterised:
            pybamm.logger.debug(
                f"Model '{model.name}' is being re-processed, "
                "which makes it unable to process symbols using `model.process_symbol`"
            )
            model.disable_symbol_processing(
                ModelSolutionObservability.REPARAMETERISED_MODEL
            )

        model.is_parameterised = True
        return model

    def _get_interpolant_events(self, model: pybamm.BaseModel) -> list[pybamm.Event]:
        """Add events for functions that have been defined as parameters."""
        interpolants = model._find_symbols(pybamm.Interpolant)
        interpolant_events = []
        for interpolant in interpolants:
            xs = interpolant.x
            children = interpolant.children
            for x, child in zip(xs, children, strict=False):
                interpolant_events.extend(
                    [
                        pybamm.Event(
                            f"Interpolant '{interpolant.name}' lower bound",
                            pybamm.min(child - min(x)),
                            pybamm.EventType.INTERPOLANT_EXTRAPOLATION,
                        ),
                        pybamm.Event(
                            f"Interpolant '{interpolant.name}' upper bound",
                            pybamm.min(max(x) - child),
                            pybamm.EventType.INTERPOLANT_EXTRAPOLATION,
                        ),
                    ]
                )
        return interpolant_events

    def process_boundary_conditions(
        self, model: pybamm.BaseModel
    ) -> dict[pybamm.Symbol, dict[str, tuple[pybamm.Symbol, str]]]:
        """
        Process boundary conditions for a model.

        Boundary conditions are dictionaries {"left": left bc, "right": right bc}
        in general, but may be imposed on the tabs (or *not* on the tab) for a
        small number of variables.
        """
        new_boundary_conditions: dict[
            pybamm.Symbol, dict[str, tuple[pybamm.Symbol, str]]
        ] = {}
        sides = [
            "left",
            "right",
            "negative tab",
            "positive tab",
            "no tab",
            "top",
            "bottom",
            "x_min",
            "x_max",
            "y_min",
            "y_max",
            "z_min",
            "z_max",
            "r_min",
            "r_max",
        ]
        for variable, bcs in model.boundary_conditions.items():
            processed_variable = self.process_symbol(variable)
            new_boundary_conditions[processed_variable] = {}
            for side in sides:
                try:
                    bc, typ = bcs[side]
                    pybamm.logger.verbose(
                        f"Processing parameters for {variable!r} ({side} bc)"
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
                        raise err

        return new_boundary_conditions

    def process_geometry(self, geometry: dict) -> None:
        """
        Assign parameter values to a geometry (inplace).

        Parameters
        ----------
        geometry : dict
            Geometry specs to assign parameter values to.

        Example
        -------
        >>> from pybamm.parameters import ParameterStore, ParameterSubstitutor
        >>> store = pybamm.ParameterValues("Marquis2019").store
        >>> processor = ParameterSubstitutor(store)
        >>> geometry = pybamm.battery_geometry()
        >>> processor.process_geometry(geometry)
        """

        def process_and_check(sym: Any) -> pybamm.Symbol:
            new_sym = self.process_symbol(sym)
            leaves = new_sym.post_order(filter=lambda node: len(node.children) == 0)
            for leaf in leaves:
                if not isinstance(leaf, pybamm.Scalar) and not isinstance(
                    leaf, pybamm.InputParameter
                ):
                    raise ValueError(
                        "Geometry parameters must be Scalars or InputParameters after parameter processing"
                    )
            return new_sym

        for domain in geometry:
            for spatial_variable, spatial_limits in geometry[domain].items():
                # process tab information if using 1 or 2D current collectors
                if spatial_variable == "tabs":
                    for tab, position_info in spatial_limits.items():
                        for position_size, sym in position_info.items():
                            geometry[domain]["tabs"][tab][position_size] = (
                                process_and_check(sym)
                            )
                else:
                    for lim, sym in spatial_limits.items():
                        geometry[domain][spatial_variable][lim] = process_and_check(sym)

    def evaluate(self, symbol: pybamm.Symbol, inputs: dict | None = None) -> Any:
        """
        Process and evaluate a symbol.

        Parameters
        ----------
        symbol : pybamm.Symbol
            Symbol or Expression tree to evaluate.
        inputs : dict, optional
            Input parameter values for evaluation.

        Returns
        -------
        number or array
            The evaluated symbol.

        Raises
        ------
        ValueError
            If symbol does not evaluate to a constant.

        Example
        -------
        >>> from pybamm.parameters import ParameterStore, ParameterSubstitutor
        >>> store = ParameterStore({"Current [A]": 5.0})
        >>> processor = ParameterSubstitutor(store)
        >>> param = pybamm.Parameter("Current [A]")
        >>> result = processor.evaluate(param)  # Returns evaluated value
        """
        processed_symbol = self.process_symbol(symbol)
        if processed_symbol.is_constant():
            return processed_symbol.evaluate()
        else:
            try:
                return processed_symbol.evaluate(inputs=inputs)
            except Exception as exc:
                raise ValueError(
                    "symbol must evaluate to a constant scalar or array"
                ) from exc
