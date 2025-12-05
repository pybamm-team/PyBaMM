import pybamm


class SymbolReplacer:
    """
    Helper class to replace all instances of one or more symbols in an expression tree
    with another symbol, as defined by the dictionary `symbol_replacement_map`

    Parameters
    ----------
    symbol_replacement_map : dict {:class:`pybamm.Symbol` -> :class:`pybamm.Symbol`}
        Map of which symbols should be replaced by which.
    processed_symbols: dict {:class:`pybamm.Symbol` -> :class:`pybamm.Symbol`}, optional
        cached replaced symbols
    process_initial_conditions: bool, optional
        Whether to process initial conditions, default is True
    """

    def __init__(
        self,
        symbol_replacement_map,
        processed_symbols=None,
        process_initial_conditions=True,
    ):
        self._symbol_replacement_map = symbol_replacement_map
        self._processed_symbols = processed_symbols or {}
        self.process_initial_conditions = process_initial_conditions

    def process_model(self, unprocessed_model, inplace=True):
        """Replace all instances of a symbol in a model.

        Parameters
        ----------
        unprocessed_model : :class:`pybamm.BaseModel`
            Model to assign parameter values for
        inplace: bool, optional
            If True, replace the parameters in the model in place. Otherwise, return a
            new model with parameter values set. Default is True.
        """
        pybamm.logger.info(f"Start replacing symbols in {unprocessed_model.name}")

        # set up inplace vs not inplace
        if inplace:
            # any changes to unprocessed_model attributes will change model attributes
            # since they point to the same object
            model = unprocessed_model
        else:
            # create a copy of the model
            model = unprocessed_model.new_copy()

        new_rhs = {}
        for variable, equation in unprocessed_model.rhs.items():
            pybamm.logger.verbose(f"Replacing symbols in {variable!r} (rhs)")
            new_rhs[self.process_symbol(variable)] = self.process_symbol(equation)
        model.rhs = new_rhs

        new_algebraic = {}
        for variable, equation in unprocessed_model.algebraic.items():
            pybamm.logger.verbose(f"Replacing symbols in {variable!r} (algebraic)")
            new_algebraic[self.process_symbol(variable)] = self.process_symbol(equation)
        model.algebraic = new_algebraic

        new_initial_conditions = {}
        for variable, equation in unprocessed_model.initial_conditions.items():
            pybamm.logger.verbose(
                f"Replacing symbols in {variable!r} (initial conditions)"
            )
            if self.process_initial_conditions:
                new_initial_conditions[self.process_symbol(variable)] = (
                    self.process_symbol(equation)
                )
            else:
                new_initial_conditions[self.process_symbol(variable)] = equation
        model.initial_conditions = new_initial_conditions

        model.boundary_conditions = self.process_boundary_conditions(unprocessed_model)

        new_variables = {}
        for variable, equation in unprocessed_model.variables.items():
            pybamm.logger.verbose(f"Replacing symbols in {variable!r} (variables)")
            new_variables[variable] = self.process_symbol(equation)
        model.variables = new_variables

        new_events = []
        for event in unprocessed_model.events:
            pybamm.logger.verbose(f"Replacing symbols in event'{event.name}''")
            new_events.append(
                pybamm.Event(
                    event.name, self.process_symbol(event.expression), event.event_type
                )
            )
        model.events = new_events

        pybamm.logger.info(f"Finish replacing symbols in {model.name}")

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
                    pybamm.logger.verbose(
                        f"Replacing symbols in {variable!r} ({side} bc)"
                    )
                    processed_bc = (self.process_symbol(bc), typ)
                    new_boundary_conditions[processed_variable][side] = processed_bc
                except KeyError as err:
                    # don't raise error if the key error comes from the side not being
                    # found
                    if err.args[0] in side:
                        pass
                    # do raise error otherwise (e.g. can't process symbol)
                    else:  # pragma: no cover
                        raise KeyError(err) from err

        return new_boundary_conditions

    def process_symbol(self, symbol):
        """
        This function recurses down the tree, replacing any symbols in
        self._symbol_replacement_map with their corresponding value

        Parameters
        ----------
        symbol : :class:`pybamm.Symbol`
            The symbol to replace

        Returns
        -------
        :class:`pybamm.Symbol`
            Symbol with all replacements performed
        """

        _processed_symbol = self._processed_symbols.get(symbol)
        if _processed_symbol is not None:
            return _processed_symbol

        replaced_symbol = self._process_symbol(symbol)
        self._processed_symbols[symbol] = replaced_symbol
        return replaced_symbol

    def _process_symbol(self, symbol):
        """See :meth:`Simplification.process_symbol()`."""
        _processed_symbol = self._processed_symbols.get(symbol)
        if _processed_symbol is not None:
            return _processed_symbol

        if symbol in self._symbol_replacement_map:
            return self._symbol_replacement_map[symbol]

        if isinstance(symbol, pybamm.BinaryOperator):
            left, right = symbol.children
            # process children
            new_left = self.process_symbol(left)
            new_right = self.process_symbol(right)
            # Return a new copy with the replaced symbols
            return symbol._binary_new_copy(new_left, new_right)

        elif isinstance(symbol, pybamm.UnaryOperator):
            new_child = self.process_symbol(symbol.child)
            # Return a new copy with the replaced symbols
            return symbol._unary_new_copy(new_child)

        elif isinstance(symbol, pybamm.Function):
            new_children = [self.process_symbol(child) for child in symbol.children]
            # Return a new copy with the replaced symbols
            return symbol._function_new_copy(new_children)

        elif isinstance(symbol, pybamm.Concatenation):
            new_children = [self.process_symbol(child) for child in symbol.children]
            # Return a new copy with the replaced symbols
            return symbol._concatenation_new_copy(new_children)

        else:
            # Only other option is that the symbol is a leaf (doesn't have children)
            # In this case, since we have already ruled out that the symbol is one of
            # the symbols that needs to be replaced, we can just return the symbol
            return symbol


class VariableReplacementMap:
    """
    A simple dict-like object that efficiently resolves :class:`pybamm.Symbol`s by name.

    This class provides a lightweight alternative to using a full `dict` mapping
    :class:`pybamm.Symbol` objects to replacement symbols. Instead of requiring actual
    :class:`pybamm.Symbol` instances as keys (which can be expensive to create), this
    class uses variable names (strings) as keys and matches them to `pybamm.Variable`
    objects by name lookup.

    This avoids creating unnecessary :class:`pybamm.Symbol` objects, which can be
    expensive for many variables, while still providing dict-like access patterns.

    Parameters
    ----------
    symbol_replacement_map : dict[str, :class:`pybamm.Symbol`]
        Dictionary mapping variable names (strings) to their replacement symbols.
    """

    __slots__ = ["_symbol_replacement_map"]

    def __init__(self, symbol_replacement_map: dict[str, pybamm.Symbol]):
        self._symbol_replacement_map = symbol_replacement_map

    def __getitem__(self, symbol):
        return self._symbol_replacement_map[symbol.name]

    def __contains__(self, symbol):
        return self.get(symbol) is not None

    def get(self, symbol, default=None):
        if not isinstance(symbol, pybamm.Variable):
            return default

        name = symbol.name
        value = self._symbol_replacement_map.get(name)

        # Check exact variable match
        if value is not None and pybamm.Variable(name) == symbol:
            return value
        return default
