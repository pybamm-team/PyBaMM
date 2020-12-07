#
# Simplify a symbol
#
import pybamm


class SymbolReplacer(object):
    """
    Helper class to replace all instances of one or more symbols in an expression tree
    with another symbol, as defined by the dictionary `symbol_replacement_map`

    Parameters
    ----------
    symbol_replacement_map : dict {:class:`pybamm.Symbol` -> :class:`pybamm.Symbol`}
        Map of which symbols should be replaced by which.
    processed_symbols: dict {variable ids -> :class:`pybamm.Symbol`}, optional
        cached replaced symbols
    """

    def __init__(self, symbol_replacement_map, processed_symbols=None):
        self._symbol_replacement_map = symbol_replacement_map
        self._symbol_replacement_map_ids = {
            symbol_in.id: symbol_out
            for symbol_in, symbol_out in symbol_replacement_map.items()
        }
        self._processed_symbols = processed_symbols or {}

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
        pybamm.logger.info(
            "Start replacing symbols in {}".format(unprocessed_model.name)
        )

        # set up inplace vs not inplace
        if inplace:
            # any changes to unprocessed_model attributes will change model attributes
            # since they point to the same object
            model = unprocessed_model
        else:
            # create a blank model of the same class
            model = unprocessed_model.new_empty_copy()

        new_rhs = {}
        for variable, equation in unprocessed_model.rhs.items():
            pybamm.logger.debug("Replacing symbols in {!r} (rhs)".format(variable))
            new_rhs[self.process_symbol(variable)] = self.process_symbol(equation)
        model.rhs = new_rhs

        new_algebraic = {}
        for variable, equation in unprocessed_model.algebraic.items():
            pybamm.logger.debug(
                "Replacing symbols in {!r} (algebraic)".format(variable)
            )
            new_algebraic[self.process_symbol(variable)] = self.process_symbol(equation)
        model.algebraic = new_algebraic

        new_initial_conditions = {}
        for variable, equation in unprocessed_model.initial_conditions.items():
            pybamm.logger.debug(
                "Replacing symbols in {!r} (initial conditions)".format(variable)
            )
            new_initial_conditions[self.process_symbol(variable)] = self.process_symbol(
                equation
            )
        model.initial_conditions = new_initial_conditions

        model.boundary_conditions = self.process_boundary_conditions(unprocessed_model)

        new_variables = {}
        for variable, equation in unprocessed_model.variables.items():
            pybamm.logger.debug(
                "Replacing symbols in {!r} (variables)".format(variable)
            )
            new_variables[variable] = self.process_symbol(equation)
        model.variables = new_variables

        new_events = []
        for event in unprocessed_model.events:
            pybamm.logger.debug("Replacing symbols in event'{}''".format(event.name))
            new_events.append(
                pybamm.Event(
                    event.name, self.process_symbol(event.expression), event.event_type
                )
            )
        model.events = new_events

        # Set external variables
        model.external_variables = [
            self.process_symbol(var) for var in unprocessed_model.external_variables
        ]

        # Process timescale
        model.timescale = self.process_symbol(unprocessed_model.timescale)

        # Process length scales
        new_length_scales = {}
        for domain, scale in unprocessed_model.length_scales.items():
            new_length_scales[domain] = self.process_symbol(scale)
        model.length_scales = new_length_scales

        pybamm.logger.info("Finish replacing symbols in {}".format(model.name))

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
                        "Replacing symbols in {!r} ({} bc)".format(variable, side)
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
                        raise KeyError(err)

        return new_boundary_conditions

    def process_symbol(self, symbol):
        """
        This function recurses down the tree, replacing any symbols in
        self._symbol_replacement_map.keys() with their corresponding value

        Parameters
        ----------
        symbol : :class:`pybamm.Symbol`
            The symbol to replace

        Returns
        -------
        :class:`pybamm.Symbol`
            Symbol with all replacements performed
        """

        try:
            return self._processed_symbols[symbol.id]
        except KeyError:
            replaced_symbol = self._process_symbol(symbol)

            self._processed_symbols[symbol.id] = replaced_symbol

            return replaced_symbol

    def _process_symbol(self, symbol):
        """ See :meth:`Simplification.process_symbol()`. """
        if symbol.id in self._symbol_replacement_map_ids.keys():
            return self._symbol_replacement_map_ids[symbol.id]

        elif isinstance(symbol, pybamm.BinaryOperator):
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
            # the symbols that needs to be replaced, we can just return a new copy of
            # the symbol
            return symbol.new_copy()
