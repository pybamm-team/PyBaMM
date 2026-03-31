"""Internal utilities for EIS simulation.

This module contains the SymbolReplacer class used to prepare models for
frequency-domain EIS calculations.
"""

import pybamm


class SymbolReplacer:
    """Replace symbols in a PyBaMM expression tree according to a mapping.

    Parameters
    ----------
    symbol_replacement_map : dict
        Map of ``{old_symbol: new_symbol}`` pairs.
    processed_symbols : dict, optional
        Cache of already-processed symbols.
    process_initial_conditions : bool, optional
        Whether to process initial conditions (default True).
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
        """Replace all mapped symbols throughout a model.

        Parameters
        ----------
        unprocessed_model : :class:`pybamm.BaseModel`
            The model to process.
        inplace : bool, optional
            If True, modify the model in place. Otherwise return a new copy.
        """
        if inplace:
            model = unprocessed_model
        else:
            model = unprocessed_model.new_copy()

        model.rhs = {
            self.process_symbol(var): self.process_symbol(eq)
            for var, eq in unprocessed_model.rhs.items()
        }

        model.algebraic = {
            self.process_symbol(var): self.process_symbol(eq)
            for var, eq in unprocessed_model.algebraic.items()
        }

        new_ics = {}
        for var, eq in unprocessed_model.initial_conditions.items():
            new_var = self.process_symbol(var)
            new_ics[new_var] = (
                self.process_symbol(eq) if self.process_initial_conditions else eq
            )
        model.initial_conditions = new_ics

        model.boundary_conditions = self._process_boundary_conditions(
            unprocessed_model
        )

        model.variables = {
            name: self.process_symbol(eq)
            for name, eq in unprocessed_model.variables.items()
        }

        model.events = [
            pybamm.Event(
                event.name,
                self.process_symbol(event.expression),
                event.event_type,
            )
            for event in unprocessed_model.events
        ]

        return model

    def _process_boundary_conditions(self, model):
        new_bcs = {}
        for variable, bcs in model.boundary_conditions.items():
            processed_variable = self.process_symbol(variable)
            new_bcs[processed_variable] = {}
            for side, (bc, typ) in bcs.items():
                new_bcs[processed_variable][side] = (
                    self.process_symbol(bc),
                    typ,
                )

        return new_bcs

    def process_symbol(self, symbol):
        """Process a single symbol, using cache for deduplication."""
        try:
            return self._processed_symbols[symbol]
        except KeyError:
            result = self._process_symbol(symbol)
            self._processed_symbols[symbol] = result
            return result

    def _process_symbol(self, symbol):
        if symbol in self._symbol_replacement_map:
            return self._symbol_replacement_map[symbol]

        if isinstance(symbol, pybamm.BinaryOperator):
            left, right = symbol.children
            return symbol._binary_new_copy(
                self.process_symbol(left), self.process_symbol(right)
            )

        if isinstance(symbol, pybamm.UnaryOperator):
            return symbol._unary_new_copy(self.process_symbol(symbol.child))

        if isinstance(symbol, pybamm.Function):
            new_children = [self.process_symbol(c) for c in symbol.children]
            return symbol._function_new_copy(new_children)

        if isinstance(symbol, pybamm.Concatenation):
            new_children = [self.process_symbol(c) for c in symbol.children]
            return symbol._concatenation_new_copy(new_children)

        return symbol
