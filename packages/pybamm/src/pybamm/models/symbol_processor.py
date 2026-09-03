#
# SymbolProcessor class for processing symbols with parameter values and discretisation
#
from __future__ import annotations

import copy

import pybamm


class SymbolProcessor:
    """
    Processes symbols for a model by applying parameter values and discretisation.

    This class provides a convenient way to process symbols using both
    :class:`pybamm.ParameterValues` and :class:`pybamm.Discretisation` objects.
    Once both are set, calling the processor on a symbol will first substitute
    parameters, then discretise the result. :meth:`resolve` turns a symbol written
    against variable names (:class:`pybamm.CoupledVariable`) into one over the model's
    own symbols, ready for either.

    Attributes
    ----------
    parameter_values : pybamm.ParameterValues or None
        The parameter values used to process symbols.
    discretisation : pybamm.Discretisation or None
        The discretisation used to process symbols.
    can_process_symbols : bool
        Whether symbol processing is enabled.

    Examples
    --------
    >>> processor = pybamm.SymbolProcessor()  # doctest: +SKIP
    >>> processor.parameter_values = param  # doctest: +SKIP
    >>> processor.discretisation = disc  # doctest: +SKIP
    >>> processed_symbol = processor("my variable", symbol)  # doctest: +SKIP
    """

    def __init__(self):
        """Initialize a new SymbolProcessor instance."""
        self._can_process_symbols = True
        self._discretisation = None
        self._parameter_values = None

    @staticmethod
    def resolve(symbol: pybamm.Symbol, variables: dict) -> pybamm.Symbol:
        """Replace each CoupledVariable in ``symbol`` with the entry of ``variables``
        of that name, so that a symbol written against variable names can be
        processed like any other model equation."""
        resolve = pybamm.SymbolProcessor.resolve
        if isinstance(symbol, pybamm.CoupledVariable):
            if symbol.name not in variables:
                raise ValueError(
                    f"CoupledVariable '{symbol.name}' not found in model.variables"
                )
            return resolve(variables[symbol.name], variables)
        if symbol.children:
            children = [resolve(child, variables) for child in symbol.children]
            if any(
                new is not old
                for new, old in zip(children, symbol.children, strict=True)
            ):
                return symbol.create_copy(new_children=children)
        return symbol

    def __call__(self, name: str, symbol: pybamm.Symbol) -> pybamm.Symbol:
        """
        Process a symbol by applying parameter values and discretisation.

        Parameters
        ----------
        name : str
            The name of the symbol (used for discretisation).
        symbol : pybamm.Symbol
            The symbol to process.

        Returns
        -------
        pybamm.Symbol
            The processed symbol.

        Raises
        ------
        ValueError
            If the processor cannot process symbols.
        """
        if not bool(self):
            raise ValueError(
                "Cannot process a symbol if neither `parameter_values` nor "
                "`discretisation` have been set."
            )
        if self.parameter_values:
            symbol = self.parameter_values.process_symbol(symbol)
        if self.discretisation:
            symbol = self.discretisation.process_equation(name, eqn=symbol)
        return symbol

    def __bool__(self) -> bool:
        """Return True if the processor can process symbols."""
        return (
            self._can_process_symbols
            and self._discretisation is not None
            and self._parameter_values is not None
        )

    @property
    def discretisation(self) -> pybamm.Discretisation | None:
        return self._discretisation

    @discretisation.setter
    def discretisation(self, discretisation: pybamm.Discretisation | None):
        if not isinstance(discretisation, pybamm.Discretisation):
            raise ValueError(
                "`discretisation` must be a `pybamm.Discretisation` object."
            )
        if self._discretisation is not None:
            self._can_process_symbols = False
        self._discretisation = copy.copy(discretisation)

    @property
    def parameter_values(self) -> pybamm.ParameterValues | None:
        return self._parameter_values

    @parameter_values.setter
    def parameter_values(self, parameter_values: pybamm.ParameterValues | None):
        if not isinstance(parameter_values, pybamm.ParameterValues):
            raise ValueError(
                "`parameter_values` must be a `pybamm.ParameterValues` object."
            )
        if self._parameter_values is not None:
            self._can_process_symbols = False
        self._parameter_values = parameter_values.copy()

    @property
    def can_process_symbols(self) -> bool:
        return bool(self)

    def disable(self):
        """Disable symbol processing."""
        self._can_process_symbols = False

    def copy(self):
        """Return a copy of this SymbolProcessor."""
        new_symbol_processor = SymbolProcessor()
        new_symbol_processor._can_process_symbols = self._can_process_symbols
        if self._discretisation is not None:
            new_symbol_processor._discretisation = copy.copy(self._discretisation)
        if self._parameter_values is not None:
            new_symbol_processor._parameter_values = self._parameter_values.copy()
        return new_symbol_processor
