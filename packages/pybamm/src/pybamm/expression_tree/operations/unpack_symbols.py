#
# Helper function to unpack a symbol
#
from __future__ import annotations

from collections.abc import Generator, Sequence

import pybamm


class SymbolUnpacker:
    """
    Helper class to unpack a (set of) symbol(s) to find all instances of a class.
    Uses caching to speed up the process.

    Parameters
    ----------
    classes_to_find : list of pybamm classes
        Classes to identify in the equations
    unpacked_symbols: set, optional
        cached unpacked equations
    """

    __slots__ = ("_unpacked_symbols", "classes_to_find")

    def __init__(
        self,
        classes_to_find: Sequence[pybamm.Symbol] | type[pybamm.Symbol],
        unpacked_symbols: dict | None = None,
    ):
        self.classes_to_find = classes_to_find
        self._unpacked_symbols: dict = unpacked_symbols or {}

    def unpack_list_of_symbols(
        self,
        list_of_symbols: Sequence[pybamm.Symbol] | Generator[pybamm.Symbol, None, None],
    ) -> set[pybamm.Symbol]:
        """
        Unpack a list of symbols. See :meth:`SymbolUnpacker.unpack()`

        Parameters
        ----------
        list_of_symbols : list of :class:`pybamm.Symbol`
            List of symbols to unpack

        Returns
        -------
        set of :class:`pybamm.Symbol`
            Set of unpacked symbols with class in `self.classes_to_find`
        """
        all_instances = set()
        for symbol in list_of_symbols:
            new_instances = self.unpack_symbol(symbol)
            all_instances.update(new_instances)

        return all_instances

    def unpack_parameter_values(
        self, parameter_values: pybamm.ParameterValues | dict
    ) -> set[pybamm.Symbol]:
        """
        Unpack a parameter values object.
        """
        return self.unpack_list_of_symbols(
            v for v in parameter_values.values() if isinstance(v, pybamm.Symbol)
        )

    def unpack_symbol(
        self, symbol: Sequence[pybamm.Symbol] | pybamm.Symbol
    ) -> set[pybamm.Symbol]:
        """
        This function recurses down the tree, unpacking the symbols and saving the ones
        that have a class in `self.classes_to_find`.

        Parameters
        ----------
        symbol : list of :class:`pybamm.Symbol`
            The symbols to unpack

        Returns
        -------
        set of :class:`pybamm.Symbol`
            Set of unpacked symbols with class in `self.classes_to_find`
        """

        value = self._unpacked_symbols.get(symbol)
        if value is not None:
            return value

        unpacked = self._unpack(symbol)
        self._unpacked_symbols[symbol] = unpacked
        return unpacked

    def _unpack(self, symbol) -> set[pybamm.Symbol]:
        """See :meth:`SymbolUnpacker.unpack()`."""
        # found a symbol of the right class -> return it
        if isinstance(symbol, self.classes_to_find):
            return {symbol}

        # flatten the per-dependent sets into one set
        return {found for leaf in symbol.leaves for found in self.unpack_symbol(leaf)}
