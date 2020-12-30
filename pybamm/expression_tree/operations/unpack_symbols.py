#
# Helper function to unpack a symbol
#


class SymbolUnpacker(object):
    """
    Helper class to unpack a (set of) symbol(s) to find all instances of a class.
    Uses caching to speed up the process.

    Parameters
    ----------
    classes_to_find : list of pybamm classes
        Classes to identify in the equations
    unpacked_symbols: dict {variable ids -> :class:`pybamm.Symbol`}, optional
        cached unpacked equations
    """

    def __init__(self, classes_to_find, unpacked_symbols=None):
        self.classes_to_find = classes_to_find
        self._unpacked_symbols = unpacked_symbols or {}

    def unpack_list_of_symbols(self, list_of_symbols):
        """
        Unpack a list of symbols. See :meth:`SymbolUnpacker.unpack()`

        Parameters
        ----------
        list_of_symbols : list of :class:`pybamm.Symbol`
            List of symbols to unpack

        Returns
        -------
        list of :class:`pybamm.Symbol`
            List of unpacked symbols with class in `self.classes_to_find`
        """
        all_instances = {}
        for symbol in list_of_symbols:
            new_instances = self.unpack_symbol(symbol)
            all_instances.update(new_instances)

        return all_instances

    def unpack_symbol(self, symbol):
        """
        This function recurses down the tree, unpacking the symbols and saving the ones
        that have a class in `self.classes_to_find`.

        Parameters
        ----------
        symbol : list of :class:`pybamm.Symbol`
            The symbols to unpack

        Returns
        -------
        list of :class:`pybamm.Symbol`
            List of unpacked symbols with class in `self.classes_to_find`
        """

        try:
            return self._unpacked_symbols[symbol.id]
        except KeyError:
            unpacked = self._unpack(symbol)
            self._unpacked_symbols[symbol.id] = unpacked
            return unpacked

    def _unpack(self, symbol):
        """ See :meth:`SymbolUnpacker.unpack()`. """

        children = symbol.children

        # If symbol has no children, just check its class
        if len(children) == 0:
            # found a symbol of the right class -> return it
            if isinstance(symbol, self.classes_to_find):
                return {symbol.id: symbol}
            # otherwise return empty dictionary
            else:
                return {}

        else:
            # iterate over all children
            found_vars = {}
            for child in children:
                # call back unpack_symbol to cache values
                child_vars = self.unpack_symbol(child)
                found_vars.update(child_vars)
            return found_vars
