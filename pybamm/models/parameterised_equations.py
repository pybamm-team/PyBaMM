#
# Parameterized equations class
#
import pybamm


class _ParameterisedEquations(pybamm._BaseProcessedEquations):
    """
    Class containing equations with parameters set.

    **Extends:** :class:`pybamm.BaseProcessedEquations`
    """

    def __init__(self, parameter_values, *args):
        # Save parameter values used to create this model
        self._parameter_values = parameter_values

        super().__init__(*args)

    def variables_update_function(self, variable):
        return self._parameter_values.process_symbol(variable)


class _ReplacedEquations(pybamm._BaseProcessedEquations):
    """
    Class containing equations with replacement performed.

    **Extends:** :class:`pybamm.BaseProcessedEquations`
    """

    def __init__(self, symbol_replacer, *args):
        # Save parameter values used to create this model
        self._symbol_replacer = symbol_replacer

        super().__init__(*args)

    def variables_update_function(self, variable):
        return self._symbol_replacer.process_symbol(variable)
