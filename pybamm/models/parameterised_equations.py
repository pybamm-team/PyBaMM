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
        self._parameter_values = parameter_values.copy_with_processed_symbols()

        super().__init__(*args)

    def variables_update_function(self, variable):
        return self._parameter_values.process_symbol(variable)
