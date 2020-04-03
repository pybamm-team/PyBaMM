#
# Solution class
#
import pybamm


class CasadiSolution(pybamm._BaseSolution):
    """
    (Semi-private) class containing the solution of, and various attributes associated
    with, a PyBaMM model, in the case where the solution depends on as-yet-unspecified
    inputs. This class is automatically created by the `CasadiSolution`class,
    and should never be called from outside the `CasadiSolution` class.

    **Extends**: :class:`pybamm._BaseSolution`
    """

    def __init__(
        self, t, y, t_event=None, y_event=None, termination="final time", copy_this=None
    ):
        super().__init__(t, y, t_event, y_event, termination, copy_this)

    @property
    def inputs(self):
        "Values of the inputs"
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        "Updates the input values"
        self._inputs = inputs

    def update(self, variables):
        """Add ProcessedVariables to the dictionary of variables in the solution"""
        # Convert single entry to list
        if isinstance(variables, str):
            variables = [variables]
        # Process
        for key in variables:
            pybamm.logger.debug("Post-processing {}".format(key))
            var = pybamm.ProcessedCasadiVariable(self.model.variables[key], self)

            # Save variable and data
            self._variables[key] = var

