#
# Dummy solver class, for empty models
#
import pybamm
import numpy as np


class DummySolver(pybamm.BaseSolver):
    """Dummy solver class for empty models. """

    def __init__(self):
        super().__init__()
        self.name = "Dummy solver"

    def _integrate(self, model, t_eval, inputs=None):
        """
        Solve an empty model.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate.
        t_eval : :class:`numpy.array`, size (k,)
            The times at which to compute the solution
        inputs : dict, optional
            Any input parameters to pass to the model when solving

        Returns
        -------
        object
            An object containing the times and values of the solution, as well as
            various diagnostic messages.

        """
        y_sol = np.zeros((1, t_eval.size))
        return pybamm.Solution(t_eval, y_sol, termination="final time")
