#
# Processed Variable class
#

import numpy as np


class ProcessedVariable(object):
    """
    An object that can be evaluated at arbitrary (scalars or vectors) t and x, and
    returns the (interpolated) value of the base variable at that t and x

    Parameters
    ----------
    base_variable : :class:`pybamm.Symbol`
        A base variable with a method `evaluate(t,y)` that returns the value of that
        variable. Note that this can be any kind of node in the expression tree, not
        just a :class:`pybamm.Variable`.
        When evaluated, returns an array of size (m,n)
    x_sol : array_like, size (n,)

    t_sol : array_like, size (m,)
        The time vector returned by the solver
    y_sol : array_like, size (m, k)
        The solution vector returned by the solver. Can include solution values that
        other than those that get read by base_variable.evaluate() (i.e. k>=n)
    """

    def __init__(self, base_variable, t_sol, y_sol, x_sol=(None,)):
        # initialise empty array of the correct size
        entries = np.empty((len(x_sol), len(t_sol)))
        # Evaluate the base_variable index-by-index
        for idx in range(len(t_sol)):
            entries[:, idx] = base_variable.evaluate(t_sol[idx], y_sol[:, idx])

        # assign attributes
        self.entries = entries
        self.x_sol = x_sol
        self.t_sol = t_sol
