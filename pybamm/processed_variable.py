#
# Processed Variable class
#
import numpy as np
import scipy.interpolate as interp


class ProcessedVariable(object):
    """
    An object that can be evaluated at arbitrary (scalars or vectors) x and t, and
    returns the (interpolated) value of the base variable at that x and t.

    Parameters
    ----------
    base_variable : :class:`pybamm.Symbol`
        A base variable with a method `evaluate(t,y)` that returns the value of that
        variable. Note that this can be any kind of node in the expression tree, not
        just a :class:`pybamm.Variable`.
        When evaluated, returns an array of size (m,n)
    t_sol : array_like, size (m,)
        The time vector returned by the solver
    y_sol : array_like, size (m, k)
        The solution vector returned by the solver. Can include solution values that
        other than those that get read by base_variable.evaluate() (i.e. k>=n)
    mesh : :class:`pybamm.Mesh`
        The mesh used to solve, used here to calculate the reference x values for
        interpolation
    interp_kind : string
        The method to use for interpolation

    """

    def __init__(self, base_variable, t_sol, y_sol, mesh=None, interp_kind="linear"):
        if mesh is not None and base_variable.domain != []:
            # Process the discretisation to get x values
            x_sol = np.concatenate([mesh[dom][0].nodes for dom in base_variable.domain])
            len_x = len(x_sol)
        else:
            # No discretisation provided, or variable has no domain (function of t only)
            x_sol = None
            len_x = 1

        # initialise empty array of the correct size
        entries = np.empty((len_x, len(t_sol)))
        # Evaluate the base_variable index-by-index
        for idx in range(len(t_sol)):
            entries[:, idx] = base_variable.evaluate(t_sol[idx], y_sol[:, idx])

        # assign attributes for reference
        self.entries = entries
        self.x_sol = x_sol
        self.t_sol = t_sol

        # set up interpolation
        # self._interpolation_function = interp.interp2d(
        #     x_sol, t_sol, entries, kind=interp_kind
        # )

    def evaluate(self, x, t):
        "Evaluate the variable at arbitrary x and t, using interpolation"
        return self._interpolation_function(x, t)
