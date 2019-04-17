#
# Processed Variable class
#
import numpy as np
import scipy.interpolate as interp


class ProcessedVariable(object):
    """
    An object that can be evaluated at arbitrary (scalars or vectors) t and x, and
    returns the (interpolated) value of the base variable at that t and x.

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
    interp_kind : str
        The method to use for interpolation

    Notes
    -----
    The default behaviour of :meth:`scipy.interpolate.interp1d` and
    :meth:`scipy.interpolate.interp2d` is followed for extrapolation to points outside
    the domain:

    - in 1D, a `ValueError` is raised if a point outside the interpolation range is \
        requested. The standard use case for 1D interpolation is a function of time, \
        which we never expect to extrapolate outside the domain
    - in 2D, the nearest neighbour is returned if a point outside the interpolation \
        range is requested. This is useful since the x values used for interpolation \
        don't always go from x=0 to x=1 (e.g. if x is nodes in a Finite Volumes mesh). \
        However, this can lead to values far from the interpolation range being very \
        inaccurate, without any errors being raised.

    A possible improvement in 2D would be to calculate the values at x=0 and x=1, add
    these to the `entries` matrix, and then disallow extrapolation outside of the
    interpolation range

    """

    def __init__(self, base_variable, t_sol, y_sol, mesh=None, interp_kind="linear"):
        if base_variable.domain != []:
            if mesh is not None:
                # Process the discretisation to get x values
                x_sol = np.concatenate(
                    [mesh[dom][0].nodes for dom in base_variable.domain]
                )
                len_x = len(x_sol)
            else:
                # We must provide a mesh for reference x values  for interpolation
                raise ValueError("mesh must be provided for intepolation")
        else:
            # No discretisation provided, or variable has no domain (function of t only)
            # We don't need x values for interpolation
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
        if x_sol is None:
            self._interpolation_function = interp.interp1d(
                t_sol, entries, kind=interp_kind
            )
        else:
            self._interpolation_function = interp.interp2d(
                t_sol, x_sol, entries, kind=interp_kind
            )

    def __call__(self, t, x=None):
        "Evaluate the variable at arbitrary t (and x), using interpolation"
        if self.x_sol is None:
            return self._interpolation_function(t)
        else:
            return self._interpolation_function(t, x)
