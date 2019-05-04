#
# Processed Variable class
#
import numbers
import numpy as np
import scipy.interpolate as interp


def post_process_variables(variables, t_sol, y_sol, mesh=None, interp_kind="linear"):
    """
    Post-process all variables in a model

    Parameters
    ----------
    variables : dict
        Dictionary of variables
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

    Returns
    -------
    dict
        Dictionary of processed variables
    """
    processed_variables = {}
    for var, eqn in variables.items():
        try:
            processed_variables[var] = ProcessedVariable(
                eqn, t_sol, y_sol, mesh, interp_kind
            )
        except ValueError:
            print("'{}' was not processed".format(var))
            pass
    return processed_variables


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
    """

    def __init__(self, base_variable, t_sol, y_sol, mesh=None, interp_kind="linear"):
        self.base_variable = base_variable
        self.t_sol = t_sol
        self.y_sol = y_sol
        self.mesh = mesh
        self.interp_kind = interp_kind

        self.base_eval = base_variable.evaluate(t_sol[0], y_sol[:, 0])

        if isinstance(self.base_eval, numbers.Number):
            self.type = "number"
            self.entries = self.base_eval * np.ones_like(t_sol)
        elif len(self.base_eval.shape) == 0 or self.base_eval.shape[0] == 1:
            self.initialise_vector()
        else:
            self.initialise_matrix()

    def initialise_vector(self):
        self.type = "vector"
        # initialise empty array of the correct size
        entries = np.empty(len(self.t_sol))
        # Evaluate the base_variable index-by-index
        for idx in range(len(self.t_sol)):
            entries[idx] = self.base_variable.evaluate(
                self.t_sol[idx], self.y_sol[:, idx]
            )

        # No discretisation provided, or variable has no domain (function of t only)
        self._interpolation_function = interp.interp1d(
            self.t_sol, entries, kind=self.interp_kind
        )

        self.entries = entries

    def initialise_matrix(self):
        self.type = "matrix"
        len_x = self.base_eval.shape[0]
        entries = np.empty((len_x, len(self.t_sol)))

        # Evaluate the base_variable index-by-index
        for idx in range(len(self.t_sol)):
            entries[:, idx] = self.base_variable.evaluate(
                self.t_sol[idx], self.y_sol[:, idx]
            )

        if self.mesh is not None:
            # Process the discretisation to get x values
            nodes = self.mesh.combine_submeshes(*self.base_variable.domain)[0].nodes
            edges = self.mesh.combine_submeshes(*self.base_variable.domain)[0].edges
            if entries.shape[0] == len(nodes):
                x_sol = nodes
            elif entries.shape[0] == len(edges) - 2:
                x_sol = edges[1:-1]
            else:
                raise ValueError
        else:
            # We must provide a mesh for reference x values  for interpolation
            raise ValueError("mesh must be provided for intepolation")

        # assign attributes for reference
        self.entries = entries
        self.x_sol = x_sol

        # set up interpolation
        self._interpolation_function = interp.interp2d(
            self.t_sol, x_sol, entries, kind=self.interp_kind
        )

    def __call__(self, t, x=None):
        "Evaluate the variable at arbitrary t (and x), using interpolation"
        if self.type == "number":
            return self.value * np.ones_like(t)
        elif self.type == "vector":
            out = self._interpolation_function(t)
            # make sure the output is 1D
            if len(out.shape) == 2:
                return out[0]
            else:
                return out
        elif self.type == "matrix":
            return self._interpolation_function(t, x)
