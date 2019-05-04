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
        self.domain = base_variable.domain

        self.base_eval = base_variable.evaluate(t_sol[0], y_sol[:, 0])

        if isinstance(self.base_eval, numbers.Number):
            self.dimensions = 0
            self.entries = self.base_eval * np.ones_like(t_sol)
        elif len(self.base_eval.shape) == 0 or self.base_eval.shape[0] == 1:
            self.initialise_1D()
        else:
            if len(self.mesh.combine_submeshes(*self.domain)) == 1:
                self.initialise_2D()
            else:
                self.initialise_3D()

    def initialise_1D(self):
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
        self.dimensions = 1

    def initialise_2D(self):
        len_space = self.base_eval.shape[0]
        entries = np.empty((len_space, len(self.t_sol)))

        # Evaluate the base_variable index-by-index
        for idx in range(len(self.t_sol)):
            entries[:, idx] = self.base_variable.evaluate(
                self.t_sol[idx], self.y_sol[:, idx]
            )

        # Process the discretisation to get x values
        nodes = self.mesh.combine_submeshes(*self.domain)[0].nodes
        edges = self.mesh.combine_submeshes(*self.domain)[0].edges
        if entries.shape[0] == len(nodes):
            space = nodes
        elif entries.shape[0] == len(edges) - 2:
            space = edges[1:-1]
        else:
            raise ValueError

        # assign attributes for reference (either x_sol or r_sol)
        self.entries = entries
        self.dimensions = 2
        if any("particle" in dom for dom in self.domain):
            self.scale = "micro"
            self.r_sol = space
        else:
            self.scale = "macro"
            self.x_sol = space

        # set up interpolation
        # note that the order of 't' and 'space' is the reverse of what you'd expect
        self._interpolation_function = interp.interp2d(
            self.t_sol, space, entries, kind=self.interp_kind
        )

    def initialise_3D(self):
        len_x = len(self.mesh.combine_submeshes(*self.domain))
        len_r = self.base_eval.shape[0] // len_x
        entries = np.empty((len_r, len_x, len(self.t_sol)))

        # Evaluate the base_variable index-by-index
        for idx in range(len(self.t_sol)):
            entries[:, :, idx] = np.reshape(
                self.base_variable.evaluate(self.t_sol[idx], self.y_sol[:, idx]),
                [len_r, len_x],
            )
        # Process the discretisation to get x values
        nodes = self.mesh.combine_submeshes(*self.domain)[0].nodes
        edges = self.mesh.combine_submeshes(*self.domain)[0].edges
        if entries.shape[0] == len(nodes):
            r_sol = nodes
        elif entries.shape[0] == len(edges) - 2:
            r_sol = edges[1:-1]
        else:
            raise ValueError

        # Get x values
        if self.domain == ["negative particle"]:
            x_sol = self.mesh["negative electrode"][0].nodes
        elif self.domain == ["positive particle"]:
            x_sol = self.mesh["positive electrode"][0].nodes

        # assign attributes for reference
        self.entries = entries
        self.dimensions = 3
        self.x_sol = x_sol
        self.r_sol = r_sol

        # set up interpolation
        self._interpolation_function = interp.RegularGridInterpolator(
            (r_sol, x_sol, self.t_sol), entries, method=self.interp_kind
        )

    def __call__(self, t, x=None, r=None):
        "Evaluate the variable at arbitrary t (and x and/or r), using interpolation"
        if self.dimensions == 0:
            return self.value * np.ones_like(t)
        elif self.dimensions == 1:
            return self._interpolation_function(t)
        elif self.dimensions == 2:
            if self.scale == "micro":
                return self._interpolation_function(t, r)
            else:
                return self._interpolation_function(t, x)
        elif self.dimensions == 3:
            import ipdb

            ipdb.set_trace()
            return self._interpolation_function(t, x, r)
