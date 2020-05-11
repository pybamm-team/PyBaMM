#
# Processed Variable class
#
import numbers
import numpy as np
import pybamm
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
    solution : :class:`pybamm.Solution`
        The solution object to be used to create the processed variables
    known_evals : dict
        Dictionary of known evaluations, to be used to speed up finding the solution
    """

    def __init__(self, base_variable, solution, known_evals=None):
        self.base_variable = base_variable
        self.t_sol = solution.t
        self.u_sol = solution.y
        self.mesh = base_variable.mesh
        self.inputs = solution.inputs
        self.domain = base_variable.domain
        self.auxiliary_domains = base_variable.auxiliary_domains
        self.known_evals = known_evals

        if self.known_evals:
            self.base_eval, self.known_evals[solution.t[0]] = base_variable.evaluate(
                solution.t[0],
                solution.y[:, 0],
                inputs={name: inp[0] for name, inp in solution.inputs.items()},
                known_evals=self.known_evals[solution.t[0]],
            )
        else:
            self.base_eval = base_variable.evaluate(
                solution.t[0],
                solution.y[:, 0],
                inputs={name: inp[0] for name, inp in solution.inputs.items()},
            )

        # handle 2D (in space) finite element variables differently
        if (
            self.mesh
            and "current collector" in self.domain
            and isinstance(self.mesh[0], pybamm.ScikitSubMesh2D)
        ):
            if len(solution.t) == 1:
                # space only (steady solution)
                self.initialise_2D_fixed_t_scikit_fem()
            else:
                self.initialise_2D_scikit_fem()

        # check variable shape
        else:
            if len(solution.t) == 1:
                raise pybamm.SolverError(
                    "Solution time vector must have length > 1. Check whether "
                    "simulation terminated too early."
                )
            elif (
                isinstance(self.base_eval, numbers.Number)
                or len(self.base_eval.shape) == 0
                or self.base_eval.shape[0] == 1
            ):
                self.initialise_0D()
            else:
                n = self.mesh[0].npts
                base_shape = self.base_eval.shape[0]
                # Try some shapes that could make the variable a 1D variable
                if base_shape in [n, n + 1]:
                    self.initialise_1D()
                else:
                    # Try some shapes that could make the variable a 2D variable
                    first_dim_nodes = self.mesh[0].nodes
                    first_dim_edges = self.mesh[0].edges
                    second_dim_pts = self.base_variable.secondary_mesh[0].nodes
                    if self.base_eval.size // len(second_dim_pts) in [
                        len(first_dim_nodes),
                        len(first_dim_edges),
                    ]:
                        self.initialise_2D()
                    else:
                        # Raise error for 3D variable
                        raise NotImplementedError(
                            "Shape not recognized for {} ".format(base_variable)
                            + "(note processing of 3D variables is not yet implemented)"
                        )

    def initialise_0D(self):
        # initialise empty array of the correct size
        entries = np.empty(len(self.t_sol))
        # Evaluate the base_variable index-by-index
        for idx in range(len(self.t_sol)):
            t = self.t_sol[idx]
            u = self.u_sol[:, idx]
            inputs = {name: inp[idx] for name, inp in self.inputs.items()}
            if self.known_evals:
                entries[idx], self.known_evals[t] = self.base_variable.evaluate(
                    t, u, inputs=inputs, known_evals=self.known_evals[t]
                )
            else:
                entries[idx] = self.base_variable.evaluate(t, u, inputs=inputs)

        # No discretisation provided, or variable has no domain (function of t only)
        self._interpolation_function = interp.interp1d(
            self.t_sol, entries, kind="linear", fill_value=np.nan, bounds_error=False
        )

        self.entries = entries
        self.dimensions = 0

    def initialise_1D(self):
        len_space = self.base_eval.shape[0]
        entries = np.empty((len_space, len(self.t_sol)))

        # Evaluate the base_variable index-by-index
        for idx in range(len(self.t_sol)):
            t = self.t_sol[idx]
            u = self.u_sol[:, idx]
            inputs = {name: inp[idx] for name, inp in self.inputs.items()}
            if self.known_evals:
                eval_and_known_evals = self.base_variable.evaluate(
                    t, u, inputs=inputs, known_evals=self.known_evals[t]
                )
                entries[:, idx] = eval_and_known_evals[0][:, 0]
                self.known_evals[t] = eval_and_known_evals[1]
            else:
                entries[:, idx] = self.base_variable.evaluate(t, u, inputs=inputs)[:, 0]

        # Get node and edge values
        nodes = self.mesh[0].nodes
        edges = self.mesh[0].edges
        if entries.shape[0] == len(nodes):
            space = nodes
        elif entries.shape[0] == len(edges):
            space = edges

        # add points outside domain for extrapolation to boundaries
        extrap_space_left = np.array([2 * space[0] - space[1]])
        extrap_space_right = np.array([2 * space[-1] - space[-2]])
        space = np.concatenate([extrap_space_left, space, extrap_space_right])
        extrap_entries_left = 2 * entries[0] - entries[1]
        extrap_entries_right = 2 * entries[-1] - entries[-2]
        entries_for_interp = np.vstack(
            [extrap_entries_left, entries, extrap_entries_right]
        )

        # assign attributes for reference (either x_sol or r_sol)
        self.entries = entries
        self.dimensions = 1
        if self.domain[0] in ["negative particle", "positive particle"]:
            self.first_dimension = "r"
            self.r_sol = space
        elif self.domain[0] in [
            "negative electrode",
            "separator",
            "positive electrode",
        ]:
            self.first_dimension = "x"
            self.x_sol = space
        elif self.domain == ["current collector"]:
            self.first_dimension = "z"
            self.z_sol = space
        else:
            self.first_dimension = "x"
            self.x_sol = space

        self.first_dim_pts = edges
        self.internal_boundaries = self.mesh[0].internal_boundaries

        # set up interpolation
        # note that the order of 't' and 'space' is the reverse of what you'd expect

        self._interpolation_function = interp.interp2d(
            self.t_sol, space, entries_for_interp, kind="linear", fill_value=np.nan
        )

    def initialise_2D(self):
        """
        Initialise a 2D object that depends on x and r, or x and z.
        """
        first_dim_nodes = self.mesh[0].nodes
        first_dim_edges = self.mesh[0].edges
        second_dim_pts = self.base_variable.secondary_mesh[0].nodes
        if self.base_eval.size // len(second_dim_pts) == len(first_dim_nodes):
            first_dim_pts = first_dim_nodes
        elif self.base_eval.size // len(second_dim_pts) == len(first_dim_edges):
            first_dim_pts = first_dim_edges

        # Process r-x or x-z
        if self.domain[0] in [
            "negative particle",
            "positive particle",
        ] and self.auxiliary_domains["secondary"][0] in [
            "negative electrode",
            "positive electrode",
        ]:
            self.first_dimension = "r"
            self.second_dimension = "x"
            self.r_sol = first_dim_pts
            self.x_sol = second_dim_pts
        elif self.domain[0] in [
            "negative electrode",
            "separator",
            "positive electrode",
        ] and self.auxiliary_domains["secondary"] == ["current collector"]:
            self.first_dimension = "x"
            self.second_dimension = "z"
            self.x_sol = first_dim_pts
            self.z_sol = second_dim_pts
        else:
            raise pybamm.DomainError(
                "Cannot process 3D object with domain '{}' "
                "and auxiliary_domains '{}'".format(self.domain, self.auxiliary_domains)
            )

        first_dim_size = len(first_dim_pts)
        second_dim_size = len(second_dim_pts)
        entries = np.empty((first_dim_size, second_dim_size, len(self.t_sol)))

        # Evaluate the base_variable index-by-index
        for idx in range(len(self.t_sol)):
            t = self.t_sol[idx]
            u = self.u_sol[:, idx]
            inputs = {name: inp[idx] for name, inp in self.inputs.items()}
            if self.known_evals:
                eval_and_known_evals = self.base_variable.evaluate(
                    t, u, inputs=inputs, known_evals=self.known_evals[t]
                )
                entries[:, :, idx] = np.reshape(
                    eval_and_known_evals[0],
                    [first_dim_size, second_dim_size],
                    order="F",
                )
                self.known_evals[t] = eval_and_known_evals[1]
            else:
                entries[:, :, idx] = np.reshape(
                    self.base_variable.evaluate(t, u, inputs=inputs),
                    [first_dim_size, second_dim_size],
                    order="F",
                )

        # assign attributes for reference
        self.entries = entries
        self.dimensions = 2
        self.first_dim_pts = first_dim_pts
        self.second_dim_pts = second_dim_pts

        # set up interpolation
        self._interpolation_function = interp.RegularGridInterpolator(
            (first_dim_pts, second_dim_pts, self.t_sol),
            entries,
            method="linear",
            fill_value=np.nan,
        )

    def initialise_2D_fixed_t_scikit_fem(self):
        y_sol = self.mesh[0].edges["y"]
        len_y = len(y_sol)
        z_sol = self.mesh[0].edges["z"]
        len_z = len(z_sol)

        # Evaluate the base_variable
        inputs = {name: inp[0] for name, inp in self.inputs.items()}

        entries = np.reshape(
            self.base_variable.evaluate(0, self.u_sol, inputs=inputs), [len_y, len_z]
        )

        # assign attributes for reference
        self.entries = entries
        self.dimensions = 2
        self.y_sol = y_sol
        self.z_sol = z_sol
        self.first_dimension = "y"
        self.second_dimension = "z"
        self.first_dim_pts = y_sol
        self.second_dim_pts = z_sol

        # set up interpolation
        self._interpolation_function = interp.interp2d(
            y_sol, z_sol, entries, kind="linear", fill_value=np.nan
        )

    def initialise_2D_scikit_fem(self):
        y_sol = self.mesh[0].edges["y"]
        len_y = len(y_sol)
        z_sol = self.mesh[0].edges["z"]
        len_z = len(z_sol)
        entries = np.empty((len_y, len_z, len(self.t_sol)))

        # Evaluate the base_variable index-by-index
        for idx in range(len(self.t_sol)):
            t = self.t_sol[idx]
            u = self.u_sol[:, idx]
            inputs = {name: inp[idx] for name, inp in self.inputs.items()}

            if self.known_evals:
                eval_and_known_evals = self.base_variable.evaluate(
                    t, u, inputs=inputs, known_evals=self.known_evals[t]
                )
                entries[:, :, idx] = np.reshape(eval_and_known_evals[0], [len_y, len_z])
                self.known_evals[t] = eval_and_known_evals[1]
            else:
                entries[:, :, idx] = np.reshape(
                    self.base_variable.evaluate(t, u, inputs=inputs), [len_y, len_z]
                )

        # assign attributes for reference
        self.entries = entries
        self.dimensions = 2
        self.y_sol = y_sol
        self.z_sol = z_sol
        self.first_dimension = "y"
        self.second_dimension = "z"
        self.first_dim_pts = y_sol
        self.second_dim_pts = z_sol

        # set up interpolation
        self._interpolation_function = interp.RegularGridInterpolator(
            (y_sol, z_sol, self.t_sol), entries, method="linear", fill_value=np.nan
        )

    def __call__(self, t=None, x=None, r=None, y=None, z=None, warn=True):
        """
        Evaluate the variable at arbitrary t (and x, r, y and/or z), using interpolation
        """
        if self.dimensions == 0:
            out = self._interpolation_function(t)
        elif self.dimensions == 1:
            out = self.call_1D(t, x, r, z)
        elif self.dimensions == 2:
            if t is None:
                out = self._interpolation_function(y, z)
            else:
                out = self.call_2D(t, x, r, y, z)
        if warn is True and np.isnan(out).any():
            pybamm.logger.warning(
                "Calling variable outside interpolation range (returns 'nan')"
            )
        return out

    def call_1D(self, t, x, r, z):
        "Evaluate a 1D variable"
        spatial_var = eval_dimension_name(self.first_dimension, x, r, None, z)
        return self._interpolation_function(t, spatial_var)

    def call_2D(self, t, x, r, y, z):
        "Evaluate a 2D variable"
        first_dim = eval_dimension_name(self.first_dimension, x, r, y, z)
        second_dim = eval_dimension_name(self.second_dimension, x, r, y, z)
        if isinstance(first_dim, np.ndarray):
            if isinstance(second_dim, np.ndarray) and isinstance(t, np.ndarray):
                first_dim = first_dim[:, np.newaxis, np.newaxis]
                second_dim = second_dim[:, np.newaxis]
            elif isinstance(second_dim, np.ndarray) or isinstance(t, np.ndarray):
                first_dim = first_dim[:, np.newaxis]
        else:
            if isinstance(second_dim, np.ndarray) and isinstance(t, np.ndarray):
                second_dim = second_dim[:, np.newaxis]

        return self._interpolation_function((first_dim, second_dim, t))

    @property
    def data(self):
        "Same as entries, but different name"
        return self.entries


def eval_dimension_name(name, x, r, y, z):
    if name == "x":
        out = x
    elif name == "r":
        out = r
    elif name == "y":
        out = y
    elif name == "z":
        out = z

    if out is None:
        raise ValueError("inputs {} cannot be None".format(name))
    else:
        return out
