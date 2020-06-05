#
# Processed Variable class
#
import numbers
import numpy as np
import pybamm
import scipy.interpolate as interp


def make_interp2D_fun(input, interpolant):
    """
    Calls and returns a 2D interpolant of the correct shape depending on the
    shape of the input
    """
    first_dim, second_dim, _ = input
    if isinstance(first_dim, np.ndarray) and isinstance(second_dim, np.ndarray):
        first_dim = first_dim[:, 0, 0]
        second_dim = second_dim[:, 0]
        return interpolant(second_dim, first_dim)
    elif isinstance(first_dim, np.ndarray):
        first_dim = first_dim[:, 0]
        return interpolant(second_dim, first_dim)[:, 0]
    elif isinstance(second_dim, np.ndarray):
        second_dim = second_dim[:, 0]
        return interpolant(second_dim, first_dim)
    else:
        return interpolant(second_dim, first_dim)[0]


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
    warn : bool, optional
        Whether to raise warnings when trying to evaluate time and length scales.
        Default is True.
    """

    def __init__(self, base_variable, solution, known_evals=None, warn=True):
        self.base_variable = base_variable
        self.t_sol = solution.t
        self.u_sol = solution.y
        self.mesh = base_variable.mesh
        self.inputs = solution.inputs
        self.domain = base_variable.domain
        self.auxiliary_domains = base_variable.auxiliary_domains
        self.known_evals = known_evals
        self.warn = warn

        # Set timescale
        self.timescale = solution.model.timescale.evaluate()
        self.t_pts = self.t_sol * self.timescale

        # Store spatial variables to get scales
        self.spatial_vars = {}
        if solution.model:
            for var in ["x", "y", "z", "r_n", "r_p"]:
                if (
                    var in solution.model.variables
                    and var + " [m]" in solution.model.variables
                ):
                    self.spatial_vars[var] = solution.model.variables[var]
                    self.spatial_vars[var + " [m]"] = solution.model.variables[
                        var + " [m]"
                    ]

        # Evaluate base variable at initial time
        if self.known_evals:
            self.base_eval, self.known_evals[solution.t[0]] = base_variable.evaluate(
                solution.t[0],
                solution.y[:, 0],
                inputs={name: inp[:, 0] for name, inp in solution.inputs.items()},
                known_evals=self.known_evals[solution.t[0]],
            )
        else:
            self.base_eval = base_variable.evaluate(
                solution.t[0],
                solution.y[:, 0],
                inputs={name: inp[:, 0] for name, inp in solution.inputs.items()},
            )

        # handle 2D (in space) finite element variables differently
        if (
            self.mesh
            and "current collector" in self.domain
            and isinstance(self.mesh, pybamm.ScikitSubMesh2D)
        ):
            self.initialise_2D_scikit_fem()

        # check variable shape
        else:
            if (
                isinstance(self.base_eval, numbers.Number)
                or len(self.base_eval.shape) == 0
                or self.base_eval.shape[0] == 1
            ):
                self.initialise_0D()
            else:
                n = self.mesh.npts
                base_shape = self.base_eval.shape[0]
                # Try some shapes that could make the variable a 1D variable
                if base_shape in [n, n + 1]:
                    self.initialise_1D()
                else:
                    # Try some shapes that could make the variable a 2D variable
                    first_dim_nodes = self.mesh.nodes
                    first_dim_edges = self.mesh.edges
                    second_dim_pts = self.base_variable.secondary_mesh.nodes
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
            inputs = {name: inp[:, idx] for name, inp in self.inputs.items()}
            if self.known_evals:
                entries[idx], self.known_evals[t] = self.base_variable.evaluate(
                    t, u, inputs=inputs, known_evals=self.known_evals[t]
                )
            else:
                entries[idx] = self.base_variable.evaluate(t, u, inputs=inputs)

        # set up interpolation
        if len(self.t_sol) == 1:
            # Variable is just a scalar value, but we need to create a callable
            # function to be consitent with other processed variables
            def fun(t):
                return entries

            self._interpolation_function = fun
        else:
            self._interpolation_function = interp.interp1d(
                self.t_pts,
                entries,
                kind="linear",
                fill_value=np.nan,
                bounds_error=False,
            )

        self.entries = entries
        self.dimensions = 0

    def initialise_1D(self, fixed_t=False):
        len_space = self.base_eval.shape[0]
        entries = np.empty((len_space, len(self.t_sol)))

        # Evaluate the base_variable index-by-index
        for idx in range(len(self.t_sol)):
            t = self.t_sol[idx]
            u = self.u_sol[:, idx]
            inputs = {name: inp[:, idx] for name, inp in self.inputs.items()}
            if self.known_evals:
                eval_and_known_evals = self.base_variable.evaluate(
                    t, u, inputs=inputs, known_evals=self.known_evals[t]
                )
                entries[:, idx] = eval_and_known_evals[0][:, 0]
                self.known_evals[t] = eval_and_known_evals[1]
            else:
                entries[:, idx] = self.base_variable.evaluate(t, u, inputs=inputs)[:, 0]

        # Get node and edge values
        nodes = self.mesh.nodes
        edges = self.mesh.edges
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

        # assign attributes for reference
        self.first_dim_pts = space * self.get_spatial_scale(
            self.first_dimension, self.domain[0]
        )
        self.internal_boundaries = self.mesh.internal_boundaries

        # set up interpolation
        if len(self.t_sol) == 1:
            # function of space only
            interpolant = interp.interp1d(
                self.first_dim_pts,
                entries_for_interp[:, 0],
                kind="linear",
                fill_value=np.nan,
                bounds_error=False,
            )

            def interp_fun(t, z):
                if isinstance(z, np.ndarray):
                    return interpolant(z)[:, np.newaxis]
                else:
                    return interpolant(z)

            self._interpolation_function = interp_fun
        else:
            # function of space and time. Note that the order of 't' and 'space'
            # is the reverse of what you'd expect
            self._interpolation_function = interp.interp2d(
                self.t_pts,
                self.first_dim_pts,
                entries_for_interp,
                kind="linear",
                fill_value=np.nan,
                bounds_error=False,
            )

    def initialise_2D(self):
        """
        Initialise a 2D object that depends on x and r, or x and z.
        """
        first_dim_nodes = self.mesh.nodes
        first_dim_edges = self.mesh.edges
        second_dim_pts = self.base_variable.secondary_mesh.nodes
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
            inputs = {name: inp[:, idx] for name, inp in self.inputs.items()}
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
        self.first_dim_pts = first_dim_pts * self.get_spatial_scale(
            self.first_dimension, self.domain[0]
        )
        self.second_dim_pts = second_dim_pts * self.get_spatial_scale(
            self.second_dimension
        )

        # set up interpolation
        if len(self.t_sol) == 1:
            # function of space only. Note the order of the points is the reverse
            # of what you'd expect
            interpolant = interp.interp2d(
                self.second_dim_pts,
                self.first_dim_pts,
                entries[:, :, 0],
                kind="linear",
                fill_value=np.nan,
                bounds_error=False,
            )

            def interp_fun(input):
                return make_interp2D_fun(input, interpolant)

            self._interpolation_function = interp_fun
        else:
            # function of space and time.
            self._interpolation_function = interp.RegularGridInterpolator(
                (self.first_dim_pts, self.second_dim_pts, self.t_pts),
                entries,
                method="linear",
                fill_value=np.nan,
                bounds_error=False,
            )

    def initialise_2D_scikit_fem(self):
        y_sol = self.mesh.edges["y"]
        len_y = len(y_sol)
        z_sol = self.mesh.edges["z"]
        len_z = len(z_sol)
        entries = np.empty((len_y, len_z, len(self.t_sol)))

        # Evaluate the base_variable index-by-index
        for idx in range(len(self.t_sol)):
            t = self.t_sol[idx]
            u = self.u_sol[:, idx]
            inputs = {name: inp[:, idx] for name, inp in self.inputs.items()}

            if self.known_evals:
                eval_and_known_evals = self.base_variable.evaluate(
                    t, u, inputs=inputs, known_evals=self.known_evals[t]
                )
                entries[:, :, idx] = np.reshape(
                    eval_and_known_evals[0], [len_y, len_z], order="F"
                )
                self.known_evals[t] = eval_and_known_evals[1]
            else:
                entries[:, :, idx] = np.reshape(
                    self.base_variable.evaluate(t, u, inputs=inputs),
                    [len_y, len_z],
                    order="F",
                )

        # assign attributes for reference
        self.entries = entries
        self.dimensions = 2
        self.y_sol = y_sol
        self.z_sol = z_sol
        self.first_dimension = "y"
        self.second_dimension = "z"
        self.first_dim_pts = y_sol * self.get_spatial_scale("y")
        self.second_dim_pts = z_sol * self.get_spatial_scale("z")

        # set up interpolation
        if len(self.t_sol) == 1:
            # function of space only. Note the order of the points is the reverse
            # of what you'd expect
            interpolant = interp.interp2d(
                self.second_dim_pts,
                self.first_dim_pts,
                entries,
                kind="linear",
                fill_value=np.nan,
                bounds_error=False,
            )

            def interp_fun(input):
                return make_interp2D_fun(input, interpolant)

            self._interpolation_function = interp_fun
        else:
            # function of space and time.
            self._interpolation_function = interp.RegularGridInterpolator(
                (self.first_dim_pts, self.second_dim_pts, self.t_pts),
                entries,
                method="linear",
                fill_value=np.nan,
                bounds_error=False,
            )

    def __call__(self, t=None, x=None, r=None, y=None, z=None, warn=True):
        """
        Evaluate the variable at arbitrary *dimensional* t (and x, r, y and/or z),
        using interpolation
        """
        # If t is None and there is only one value of time in the soluton (i.e.
        # the solution is independent of time) then we set t equal to the value
        # stored in the solution. If the variable is constant (doesn't depend on
        # time) evaluate arbitrarily at the first value of t. Otherwise, raise
        # an error
        if t is None:
            if len(self.t_pts) == 1:
                t = self.t_pts
            elif self.base_variable.is_constant():
                t = self.t_pts[0]
            else:
                raise ValueError(
                    "t cannot be None for variable {}".format(self.base_variable)
                )

        # Call interpolant of correct spatial dimension
        if self.dimensions == 0:
            out = self._interpolation_function(t)
        elif self.dimensions == 1:
            out = self.call_1D(t, x, r, z)
        elif self.dimensions == 2:
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

    def get_spatial_scale(self, name, domain=None):
        "Returns the spatial scale for a named spatial variable"
        # Different scale in negative and positive particles
        if domain == "negative particle":
            name = "r_n"
        elif domain == "positive particle":
            name = "r_p"

        # Try to get length scale
        if name + " [m]" in self.spatial_vars and name in self.spatial_vars:
            scale = (
                self.spatial_vars[name + " [m]"] / self.spatial_vars[name]
            ).evaluate()[-1]
        else:
            if self.warn:
                pybamm.logger.warning(
                    "No scale set for spatial variable {}. "
                    "Using default of 1 [m].".format(name)
                )
            scale = 1
        return scale

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
