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
    base_variables : list of :class:`pybamm.Symbol`
        A list of base variables with a method `evaluate(t,y)`, each entry of which
        returns the value of that variable for that particular sub-solution.
        A Solution can be comprised of sub-solutions which are the solutions of
        different models.
        Note that this can be any kind of node in the expression tree, not
        just a :class:`pybamm.Variable`.
        When evaluated, returns an array of size (m,n)
    base_variable_casadis : list of :class:`casadi.Function`
        A list of casadi functions. When evaluated, returns the same thing as
        `base_Variable.evaluate` (but more efficiently).
    solution : :class:`pybamm.Solution`
        The solution object to be used to create the processed variables
    warn : bool, optional
        Whether to raise warnings when trying to evaluate time and length scales.
        Default is True.
    """

    def __init__(self, base_variables, base_variables_casadi, solution, warn=True):
        self.base_variables = base_variables
        self.base_variables_casadi = base_variables_casadi

        self.all_ts = solution.all_ts
        self.all_ys = solution.all_ys
        self.all_inputs_casadi = solution.all_inputs_casadi

        self.mesh = base_variables[0].mesh
        self.domain = base_variables[0].domain
        self.auxiliary_domains = base_variables[0].auxiliary_domains
        self.warn = warn

        # Set timescale
        self.timescale = solution.timescale_eval
        self.t_pts = solution.t * self.timescale

        # Store length scales
        self.length_scales = solution.length_scales_eval

        # Evaluate base variable at initial time
        self.base_eval = self.base_variables_casadi[0](
            self.all_ts[0][0], self.all_ys[0][:, 0], self.all_inputs_casadi[0]
        ).full()

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
                    second_dim_pts = self.base_variables[0].secondary_mesh.nodes
                    if self.base_eval.size // len(second_dim_pts) in [
                        len(first_dim_nodes),
                        len(first_dim_edges),
                    ]:
                        self.initialise_2D()
                    else:
                        # Raise error for 3D variable
                        raise NotImplementedError(
                            "Shape not recognized for {} ".format(base_variables[0])
                            + "(note processing of 3D variables is not yet implemented)"
                        )

    def initialise_0D(self):
        # initialise empty array of the correct size
        entries = np.empty(len(self.t_pts))
        idx = 0
        # Evaluate the base_variable index-by-index
        for ts, ys, inputs, base_var_casadi in zip(
            self.all_ts, self.all_ys, self.all_inputs_casadi, self.base_variables_casadi
        ):
            for inner_idx, t in enumerate(ts):
                t = ts[inner_idx]
                y = ys[:, inner_idx]
                entries[idx] = base_var_casadi(t, y, inputs).full()[0, 0]
                idx += 1

        # set up interpolation
        if len(self.t_pts) == 1:
            # Variable is just a scalar value, but we need to create a callable
            # function to be consistent with other processed variables
            self._interpolation_function = Interpolant0D(entries)
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
        entries = np.empty((len_space, len(self.t_pts)))

        # Evaluate the base_variable index-by-index
        idx = 0
        for ts, ys, inputs, base_var_casadi in zip(
            self.all_ts, self.all_ys, self.all_inputs_casadi, self.base_variables_casadi
        ):
            for inner_idx, t in enumerate(ts):
                t = ts[inner_idx]
                y = ys[:, inner_idx]
                entries[:, idx] = base_var_casadi(t, y, inputs).full()[:, 0]
                idx += 1

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
        length_scale = self.get_spatial_scale(self.first_dimension, self.domain[0])
        pts_for_interp = space * length_scale
        self.internal_boundaries = [
            bnd * length_scale for bnd in self.mesh.internal_boundaries
        ]

        # Set first_dim_pts to edges for nicer plotting
        self.first_dim_pts = edges * length_scale

        # set up interpolation
        if len(self.t_pts) == 1:
            # function of space only
            self._interpolation_function = Interpolant1D(
                pts_for_interp, entries_for_interp
            )
        else:
            # function of space and time. Note that the order of 't' and 'space'
            # is the reverse of what you'd expect
            self._interpolation_function = interp.interp2d(
                self.t_pts,
                pts_for_interp,
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
        second_dim_nodes = self.base_variables[0].secondary_mesh.nodes
        second_dim_edges = self.base_variables[0].secondary_mesh.edges
        if self.base_eval.size // len(second_dim_nodes) == len(first_dim_nodes):
            first_dim_pts = first_dim_nodes
        elif self.base_eval.size // len(second_dim_nodes) == len(first_dim_edges):
            first_dim_pts = first_dim_edges

        second_dim_pts = second_dim_nodes
        first_dim_size = len(first_dim_pts)
        second_dim_size = len(second_dim_pts)
        entries = np.empty((first_dim_size, second_dim_size, len(self.t_pts)))

        # Evaluate the base_variable index-by-index
        idx = 0
        for ts, ys, inputs, base_var_casadi in zip(
            self.all_ts, self.all_ys, self.all_inputs_casadi, self.base_variables_casadi
        ):
            for inner_idx, t in enumerate(ts):
                t = ts[inner_idx]
                y = ys[:, inner_idx]
                entries[:, :, idx] = np.reshape(
                    base_var_casadi(t, y, inputs).full(),
                    [first_dim_size, second_dim_size],
                    order="F",
                )
                idx += 1

        # add points outside first dimension domain for extrapolation to
        # boundaries
        extrap_space_first_dim_left = np.array(
            [2 * first_dim_pts[0] - first_dim_pts[1]]
        )
        extrap_space_first_dim_right = np.array(
            [2 * first_dim_pts[-1] - first_dim_pts[-2]]
        )
        first_dim_pts = np.concatenate(
            [extrap_space_first_dim_left, first_dim_pts, extrap_space_first_dim_right]
        )
        extrap_entries_left = np.expand_dims(2 * entries[0] - entries[1], axis=0)
        extrap_entries_right = np.expand_dims(2 * entries[-1] - entries[-2], axis=0)
        entries_for_interp = np.concatenate(
            [extrap_entries_left, entries, extrap_entries_right], axis=0
        )

        # add points outside second dimension domain for extrapolation to
        # boundaries
        extrap_space_second_dim_left = np.array(
            [2 * second_dim_pts[0] - second_dim_pts[1]]
        )
        extrap_space_second_dim_right = np.array(
            [2 * second_dim_pts[-1] - second_dim_pts[-2]]
        )
        second_dim_pts = np.concatenate(
            [
                extrap_space_second_dim_left,
                second_dim_pts,
                extrap_space_second_dim_right,
            ]
        )
        extrap_entries_second_dim_left = np.expand_dims(
            2 * entries_for_interp[:, 0, :] - entries_for_interp[:, 1, :], axis=1
        )
        extrap_entries_second_dim_right = np.expand_dims(
            2 * entries_for_interp[:, -1, :] - entries_for_interp[:, -2, :], axis=1
        )
        entries_for_interp = np.concatenate(
            [
                extrap_entries_second_dim_left,
                entries_for_interp,
                extrap_entries_second_dim_right,
            ],
            axis=1,
        )

        # Process r-x or x-z
        if self.domain[0] in [
            "negative particle",
            "positive particle",
            "working particle",
        ] and self.auxiliary_domains["secondary"][0] in [
            "negative electrode",
            "positive electrode",
            "working electrode",
        ]:
            self.first_dimension = "r"
            self.second_dimension = "x"
            self.r_sol = first_dim_pts
            self.x_sol = second_dim_pts
        elif (
            self.domain[0]
            in [
                "negative electrode",
                "separator",
                "positive electrode",
            ]
            and self.auxiliary_domains["secondary"] == ["current collector"]
        ):
            self.first_dimension = "x"
            self.second_dimension = "z"
            self.x_sol = first_dim_pts
            self.z_sol = second_dim_pts
        else:
            raise pybamm.DomainError(
                "Cannot process 3D object with domain '{}' "
                "and auxiliary_domains '{}'".format(self.domain, self.auxiliary_domains)
            )

        # assign attributes for reference
        self.entries = entries
        self.dimensions = 2
        first_length_scale = self.get_spatial_scale(
            self.first_dimension, self.domain[0]
        )
        first_dim_pts_for_interp = first_dim_pts * first_length_scale

        second_length_scale = self.get_spatial_scale(
            self.second_dimension, self.auxiliary_domains["secondary"][0]
        )
        second_dim_pts_for_interp = second_dim_pts * second_length_scale

        # Set pts to edges for nicer plotting
        self.first_dim_pts = first_dim_edges * first_length_scale
        self.second_dim_pts = second_dim_edges * second_length_scale

        # set up interpolation
        if len(self.t_pts) == 1:
            # function of space only. Note the order of the points is the reverse
            # of what you'd expect
            self._interpolation_function = Interpolant2D(
                first_dim_pts_for_interp, second_dim_pts_for_interp, entries_for_interp
            )
        else:
            # function of space and time.
            self._interpolation_function = interp.RegularGridInterpolator(
                (first_dim_pts_for_interp, second_dim_pts_for_interp, self.t_pts),
                entries_for_interp,
                method="linear",
                fill_value=np.nan,
                bounds_error=False,
            )

    def initialise_2D_scikit_fem(self):
        y_sol = self.mesh.edges["y"]
        len_y = len(y_sol)
        z_sol = self.mesh.edges["z"]
        len_z = len(z_sol)
        entries = np.empty((len_y, len_z, len(self.t_pts)))

        # Evaluate the base_variable index-by-index
        idx = 0
        for ts, ys, inputs, base_var_casadi in zip(
            self.all_ts, self.all_ys, self.all_inputs_casadi, self.base_variables_casadi
        ):
            for inner_idx, t in enumerate(ts):
                t = ts[inner_idx]
                y = ys[:, inner_idx]
                entries[:, :, idx] = np.reshape(
                    base_var_casadi(t, y, inputs).full(),
                    [len_y, len_z],
                    order="F",
                )
                idx += 1

        # assign attributes for reference
        self.entries = entries
        self.dimensions = 2
        self.y_sol = y_sol
        self.z_sol = z_sol
        self.first_dimension = "y"
        self.second_dimension = "z"
        self.first_dim_pts = y_sol * self.get_spatial_scale("y", "current collector")
        self.second_dim_pts = z_sol * self.get_spatial_scale("z", "current collector")

        # set up interpolation
        if len(self.t_pts) == 1:
            # function of space only. Note the order of the points is the reverse
            # of what you'd expect
            self._interpolation_function = Interpolant2D(
                self.first_dim_pts, self.second_dim_pts, entries
            )
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
            elif len(self.base_variables) == 1 and self.base_variables[0].is_constant():
                t = self.t_pts[0]
            else:
                raise ValueError(
                    "t cannot be None for variable {}".format(self.base_variables)
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
        """Evaluate a 1D variable"""
        spatial_var = eval_dimension_name(self.first_dimension, x, r, None, z)
        return self._interpolation_function(t, spatial_var)

    def call_2D(self, t, x, r, y, z):
        """Evaluate a 2D variable"""
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

    def get_spatial_scale(self, name, domain):
        """Returns the spatial scale for a named spatial variable"""
        try:
            if name == "y" and domain == "current collector":
                return self.length_scales["current collector y"]
            elif name == "z" and domain == "current collector":
                return self.length_scales["current collector z"]
            else:
                return self.length_scales[domain]
        except KeyError:
            if self.warn:  # pragma: no cover
                pybamm.logger.warning(
                    "No length scale set for {}. "
                    "Using default of 1 [m].".format(domain)
                )
            return 1

    @property
    def data(self):
        """Same as entries, but different name"""
        return self.entries


class Interpolant0D:
    def __init__(self, entries):
        self.entries = entries

    def __call__(self, t):
        return self.entries


class Interpolant1D:
    def __init__(self, pts_for_interp, entries_for_interp):
        self.interpolant = interp.interp1d(
            pts_for_interp,
            entries_for_interp[:, 0],
            kind="linear",
            fill_value=np.nan,
            bounds_error=False,
        )

    def __call__(self, t, z):
        if isinstance(z, np.ndarray):
            return self.interpolant(z)[:, np.newaxis]
        else:
            return self.interpolant(z)


class Interpolant2D:
    def __init__(
        self, first_dim_pts_for_interp, second_dim_pts_for_interp, entries_for_interp
    ):
        self.interpolant = interp.interp2d(
            second_dim_pts_for_interp,
            first_dim_pts_for_interp,
            entries_for_interp[:, :, 0],
            kind="linear",
            fill_value=np.nan,
            bounds_error=False,
        )

    def __call__(self, input):
        """
        Calls and returns a 2D interpolant of the correct shape depending on the
        shape of the input
        """
        first_dim, second_dim, _ = input
        if isinstance(first_dim, np.ndarray) and isinstance(second_dim, np.ndarray):
            first_dim = first_dim[:, 0, 0]
            second_dim = second_dim[:, 0]
            return self.interpolant(second_dim, first_dim)
        elif isinstance(first_dim, np.ndarray):
            first_dim = first_dim[:, 0]
            return self.interpolant(second_dim, first_dim)[:, 0]
        elif isinstance(second_dim, np.ndarray):
            second_dim = second_dim[:, 0]
            return self.interpolant(second_dim, first_dim)
        else:
            return self.interpolant(second_dim, first_dim)[0]


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
