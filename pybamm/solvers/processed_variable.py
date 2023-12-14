#
# Processed Variable class
#
import casadi
import numpy as np
import pybamm
from scipy.integrate import cumulative_trapezoid
import xarray as xr


class ProcessedVariable:
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

    def __init__(
        self,
        base_variables,
        base_variables_casadi,
        solution,
        warn=True,
        cumtrapz_ic=None,
    ):
        self.base_variables = base_variables
        self.base_variables_casadi = base_variables_casadi

        self.all_ts = solution.all_ts
        self.all_ys = solution.all_ys
        self.all_inputs = solution.all_inputs
        self.all_inputs_casadi = solution.all_inputs_casadi

        self.mesh = base_variables[0].mesh
        self.domain = base_variables[0].domain
        self.domains = base_variables[0].domains
        self.warn = warn
        self.cumtrapz_ic = cumtrapz_ic

        # Process spatial variables
        geometry = solution.all_models[0].geometry
        self.spatial_variables = {}
        for domain_level, domain_names in self.domains.items():
            variables = []
            for domain in domain_names:
                variables += list(geometry[domain].keys())
            self.spatial_variables[domain_level] = variables

        # Sensitivity starts off uninitialized, only set when called
        self._sensitivities = None
        self.solution_sensitivities = solution.sensitivities

        # Store time
        self.t_pts = solution.t

        # Evaluate base variable at initial time
        self.base_eval_shape = self.base_variables[0].shape
        self.base_eval_size = self.base_variables[0].size

        # handle 2D (in space) finite element variables differently
        if (
            self.mesh
            and "current collector" in self.domain
            and isinstance(self.mesh, pybamm.ScikitSubMesh2D)
        ):
            self.initialise_2D_scikit_fem()

        # check variable shape
        else:
            if len(self.base_eval_shape) == 0 or self.base_eval_shape[0] == 1:
                self.initialise_0D()
            else:
                n = self.mesh.npts
                base_shape = self.base_eval_shape[0]
                # Try some shapes that could make the variable a 1D variable
                if base_shape in [n, n + 1]:
                    self.initialise_1D()
                else:
                    # Try some shapes that could make the variable a 2D variable
                    first_dim_nodes = self.mesh.nodes
                    first_dim_edges = self.mesh.edges
                    second_dim_pts = self.base_variables[0].secondary_mesh.nodes
                    if self.base_eval_size // len(second_dim_pts) in [
                        len(first_dim_nodes),
                        len(first_dim_edges),
                    ]:
                        self.initialise_2D()
                    else:
                        # Raise error for 3D variable
                        raise NotImplementedError(
                            f"Shape not recognized for {base_variables[0]}"
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
                entries[idx] = float(base_var_casadi(t, y, inputs))

                idx += 1

        if self.cumtrapz_ic is not None:
            entries = cumulative_trapezoid(
                entries, self.t_pts, initial=float(self.cumtrapz_ic)
            )

        # set up interpolation
        self._xr_data_array = xr.DataArray(entries, coords=[("t", self.t_pts)])

        self.entries = entries
        self.dimensions = 0

    def initialise_1D(self, fixed_t=False):
        len_space = self.base_eval_shape[0]
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
        self.spatial_variable_names = {
            k: self._process_spatial_variable_names(v)
            for k, v in self.spatial_variables.items()
        }
        self.first_dimension = self.spatial_variable_names["primary"]

        # assign attributes for reference
        pts_for_interp = space
        self.internal_boundaries = self.mesh.internal_boundaries

        # Set first_dim_pts to edges for nicer plotting
        self.first_dim_pts = edges

        # set up interpolation
        self._xr_data_array = xr.DataArray(
            entries_for_interp,
            coords=[(self.first_dimension, pts_for_interp), ("t", self.t_pts)],
        )

    def initialise_2D(self):
        """
        Initialise a 2D object that depends on x and r, x and z, x and R, or R and r.
        """
        first_dim_nodes = self.mesh.nodes
        first_dim_edges = self.mesh.edges
        second_dim_nodes = self.base_variables[0].secondary_mesh.nodes
        second_dim_edges = self.base_variables[0].secondary_mesh.edges
        if self.base_eval_size // len(second_dim_nodes) == len(first_dim_nodes):
            first_dim_pts = first_dim_nodes
        elif self.base_eval_size // len(second_dim_nodes) == len(first_dim_edges):
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

        self.spatial_variable_names = {
            k: self._process_spatial_variable_names(v)
            for k, v in self.spatial_variables.items()
        }

        self.first_dimension = self.spatial_variable_names["primary"]
        self.second_dimension = self.spatial_variable_names["secondary"]

        # assign attributes for reference
        self.entries = entries
        self.dimensions = 2
        first_dim_pts_for_interp = first_dim_pts
        second_dim_pts_for_interp = second_dim_pts

        # Set pts to edges for nicer plotting
        self.first_dim_pts = first_dim_edges
        self.second_dim_pts = second_dim_edges

        # set up interpolation
        self._xr_data_array = xr.DataArray(
            entries_for_interp,
            coords={
                self.first_dimension: first_dim_pts_for_interp,
                self.second_dimension: second_dim_pts_for_interp,
                "t": self.t_pts,
            },
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
                    order="C",
                )
                idx += 1

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
        self._xr_data_array = xr.DataArray(
            entries,
            coords={"y": y_sol, "z": z_sol, "t": self.t_pts},
        )

    def _process_spatial_variable_names(self, spatial_variable):
        if len(spatial_variable) == 0:
            return None

        # Extract names
        raw_names = []
        for var in spatial_variable:
            # Ignore tabs in domain names
            if var == "tabs":
                continue
            if isinstance(var, str):
                raw_names.append(var)
            else:
                raw_names.append(var.name)

        # Rename battery variables to match PyBaMM convention
        if all([var.startswith("r") for var in raw_names]):
            return "r"
        elif all([var.startswith("x") for var in raw_names]):
            return "x"
        elif all([var.startswith("R") for var in raw_names]):
            return "R"
        elif len(raw_names) == 1:
            return raw_names[0]
        else:
            raise NotImplementedError(
                f"Spatial variable name not recognized for {spatial_variable}"
            )

    def __call__(self, t=None, x=None, r=None, y=None, z=None, R=None, warn=True):
        """
        Evaluate the variable at arbitrary *dimensional* t (and x, r, y, z and/or R),
        using interpolation
        """
        kwargs = {"t": t, "x": x, "r": r, "y": y, "z": z, "R": R}
        # Remove any None arguments
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        # Use xarray interpolation, return numpy array
        return self._xr_data_array.interp(**kwargs).values

    @property
    def data(self):
        """Same as entries, but different name"""
        return self.entries

    @property
    def sensitivities(self):
        """
        Returns a dictionary of sensitivities for each input parameter.
        The keys are the input parameters, and the value is a matrix of size
        (n_x * n_t, n_p), where n_x is the number of states, n_t is the number of time
        points, and n_p is the size of the input parameter
        """
        # No sensitivities if there are no inputs
        if len(self.all_inputs[0]) == 0:
            return {}
        # Otherwise initialise and return sensitivities
        if self._sensitivities is None:
            if self.solution_sensitivities != {}:
                self.initialise_sensitivity_explicit_forward()
            else:
                raise ValueError(
                    "Cannot compute sensitivities. The 'calculate_sensitivities' "
                    "argument of the solver.solve should be changed from 'None' to "
                    "allow sensitivities calculations. Check solver documentation for "
                    "details."
                )
        return self._sensitivities

    def initialise_sensitivity_explicit_forward(self):
        "Set up the sensitivity dictionary"
        inputs_stacked = self.all_inputs_casadi[0]

        # Set up symbolic variables
        t_casadi = casadi.MX.sym("t")
        y_casadi = casadi.MX.sym("y", self.all_ys[0].shape[0])
        p_casadi = {
            name: casadi.MX.sym(name, value.shape[0])
            for name, value in self.all_inputs[0].items()
        }

        p_casadi_stacked = casadi.vertcat(*[p for p in p_casadi.values()])

        # Convert variable to casadi format for differentiating
        var_casadi = self.base_variables[0].to_casadi(
            t_casadi, y_casadi, inputs=p_casadi
        )
        dvar_dy = casadi.jacobian(var_casadi, y_casadi)
        dvar_dp = casadi.jacobian(var_casadi, p_casadi_stacked)

        # Convert to functions and evaluate index-by-index
        dvar_dy_func = casadi.Function(
            "dvar_dy", [t_casadi, y_casadi, p_casadi_stacked], [dvar_dy]
        )
        dvar_dp_func = casadi.Function(
            "dvar_dp", [t_casadi, y_casadi, p_casadi_stacked], [dvar_dp]
        )
        for index, (ts, ys) in enumerate(zip(self.all_ts, self.all_ys)):
            for idx, t in enumerate(ts):
                u = ys[:, idx]
                next_dvar_dy_eval = dvar_dy_func(t, u, inputs_stacked)
                next_dvar_dp_eval = dvar_dp_func(t, u, inputs_stacked)
                if index == 0 and idx == 0:
                    dvar_dy_eval = next_dvar_dy_eval
                    dvar_dp_eval = next_dvar_dp_eval
                else:
                    dvar_dy_eval = casadi.diagcat(dvar_dy_eval, next_dvar_dy_eval)
                    dvar_dp_eval = casadi.vertcat(dvar_dp_eval, next_dvar_dp_eval)

        # Compute sensitivity
        dy_dp = self.solution_sensitivities["all"]
        S_var = dvar_dy_eval @ dy_dp + dvar_dp_eval

        sensitivities = {"all": S_var}

        # Add the individual sensitivity
        start = 0
        for name, inp in self.all_inputs[0].items():
            end = start + inp.shape[0]
            sensitivities[name] = S_var[:, start:end]
            start = end

        # Save attribute
        self._sensitivities = sensitivities
