import bisect

import casadi
import numpy as np
import xarray as xr
from pybammsolvers import idaklu

import pybamm

from .base_processed_variable import BaseProcessedVariable


class ProcessedVariable(BaseProcessedVariable):
    """
    An object that can be evaluated at arbitrary (scalars or vectors) t and x, and
    returns the (interpolated) value of the base variable at that t and x.

    Parameters
    ----------
    name : str
        The name of the variable
    base_variables : list of :class:`pybamm.Symbol`
        A list of base variables with a method `evaluate(t,y)`, each entry of which
        returns the value of that variable for that particular sub-solution.
        A Solution can be comprised of sub-solutions which are the solutions of
        different models.
        Note that this can be any kind of node in the expression tree, not
        just a :class:`pybamm.Variable`.
        When evaluated, returns an array of size (m,n)
    base_variables_casadi : list of :class:`casadi.Function`
        A list of casadi functions. When evaluated, returns the same thing as
        `base_Variable.evaluate` (but more efficiently).
    solution : :class:`pybamm.Solution`
        The solution object to be used to create the processed variables
    time_integral : :class:`pybamm.ProcessedVariableTimeIntegral`, optional
        Not none if the variable is to be time-integrated (default is None)
    """

    def __init__(
        self,
        name: str,
        base_variables,
        base_variables_casadi,
        solution,
        time_integral: pybamm.ProcessedVariableTimeIntegral | None = None,
    ):
        self._name = name
        self.base_variables = base_variables
        self.base_variables_casadi = base_variables_casadi

        self.all_ts = solution.all_ts
        self.all_ys = solution.all_ys
        self.all_yps = solution.all_yps
        self.all_inputs = solution.all_inputs
        self.all_inputs_casadi = solution.all_inputs_casadi

        self.mesh = base_variables[0].mesh
        self.domain = base_variables[0].domain
        self.domains = base_variables[0].domains
        self.time_integral = time_integral

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
        self.all_solution_sensitivities = solution._all_sensitivities

        # Store time
        self.t_pts = solution.t

        # Evaluate base variable at initial time
        self.base_eval_shape = self.base_variables[0].shape
        self.base_eval_size = self.base_variables[0].size

        self._xr_array_raw = None
        self._entries_raw = None
        self._entries_for_interp_raw = None
        self._coords_raw = None

    def initialise(self):
        if self.entries_raw_initialized:
            return

        entries = self.observe_raw()

        t = self.t_pts
        entries_for_interp, coords = self._interp_setup(entries, t)

        self._entries_raw = entries
        self._entries_for_interp_raw = entries_for_interp
        self._coords_raw = coords

    def observe_and_interp(self, t, fill_value):
        """
        Interpolate the variable at the given time points and y values.
        t must be a sorted array of time points.
        """

        entries = self._observe_hermite(t)
        processed_entries = self._observe_postfix(entries, t)

        tf = self.t_pts[-1]
        if t[-1] > tf and fill_value != "extrapolate":
            # fill the rest
            idx = np.searchsorted(t, tf, side="right")
            processed_entries[..., idx:] = fill_value

        return processed_entries

    def observe_raw(self):
        """
        Evaluate the base variable at the given time points and y values.
        """
        t = self.t_pts
        return self._observe_postfix(self._observe_raw(), t)

    def _setup_inputs(self, t, full_range):
        pybamm.logger.debug("Setting up C++ interpolation inputs")

        ts = self.all_ts
        ys = self.all_ys
        yps = self.all_yps
        inputs = self.all_inputs_casadi

        # Remove all empty ts
        idxs = np.where([ti.size > 0 for ti in ts])[0]

        # Find the indices of the time points to observe
        if not full_range:
            ts_nonempty = [ts[idx] for idx in idxs]
            idxs_subset = _find_ts_indices(ts_nonempty, t)
            idxs = idxs[idxs_subset]

        # Extract the time points and inputs
        ts = [ts[idx] for idx in idxs]
        ys = [ys[idx] for idx in idxs]
        if self.hermite_interpolation:
            yps = [yps[idx] for idx in idxs]
        inputs = [self.all_inputs_casadi[idx] for idx in idxs]

        is_f_contiguous = _is_f_contiguous(ys)

        ts = idaklu.VectorRealtypeNdArray(ts)
        ys = idaklu.VectorRealtypeNdArray(ys)
        if self.hermite_interpolation:
            yps = idaklu.VectorRealtypeNdArray(yps)
        else:
            yps = None
        inputs = idaklu.VectorRealtypeNdArray(inputs)

        # Generate the serialized C++ functions only once
        funcs_unique = {}
        funcs = [None] * len(idxs)
        for i in range(len(idxs)):
            vars = self.base_variables_casadi[idxs[i]]
            if vars not in funcs_unique:
                funcs_unique[vars] = vars.serialize()
            funcs[i] = funcs_unique[vars]

        return ts, ys, yps, funcs, inputs, is_f_contiguous

    def _observe_hermite(self, t):
        pybamm.logger.debug("Observing and Hermite interpolating the variable")

        ts, ys, yps, funcs, inputs, _ = self._setup_inputs(t, full_range=False)
        shapes = self._shape(t)
        return idaklu.observe_hermite_interp(t, ts, ys, yps, inputs, funcs, shapes)

    def _observe_raw(self):
        pybamm.logger.debug("Observing the variable raw data")
        t = self.t_pts
        ts, ys, _, funcs, inputs, is_f_contiguous = self._setup_inputs(
            t, full_range=True
        )
        shapes = self._shape(self.t_pts)

        return idaklu.observe(ts, ys, inputs, funcs, is_f_contiguous, shapes)

    def _observe_postfix(self, entries, t):
        return entries

    def _interp_setup(self, entries, t):
        raise NotImplementedError  # pragma: no cover

    def _shape(self, t):
        raise NotImplementedError  # pragma: no cover

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

    def __call__(
        self,
        t=None,
        x=None,
        r=None,
        y=None,
        z=None,
        R=None,
        fill_value=np.nan,
    ):
        # Check to see if we are interpolating exactly onto the raw solution time points
        t_observe, observe_raw = self._check_observe_raw(t)

        # Check if the time points are sorted and unique
        is_sorted = observe_raw or _is_sorted(t_observe)

        # Sort them if not
        if not is_sorted:
            idxs_sort = np.argsort(t_observe)
            t_observe = t_observe[idxs_sort]

        hermite_time_interp = self.hermite_interpolation and not observe_raw

        if hermite_time_interp:
            entries = self.observe_and_interp(t_observe, fill_value)

        spatial_interp = any(a is not None for a in [x, r, y, z, R])

        xr_interp = spatial_interp or not hermite_time_interp

        if xr_interp:
            if hermite_time_interp:
                # Already interpolated in time
                t = None
                entries_for_interp, coords = self._interp_setup(entries, t_observe)
            else:
                self.initialise()
                entries_for_interp, coords = (
                    self._entries_for_interp_raw,
                    self._coords_raw,
                )

            if self.time_integral is None:
                processed_entries = self._xr_interpolate(
                    entries_for_interp,
                    coords,
                    observe_raw,
                    t,
                    x,
                    r,
                    y,
                    z,
                    R,
                    fill_value,
                )
            else:
                processed_entries = entries_for_interp
        else:
            processed_entries = entries

        if not is_sorted:
            idxs_unsort = np.empty_like(idxs_sort)
            idxs_unsort[idxs_sort] = np.arange(len(t_observe))

            processed_entries = processed_entries[..., idxs_unsort]

        # Remove a singleton time dimension if we interpolate in time with hermite
        if hermite_time_interp and t_observe.size == 1:
            processed_entries = np.squeeze(processed_entries, axis=-1)

        return processed_entries

    def _xr_interpolate(
        self,
        entries_for_interp,
        coords,
        observe_raw,
        t=None,
        x=None,
        r=None,
        y=None,
        z=None,
        R=None,
        fill_value=None,
    ):
        """
        Evaluate the variable at arbitrary *dimensional* t (and x, r, y, z and/or R),
        using interpolation
        """
        if observe_raw:
            if not self.xr_array_raw_initialized:
                self._xr_array_raw = xr.DataArray(entries_for_interp, coords=coords)
            xr_data_array = self._xr_array_raw
        else:
            xr_data_array = xr.DataArray(entries_for_interp, coords=coords)

        kwargs = {"t": t, "x": x, "r": r, "y": y, "z": z, "R": R}

        # Remove any None arguments
        kwargs = {key: value for key, value in kwargs.items() if value is not None}

        # Use xarray interpolation, return numpy array
        out = xr_data_array.interp(**kwargs, kwargs={"fill_value": fill_value}).values

        return out

    def _check_observe_raw(self, t):
        """
        Checks if the raw data should be observed exactly at the solution time points

        Args:
            t (np.ndarray, list, None): time points to observe

        Returns:
            t_observe (np.ndarray): time points to observe
            observe_raw (bool): True if observing the raw data
        """
        # if this is a time integral variable, t must be None and we observe either the
        # data times (for a discrete sum) or the solution times (for a continuous sum)
        if self.time_integral is not None:
            if self.time_integral.method == "discrete":
                # discrete sum should be observed at the discrete times
                t = self.time_integral.discrete_times
            else:
                # assume we can do a sufficiently accurate trapezoidal integration at t_pts
                t = self.t_pts

        observe_raw = (t is None) or (
            np.asarray(t).size == len(self.t_pts) and np.all(t == self.t_pts)
        )

        if observe_raw:
            t_observe = self.t_pts
        elif not isinstance(t, np.ndarray):
            if not isinstance(t, list):
                t = [t]
            t_observe = np.array(t)
        else:
            t_observe = t

        if t_observe[0] < self.t_pts[0]:
            raise ValueError(
                "The interpolation points must be greater than or equal to the initial solution time"
            )

        return t_observe, observe_raw

    @property
    def entries(self):
        """
        Returns the raw data entries of the processed variable. If the processed
        variable has not been initialized (i.e. the entries have not been
        calculated), then the processed variable is initialized first.
        """
        self.initialise()
        return self._entries_raw

    @property
    def data(self):
        """Same as entries, but different name"""
        return self.entries

    @property
    def entries_raw_initialized(self):
        return self._entries_raw is not None

    @property
    def xr_array_raw_initialized(self):
        return self._xr_array_raw is not None

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
            if self.all_solution_sensitivities:
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

        all_S_var = []
        for ts, ys, inputs_stacked, inputs, base_variable, dy_dp in zip(
            self.all_ts,
            self.all_ys,
            self.all_inputs_casadi,
            self.all_inputs,
            self.base_variables,
            self.all_solution_sensitivities["all"],
            strict=True,
        ):
            # Set up symbolic variables
            t_casadi = casadi.MX.sym("t")
            y_casadi = casadi.MX.sym("y", ys.shape[0])
            p_casadi = {
                name: casadi.MX.sym(name, value.shape[0])
                for name, value in inputs.items()
            }

            p_casadi_stacked = casadi.vertcat(*[p for p in p_casadi.values()])

            # Convert variable to casadi format for differentiating
            var_casadi = base_variable.to_casadi(t_casadi, y_casadi, inputs=p_casadi)
            dvar_dy = casadi.jacobian(var_casadi, y_casadi)
            dvar_dp = casadi.jacobian(var_casadi, p_casadi_stacked)

            # Convert to functions and evaluate index-by-index
            dvar_dy_func = casadi.Function(
                "dvar_dy", [t_casadi, y_casadi, p_casadi_stacked], [dvar_dy]
            )
            dvar_dp_func = casadi.Function(
                "dvar_dp", [t_casadi, y_casadi, p_casadi_stacked], [dvar_dp]
            )
            dvar_dy_eval = casadi.diagcat(
                *[
                    dvar_dy_func(t, ys[:, idx], inputs_stacked)
                    for idx, t in enumerate(ts)
                ]
            )
            dvar_dp_eval = casadi.vertcat(
                *[
                    dvar_dp_func(t, ys[:, idx], inputs_stacked)
                    for idx, t in enumerate(ts)
                ]
            )

            # Compute sensitivity
            S_var = dvar_dy_eval @ dy_dp + dvar_dp_eval

            if self.time_integral is not None:
                S_var = self.time_integral.postfix_sensitivities(
                    self._name, self.data, ts, inputs, S_var
                )

            all_S_var.append(S_var)

        S_var = np.vstack(all_S_var)
        sensitivities = {"all": S_var}

        # Add the individual sensitivity
        for i, name in enumerate(self.all_inputs[0].keys()):
            sensitivities[name] = S_var[:, i : i + 1].reshape(-1)

        # Save attribute
        self._sensitivities = sensitivities

    def _is_discrete_time_method(self):
        """Check if using discrete time integral method"""
        return (
            self.time_integral is not None and self.time_integral.method == "discrete"
        )

    @property
    def hermite_interpolation(self):
        return self.all_yps is not None

    def as_computed(self):
        """
        Allows a ProcessedVariable to be converted to a ComputedProcessedVariable for
        use together, e.g. when using a last state solution with a new simulation running
        with output variables in the solver.
        """

        def _stub_solution(self):
            """
            Return a lightweight object that looks like the parts of
            `pybamm.Solution` required by ProcessedVariableComputed, but without
            keeping the full state vector in memory.
            """

            class StubSolution:
                def __init__(self, ts, ys, inputs, inputs_casadi, sensitivities, t_pts):
                    self.all_ts = ts
                    self.all_ys = ys
                    self.all_inputs = inputs
                    self.all_inputs_casadi = inputs_casadi
                    self.sensitivities = sensitivities
                    self.t = t_pts

            return StubSolution(
                self.all_ts,
                self.all_ys,
                self.all_inputs,
                self.all_inputs_casadi,
                self.sensitivities,
                self.t_pts,
            )

        entries = self.entries  # shape: (..., n_t)

        # Move time to axis 0, then flatten spatial dims per timestep
        reshaped = np.moveaxis(entries, -1, 0)  # shape: (n_t, ...)
        base_data = [reshaped.reshape(reshaped.shape[0], -1)]  # (n_t, n_vars)

        cpv = pybamm.ProcessedVariableComputed(
            self.base_variables,
            self.base_variables_casadi,
            base_data,
            _stub_solution(self),
        )

        # add sensitivities if they exist
        if self.sensitivities:
            # TODO: test once #5058 is fixed
            cpv._sensitivities = self.sensitivities  # pragma: no cover

        return cpv


class ProcessedVariable0D(ProcessedVariable):
    def __init__(
        self,
        name: str,
        base_variables,
        base_variables_casadi,
        solution,
        time_integral: pybamm.ProcessedVariableTimeIntegral | None = None,
    ):
        self.dimensions = 0
        super().__init__(
            name,
            base_variables,
            base_variables_casadi,
            solution,
            time_integral=time_integral,
        )

    def _observe_postfix(self, entries, t):
        if self.time_integral is None:
            return entries
        return self.time_integral.postfix(
            entries, self.t_pts, self.all_inputs_casadi[0]
        )

    def _interp_setup(self, entries, t):
        # save attributes for interpolation
        entries_for_interp = entries
        coords_for_interp = {"t": t}

        return entries_for_interp, coords_for_interp

    def _shape(self, t):
        return [len(t)]


class ProcessedVariable1D(ProcessedVariable):
    """
    An object that can be evaluated at arbitrary (scalars or vectors) t and x, and
    returns the (interpolated) value of the base variable at that t and x.

    Parameters
    ----------
    variable : str
        The name of the variable
    base_variables : list of :class:`pybamm.Symbol`
        A list of base variables with a method `evaluate(t,y)`, each entry of which
        returns the value of that variable for that particular sub-solution.
        A Solution can be comprised of sub-solutions which are the solutions of
        different models.
        Note that this can be any kind of node in the expression tree, not
        just a :class:`pybamm.Variable`.
        When evaluated, returns an array of size (m,n)
    base_variables_casadi : list of :class:`casadi.Function`
        A list of casadi functions. When evaluated, returns the same thing as
        `base_Variable.evaluate` (but more efficiently).
    solution : :class:`pybamm.Solution`
        The solution object to be used to create the processed variables
    """

    def __init__(
        self,
        name: str,
        base_variables,
        base_variables_casadi,
        solution,
        time_integral: pybamm.ProcessedVariableTimeIntegral | None = None,
    ):
        self.dimensions = 1
        super().__init__(
            name,
            base_variables,
            base_variables_casadi,
            solution,
            time_integral=time_integral,
        )

    def _interp_setup(self, entries, t):
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

        # save attributes for interpolation
        coords_for_interp = {self.first_dimension: pts_for_interp, "t": t}

        return entries_for_interp, coords_for_interp

    def _shape(self, t):
        t_size = len(t)
        space_size = self.base_eval_shape[0]
        return [space_size, t_size]


class ProcessedVariable2D(ProcessedVariable):
    """
    An object that can be evaluated at arbitrary (scalars or vectors) t and x, and
    returns the (interpolated) value of the base variable at that t and x.

    Parameters
    ----------
    variable : str
        The name of the variable
    base_variables : list of :class:`pybamm.Symbol`
        A list of base variables with a method `evaluate(t,y)`, each entry of which
        returns the value of that variable for that particular sub-solution.
        A Solution can be comprised of sub-solutions which are the solutions of
        different models.
        Note that this can be any kind of node in the expression tree, not
        just a :class:`pybamm.Variable`.
        When evaluated, returns an array of size (m,n)
    base_variables_casadi : list of :class:`casadi.Function`
        A list of casadi functions. When evaluated, returns the same thing as
        `base_Variable.evaluate` (but more efficiently).
    solution : :class:`pybamm.Solution`
        The solution object to be used to create the processed variables
    """

    def __init__(
        self,
        name: str,
        base_variables,
        base_variables_casadi,
        solution,
        time_integral: pybamm.ProcessedVariableTimeIntegral | None = None,
    ):
        self.dimensions = 2
        super().__init__(
            name,
            base_variables,
            base_variables_casadi,
            solution,
            time_integral=time_integral,
        )
        first_dim_nodes = self.mesh.nodes
        first_dim_edges = self.mesh.edges
        second_dim_nodes = self.base_variables[0].secondary_mesh.nodes
        if self.base_eval_size // len(second_dim_nodes) == len(first_dim_nodes):
            first_dim_pts = first_dim_nodes
        elif self.base_eval_size // len(second_dim_nodes) == len(first_dim_edges):
            first_dim_pts = first_dim_edges

        second_dim_pts = second_dim_nodes
        self.first_dim_size = len(first_dim_pts)
        self.second_dim_size = len(second_dim_pts)

    def _interp_setup(self, entries, t):
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
        first_dim_pts_for_interp = first_dim_pts
        second_dim_pts_for_interp = second_dim_pts

        # Set pts to edges for nicer plotting
        self.first_dim_pts = first_dim_edges
        self.second_dim_pts = second_dim_edges

        # save attributes for interpolation
        coords_for_interp = {
            self.first_dimension: first_dim_pts_for_interp,
            self.second_dimension: second_dim_pts_for_interp,
            "t": t,
        }

        return entries_for_interp, coords_for_interp

    def _shape(self, t):
        first_dim_size = self.first_dim_size
        second_dim_size = self.second_dim_size
        t_size = len(t)
        return [first_dim_size, second_dim_size, t_size]


class ProcessedVariable2DSciKitFEM(ProcessedVariable2D):
    """
    An object that can be evaluated at arbitrary (scalars or vectors) t and x, and
    returns the (interpolated) value of the base variable at that t and x.

    Parameters
    ----------
    variable : str
        The name of the variable
    base_variables : list of :class:`pybamm.Symbol`
        A list of base variables with a method `evaluate(t,y)`, each entry of which
        returns the value of that variable for that particular sub-solution.
        A Solution can be comprised of sub-solutions which are the solutions of
        different models.
        Note that this can be any kind of node in the expression tree, not
        just a :class:`pybamm.Variable`.
        When evaluated, returns an array of size (m,n)
    base_variables_casadi : list of :class:`casadi.Function`
        A list of casadi functions. When evaluated, returns the same thing as
        `base_Variable.evaluate` (but more efficiently).
    solution : :class:`pybamm.Solution`
        The solution object to be used to create the processed variables
    """

    def __init__(
        self,
        name: str,
        base_variables,
        base_variables_casadi,
        solution,
        time_integral: pybamm.ProcessedVariableTimeIntegral | None = None,
    ):
        self.dimensions = 2
        super(ProcessedVariable2D, self).__init__(
            name,
            base_variables,
            base_variables_casadi,
            solution,
            time_integral=time_integral,
        )
        y_sol = self.mesh.edges["y"]
        z_sol = self.mesh.edges["z"]

        self.first_dim_size = len(y_sol)
        self.second_dim_size = len(z_sol)

    def _interp_setup(self, entries, t):
        y_sol = self.mesh.edges["y"]
        z_sol = self.mesh.edges["z"]

        # assign attributes for reference
        self.y_sol = y_sol
        self.z_sol = z_sol
        self.first_dimension = "y"
        self.second_dimension = "z"
        self.first_dim_pts = y_sol
        self.second_dim_pts = z_sol

        # save attributes for interpolation
        coords_for_interp = {"y": y_sol, "z": z_sol, "t": t}

        return entries, coords_for_interp

    def _observe_postfix(self, entries, t):
        shape = entries.shape
        entries = entries.transpose(1, 0, 2).reshape(shape)
        return entries


class ProcessedVariable2DFVM(ProcessedVariable):
    def __init__(
        self,
        name: str,
        base_variables,
        base_variables_casadi,
        solution,
        time_integral: pybamm.ProcessedVariableTimeIntegral | None = None,
    ):
        self.dimensions = 2
        super().__init__(
            name,
            base_variables,
            base_variables_casadi,
            solution,
            time_integral=time_integral,
        )
        num_nodes_lr = len(self.mesh.nodes_lr)
        num_nodes_tb = len(self.mesh.nodes_tb)
        num_edges_lr = len(self.mesh.edges_lr)
        num_edges_tb = len(self.mesh.edges_tb)

        if not self.base_variables[0].evaluates_on_edges("primary"):
            self.first_dim_size = num_nodes_lr
            self.second_dim_size = num_nodes_tb
            self.first_dim_pts = self.mesh.nodes_lr
            self.second_dim_pts = self.mesh.nodes_tb
        elif base_variables[0].evaluates_on_edges("primary") == "lr":
            # Evaluates on edges in the LR direction
            # Note that if the variable has the same number of nodes in the LR direction and the TB direction,
            # Then we assume it evaluates on edges in the LR direction for lack of a better option
            self.first_dim_size = num_edges_lr
            self.second_dim_size = num_nodes_tb
            self.first_dim_pts = self.mesh.edges_lr
            self.second_dim_pts = self.mesh.nodes_tb
        elif base_variables[0].evaluates_on_edges("primary") == "tb":
            # Evaluates on edges in the TB direction
            self.first_dim_size = num_nodes_lr
            self.second_dim_size = num_edges_tb
            self.first_dim_pts = self.mesh.nodes_lr
            self.second_dim_pts = self.mesh.edges_tb
        elif base_variables[0].evaluates_on_edges("primary"):
            self.first_dim_size = num_edges_lr
            self.second_dim_size = num_edges_tb
            self.first_dim_pts = self.mesh.edges_lr
            self.second_dim_pts = self.mesh.edges_tb
        else:
            raise ValueError(
                f"ProcessedVariable2DFVM: Invalid shape {base_variables[0].shape}"
            )

    def _interp_setup(self, entries, t):
        self.first_dimension = "x"
        self.second_dimension = "z"
        coords_for_interp = {
            self.first_dimension: self.first_dim_pts,
            self.second_dimension: self.second_dim_pts,
            "t": t,
        }
        return entries, coords_for_interp

    def _shape(self, t):
        return [self.first_dim_size, self.second_dim_size, len(t)]


class ProcessedVariableRawFVM(ProcessedVariable):
    def _shape(self, t):
        return [self.base_variables[0].size, len(t)]

    def initialise(self):
        if self.entries_raw_initialized:
            return
        entries = self.observe_raw()
        self._entries_raw = entries


class ProcessedVariable3D(ProcessedVariable):
    """
    An object that can be evaluated at arbitrary (scalars or vectors) t and x, and
    returns the (interpolated) value of the base variable at that t and x.

    Parameters
    ----------
    variable : str
        The name of the variable
    base_variables : list of :class:`pybamm.Symbol`
        A list of base variables with a method `evaluate(t,y)`, each entry of which
        returns the value of that variable for that particular sub-solution.
        A Solution can be comprised of sub-solutions which are the solutions of
        different models.
        Note that this can be any kind of node in the expression tree, not
        just a :class:`pybamm.Variable`.
        When evaluated, returns an array of size (m,n)
    base_variables_casadi : list of :class:`casadi.Function`
        A list of casadi functions. When evaluated, returns the same thing as
        `base_Variable.evaluate` (but more efficiently).
    solution : :class:`pybamm.Solution`
        The solution object to be used to create the processed variables
    """

    def __init__(
        self,
        name: str,
        base_variables,
        base_variables_casadi,
        solution,
        time_integral: pybamm.ProcessedVariableTimeIntegral | None = None,
    ):
        self.dimensions = 3
        super().__init__(
            name,
            base_variables,
            base_variables_casadi,
            solution,
            time_integral=time_integral,
        )
        first_dim_nodes = self.mesh.nodes
        first_dim_edges = self.mesh.edges
        second_dim_nodes = self.base_variables[0].secondary_mesh.nodes
        third_dim_nodes = self.base_variables[0].tertiary_mesh.nodes
        if self.base_eval_size // (len(second_dim_nodes) * len(third_dim_nodes)) == len(
            first_dim_nodes
        ):
            first_dim_pts = first_dim_nodes
        elif self.base_eval_size // (
            len(second_dim_nodes) * len(third_dim_nodes)
        ) == len(first_dim_edges):
            first_dim_pts = first_dim_edges

        second_dim_pts = second_dim_nodes
        third_dim_pts = third_dim_nodes
        self.first_dim_size = len(first_dim_pts)
        self.second_dim_size = len(second_dim_pts)
        self.third_dim_size = len(third_dim_pts)

    def _interp_setup(self, entries, t):
        """
        Initialise a 3D object that depends on x, y, and z, or x, r, and R.
        """
        first_dim_nodes = self.mesh.nodes
        first_dim_edges = self.mesh.edges
        second_dim_nodes = self.base_variables[0].secondary_mesh.nodes
        second_dim_edges = self.base_variables[0].secondary_mesh.edges
        third_dim_nodes = self.base_variables[0].tertiary_mesh.nodes
        third_dim_edges = self.base_variables[0].tertiary_mesh.edges
        if self.base_eval_size // (len(second_dim_nodes) * len(third_dim_nodes)) == len(
            first_dim_nodes
        ):
            first_dim_pts = first_dim_nodes
        elif self.base_eval_size // (
            len(second_dim_nodes) * len(third_dim_nodes)
        ) == len(first_dim_edges):
            first_dim_pts = first_dim_edges

        second_dim_pts = second_dim_nodes
        third_dim_pts = third_dim_nodes

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

        # add points outside tertiary dimension domain for extrapolation to
        # boundaries
        extrap_space_third_dim_left = np.array(
            [2 * third_dim_pts[0] - third_dim_pts[1]]
        )
        extrap_space_third_dim_right = np.array(
            [2 * third_dim_pts[-1] - third_dim_pts[-2]]
        )
        third_dim_pts = np.concatenate(
            [
                extrap_space_third_dim_left,
                third_dim_pts,
                extrap_space_third_dim_right,
            ]
        )
        extrap_entries_third_dim_left = np.expand_dims(
            2 * entries_for_interp[:, :, 0] - entries_for_interp[:, :, 1], axis=2
        )
        extrap_entries_third_dim_right = np.expand_dims(
            2 * entries_for_interp[:, :, -1] - entries_for_interp[:, :, -2], axis=2
        )
        entries_for_interp = np.concatenate(
            [
                extrap_entries_third_dim_left,
                entries_for_interp,
                extrap_entries_third_dim_right,
            ],
            axis=2,
        )

        self.spatial_variable_names = {
            k: self._process_spatial_variable_names(v)
            for k, v in self.spatial_variables.items()
        }

        self.first_dimension = self.spatial_variable_names["primary"]
        self.second_dimension = self.spatial_variable_names["secondary"]
        self.third_dimension = self.spatial_variable_names["tertiary"]

        # assign attributes for reference
        first_dim_pts_for_interp = first_dim_pts
        second_dim_pts_for_interp = second_dim_pts
        third_dim_pts_for_interp = third_dim_pts

        # Set pts to edges for nicer plotting
        self.first_dim_pts = first_dim_edges
        self.second_dim_pts = second_dim_edges
        self.third_dim_pts = third_dim_edges

        # save attributes for interpolation
        coords_for_interp = {
            self.first_dimension: first_dim_pts_for_interp,
            self.second_dimension: second_dim_pts_for_interp,
            self.third_dimension: third_dim_pts_for_interp,
            "t": t,
        }

        return entries_for_interp, coords_for_interp

    def _shape(self, t):
        first_dim_size = self.first_dim_size
        second_dim_size = self.second_dim_size
        third_dim_size = self.third_dim_size
        t_size = len(t)
        return [first_dim_size, second_dim_size, third_dim_size, t_size]


class ProcessedVariable3DSciKitFEM(ProcessedVariable3D):
    """
    An object that can be evaluated at arbitrary (scalars or vectors) t and x, and
    returns the (interpolated) value of the base variable at that t and x.

    Parameters
    ----------
    variable : str
        The name of the variable
    base_variables : list of :class:`pybamm.Symbol`
        A list of base variables with a method `evaluate(t,y)`, each entry of which
        returns the value of that variable for that particular sub-solution.
        A Solution can be comprised of sub-solutions which are the solutions of
        different models.
        Note that this can be any kind of node in the expression tree, not
        just a :class:`pybamm.Variable`.
        When evaluated, returns an array of size (m,n)
    base_variables_casadi : list of :class:`casadi.Function`
        A list of casadi functions. When evaluated, returns the same thing as
        `base_Variable.evaluate` (but more efficiently).
    solution : :class:`pybamm.Solution`
        The solution object to be used to create the processed variables
    """

    def __init__(
        self,
        name: str,
        base_variables,
        base_variables_casadi,
        solution,
        time_integral: pybamm.ProcessedVariableTimeIntegral | None = None,
    ):
        self.dimensions = 3
        super(ProcessedVariable3D, self).__init__(
            name,
            base_variables,
            base_variables_casadi,
            solution,
            time_integral=time_integral,
        )
        x_nodes = self.mesh.nodes
        x_edges = self.mesh.edges
        y_sol = self.base_variables[0].secondary_mesh.edges["y"]
        z_sol = self.base_variables[0].secondary_mesh.edges["z"]
        if self.base_eval_size // (len(y_sol) * len(z_sol)) == len(x_nodes):
            x_sol = x_nodes
        elif self.base_eval_size // (len(y_sol) * len(z_sol)) == len(x_edges):
            x_sol = x_edges

        self.first_dim_size = len(x_sol)
        self.second_dim_size = len(y_sol)
        self.third_dim_size = len(z_sol)

    def _interp_setup(self, entries, t):
        x_nodes = self.mesh.nodes
        x_edges = self.mesh.edges
        y_sol = self.base_variables[0].secondary_mesh.edges["y"]
        z_sol = self.base_variables[0].secondary_mesh.edges["z"]
        if self.base_eval_size // (len(y_sol) * len(z_sol)) == len(x_nodes):
            x_sol = x_nodes
        elif self.base_eval_size // (len(y_sol) * len(z_sol)) == len(x_edges):
            x_sol = x_edges

        # assign attributes for reference
        self.x_sol = x_sol
        self.y_sol = y_sol
        self.z_sol = z_sol
        self.first_dimension = "x"
        self.second_dimension = "y"
        self.third_dimension = "z"
        self.first_dim_pts = x_sol
        self.second_dim_pts = y_sol
        self.third_dim_pts = z_sol

        # save attributes for interpolation
        coords_for_interp = {"x": x_sol, "y": y_sol, "z": z_sol, "t": t}

        return entries, coords_for_interp

    def _observe_postfix(self, entries, t):
        shape = entries.shape
        entries = entries.transpose(0, 2, 1, 3).reshape(shape)
        return entries


class ProcessedVariableUnstructured(ProcessedVariable):
    """
    A processed variable for data on an unstructured mesh (e.g., from a FEM solution).
    This class correctly uses scipy's LinearNDInterpolator for spatial interpolation,
    which is the required method for scattered data points from a FEM mesh.

    Parameters
    ----------
    name : str
        The name of the variable
    base_variables : list of :class:`pybamm.Symbol`
        A list of base variables with a method `evaluate(t,y)`, each entry of which
        returns the value of that variable for that particular sub-solution.
        A Solution can be comprised of sub-solutions which are the solutions of
        different models.
        base_variables : list of :class:`pybamm.Symbol`
        A list of base variables with a method `evaluate(t,y)`, each entry of which
        returns the value of that variable for that particular sub-solution.
        A Solution can be comprised of sub-solutions which are the solutions of
        different models.
        Note that this can be any kind of node in the expression tree, not
        just a :class:`pybamm.Variable`.
        When evaluated, returns an array of size (m,n)
    base_variables_casadi : list of :class:`casadi.Function`
        A list of casadi functions. When evaluated, returns the same thing as
        `base_Variable.evaluate` (but more efficiently).
    solution : :class:`pybamm.Solution`
        The solution object to be used to create the processed variables
    time_integral : pybamm.ProcessedVariableTimeIntegral, optional
        An optional time integral object to handle time integration of the variable.
        If provided, the processed variable will handle time integration using this object.
    """

    def __init__(
        self,
        name: str,
        base_variables,
        base_variables_casadi,
        solution,
        time_integral: pybamm.ProcessedVariableTimeIntegral | None = None,
    ):
        self.dimensions = base_variables[0].mesh.dimension
        super().__init__(
            name,
            base_variables,
            base_variables_casadi,
            solution,
            time_integral=time_integral,
        )
        self._time_interpolator = None

    def initialise(self):
        if self.entries_raw_initialized:
            return
        self._entries_raw = self.observe_raw()

        from scipy.interpolate import interp1d

        self._time_interpolator = interp1d(
            self.t_pts,
            self._entries_raw,
            kind="linear",
            axis=1,  # Interpolate along the time axis
            bounds_error=False,
            fill_value="extrapolate",
        )

    def _shape(self, t):
        return [self.mesh.npts, len(t)]

    def __call__(self, t, x=None, y=None, z=None, r=None, R=None, fill_value=np.nan):
        from scipy.interpolate import LinearNDInterpolator

        self.initialise()

        data_at_t = self._time_interpolator(t)

        spatial_coords_provided = any(c is not None for c in [x, y, z])
        if not spatial_coords_provided:
            return data_at_t  # Return all node data if no spatial coords are given

        node_coords = self.mesh.nodes

        eval_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        output_shape = x.shape

        # If t was a single value (scalar), data_at_t is 1D array of (n_nodes)
        if isinstance(t, int | float):
            spatial_interpolator = LinearNDInterpolator(
                points=node_coords, values=data_at_t, fill_value=fill_value
            )
            result = spatial_interpolator(eval_points)
        else:  # If t was a vector, we must create an interpolator for each time step
            result = np.empty((len(eval_points), len(t)))
            for i in range(len(t)):
                spatial_interpolator = LinearNDInterpolator(
                    points=node_coords, values=data_at_t[:, i], fill_value=fill_value
                )
                result[:, i] = spatial_interpolator(eval_points)

        final_shape = (*output_shape, len(t) if not isinstance(t, int | float) else 1)
        return result.reshape(final_shape).squeeze()


def process_variable(name: str, base_variables, *args, **kwargs):
    mesh = base_variables[0].mesh
    domain = base_variables[0].domain

    # Evaluate base variable at initial time
    base_eval_shape = base_variables[0].shape
    base_eval_size = base_variables[0].size

    if isinstance(mesh, pybamm.ScikitFemSubMesh3D):
        return ProcessedVariableUnstructured(name, base_variables, *args, **kwargs)

    # handle 2D (in space) finite element variables differently
    if (
        mesh
        and "current collector" in domain
        and isinstance(mesh, pybamm.ScikitSubMesh2D)
    ):
        return ProcessedVariable2DSciKitFEM(name, base_variables, *args, **kwargs)
    if hasattr(base_variables[0], "secondary_mesh"):
        if "current collector" in base_variables[0].domains["secondary"] and isinstance(
            base_variables[0].secondary_mesh, pybamm.ScikitSubMesh2D
        ):
            return ProcessedVariable3DSciKitFEM(name, base_variables, *args, **kwargs)

    if mesh and hasattr(mesh, "edges_lr") and hasattr(mesh, "edges_tb"):
        return ProcessedVariable2DFVM(name, base_variables, *args, **kwargs)

    # check variable shape
    if len(base_eval_shape) == 0 or base_eval_shape[0] == 1:
        return ProcessedVariable0D(name, base_variables, *args, **kwargs)

    if mesh is None:
        return ProcessedVariable2DFVM(name, base_variables, *args, **kwargs)
    n = mesh.npts
    base_shape = base_eval_shape[0]
    # Try some shapes that could make the variable a 1D variable
    if base_shape in [n, n + 1]:
        return ProcessedVariable1D(name, base_variables, *args, **kwargs)

    # Try some shapes that could make the variable a 2D variable
    first_dim_nodes = mesh.nodes
    first_dim_edges = mesh.edges
    try:
        second_dim_pts = base_variables[0].secondary_mesh.nodes
    except AttributeError:
        try:
            return ProcessedVariable2DFVM(name, base_variables, *args, **kwargs)
        except AttributeError:
            return ProcessedVariableRawFVM(name, base_variables, *args, **kwargs)
    if base_eval_size // len(second_dim_pts) in [
        len(first_dim_nodes),
        len(first_dim_edges),
    ]:
        return ProcessedVariable2D(name, base_variables, *args, **kwargs)

    # Try some shapes that could make the variable a 3D variable
    tertiary_pts = base_variables[0].tertiary_mesh.nodes
    if base_eval_size // (len(second_dim_pts) * len(tertiary_pts)) in [
        len(first_dim_nodes),
        len(first_dim_edges),
    ]:
        return ProcessedVariable3D(name, base_variables, *args, **kwargs)

    raise NotImplementedError(f"Shape not recognized for {base_variables[0]}")


def _is_f_contiguous(all_ys):
    """
    Check if all the ys are f-contiguous in memory

    Args:
        all_ys (list of np.ndarray): list of all ys

    Returns:
        bool: True if all ys are f-contiguous
    """

    return all(isinstance(y, np.ndarray) and y.data.f_contiguous for y in all_ys)


def _is_sorted(t):
    """
    Check if an array is sorted

    Args:
        t (np.ndarray): array to check

    Returns:
        bool: True if array is sorted
    """
    return np.all(t[:-1] <= t[1:])


def _find_ts_indices(ts, t):
    """
    Parameters:
    - ts: A list of numpy arrays (each sorted) whose values are successively increasing.
    - t: A sorted list or array of values to find within ts.

    Returns:
    - indices: A list of indices from `ts` such that at least one value of `t` falls within ts[idx].
    """

    indices = []

    # Get the minimum and maximum values of the target values `t`
    t_min, t_max = t[0], t[-1]

    # Step 1: Use binary search to find the range of `ts` arrays where t_min and t_max could lie
    low_idx = bisect.bisect_left([ts_arr[-1] for ts_arr in ts], t_min)
    high_idx = bisect.bisect_right([ts_arr[0] for ts_arr in ts], t_max)

    # Step 2: Iterate over the identified range
    for idx in range(low_idx, high_idx):
        ts_min, ts_max = ts[idx][0], ts[idx][-1]

        # Binary search within `t` to check if any value falls within [ts_min, ts_max]
        i = bisect.bisect_left(t, ts_min)
        if i < len(t) and t[i] <= ts_max:
            # At least one value of t is within ts[idx]
            indices.append(idx)

    # extrapolating
    if (t[-1] > ts[-1][-1]) and (len(indices) == 0 or indices[-1] != len(ts) - 1):
        indices.append(len(ts) - 1)

    return indices
