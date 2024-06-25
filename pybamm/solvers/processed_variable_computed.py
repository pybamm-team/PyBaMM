#
# Processed Variable class
#
import casadi
import numpy as np
import pybamm
from scipy.integrate import cumulative_trapezoid
import xarray as xr


class ProcessedVariableComputed:
    """
    An object that can be evaluated at arbitrary (scalars or vectors) t and x, and
    returns the (interpolated) value of the base variable at that t and x.

    The 'Computed' variant of ProcessedVariable deals with variables that have
    been derived at solve time (see the 'output_variables' solver option),
    where the full state-vector is not itself propogated and returned.

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
    base_variable_data : list of :numpy:array
        A list of numpy arrays, the returned evaluations.
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
        base_variables_data,
        solution,
        warn=True,
        cumtrapz_ic=None,
    ):
        self.base_variables = base_variables
        self.base_variables_casadi = base_variables_casadi
        self.base_variables_data = base_variables_data

        self.all_ts = solution.all_ts
        self.all_ys = solution.all_ys
        self.all_inputs = solution.all_inputs
        self.all_inputs_casadi = solution.all_inputs_casadi

        self.mesh = base_variables[0].mesh
        self.domain = base_variables[0].domain
        self.domains = base_variables[0].domains
        self.warn = warn
        self.cumtrapz_ic = cumtrapz_ic

        # Sensitivity starts off uninitialized, only set when called
        self._sensitivities = None
        self.solution_sensitivities = solution.sensitivities

        # Store time
        self.t_pts = solution.t

        # Evaluate base variable at initial time
        self.base_eval_shape = self.base_variables[0].shape
        self.base_eval_size = self.base_variables[0].size
        self.unroll_params = {}

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
                            f"Shape not recognized for {base_variables[0]} "
                            + "(note processing of 3D variables is not yet implemented)"
                        )

    def add_sensitivity(self, param, data):
        # unroll from sparse representation into n-d matrix
        # Note: then flatten and convert to casadi.DM for consistency with
        #       full state-vector ProcessedVariable sensitivities
        self._sensitivities[param] = casadi.DM(self.unroll(data).flatten())

    def _unroll_nnz(self, realdata=None):
        # unroll in nnz != numel, otherwise copy
        if realdata is None:
            realdata = self.base_variables_data
        if isinstance(self.base_variables_casadi[0], casadi.Function):  # casadi fcn
            sp = self.base_variables_casadi[0](0, 0, 0).sparsity()
            nnz = sp.nnz()
            numel = sp.numel()
            row = sp.row()
        elif "nnz" in dir(self.base_variables_casadi[0]):  # IREE fcn
            sp = self.base_variables_casadi[0]
            nnz = sp.nnz
            numel = sp.numel
            row = sp.row
        if nnz != numel:
            data = [None] * len(realdata)
            for datak in range(len(realdata)):
                data[datak] = np.zeros(self.base_eval_shape[0] * len(self.t_pts))
                var_data = realdata[0].flatten()
                k = 0
                for t_i in range(len(self.t_pts)):
                    base = t_i * numel
                    for r in row:
                        data[datak][base + r] = var_data[k]
                        k = k + 1
        else:
            data = realdata
        return data

    def unroll_0D(self, realdata=None):
        if realdata is None:
            realdata = self.base_variables_data
        return np.concatenate(realdata, axis=0).flatten()

    def unroll_1D(self, realdata=None):
        len_space = self.base_eval_shape[0]
        return (
            np.concatenate(self._unroll_nnz(realdata), axis=0)
            .reshape((len(self.t_pts), len_space))
            .transpose()
        )

    def unroll_2D(self, realdata=None, n_dim1=None, n_dim2=None, axis_swaps=None):
        # initialise settings on first run
        if axis_swaps is None:
            axis_swaps = []
        if not self.unroll_params:
            self.unroll_params["n_dim1"] = n_dim1
            self.unroll_params["n_dim2"] = n_dim2
            self.unroll_params["axis_swaps"] = axis_swaps
        # use stored settings on subsequent runs
        if not n_dim1:
            n_dim1 = self.unroll_params["n_dim1"]
            n_dim2 = self.unroll_params["n_dim2"]
            axis_swaps = self.unroll_params["axis_swaps"]
        entries = np.concatenate(self._unroll_nnz(realdata), axis=0).reshape(
            (len(self.t_pts), n_dim1, n_dim2)
        )
        for a, b in axis_swaps:
            entries = np.moveaxis(entries, a, b)
        return entries

    def unroll(self, realdata=None):
        if self.dimensions == 0:
            return self.unroll_0D(realdata=realdata)
        elif self.dimensions == 1:
            return self.unroll_1D(realdata=realdata)
        elif self.dimensions == 2:
            return self.unroll_2D(realdata=realdata)
        else:
            # Raise error for 3D variable
            raise NotImplementedError(f"Unsupported data dimension: {self.dimensions}")

    def initialise_0D(self):
        entries = self.unroll_0D()

        if self.cumtrapz_ic is not None:
            entries = cumulative_trapezoid(
                entries, self.t_pts, initial=float(self.cumtrapz_ic)
            )

        # set up interpolation
        self._xr_data_array = xr.DataArray(entries, coords=[("t", self.t_pts)])

        self.entries = entries
        self.dimensions = 0

    def initialise_1D(self, fixed_t=False):
        entries = self.unroll_1D()

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
        if self.domain[0].endswith("particle"):
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
        elif self.domain[0].endswith("particle size"):
            self.first_dimension = "R"
            self.R_sol = space
        else:
            self.first_dimension = "x"
            self.x_sol = space

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

        entries = self.unroll_2D(
            realdata=None,
            n_dim1=second_dim_size,
            n_dim2=first_dim_size,
            axis_swaps=[(0, 2), (0, 1)],
        )

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

        # Process r-x, x-z, r-R, R-x, or R-z
        if self.domain[0].endswith("particle") and self.domains["secondary"][
            0
        ].endswith("electrode"):
            self.first_dimension = "r"
            self.second_dimension = "x"
            self.r_sol = first_dim_pts
            self.x_sol = second_dim_pts
        elif self.domain[0] in [
            "negative electrode",
            "separator",
            "positive electrode",
        ] and self.domains["secondary"] == ["current collector"]:
            self.first_dimension = "x"
            self.second_dimension = "z"
            self.x_sol = first_dim_pts
            self.z_sol = second_dim_pts
        elif self.domain[0].endswith("particle") and self.domains["secondary"][
            0
        ].endswith("particle size"):
            self.first_dimension = "r"
            self.second_dimension = "R"
            self.r_sol = first_dim_pts
            self.R_sol = second_dim_pts
        elif self.domain[0].endswith("particle size") and self.domains["secondary"][
            0
        ].endswith("electrode"):
            self.first_dimension = "R"
            self.second_dimension = "x"
            self.R_sol = first_dim_pts
            self.x_sol = second_dim_pts
        elif self.domain[0].endswith("particle size") and self.domains["secondary"] == [
            "current collector"
        ]:
            self.first_dimension = "R"
            self.second_dimension = "z"
            self.R_sol = first_dim_pts
            self.z_sol = second_dim_pts
        else:  # pragma: no cover
            raise pybamm.DomainError(
                f"Cannot process 2D object with domains '{self.domains}'."
            )

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
        entries = self.unroll_2D(
            realdata=None,
            n_dim1=len_y,
            n_dim2=len_z,
            axis_swaps=[(0, 2)],
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
        self._xr_data_array = xr.DataArray(
            entries,
            coords={"y": y_sol, "z": z_sol, "t": self.t_pts},
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
        return self._sensitivities
