import pybamm

from scipy.sparse import (
    diags,
    eye,
    kron,
    csr_matrix,
    hstack,
    block_diag,
    spdiags,
    lil_matrix,
)
import numpy as np


class FiniteVolume3D(pybamm.SpatialMethod):
    """
    A class which implements the steps specific to the finite volume method during
    discretisation.

    For broadcast and mass_matrix, we follow the default behaviour from SpatialMethod.

    Parameters
    ----------
    options : dict-like, optional
        A dictionary of options to be passed to the spatial method. The only option
        currently available is "extrapolation", which has options for "order" and "use_bcs".
        It sets the order separately for `pybamm.BoundaryValue` and `pybamm.BoundaryGradient`.
        Default is "linear" for the value and quadratic for the gradient.
    """

    def __init__(self, options=None):
        super().__init__(options)

    def build(self, mesh):
        super().build(mesh)

        # add npts_for_broadcast to mesh domains for this particular discretisation
        for dom in mesh.keys():
            mesh[dom].npts_for_broadcast_to_nodes = mesh[dom].npts

    def compute_spiral_metric(self, submesh):
        """
        Compute the custom metric for the spiral coordinate system.
        This function should return a scaling factor for each node or edge
        based on the local geometry of the spiral-wound structure.
        """
        r_nodes = submesh.nodes_x
        spiral_metric = 1 + 0.1 * np.sin(2 * np.pi * r_nodes / r_nodes.max())

        return spiral_metric

    def spatial_variable(self, symbol):
        """
        Creates a discretised spatial variable compatible with
        the FiniteVolume method.

        Parameters
        ----------
        symbol : :class:`pybamm.SpatialVariable`
            The spatial variable to be discretised.

        Returns
        -------
        :class:`pybamm.Vector`
            Contains the discretised spatial variable
        """
        symbol_mesh = self.mesh[symbol.domain]
        symbol_direction = symbol.direction
        if symbol_mesh.dimension != 3:
            raise ValueError(f"Spatial variable {symbol} is not in 3D")
        repeats = self._get_auxiliary_domain_repeats(symbol.domains)

        if symbol.evaluates_on_edges("primary"):
            X, Y, Z = np.meshgrid(
                symbol_mesh.edges_x,
                symbol_mesh.edges_y,
                symbol_mesh.edges_z,
                indexing="ij",
            )
            x = X.flatten(order="F")
            y = Y.flatten(order="F")
            z = Z.flatten(order="F")
        else:
            X, Y, Z = np.meshgrid(
                symbol_mesh.nodes_x,
                symbol_mesh.nodes_y,
                symbol_mesh.nodes_z,
                indexing="ij",
            )
            x = X.flatten(order="F")
            y = Y.flatten(order="F")
            z = Z.flatten(order="F")
        if symbol_direction == "x":
            entries = np.tile(x, repeats)
        elif symbol_direction == "y":
            entries = np.tile(y, repeats)
        elif symbol_direction == "z":
            entries = np.tile(z, repeats)
        else:
            if symbol_direction not in ["x", "y", "z"]:
                raise ValueError(
                    f"Symbol direction '{symbol_direction}' not supported for direct construction "
                    "as a spatial variable vector. Discretise the variable directly."
                )

        return pybamm.Vector(entries, domains=symbol.domains)

    def _gradient(self, symbol, discretised_symbol, boundary_conditions, direction):
        """
        Gradient with a specific direction (x, y, or z)
        """
        domain = symbol.domain

        # Add Dirichlet boundary conditions, if defined
        if direction == "x":
            relevant_bcs = ["left", "right"]
        elif direction == "y":
            relevant_bcs = ["front", "back"]
        elif direction == "z":
            relevant_bcs = ["bottom", "top"]
        else:
            raise ValueError(f"Direction {direction} not supported")

        bcs = {}
        if symbol in boundary_conditions:
            bcs = {
                key: boundary_conditions[symbol][key]
                for key in relevant_bcs
                if key in boundary_conditions[symbol]
            }

        if any(bc[1] == "Dirichlet" for bc in bcs.values()):
            discretised_symbol, domain = self.add_ghost_nodes(
                symbol, discretised_symbol, bcs
            )

        grad_mat = self.gradient_matrix(domain, symbol.domains, direction)

        grad = grad_mat @ discretised_symbol
        grad.copy_domains(symbol)

        # Add Neumann boundary conditions, if defined
        if symbol in boundary_conditions:
            bcs = {
                key: boundary_conditions[symbol][key]
                for key in relevant_bcs
                if key in boundary_conditions[symbol]
            }
            if any(bc[1] == "Neumann" for bc in bcs.values()):
                grad = self.add_neumann_values(symbol, grad, bcs, domain)

        return grad

    def gradient(self, symbol, discretised_symbol, boundary_conditions):
        """
        3D finite-volume gradient
        """
        grad_x = self._gradient(symbol, discretised_symbol, boundary_conditions, "x")
        grad_y = self._gradient(symbol, discretised_symbol, boundary_conditions, "y")
        grad_z = self._gradient(symbol, discretised_symbol, boundary_conditions, "z")

        grad = pybamm.VectorField3D(grad_x, grad_y, grad_z)
        return grad

    def gradient_matrix(self, domain, domains, direction):
        """
        Gradient matrix for finite volumes in 3D.
        Following PyBaMM convention: grad(y) = (y[1:] - y[:-1])/dx
        """
        if direction not in ["x", "y", "z"]:
            raise ValueError(
                f"Invalid direction '{direction}'. Must be 'x', 'y', or 'z'"
            )

        submesh = self.mesh[domain]

        # Create matrix using submesh
        n_x = submesh.npts_x
        n_y = submesh.npts_y
        n_z = submesh.npts_z

        # Use d_nodes for gradient (distance between node centers)
        e_x = 1 / submesh.d_nodes_x
        e_y = 1 / submesh.d_nodes_y
        e_z = 1 / submesh.d_nodes_z

        if direction == "x":
            # Gradient in x-direction: shape (n_x-1)*n_y*n_z x n_x*n_y*n_z
            if submesh.coord_sys == "cylindrical polar":
                r_nodes = submesh.nodes_x
                r_weights = r_nodes[:-1]  # Use left node for face weighting
                sub_matrix = diags(
                    [-e_x * r_weights, e_x * r_weights],
                    [0, 1],
                    shape=(n_x - 1, n_x),
                )
                sub_matrix = block_diag([sub_matrix] * (n_y * n_z))
            elif submesh.coord_sys == "spherical polar":
                r_nodes = submesh.nodes_x
                r_weights = r_nodes[:-1] ** 2  # Use left node for face weighting
                sub_matrix = diags(
                    [-e_x * r_weights, e_x * r_weights],
                    [0, 1],
                    shape=(n_x - 1, n_x),
                )
                sub_matrix = block_diag([sub_matrix] * (n_y * n_z))
            elif submesh.coord_sys == "spiral":
                spiral_metric = self.compute_spiral_metric(submesh)
                spiral_metric_faces = 0.5 * (spiral_metric[:-1] + spiral_metric[1:])
                if len(e_x) == 1:
                    e_x_broadcast = np.full(len(spiral_metric_faces), e_x[0])
                else:
                    e_x_broadcast = e_x[: len(spiral_metric_faces)]
                sub_matrix = diags(
                    [
                        -e_x_broadcast * spiral_metric_faces,
                        e_x_broadcast * spiral_metric_faces,
                    ],
                    [0, 1],
                    shape=(n_x - 1, n_x),
                )
                sub_matrix = block_diag([sub_matrix] * (n_y * n_z))
            else:
                sub_matrix = diags([-e_x, e_x], [0, 1], shape=(n_x - 1, n_x))
                sub_matrix = block_diag([sub_matrix] * (n_y * n_z))

        elif direction == "y":
            # Gradient in y-direction: shape n_x*(n_y-1)*n_z x n_x*n_y*n_z
            if submesh.coord_sys == "cylindrical polar":
                r_nodes = submesh.nodes_x
                r_weights = np.tile(r_nodes, n_y - 1)
                e_y_weighted = (
                    np.repeat(e_y, n_x) / r_weights[: len(np.repeat(e_y, n_x))]
                )
                sub_matrix = diags(
                    [-e_y_weighted, e_y_weighted],
                    [0, n_x],
                    shape=(n_x * (n_y - 1), n_x * n_y),
                )
                sub_matrix = block_diag([sub_matrix] * n_z)
            elif submesh.coord_sys == "spherical polar":
                r_nodes = submesh.nodes_x
                theta_nodes = submesh.nodes_y
                # Spherical coordinate weighting
                r_weights = np.tile(r_nodes, n_y - 1)
                sin_weights = np.repeat(np.sin(theta_nodes[:-1]), n_x)
                e_y_weighted = np.repeat(e_y, n_x) / (
                    r_weights[: len(np.repeat(e_y, n_x))] * sin_weights
                )
                sub_matrix = diags(
                    [-e_y_weighted, e_y_weighted],
                    [0, n_x],
                    shape=(n_x * (n_y - 1), n_x * n_y),
                )
                sub_matrix = block_diag([sub_matrix] * n_z)
            else:
                # Standard Cartesian case
                e_y_repeated = np.repeat(e_y, n_x)
                sub_matrix = diags(
                    [-e_y_repeated, e_y_repeated],
                    [0, n_x],
                    shape=(n_x * (n_y - 1), n_x * n_y),
                )
                sub_matrix = block_diag([sub_matrix] * n_z)

        elif direction == "z":
            # Gradient in z-direction: shape n_x*n_y*(n_z-1) x n_x*n_y*n_z
            if submesh.coord_sys in ["cylindrical polar", "spherical polar"]:
                # For cylindrical and spherical, z-direction is typically standard
                e_z_repeated = np.repeat(e_z, n_x * n_y)
                sub_matrix = diags(
                    [-e_z_repeated, e_z_repeated],
                    [0, n_x * n_y],
                    shape=(n_x * n_y * (n_z - 1), n_x * n_y * n_z),
                )
            else:
                # Standard Cartesian case
                e_z_repeated = np.repeat(e_z, n_x * n_y)
                sub_matrix = diags(
                    [-e_z_repeated, e_z_repeated],
                    [0, n_x * n_y],
                    shape=(n_x * n_y * (n_z - 1), n_x * n_y * n_z),
                )

        # number of repeats
        second_dim_repeats = self._get_auxiliary_domain_repeats(domains)

        matrix = csr_matrix(kron(eye(second_dim_repeats), sub_matrix))
        return pybamm.Matrix(matrix)

    def divergence(self, symbol, discretised_symbol, boundary_conditions):
        """Matrix-vector multiplication to implement the divergence operator in 3D."""
        divergence_matrix_x = self.divergence_matrix(symbol.domains, "x")
        divergence_matrix_y = self.divergence_matrix(symbol.domains, "y")
        divergence_matrix_z = self.divergence_matrix(symbol.domains, "z")

        grad_x = discretised_symbol.x_field
        grad_y = discretised_symbol.y_field
        grad_z = discretised_symbol.z_field

        div_x = divergence_matrix_x @ grad_x
        div_y = divergence_matrix_y @ grad_y
        div_z = divergence_matrix_z @ grad_z

        return div_x + div_y + div_z

    def divergence_matrix(self, domains, direction):
        """
        Divergence matrix for a specific direction (x, y, or z)
        Following PyBaMM convention: div(N) = (N[1:] - N[:-1])/dx
        """

        submesh = self.mesh[domains["primary"]]
        n_x = submesh.npts_x
        n_y = submesh.npts_y
        n_z = submesh.npts_z

        # Use d_edges for divergence (distance between adjacent edges)
        e_x = 1 / submesh.d_edges_x
        e_y = 1 / submesh.d_edges_y
        e_z = 1 / submesh.d_edges_z

        if direction == "x":
            # Divergence in x-direction: shape n_x*n_y*n_z x (n_x-1)*n_y*n_z
            if submesh.coord_sys == "cylindrical polar":
                r_nodes = submesh.nodes_x
                r_weights = r_nodes
                sub_matrix = diags(
                    [-e_x * r_weights[:-1], e_x * r_weights[1:]],
                    [0, 1],
                    shape=(n_x, n_x - 1),
                )
                sub_matrix = block_diag([sub_matrix] * (n_y * n_z))
            elif submesh.coord_sys == "spherical polar":
                r_nodes = submesh.nodes_x
                r_weights = r_nodes**2
                sub_matrix = diags(
                    [-e_x * r_weights[:-1], e_x * r_weights[1:]],
                    [0, 1],
                    shape=(n_x, n_x - 1),
                )
                sub_matrix = block_diag([sub_matrix] * (n_y * n_z))
            elif submesh.coord_sys == "spiral":
                spiral_metric = self.compute_spiral_metric(submesh)
                sub_matrix = diags(
                    [-e_x * spiral_metric[:-1], e_x * spiral_metric[1:]],
                    [0, 1],
                    shape=(n_x, n_x - 1),
                )
                sub_matrix = block_diag([sub_matrix] * (n_y * n_z))
            else:
                sub_matrix = diags([e_x, -e_x], [0, -1], shape=(n_x, n_x - 1))
                sub_matrix = block_diag([sub_matrix] * (n_y * n_z))

        elif direction == "y":
            # Divergence in y-direction: shape n_x*n_y*n_z x n_x*(n_y-1)*n_z
            if submesh.coord_sys == "cylindrical polar":
                r_nodes = submesh.nodes_x
                r_weights = np.tile(r_nodes, n_y)
                e_y_weighted = (
                    np.repeat(e_y, n_x) / r_weights[: len(np.repeat(e_y, n_x))]
                )
                sub_matrix = diags(
                    [-e_y_weighted, e_y_weighted],
                    [0, n_x],
                    shape=(n_x * n_y, n_x * (n_y - 1)),
                )
                sub_matrix = block_diag([sub_matrix] * n_z)
            elif submesh.coord_sys == "spherical polar":
                r_nodes = submesh.nodes_x
                theta_nodes = submesh.nodes_y
                r_weights = np.tile(r_nodes, n_y)
                sin_weights = np.repeat(np.sin(theta_nodes), n_x)
                e_y_weighted = np.repeat(e_y, n_x) / (
                    r_weights[: len(np.repeat(e_y, n_x))] * sin_weights
                )
                sub_matrix = diags(
                    [-e_y_weighted, e_y_weighted],
                    [0, n_x],
                    shape=(n_x * n_y, n_x * (n_y - 1)),
                )
                sub_matrix = block_diag([sub_matrix] * n_z)
            else:
                e_y_repeated = np.repeat(e_y, n_x)
                sub_matrix_plane = diags(
                    [e_y_repeated, -e_y_repeated],
                    [0, -n_x],
                    shape=(n_x * n_y, n_x * (n_y - 1)),
                )
                sub_matrix = block_diag([sub_matrix_plane] * n_z)

        elif direction == "z":
            if submesh.coord_sys in ["cylindrical polar", "spherical polar"]:
                e_z_repeated = np.repeat(e_z, n_x * n_y)
                sub_matrix = diags(
                    [e_z_repeated, -e_z_repeated],
                    [0, -n_x * n_y],
                    shape=(n_x * n_y * n_z, n_x * n_y * (n_z - 1)),
                )
            else:  # Cartesian
                e_z_repeated = np.repeat(e_z, n_x * n_y)
                sub_matrix = diags(
                    [e_z_repeated, -e_z_repeated],
                    [0, -n_x * n_y],
                    shape=(n_x * n_y * n_z, n_x * n_y * (n_z - 1)),
                )

        second_dim_repeats = self._get_auxiliary_domain_repeats(domains)
        if not isinstance(sub_matrix, csr_matrix):
            sub_matrix = csr_matrix(sub_matrix)
        matrix = csr_matrix(kron(eye(second_dim_repeats), sub_matrix))

        return pybamm.Matrix(matrix)

    def integral(
        self, child, discretised_child, integration_dimension, integration_variable
    ):
        """matrix-vector product to implement 3D integration operator."""
        integration_matrix = self.definite_integral_matrix(
            child,
            integration_dimension=integration_dimension,
            integration_variable=integration_variable,
        )

        if len(integration_variable) > 1:
            dirs = [v.direction for v in integration_variable]
            if len(set(dirs)) != len(dirs):
                raise ValueError("Integration variables must be in distinct directions")

            for _i, var in enumerate(integration_variable[1:], 1):
                direction = var.direction
                one_dimensional_matrix = self.one_dimensional_integral_matrix(
                    child, direction
                )
                if integration_matrix.shape[1] != one_dimensional_matrix.shape[0]:
                    repeats_needed = (
                        integration_matrix.shape[1] // one_dimensional_matrix.shape[0]
                    )
                    if repeats_needed > 1:
                        one_dimensional_matrix = kron(
                            eye(repeats_needed), one_dimensional_matrix
                        )
                integration_matrix = one_dimensional_matrix @ integration_matrix

        domains = child.domains
        second_dim_repeats = self._get_auxiliary_domain_repeats(domains)
        integration_matrix = kron(eye(second_dim_repeats), integration_matrix)
        return pybamm.Matrix(integration_matrix) @ discretised_child

    def laplacian(self, symbol, discretised_symbol, boundary_conditions):
        """
        Laplacian operator in 3D, implemented as div(grad(.))
        """
        grad = self.gradient(symbol, discretised_symbol, boundary_conditions)
        return self.divergence(grad, grad, boundary_conditions)

    def definite_integral_matrix(
        self,
        child,
        vector_type="row",
        integration_dimension="primary",
        integration_variable=None,
    ):
        """
        Create integration matrix for 3D finite volume method with support for all coordinate systems.
        """
        if integration_variable is None:
            raise ValueError("Integration variable must be provided for 3D integration")

        integration_direction = integration_variable[0].direction
        domains = child.domains
        domain = child.domains[integration_dimension]
        submesh = self.mesh[domain]

        n_x = submesh.npts_x
        n_y = submesh.npts_y
        n_z = submesh.npts_z

        if integration_dimension == "primary":
            submesh = self.mesh[domains["primary"]]

            if integration_direction == "x":
                d_edges = submesh.d_edges_x
                if submesh.coord_sys == "cylindrical polar":
                    r_nodes = submesh.nodes_x
                    weights = r_nodes  # r weighting for cylindrical
                elif submesh.coord_sys == "spherical polar":
                    r_nodes = submesh.nodes_x
                    weights = r_nodes**2  # r² weighting for spherical
                elif submesh.coord_sys == "spiral":
                    spiral_metric = self.compute_spiral_metric(submesh)
                    weights = spiral_metric
                else:
                    weights = np.ones_like(d_edges)

                # Create integration matrix for x-direction
                cols_list = []
                rows_list = []
                for k in range(n_z):
                    for j in range(n_y):
                        row_idx = k * n_y + j
                        col_start = k * n_x * n_y + j * n_x
                        cols_list.append(np.arange(col_start, col_start + n_x))
                        rows_list.append(np.full(n_x, row_idx))

                cols = np.concatenate(cols_list)
                rows = np.concatenate(rows_list)
                sub_matrix = csr_matrix(
                    (np.tile(d_edges * weights, n_y * n_z), (rows, cols)),
                    shape=(n_y * n_z, n_x * n_y * n_z),
                )

            elif integration_direction == "y":
                d_edges = submesh.d_edges_y
                if submesh.coord_sys == "cylindrical polar":
                    r_nodes = submesh.nodes_x
                    weights = np.ones_like(
                        d_edges
                    )  # No additional weighting for y in cylindrical
                elif submesh.coord_sys == "spherical polar":
                    theta_nodes = submesh.nodes_y
                    weights = np.sin(theta_nodes)
                elif submesh.coord_sys == "spiral":
                    spiral_metric = self.compute_spiral_metric(submesh)
                    weights = np.ones_like(d_edges)
                else:
                    weights = np.ones_like(d_edges)

                # Create integration matrix for y-direction
                cols_list = []
                rows_list = []
                for k in range(n_z):
                    for i in range(n_x):
                        row_idx = k * n_x + i
                        cols_k = []
                        for j in range(n_y):
                            col_idx = k * n_x * n_y + j * n_x + i
                            cols_k.append(col_idx)
                        cols_list.append(cols_k)
                        rows_list.append(np.full(n_y, row_idx))

                cols = np.concatenate(cols_list)
                rows = np.concatenate(rows_list)
                sub_matrix = csr_matrix(
                    (np.tile(d_edges * weights, n_x * n_z), (rows, cols)),
                    shape=(n_x * n_z, n_x * n_y * n_z),
                )

            elif integration_direction == "z":
                d_edges = submesh.d_edges_z
                if submesh.coord_sys == "cylindrical polar":
                    weights = np.ones_like(
                        d_edges
                    )  # No additional weighting for z in cylindrical
                elif submesh.coord_sys == "spherical polar":
                    weights = np.ones_like(
                        d_edges
                    )  # No additional weighting for z in spherical
                elif submesh.coord_sys == "spiral":
                    spiral_metric = self.compute_spiral_metric(submesh)
                    weights = np.ones_like(d_edges)
                else:
                    weights = np.ones_like(d_edges)

                # Create integration matrix for z-direction
                cols_list = []
                rows_list = []
                for j in range(n_y):
                    for i in range(n_x):
                        row_idx = j * n_x + i
                        cols_j = []
                        for k in range(n_z):
                            col_idx = k * n_x * n_y + j * n_x + i
                            cols_j.append(col_idx)
                        cols_list.append(cols_j)
                        rows_list.append(np.full(n_z, row_idx))

                cols = np.concatenate(cols_list)
                rows = np.concatenate(rows_list)
                sub_matrix = csr_matrix(
                    (np.tile(d_edges * weights, n_x * n_y), (rows, cols)),
                    shape=(n_x * n_y, n_x * n_y * n_z),
                )
        else:
            raise NotImplementedError(
                "Only primary integration dimension is implemented for 3D integration"
            )

        return sub_matrix

    def one_dimensional_integral_matrix(self, child, direction):
        """
        One-dimensional integral matrix for 3D finite volumes.
        """
        submesh = self.mesh[child.domain]
        domains = child.domains

        if direction == "x":
            d_edges = submesh.d_edges_y * submesh.d_edges_z
        elif direction == "y":
            d_edges = submesh.d_edges_x * submesh.d_edges_z
        elif direction == "z":
            d_edges = submesh.d_edges_x * submesh.d_edges_y

        # repeat matrix for each node in secondary dimensions
        second_dim_repeats = self._get_auxiliary_domain_repeats(domains)
        matrix = kron(eye(second_dim_repeats), d_edges)

        return matrix

    def boundary_integral(self, child, discretised_child, region):
        """
        Boundary integral operator for 3D.
        """
        symbol = pybamm.BoundaryValue(child, region)
        boundary_value = self.boundary_value_or_flux(symbol, discretised_child)

        if region in ["left", "right"]:
            direction = "x"
        elif region in ["front", "back"]:
            direction = "y"
        elif region in ["bottom", "top"]:
            direction = "z"
        else:
            raise ValueError(f"Region {region} not supported")

        integral_matrix = self.one_dimensional_integral_matrix(child, direction)
        domains = child.domains
        second_dim_repeats = self._get_auxiliary_domain_repeats(domains)
        integral_matrix = kron(eye(second_dim_repeats), integral_matrix)
        return pybamm.Matrix(integral_matrix) @ boundary_value

    def indefinite_integral(self, child, discretised_child, direction):
        """
        Indefinite integral operator for 3D.
        """
        if child.evaluates_on_edges("primary"):
            indefinite_integral_matrix = self.indefinite_integral_matrix_edges(
                child.domains, direction
            )
        else:
            submesh = self.mesh[child.domain]
            if submesh.coord_sys in ["cylindrical polar", "spherical polar", "spiral"]:
                raise NotImplementedError(
                    f"Indefinite integral on a {submesh.coord_sys} domain is not "
                    "implemented"
                )
            indefinite_integral_matrix = self.indefinite_integral_matrix_nodes(
                child.domains, direction
            )
        return indefinite_integral_matrix @ discretised_child

    def indefinite_integral_matrix_edges(self, domains, direction):
        """
        Matrix for finite-volume implementation of the indefinite integral where the
        integrand is evaluated on mesh edges in 3D.

        This follows the same logic as 1D but applies it to the primary dimension
        of the 3D mesh while keeping other dimensions fixed.
        """
        submesh = self.mesh[domains["primary"]]

        n_primary = submesh.npts_x
        n_y = submesh.npts_y
        n_z = submesh.npts_z

        du_n = submesh.d_nodes_x

        if direction == "forward":
            # Forward integration: cumulative sum from start
            du_entries = [du_n] * (n_primary - 1)
            offset = -np.arange(1, n_primary, 1)
            main_integral_matrix = spdiags(du_entries, offset, n_primary, n_primary - 1)
            bc_offset_matrix = lil_matrix((n_primary, n_primary - 1))
            bc_offset_matrix[:, 0] = du_n[0] / 2

        elif direction == "backward":
            # Backward integration: cumulative sum from end
            du_entries = [du_n] * (n_primary + 1)
            offset = np.arange(n_primary, -1, -1)
            main_integral_matrix = spdiags(du_entries, offset, n_primary, n_primary - 1)
            bc_offset_matrix = lil_matrix((n_primary, n_primary - 1))
            bc_offset_matrix[:, -1] = du_n[-1] / 2

        else:
            raise ValueError(
                f"Unknown direction: {direction}. Must be 'forward' or 'backward'"
            )

        # Combine main matrix and boundary condition offset
        sub_matrix_1d = main_integral_matrix + bc_offset_matrix

        zero_col = csr_matrix((n_primary, 1))
        sub_matrix_1d = hstack([zero_col, sub_matrix_1d, zero_col])

        sub_matrix_3d = kron(eye(n_y * n_z), sub_matrix_1d)

        second_dim_repeats = self._get_auxiliary_domain_repeats(domains)
        matrix = csr_matrix(kron(eye(second_dim_repeats), sub_matrix_3d))

        if hasattr(submesh, "length"):
            matrix = matrix * submesh.length

        return pybamm.Matrix(matrix)

    def indefinite_integral_matrix_nodes(self, domains, direction):
        """
        Matrix for finite-volume implementation of the indefinite integral where the
        integrand is evaluated on mesh nodes in 3D.

        This is a straightforward cumulative sum along the primary dimension.
        """
        submesh = self.mesh[domains["primary"]]

        n_primary = submesh.npts_x
        n_y = submesh.npts_y
        n_z = submesh.npts_z

        du_n = submesh.d_edges_x

        du_entries = [du_n] * n_primary

        if direction == "forward":
            # Forward: cumulative sum from start to current
            offset = -np.arange(1, n_primary + 1, 1)
        elif direction == "backward":
            # Backward: cumulative sum from current to end
            offset = np.arange(n_primary - 1, -1, -1)
        else:
            raise ValueError(
                f"Unknown direction: {direction}. Must be 'forward' or 'backward'"
            )

        sub_matrix_1d = spdiags(du_entries, offset, n_primary + 1, n_primary)

        sub_matrix_3d = kron(eye(n_y * n_z), sub_matrix_1d)

        second_dim_repeats = self._get_auxiliary_domain_repeats(domains)
        matrix = csr_matrix(kron(eye(second_dim_repeats), sub_matrix_3d))

        if hasattr(submesh, "length"):
            matrix = matrix * submesh.length

        return pybamm.Matrix(matrix)

    def delta_function(self, symbol, discretised_symbol):
        """3D delta function for boundary flux (symbol.side = ("x"/"y"/"z", "left"/"right"))."""
        submesh = self.mesh[symbol.domain]
        n_x, n_y, n_z = submesh.npts_x, submesh.npts_y, submesh.npts_z
        coord_dir, face_side = symbol.side

        if coord_dir == "x":
            x_idx = 0 if face_side == "left" else n_x - 1
            y_length = submesh.edges_y[-1] - submesh.edges_y[0]
            z_length = submesh.edges_z[-1] - submesh.edges_z[0]
            face_area = y_length * z_length
            dx = submesh.d_edges_x[x_idx]

            rows = []
            for z_i in range(n_z):
                base_z = z_i * (n_x * n_y)
                for y_i in range(n_y):
                    rows.append(base_z + y_i * n_x + x_idx)

            scale = face_area / dx

        elif coord_dir == "y":
            y_idx = 0 if face_side == "left" else n_y - 1
            x_length = submesh.edges_x[-1] - submesh.edges_x[0]
            z_length = submesh.edges_z[-1] - submesh.edges_z[0]
            face_area = x_length * z_length
            dy = submesh.d_edges_y[y_idx]

            rows = []
            for z_i in range(n_z):
                base_z = z_i * (n_x * n_y)
                for x_i in range(n_x):
                    rows.append(base_z + y_idx * n_x + x_i)

            scale = face_area / dy

        elif coord_dir == "z":
            z_idx = 0 if face_side == "left" else n_z - 1
            x_length = submesh.edges_x[-1] - submesh.edges_x[0]
            y_length = submesh.edges_y[-1] - submesh.edges_y[0]
            face_area = x_length * y_length
            dz = submesh.d_edges_z[z_idx]

            rows = []
            base_z = z_idx * (n_x * n_y)
            for y_i in range(n_y):
                for x_i in range(n_x):
                    rows.append(base_z + y_i * n_x + x_i)

            scale = face_area / dz

        else:
            raise ValueError(
                "symbol.side must be ('x','left'/'right'), ('y',...), or ('z',...)"
            )

        rows = np.array(rows, dtype=int)
        cols = np.zeros_like(rows)
        data = np.ones_like(rows, dtype=float)
        sub_matrix = csr_matrix((data, (rows, cols)), shape=(n_x * n_y * n_z, 1))

        repeats = self._get_auxiliary_domain_repeats(symbol.domains)
        full_matrix = kron(eye(repeats), sub_matrix)

        delta_vec = pybamm.Matrix(scale * full_matrix) @ discretised_symbol
        delta_vec.copy_domains(symbol)
        return delta_vec

    def internal_neumann_condition(
        self, left_symbol_disc, right_symbol_disc, left_mesh, right_mesh
    ):
        """
        Internal Neumann conditions between two 3D symbols on adjacent subdomains.
        """
        left_npts = left_mesh.npts
        left_npts_x = left_mesh.npts_x
        left_npts_y = left_mesh.npts_y
        left_npts_z = left_mesh.npts_z

        right_npts = right_mesh.npts
        right_npts_x = right_mesh.npts_x
        right_npts_y = right_mesh.npts_y
        right_npts_z = right_mesh.npts_z

        second_dim_repeats = self._get_auxiliary_domain_repeats(
            left_symbol_disc.domains
        )

        if second_dim_repeats != self._get_auxiliary_domain_repeats(
            right_symbol_disc.domains
        ):
            raise pybamm.DomainError(
                "Number of secondary points in subdomains do not match"
            )

        left_sub_matrix = np.zeros((1, left_npts))
        for k in range(left_npts_z):
            for j in range(left_npts_y):
                idx = (
                    k * left_npts_x * left_npts_y + j * left_npts_x + (left_npts_x - 1)
                )
                left_sub_matrix[0, idx] = 1

        left_matrix = pybamm.Matrix(
            csr_matrix(kron(eye(second_dim_repeats), left_sub_matrix))
        )

        right_sub_matrix = np.zeros((1, right_npts))
        for k in range(right_npts_z):
            for j in range(right_npts_y):
                idx = k * right_npts_x * right_npts_y + j * right_npts_x + 0
                right_sub_matrix[0, idx] = 1

        right_matrix = pybamm.Matrix(
            csr_matrix(kron(eye(second_dim_repeats), right_sub_matrix))
        )

        right_mesh_x = right_mesh.nodes_x[0]
        left_mesh_x = left_mesh.nodes_x[-1]
        dx = right_mesh_x - left_mesh_x

        dy_r = (right_matrix / dx) @ right_symbol_disc
        dy_r.clear_domains()
        dy_l = (left_matrix / dx) @ left_symbol_disc
        dy_l.clear_domains()

        return dy_r - dy_l

    def add_ghost_nodes(self, symbol, discretised_symbol, bcs):
        """
        Add ghost nodes to a symbol for 3D finite volume.

        For Dirichlet bcs, for a boundary condition "y = a at the left-hand boundary",
        we add ghost nodes with value "2*a - y_boundary" where y_boundary is the
        value at the boundary face.

        Parameters
        ----------
        symbol : :class:`pybamm.SpatialVariable`
            The variable to be discretised
        discretised_symbol : :class:`pybamm.Vector`
            Contains the discretised variable
        bcs : dict of tuples (:class:`pybamm.Scalar`, str)
            Dictionary (with keys "left", "right", "front", "back", "bottom", "top") of
            boundary conditions.

        Returns
        -------
        :class:`pybamm.Symbol`
            The discretised symbol with ghost nodes
        str
            New domain name with ghost suffix
        """
        domain = symbol.domain
        submesh = self.mesh[domain]

        n_x = submesh.npts_x
        n_y = submesh.npts_y
        n_z = submesh.npts_z
        n = submesh.npts
        second_dim_repeats = self._get_auxiliary_domain_repeats(symbol.domains)

        # Check if we have boundary conditions
        if not any(
            face in bcs.keys()
            for face in ["left", "right", "front", "back", "bottom", "top"]
        ):
            raise ValueError(f"No boundary conditions have been provided for {symbol}")

        # Get boundary conditions
        lbc_value, lbc_type = bcs.get("left", (None, None))
        rbc_value, rbc_type = bcs.get("right", (None, None))
        fbc_value, fbc_type = bcs.get("front", (None, None))
        bbc_value, bbc_type = bcs.get("back", (None, None))
        botbc_value, botbc_type = bcs.get("bottom", (None, None))
        tbc_value, tbc_type = bcs.get("top", (None, None))

        # Count ghost nodes needed in each direction
        n_ghost_x = 0
        n_ghost_y = 0
        n_ghost_z = 0

        if lbc_type == "Dirichlet":
            n_ghost_x += 1
        if rbc_type == "Dirichlet":
            n_ghost_x += 1
        if fbc_type == "Dirichlet":
            n_ghost_y += 1
        if bbc_type == "Dirichlet":
            n_ghost_y += 1
        if botbc_type == "Dirichlet":
            n_ghost_z += 1
        if tbc_type == "Dirichlet":
            n_ghost_z += 1

        # Calculate new dimensions
        new_n_x = n_x + n_ghost_x
        new_n_y = n_y + n_ghost_y
        new_n_z = n_z + n_ghost_z
        new_n = new_n_x * new_n_y * new_n_z

        # Create transformation matrix and boundary vector
        from scipy.sparse import lil_matrix
        import numpy as np

        matrix = lil_matrix((new_n * second_dim_repeats, n * second_dim_repeats))
        bcs_vector = np.zeros(new_n * second_dim_repeats)

        # Helper function to convert 3D indices to flat index
        def flat_index(i, j, k, nx, ny):
            return i + j * nx + k * nx * ny

        # Process each secondary domain repeat
        for rep in range(second_dim_repeats):
            base_old = rep * n
            base_new = rep * new_n

            # Determine offsets for ghost nodes
            x_offset = 1 if lbc_type == "Dirichlet" else 0
            y_offset = 1 if fbc_type == "Dirichlet" else 0
            z_offset = 1 if botbc_type == "Dirichlet" else 0

            # Map interior nodes
            for k in range(n_z):
                for j in range(n_y):
                    for i in range(n_x):
                        old_idx = base_old + flat_index(i, j, k, n_x, n_y)
                        new_idx = base_new + flat_index(
                            i + x_offset, j + y_offset, k + z_offset, new_n_x, new_n_y
                        )
                        matrix[new_idx, old_idx] = 1.0

            # Evaluate boundary condition values once
            lbc_val = (
                float(lbc_value.evaluate())
                if lbc_value is not None and lbc_value.evaluates_to_number()
                else 0.0
            )
            rbc_val = (
                float(rbc_value.evaluate())
                if rbc_value is not None and rbc_value.evaluates_to_number()
                else 0.0
            )
            fbc_val = (
                float(fbc_value.evaluate())
                if fbc_value is not None and fbc_value.evaluates_to_number()
                else 0.0
            )
            bbc_val = (
                float(bbc_value.evaluate())
                if bbc_value is not None and bbc_value.evaluates_to_number()
                else 0.0
            )
            botbc_val = (
                float(botbc_value.evaluate())
                if botbc_value is not None and botbc_value.evaluates_to_number()
                else 0.0
            )
            tbc_val = (
                float(tbc_value.evaluate())
                if tbc_value is not None and tbc_value.evaluates_to_number()
                else 0.0
            )

            # Add ghost nodes for each face separately
            # Left face ghost nodes (i = 0)
            if lbc_type == "Dirichlet":
                for k in range(new_n_z):
                    for j in range(new_n_y):
                        # Skip corner nodes that belong to other faces
                        k_interior = max(z_offset, min(z_offset + n_z - 1, k))
                        j_interior = max(y_offset, min(y_offset + n_y - 1, j))

                        ghost_idx = base_new + flat_index(0, j, k, new_n_x, new_n_y)
                        interior_idx = base_old + flat_index(
                            0, j_interior - y_offset, k_interior - z_offset, n_x, n_y
                        )

                        matrix[ghost_idx, interior_idx] = -1.0
                        bcs_vector[ghost_idx] = 2.0 * lbc_val

            # Right face ghost nodes (i = new_n_x - 1)
            if rbc_type == "Dirichlet":
                for k in range(new_n_z):
                    for j in range(new_n_y):
                        k_interior = max(z_offset, min(z_offset + n_z - 1, k))
                        j_interior = max(y_offset, min(y_offset + n_y - 1, j))

                        ghost_idx = base_new + flat_index(
                            new_n_x - 1, j, k, new_n_x, new_n_y
                        )
                        interior_idx = base_old + flat_index(
                            n_x - 1,
                            j_interior - y_offset,
                            k_interior - z_offset,
                            n_x,
                            n_y,
                        )

                        matrix[ghost_idx, interior_idx] = -1.0
                        bcs_vector[ghost_idx] = 2.0 * rbc_val

            # Front face ghost nodes (j = 0)
            if fbc_type == "Dirichlet":
                for k in range(new_n_z):
                    for i in range(new_n_x):
                        # Skip nodes already handled by left/right faces
                        if (i == 0 and lbc_type == "Dirichlet") or (
                            i == new_n_x - 1 and rbc_type == "Dirichlet"
                        ):
                            continue

                        k_interior = max(z_offset, min(z_offset + n_z - 1, k))
                        i_interior = max(x_offset, min(x_offset + n_x - 1, i))

                        ghost_idx = base_new + flat_index(i, 0, k, new_n_x, new_n_y)
                        interior_idx = base_old + flat_index(
                            i_interior - x_offset, 0, k_interior - z_offset, n_x, n_y
                        )

                        matrix[ghost_idx, interior_idx] = -1.0
                        bcs_vector[ghost_idx] = 2.0 * fbc_val

            # Back face ghost nodes (j = new_n_y - 1)
            if bbc_type == "Dirichlet":
                for k in range(new_n_z):
                    for i in range(new_n_x):
                        # Skip nodes already handled by left/right faces
                        if (i == 0 and lbc_type == "Dirichlet") or (
                            i == new_n_x - 1 and rbc_type == "Dirichlet"
                        ):
                            continue

                        k_interior = max(z_offset, min(z_offset + n_z - 1, k))
                        i_interior = max(x_offset, min(x_offset + n_x - 1, i))

                        ghost_idx = base_new + flat_index(
                            i, new_n_y - 1, k, new_n_x, new_n_y
                        )
                        interior_idx = base_old + flat_index(
                            i_interior - x_offset,
                            n_y - 1,
                            k_interior - z_offset,
                            n_x,
                            n_y,
                        )

                        matrix[ghost_idx, interior_idx] = -1.0
                        bcs_vector[ghost_idx] = 2.0 * bbc_val

            # Bottom face ghost nodes (k = 0)
            if botbc_type == "Dirichlet":
                for j in range(new_n_y):
                    for i in range(new_n_x):
                        # Skip nodes already handled by other faces
                        if (
                            (i == 0 and lbc_type == "Dirichlet")
                            or (i == new_n_x - 1 and rbc_type == "Dirichlet")
                            or (j == 0 and fbc_type == "Dirichlet")
                            or (j == new_n_y - 1 and bbc_type == "Dirichlet")
                        ):
                            continue

                        j_interior = max(y_offset, min(y_offset + n_y - 1, j))
                        i_interior = max(x_offset, min(x_offset + n_x - 1, i))

                        ghost_idx = base_new + flat_index(i, j, 0, new_n_x, new_n_y)
                        interior_idx = base_old + flat_index(
                            i_interior - x_offset, j_interior - y_offset, 0, n_x, n_y
                        )

                        matrix[ghost_idx, interior_idx] = -1.0
                        bcs_vector[ghost_idx] = 2.0 * botbc_val

            # Top face ghost nodes (k = new_n_z - 1)
            if tbc_type == "Dirichlet":
                for j in range(new_n_y):
                    for i in range(new_n_x):
                        # Skip nodes already handled by other faces
                        if (
                            (i == 0 and lbc_type == "Dirichlet")
                            or (i == new_n_x - 1 and rbc_type == "Dirichlet")
                            or (j == 0 and fbc_type == "Dirichlet")
                            or (j == new_n_y - 1 and bbc_type == "Dirichlet")
                        ):
                            continue

                        j_interior = max(y_offset, min(y_offset + n_y - 1, j))
                        i_interior = max(x_offset, min(x_offset + n_x - 1, i))

                        ghost_idx = base_new + flat_index(
                            i, j, new_n_z - 1, new_n_x, new_n_y
                        )
                        interior_idx = base_old + flat_index(
                            i_interior - x_offset,
                            j_interior - y_offset,
                            n_z - 1,
                            n_x,
                            n_y,
                        )

                        matrix[ghost_idx, interior_idx] = -1.0
                        bcs_vector[ghost_idx] = 2.0 * tbc_val

        # Convert to CSR format for efficiency
        matrix = matrix.tocsr()

        # Create new domain name with proper ghost cell naming
        new_domain = domain

        # Handle X-direction (left/right)
        if lbc_type == "Dirichlet" and rbc_type != "Dirichlet":
            if isinstance(new_domain, (list, tuple)):
                new_domain = [(d + "_left ghost cell", d) for d in new_domain]
            else:
                new_domain = [new_domain + "_left ghost cell", new_domain]
        elif lbc_type != "Dirichlet" and rbc_type == "Dirichlet":
            if isinstance(new_domain, (list, tuple)):
                new_domain = [(d, d + "_right ghost cell") for d in new_domain]
            else:
                new_domain = [new_domain, new_domain + "_right ghost cell"]
        elif lbc_type == "Dirichlet" and rbc_type == "Dirichlet":
            if isinstance(new_domain, (list, tuple)):
                new_domain = [
                    (d + "_left ghost cell", d, d + "_right ghost cell")
                    for d in new_domain
                ]
            else:
                new_domain = [
                    new_domain + "_left ghost cell",
                    new_domain,
                    new_domain + "_right ghost cell",
                ]

        # Handle Y-direction (front/back)
        if fbc_type == "Dirichlet" and bbc_type != "Dirichlet":
            if isinstance(new_domain, (list, tuple)):
                new_domain = [(d + "_front ghost cell", d) for d in new_domain]
            else:
                new_domain = [new_domain + "_front ghost cell", new_domain]
        elif fbc_type != "Dirichlet" and bbc_type == "Dirichlet":
            if isinstance(new_domain, (list, tuple)):
                new_domain = [(d, d + "_back ghost cell") for d in new_domain]
            else:
                new_domain = [new_domain, new_domain + "_back ghost cell"]
        elif fbc_type == "Dirichlet" and bbc_type == "Dirichlet":
            if isinstance(new_domain, (list, tuple)):
                new_domain = [
                    (d + "_front ghost cell", d, d + "_back ghost cell")
                    for d in new_domain
                ]
            else:
                new_domain = [
                    new_domain + "_front ghost cell",
                    new_domain,
                    new_domain + "_back ghost cell",
                ]

        # Handle Z-direction (bottom/top)
        if botbc_type == "Dirichlet" and tbc_type != "Dirichlet":
            if isinstance(new_domain, (list, tuple)):
                new_domain = [(d + "_bottom ghost cell", d) for d in new_domain]
            else:
                new_domain = [new_domain + "_bottom ghost cell", new_domain]
        elif botbc_type != "Dirichlet" and tbc_type == "Dirichlet":
            if isinstance(new_domain, (list, tuple)):
                new_domain = [(d, d + "_top ghost cell") for d in new_domain]
            else:
                new_domain = [new_domain, new_domain + "_top ghost cell"]
        elif botbc_type == "Dirichlet" and tbc_type == "Dirichlet":
            if isinstance(new_domain, (list, tuple)):
                new_domain = [
                    (d + "_bottom ghost cell", d, d + "_top ghost cell")
                    for d in new_domain
                ]
            else:
                new_domain = [
                    new_domain + "_bottom ghost cell",
                    new_domain,
                    new_domain + "_top ghost cell",
                ]

        # Create the result
        bcs_vector_symbol = pybamm.Vector(bcs_vector)
        bcs_vector_symbol.copy_domains(discretised_symbol)

        new_symbol = pybamm.Matrix(matrix) @ discretised_symbol + bcs_vector_symbol

        return new_symbol, new_domain

    def add_neumann_values(self, symbol, discretised_gradient, bcs, domain):
        """
        3D version of add_neumann_values.  Any Neumann BC on one of the six faces
        contributes a known flux into the gradient vector; Dirichlet BCs were handled
        earlier by ghost nodes.
        """

        def get_float_val(bc_val_obj):
            if hasattr(bc_val_obj, "evaluate"):
                return float(bc_val_obj.evaluate())
            return float(bc_val_obj)

        submesh = self.mesh[domain]
        nx, ny, nz = submesh.npts_x, submesh.npts_y, submesh.npts_z
        repeats = self._get_auxiliary_domain_repeats(symbol.domains)

        (xlv, xlt), (xrv, xrt) = (
            bcs.get("left", (None, None)),
            bcs.get("right", (None, None)),
        )
        (yfv, yft), (ybv, ybt) = (
            bcs.get("front", (None, None)),
            bcs.get("back", (None, None)),
        )
        (zbv, zbt), (ztv, ztt) = (
            bcs.get("bottom", (None, None)),
            bcs.get("top", (None, None)),
        )

        # count Neumann faces (will need to expand length by 1 in that direction)
        nnx = nx + (1 if xlt == "Neumann" else 0) + (1 if xrt == "Neumann" else 0)
        nny = ny + (1 if yft == "Neumann" else 0) + (1 if ybt == "Neumann" else 0)
        nnz = nz + (1 if zbt == "Neumann" else 0) + (1 if ztt == "Neumann" else 0)
        # Nn = nnx * nny * nnz

        # total interior gradient points is (nx-1)*ny*nz in x direction, etc,
        # but in 3D we actually have three separate gradient components; here
        # I'm assuming `discretised_gradient` already flattened one component
        # of shape ( (nx-1)*ny*nz , ) for ∂/∂x etc.  You may need to adapt
        # this to your actual storage of the 3D vector.
        # n = discretised_gradient.size // repeats

        # build 1D expansion matrices Ix, Iy, Iz for where the Neumann extended
        # gradient lives; e.g. if left Neumann, we prepend one zero, etc.
        def expand_1d(npts, didx):
            rows, cols = [], []
            for i in range(npts):
                rows.append(i + (1 if (didx and didx > 0) else 0))
                cols.append(i)
            return csr_matrix((np.ones(npts), (rows, cols)), shape=(npts + didx, npts))

        Ix = expand_1d(
            nx - 1, 1 if xlt == "Neumann" else 0
        )  # for ∂/∂x, there are nx 1 intervals
        Iy = expand_1d(
            ny, 1 if yft == "Neumann" else 0
        )  # each x gradient slice repeats ny times
        Iz = expand_1d(nz, 1 if zbt == "Neumann" else 0)

        Mx = kron(Iz, kron(Iy, Ix))

        bc_vec = pybamm.Vector(np.zeros((Mx.shape[0] * repeats,)))

        def scatter(vals, rows, length):
            out = np.zeros((length,))
            out[rows] = vals
            return out

        if xlt == "Neumann" and xlv is not None:
            xlv_val = get_float_val(xrv)
            if xlv_val != 0:
                rows = []
                for k in range(nz):
                    for j in range(ny):
                        # row index in flattened (nnx*ny*nz) vector
                        idx = k * (nny * nnx) + j * nnx + 0
                        rows.append(idx)
                val = xlv_val if isinstance(xlv_val, (int, float)) else xlv
                bc_vec += pybamm.Vector(
                    scatter(val * np.ones(len(rows)), rows, Mx.shape[0] * repeats)
                )

        if xrt == "Neumann" and xrv is not None:
            xrv_val = get_float_val(xrv)
            if xrv_val != 0:
                rows = []
                for k in range(nz):
                    for j in range(ny):
                        idx = k * (nny * nnx) + j * nnx + (nnx - 1)
                        rows.append(idx)
                val = xrv_val if isinstance(xrv_val, (int, float)) else xrv
                bc_vec += pybamm.Vector(
                    scatter(val * np.ones(len(rows)), rows, Mx.shape[0] * repeats)
                )

        if yft == "Neumann" and yfv is not None:
            yfv_val_num = get_float_val(yfv)
            if yfv_val_num != 0:
                rows = []
                for k in range(nz):
                    for i in range(
                        nx - 1 + (1 if xlt == "Neumann" or xrt == "Neumann" else 0)
                    ):
                        idx = k * (nny * nnx) + 0 * nnx + i
                        rows.append(idx)
                val = yfv_val_num
                bc_vec += pybamm.Vector(
                    scatter(val * np.ones(len(rows)), rows, Mx.shape[0] * repeats)
                )

        if ybt == "Neumann" and ybv is not None:
            ybv_val_num = get_float_val(ybv)
            if ybv_val_num != 0:
                rows = []
                for k in range(nz):
                    for i in range(
                        nx - 1 + (1 if xlt == "Neumann" or xrt == "Neumann" else 0)
                    ):
                        idx = k * (nny * nnx) + (nny - 1) * nnx + i
                        rows.append(idx)
                val = ybv_val_num
                bc_vec += pybamm.Vector(
                    scatter(val * np.ones(len(rows)), rows, Mx.shape[0] * repeats)
                )

        if zbt == "Neumann" and zbv is not None:
            zbv_val = get_float_val(zbv)
            if zbv_val != 0:
                rows = []
                for j in range(ny + (1 if yft == "Neumann" or ybt == "Neumann" else 0)):
                    for i in range(
                        nx - 1 + (1 if xlt == "Neumann" or xrt == "Neumann" else 0)
                    ):
                        idx = 0 * (nny * nnx) + j * nnx + i
                        rows.append(idx)
                val = zbv_val
                bc_vec += pybamm.Vector(
                    scatter(val * np.ones(len(rows)), rows, Mx.shape[0] * repeats)
                )

        if ztt == "Neumann" and ztv is not None:
            ztv_val = get_float_val(ztv)
            if ztv_val != 0:
                rows = []
                for j in range(ny + (1 if yft == "Neumann" or ybt == "Neumann" else 0)):
                    for i in range(
                        nx - 1 + (1 if xlt == "Neumann" or xrt == "Neumann" else 0)
                    ):
                        idx = (nnz - 1) * (nny * nnx) + j * nnx + i
                        rows.append(idx)
                val = ztv_val
                bc_vec += pybamm.Vector(
                    scatter(val * np.ones(len(rows)), rows, Mx.shape[0] * repeats)
                )

        new_grad = pybamm.Matrix(Mx) @ discretised_gradient + bc_vec
        new_grad.copy_domains(discretised_gradient)
        return new_grad

    def boundary_value_or_flux(self, symbol, discretised_child, bcs=None):
        """
        Uses extrapolation to get the boundary value or flux of a variable in the
        Finite Volume Method for 3D meshes.

        See :meth:`pybamm.SpatialMethod.boundary_value`
        """
        # Find the submesh
        submesh = self.mesh[discretised_child.domain]

        if "-" in symbol.side:
            sides = symbol.side.split("-")
            primary_side = sides[0]
            secondary_side = sides[1] if len(sides) > 1 else None
            tertiary_side = sides[2] if len(sides) > 2 else None
        else:
            primary_side = symbol.side
            secondary_side = None
            tertiary_side = None

        repeats = self._get_auxiliary_domain_repeats(discretised_child.domains)

        if bcs is None:
            bcs = {}

        extrap_order_gradient = self.options["extrapolation"]["order"]["gradient"]
        extrap_order_value = self.options["extrapolation"]["order"]["value"]
        use_bcs = self.options["extrapolation"]["use bcs"]

        n_x = submesh.npts_x
        n_y = submesh.npts_y
        n_z = submesh.npts_z

        nodes_x = submesh.nodes_x
        edges_x = submesh.edges_x
        nodes_y = submesh.nodes_y
        edges_y = submesh.edges_y
        nodes_z = submesh.nodes_z
        edges_z = submesh.edges_z

        dx0_x = nodes_x[0] - edges_x[0]
        dx1_x = submesh.d_nodes_x[0]
        dxN_x = edges_x[-1] - nodes_x[-1]
        dxNm1_x = submesh.d_nodes_x[-1]

        dx0_y = nodes_y[0] - edges_y[0]
        dx1_y = submesh.d_nodes_y[0]
        dxN_y = edges_y[-1] - nodes_y[-1]
        dxNm1_y = submesh.d_nodes_y[-1]

        dx0_z = nodes_z[0] - edges_z[0]
        dx1_z = submesh.d_nodes_z[0]
        dxN_z = edges_z[-1] - nodes_z[-1]
        dxNm1_z = submesh.d_nodes_z[-1]

        child = symbol.child

        # Initialize variables
        sub_matrix = None
        additive = pybamm.Scalar(0)

        if isinstance(symbol, pybamm.BoundaryValue):
            if use_bcs and pybamm.has_bc_of_form(child, primary_side, bcs, "Dirichlet"):
                if primary_side in ["left", "right"]:
                    sub_matrix = csr_matrix((n_y * n_z, n_x * n_y * n_z))
                elif primary_side in ["front", "back"]:
                    sub_matrix = csr_matrix((n_x * n_z, n_x * n_y * n_z))
                elif primary_side in ["bottom", "top"]:
                    sub_matrix = csr_matrix((n_x * n_y, n_x * n_y * n_z))
                additive = bcs[child][primary_side][0]

            elif primary_side == "left":
                if extrap_order_value == "linear":
                    if use_bcs and pybamm.has_bc_of_form(
                        child, primary_side, bcs, "Neumann"
                    ):
                        dx0 = dx0_x
                        row_indices = np.arange(0, n_y * n_z)
                        col_indices = np.arange(0, n_x * n_y * n_z, n_x)
                        vals = np.ones(n_y * n_z)
                        sub_matrix = csr_matrix(
                            (vals, (row_indices, col_indices)),
                            shape=(n_y * n_z, n_x * n_y * n_z),
                        )
                        additive = -dx0 * bcs[child][primary_side][0]
                    else:
                        dx0 = dx0_x
                        dx1 = dx1_x
                        row_indices = np.arange(0, n_y * n_z)
                        col_indices_0 = np.arange(0, n_x * n_y * n_z, n_x)
                        col_indices_1 = col_indices_0 + 1
                        vals_0 = np.ones(n_y * n_z) * (1 + (dx0 / dx1))
                        vals_1 = np.ones(n_y * n_z) * (-(dx0 / dx1))
                        sub_matrix = csr_matrix(
                            (
                                np.hstack([vals_0, vals_1]),
                                (
                                    np.hstack([row_indices, row_indices]),
                                    np.hstack([col_indices_0, col_indices_1]),
                                ),
                            ),
                            shape=(n_y * n_z, n_x * n_y * n_z),
                        )
                        additive = pybamm.Scalar(0)
                else:
                    raise NotImplementedError(
                        f"Order {extrap_order_value} not implemented for left boundary"
                    )

            elif primary_side == "right":
                if extrap_order_value == "linear":
                    if use_bcs and pybamm.has_bc_of_form(
                        child, primary_side, bcs, "Neumann"
                    ):
                        dxN = dxN_x
                        row_indices = np.arange(0, n_y * n_z)
                        col_indices = np.arange(n_x - 1, n_x * n_y * n_z, n_x)
                        vals = np.ones(n_y * n_z)
                        sub_matrix = csr_matrix(
                            (vals, (row_indices, col_indices)),
                            shape=(n_y * n_z, n_x * n_y * n_z),
                        )
                        additive = dxN * bcs[child][primary_side][0]
                    else:
                        dxN = dxN_x
                        dxNm1 = dxNm1_x
                        row_indices = np.arange(0, n_y * n_z)
                        col_indices_Nm1 = np.arange(n_x - 2, n_x * n_y * n_z, n_x)
                        col_indices_N = col_indices_Nm1 + 1
                        vals_Nm1 = np.ones(n_y * n_z) * (-(dxN / dxNm1))
                        vals_N = np.ones(n_y * n_z) * (1 + (dxN / dxNm1))
                        sub_matrix = csr_matrix(
                            (
                                np.hstack([vals_Nm1, vals_N]),
                                (
                                    np.hstack([row_indices, row_indices]),
                                    np.hstack([col_indices_Nm1, col_indices_N]),
                                ),
                            ),
                            shape=(n_y * n_z, n_x * n_y * n_z),
                        )
                        additive = pybamm.Scalar(0)
                else:
                    raise NotImplementedError(
                        f"Order {extrap_order_value} not implemented for right boundary"
                    )

            elif primary_side == "front":
                if extrap_order_value == "linear":
                    if use_bcs and pybamm.has_bc_of_form(
                        child, primary_side, bcs, "Neumann"
                    ):
                        dx0 = dx0_y
                        row_indices = np.arange(0, n_x * n_z)
                        col_indices = []
                        for k in range(n_z):
                            for i in range(n_x):
                                col_indices.append(k * n_x * n_y + i)
                        col_indices = np.array(col_indices)
                        vals = np.ones(n_x * n_z)
                        sub_matrix = csr_matrix(
                            (vals, (row_indices, col_indices)),
                            shape=(n_x * n_z, n_x * n_y * n_z),
                        )
                        additive = -dx0 * bcs[child][primary_side][0]
                    else:
                        dx0 = dx0_y
                        dx1 = dx1_y
                        row_indices = np.arange(0, n_x * n_z)
                        col_indices_0 = []
                        col_indices_1 = []
                        for k in range(n_z):
                            for i in range(n_x):
                                col_indices_0.append(k * n_x * n_y + i)
                                col_indices_1.append(k * n_x * n_y + n_x + i)
                        col_indices_0 = np.array(col_indices_0)
                        col_indices_1 = np.array(col_indices_1)
                        vals_0 = np.ones(n_x * n_z) * (1 + (dx0 / dx1))
                        vals_1 = np.ones(n_x * n_z) * (-(dx0 / dx1))
                        sub_matrix = csr_matrix(
                            (
                                np.hstack([vals_0, vals_1]),
                                (
                                    np.hstack([row_indices, row_indices]),
                                    np.hstack([col_indices_0, col_indices_1]),
                                ),
                            ),
                            shape=(n_x * n_z, n_x * n_y * n_z),
                        )
                        additive = pybamm.Scalar(0)
                else:
                    raise NotImplementedError(
                        f"Order {extrap_order_value} not implemented for front boundary"
                    )

            elif primary_side == "back":
                if extrap_order_value == "linear":
                    if use_bcs and pybamm.has_bc_of_form(
                        child, primary_side, bcs, "Neumann"
                    ):
                        dxN = dxN_y
                        row_indices = np.arange(0, n_x * n_z)
                        col_indices = []
                        for k in range(n_z):
                            for i in range(n_x):
                                col_indices.append(k * n_x * n_y + (n_y - 1) * n_x + i)
                        col_indices = np.array(col_indices)
                        vals = np.ones(n_x * n_z)
                        sub_matrix = csr_matrix(
                            (vals, (row_indices, col_indices)),
                            shape=(n_x * n_z, n_x * n_y * n_z),
                        )
                        additive = dxN * bcs[child][primary_side][0]
                    else:
                        dxN = dxN_y
                        dxNm1 = dxNm1_y
                        row_indices = np.arange(0, n_x * n_z)
                        col_indices_Nm1 = []
                        col_indices_N = []
                        for k in range(n_z):
                            for i in range(n_x):
                                col_indices_Nm1.append(
                                    k * n_x * n_y + (n_y - 2) * n_x + i
                                )
                                col_indices_N.append(
                                    k * n_x * n_y + (n_y - 1) * n_x + i
                                )
                        col_indices_Nm1 = np.array(col_indices_Nm1)
                        col_indices_N = np.array(col_indices_N)
                        vals_Nm1 = np.ones(n_x * n_z) * (-(dxN / dxNm1))
                        vals_N = np.ones(n_x * n_z) * (1 + (dxN / dxNm1))
                        sub_matrix = csr_matrix(
                            (
                                np.hstack([vals_Nm1, vals_N]),
                                (
                                    np.hstack([row_indices, row_indices]),
                                    np.hstack([col_indices_Nm1, col_indices_N]),
                                ),
                            ),
                            shape=(n_x * n_z, n_x * n_y * n_z),
                        )
                        additive = pybamm.Scalar(0)
                else:
                    raise NotImplementedError(
                        f"Order {extrap_order_value} not implemented for back boundary"
                    )

            elif primary_side == "bottom":
                if extrap_order_value == "linear":
                    if use_bcs and pybamm.has_bc_of_form(
                        child, primary_side, bcs, "Neumann"
                    ):
                        dx0 = dx0_z
                        row_indices = np.arange(0, n_x * n_y)
                        col_indices = np.arange(0, n_x * n_y)
                        vals = np.ones(n_x * n_y)
                        sub_matrix = csr_matrix(
                            (vals, (row_indices, col_indices)),
                            shape=(n_x * n_y, n_x * n_y * n_z),
                        )
                        additive = -dx0 * bcs[child][primary_side][0]
                    else:
                        dx0 = dx0_z
                        dx1 = dx1_z
                        row_indices = np.arange(0, n_x * n_y)
                        col_indices_0 = np.arange(0, n_x * n_y)
                        col_indices_1 = np.arange(n_x * n_y, 2 * n_x * n_y)
                        vals_0 = np.ones(n_x * n_y) * (1 + (dx0 / dx1))
                        vals_1 = np.ones(n_x * n_y) * (-(dx0 / dx1))
                        sub_matrix = csr_matrix(
                            (
                                np.hstack([vals_0, vals_1]),
                                (
                                    np.hstack([row_indices, row_indices]),
                                    np.hstack([col_indices_0, col_indices_1]),
                                ),
                            ),
                            shape=(n_x * n_y, n_x * n_y * n_z),
                        )
                        additive = pybamm.Scalar(0)
                else:
                    raise NotImplementedError(
                        f"Order {extrap_order_value} not implemented for bottom boundary"
                    )

            elif primary_side == "top":
                if extrap_order_value == "linear":
                    if use_bcs and pybamm.has_bc_of_form(
                        child, primary_side, bcs, "Neumann"
                    ):
                        dxN = dxN_z
                        row_indices = np.arange(0, n_x * n_y)
                        col_indices = np.arange((n_z - 1) * n_x * n_y, n_z * n_x * n_y)
                        vals = np.ones(n_x * n_y)
                        sub_matrix = csr_matrix(
                            (vals, (row_indices, col_indices)),
                            shape=(n_x * n_y, n_x * n_y * n_z),
                        )
                        additive = dxN * bcs[child][primary_side][0]
                    else:
                        dxN = dxN_z
                        dxNm1 = dxNm1_z
                        row_indices = np.arange(0, n_x * n_y)
                        col_indices_Nm1 = np.arange(
                            (n_z - 2) * n_x * n_y, (n_z - 1) * n_x * n_y
                        )
                        col_indices_N = np.arange(
                            (n_z - 1) * n_x * n_y, n_z * n_x * n_y
                        )
                        vals_Nm1 = np.ones(n_x * n_y) * (-(dxN / dxNm1))
                        vals_N = np.ones(n_x * n_y) * (1 + (dxN / dxNm1))
                        sub_matrix = csr_matrix(
                            (
                                np.hstack([vals_Nm1, vals_N]),
                                (
                                    np.hstack([row_indices, row_indices]),
                                    np.hstack([col_indices_Nm1, col_indices_N]),
                                ),
                            ),
                            shape=(n_x * n_y, n_x * n_y * n_z),
                        )
                        additive = pybamm.Scalar(0)
                else:
                    raise NotImplementedError(
                        f"Order {extrap_order_value} not implemented for top boundary"
                    )

            else:
                raise ValueError(f"Unknown primary side: {primary_side}")

            # Handle secondary sides (compound boundaries like edges)
            if secondary_side is not None:
                additive_secondary = pybamm.Scalar(0)

                if secondary_side == "front":
                    if use_bcs and pybamm.has_bc_of_form(
                        child, secondary_side, bcs, "Neumann"
                    ):
                        dx0 = dx0_y
                        additive_secondary = -dx0 * bcs[child][secondary_side][0]
                        if primary_side in ["left", "right"]:
                            secondary_matrix = csr_matrix(
                                ([1], ([0], [0])), shape=(1, n_y * n_z)
                            )
                        elif primary_side in ["bottom", "top"]:
                            secondary_matrix = csr_matrix(
                                ([1], ([0], [0])), shape=(1, n_x * n_y)
                            )
                    else:
                        dx0 = dx0_y
                        dx1 = dx1_y
                        if primary_side in ["left", "right"]:
                            vals = [1 + (dx0 / dx1), -(dx0 / dx1)]
                            secondary_matrix = csr_matrix(
                                (vals, ([0, 0], [0, n_z])), shape=(1, n_y * n_z)
                            )
                        elif primary_side in ["bottom", "top"]:
                            vals = [1 + (dx0 / dx1), -(dx0 / dx1)]
                            secondary_matrix = csr_matrix(
                                (vals, ([0, 0], [0, n_x])), shape=(1, n_x * n_y)
                            )

                    sub_matrix = secondary_matrix @ sub_matrix
                    additive += additive_secondary

                elif secondary_side == "back":
                    if use_bcs and pybamm.has_bc_of_form(
                        child, secondary_side, bcs, "Neumann"
                    ):
                        dxN = dxN_y
                        additive_secondary = dxN * bcs[child][secondary_side][0]
                        if primary_side in ["left", "right"]:
                            secondary_matrix = csr_matrix(
                                ([1], ([0], [n_y * n_z - n_z])), shape=(1, n_y * n_z)
                            )
                        elif primary_side in ["bottom", "top"]:
                            secondary_matrix = csr_matrix(
                                ([1], ([0], [n_x * (n_y - 1)])), shape=(1, n_x * n_y)
                            )
                    else:
                        dxN = dxN_y
                        dxNm1 = dxNm1_y
                        if primary_side in ["left", "right"]:
                            vals = [-(dxN / dxNm1), 1 + (dxN / dxNm1)]
                            secondary_matrix = csr_matrix(
                                (
                                    vals,
                                    ([0, 0], [n_y * n_z - 2 * n_z, n_y * n_z - n_z]),
                                ),
                                shape=(1, n_y * n_z),
                            )
                        elif primary_side in ["bottom", "top"]:
                            vals = [-(dxN / dxNm1), 1 + (dxN / dxNm1)]
                            secondary_matrix = csr_matrix(
                                (vals, ([0, 0], [n_x * (n_y - 2), n_x * (n_y - 1)])),
                                shape=(1, n_x * n_y),
                            )

                    sub_matrix = secondary_matrix @ sub_matrix
                    additive += additive_secondary

                elif secondary_side == "bottom":
                    if use_bcs and pybamm.has_bc_of_form(
                        child, secondary_side, bcs, "Neumann"
                    ):
                        dx0 = dx0_z
                        additive_secondary = -dx0 * bcs[child][secondary_side][0]
                        if primary_side in ["left", "right"]:
                            secondary_matrix = csr_matrix(
                                ([1], ([0], [0])), shape=(1, n_y * n_z)
                            )
                        elif primary_side in ["front", "back"]:
                            secondary_matrix = csr_matrix(
                                ([1], ([0], [0])), shape=(1, n_x * n_z)
                            )
                    else:
                        dx0 = dx0_z
                        dx1 = dx1_z
                        if primary_side in ["left", "right"]:
                            vals = [1 + (dx0 / dx1), -(dx0 / dx1)]
                            secondary_matrix = csr_matrix(
                                (vals, ([0, 0], [0, n_y])), shape=(1, n_y * n_z)
                            )
                        elif primary_side in ["front", "back"]:
                            vals = [1 + (dx0 / dx1), -(dx0 / dx1)]
                            secondary_matrix = csr_matrix(
                                (vals, ([0, 0], [0, n_x])), shape=(1, n_x * n_z)
                            )

                    sub_matrix = secondary_matrix @ sub_matrix
                    additive += additive_secondary

                elif secondary_side == "top":
                    if use_bcs and pybamm.has_bc_of_form(
                        child, secondary_side, bcs, "Neumann"
                    ):
                        dxN = dxN_z
                        additive_secondary = dxN * bcs[child][secondary_side][0]
                        if primary_side in ["left", "right"]:
                            secondary_matrix = csr_matrix(
                                ([1], ([0], [n_y * (n_z - 1)])), shape=(1, n_y * n_z)
                            )
                        elif primary_side in ["front", "back"]:
                            secondary_matrix = csr_matrix(
                                ([1], ([0], [n_x * (n_z - 1)])), shape=(1, n_x * n_z)
                            )
                    else:
                        dxN = dxN_z
                        dxNm1 = dxNm1_z
                        if primary_side in ["left", "right"]:
                            vals = [-(dxN / dxNm1), 1 + (dxN / dxNm1)]
                            secondary_matrix = csr_matrix(
                                (vals, ([0, 0], [n_y * (n_z - 2), n_y * (n_z - 1)])),
                                shape=(1, n_y * n_z),
                            )
                        elif primary_side in ["front", "back"]:
                            vals = [-(dxN / dxNm1), 1 + (dxN / dxNm1)]
                            secondary_matrix = csr_matrix(
                                (vals, ([0, 0], [n_x * (n_z - 2), n_x * (n_z - 1)])),
                                shape=(1, n_x * n_z),
                            )

                    sub_matrix = secondary_matrix @ sub_matrix
                    additive += additive_secondary

                # Handle tertiary sides (corner boundaries)
                if tertiary_side is not None:
                    additive_tertiary = pybamm.Scalar(0)

                    if tertiary_side == "bottom":
                        if use_bcs and pybamm.has_bc_of_form(
                            child, tertiary_side, bcs, "Neumann"
                        ):
                            dx0 = dx0_z
                            additive_tertiary = -dx0 * bcs[child][tertiary_side][0]
                            tertiary_matrix = csr_matrix(
                                ([1], ([0], [0])), shape=(1, 1)
                            )
                        else:
                            dx0 = dx0_z
                            dx1 = dx1_z
                            tertiary_matrix = csr_matrix(
                                ([1 + (dx0 / dx1)], ([0], [0])), shape=(1, 1)
                            )

                        sub_matrix = tertiary_matrix @ sub_matrix
                        additive += additive_tertiary

                    elif tertiary_side == "top":
                        if use_bcs and pybamm.has_bc_of_form(
                            child, tertiary_side, bcs, "Neumann"
                        ):
                            dxN = dxN_z
                            additive_tertiary = dxN * bcs[child][tertiary_side][0]
                            tertiary_matrix = csr_matrix(
                                ([1], ([0], [0])), shape=(1, 1)
                            )
                        else:
                            dxN = dxN_z
                            dxNm1 = dxNm1_z
                            tertiary_matrix = csr_matrix(
                                ([1 + (dxN / dxNm1)], ([0], [0])), shape=(1, 1)
                            )

                        sub_matrix = tertiary_matrix @ sub_matrix
                        additive += additive_tertiary

        elif isinstance(symbol, pybamm.BoundaryGradient):
            if use_bcs and pybamm.has_bc_of_form(child, primary_side, bcs, "Neumann"):
                if primary_side in ["left", "right"]:
                    sub_matrix = csr_matrix((n_y * n_z, n_x * n_y * n_z))
                elif primary_side in ["front", "back"]:
                    sub_matrix = csr_matrix((n_x * n_z, n_x * n_y * n_z))
                elif primary_side in ["bottom", "top"]:
                    sub_matrix = csr_matrix((n_x * n_y, n_x * n_y * n_z))
                additive = bcs[child][primary_side][0]

            elif primary_side == "left":
                if extrap_order_gradient == "linear":
                    dx1 = dx1_x
                    row_indices = np.arange(0, n_y * n_z)
                    col_indices_0 = np.arange(0, n_x * n_y * n_z, n_x)
                    col_indices_1 = col_indices_0 + 1
                    vals_0 = np.ones(n_y * n_z) * (-1 / dx1)
                    vals_1 = np.ones(n_y * n_z) * (1 / dx1)
                    sub_matrix = csr_matrix(
                        (
                            np.hstack([vals_0, vals_1]),
                            (
                                np.hstack([row_indices, row_indices]),
                                np.hstack([col_indices_0, col_indices_1]),
                            ),
                        ),
                        shape=(n_y * n_z, n_x * n_y * n_z),
                    )
                    additive = pybamm.Scalar(0)
                elif extrap_order_gradient == "quadratic":
                    # FIX: Implement quadratic extrapolation
                    dx1 = dx1_x
                    dx2 = submesh.d_nodes_x[1] if len(submesh.d_nodes_x) > 1 else dx1_x
                    row_indices = np.arange(0, n_y * n_z)
                    col_indices_0 = np.arange(0, n_x * n_y * n_z, n_x)
                    col_indices_1 = col_indices_0 + 1
                    col_indices_2 = col_indices_0 + 2 if n_x > 2 else col_indices_1

                    # Quadratic extrapolation coefficients
                    c0 = (2 * dx1 + dx2) / (dx1 * (dx1 + dx2))
                    c1 = -(dx1 + dx2) / (dx1 * dx2)
                    c2 = dx1 / (dx2 * (dx1 + dx2))

                    vals_0 = np.ones(n_y * n_z) * c0
                    vals_1 = np.ones(n_y * n_z) * c1
                    vals_2 = np.ones(n_y * n_z) * c2

                    sub_matrix = csr_matrix(
                        (
                            np.hstack([vals_0, vals_1, vals_2]),
                            (
                                np.hstack([row_indices, row_indices, row_indices]),
                                np.hstack(
                                    [col_indices_0, col_indices_1, col_indices_2]
                                ),
                            ),
                        ),
                        shape=(n_y * n_z, n_x * n_y * n_z),
                    )
                    additive = pybamm.Scalar(0)

                else:
                    raise NotImplementedError(
                        f"Order {extrap_order_gradient} not implemented for left gradient"
                    )

            elif primary_side == "right":
                if extrap_order_gradient == "linear":
                    dxNm1 = dxNm1_x
                    row_indices = np.arange(0, n_y * n_z)
                    col_indices_Nm1 = np.arange(n_x - 2, n_x * n_y * n_z, n_x)
                    col_indices_N = col_indices_Nm1 + 1
                    vals_Nm1 = np.ones(n_y * n_z) * (-1 / dxNm1)
                    vals_N = np.ones(n_y * n_z) * (1 / dxNm1)
                    sub_matrix = csr_matrix(
                        (
                            np.hstack([vals_Nm1, vals_N]),
                            (
                                np.hstack([row_indices, row_indices]),
                                np.hstack([col_indices_Nm1, col_indices_N]),
                            ),
                        ),
                        shape=(n_y * n_z, n_x * n_y * n_z),
                    )
                    additive = pybamm.Scalar(0)
                elif extrap_order_gradient == "quadratic":
                    dxNm1 = dxNm1_x
                    dxNm2 = (
                        submesh.d_nodes_x[-2] if len(submesh.d_nodes_x) > 1 else dxNm1_x
                    )
                    row_indices = np.arange(0, n_y * n_z)
                    col_indices_Nm2 = (
                        np.arange(n_x - 3, n_x * n_y * n_z, n_x)
                        if n_x > 2
                        else np.arange(n_x - 2, n_x * n_y * n_z, n_x)
                    )
                    col_indices_Nm1 = np.arange(n_x - 2, n_x * n_y * n_z, n_x)
                    col_indices_N = col_indices_Nm1 + 1

                    # Quadratic extrapolation coefficients for right boundary
                    c0 = dxNm1 / (dxNm2 * (dxNm1 + dxNm2))
                    c1 = -(dxNm1 + dxNm2) / (dxNm1 * dxNm2)
                    c2 = (2 * dxNm1 + dxNm2) / (dxNm1 * (dxNm1 + dxNm2))

                    vals_Nm2 = np.ones(n_y * n_z) * c0
                    vals_Nm1 = np.ones(n_y * n_z) * c1
                    vals_N = np.ones(n_y * n_z) * c2

                    sub_matrix = csr_matrix(
                        (
                            np.hstack([vals_Nm2, vals_Nm1, vals_N]),
                            (
                                np.hstack([row_indices, row_indices, row_indices]),
                                np.hstack(
                                    [col_indices_Nm2, col_indices_Nm1, col_indices_N]
                                ),
                            ),
                        ),
                        shape=(n_y * n_z, n_x * n_y * n_z),
                    )
                    additive = pybamm.Scalar(0)
                else:
                    raise NotImplementedError(
                        f"Order {extrap_order_gradient} not implemented for right gradient"
                    )

            elif primary_side == "front":
                if extrap_order_gradient == "linear":
                    dx1 = dx1_y
                    row_indices = np.arange(0, n_x * n_z)
                    col_indices_0 = []
                    col_indices_1 = []
                    for k in range(n_z):
                        for i in range(n_x):
                            col_indices_0.append(k * n_x * n_y + i)
                            col_indices_1.append(k * n_x * n_y + n_x + i)
                    col_indices_0 = np.array(col_indices_0)
                    col_indices_1 = np.array(col_indices_1)
                    vals_0 = np.ones(n_x * n_z) * (-1 / dx1)
                    vals_1 = np.ones(n_x * n_z) * (1 / dx1)
                    sub_matrix = csr_matrix(
                        (
                            np.hstack([vals_0, vals_1]),
                            (
                                np.hstack([row_indices, row_indices]),
                                np.hstack([col_indices_0, col_indices_1]),
                            ),
                        ),
                        shape=(n_x * n_z, n_x * n_y * n_z),
                    )
                    additive = pybamm.Scalar(0)
                elif extrap_order_gradient == "quadratic":
                    dx1 = dx1_y
                    dx2 = submesh.d_nodes_y[1] if len(submesh.d_nodes_y) > 1 else dx1_y
                    row_indices = np.arange(0, n_x * n_z)
                    col_indices_0 = []
                    col_indices_1 = []
                    col_indices_2 = []

                    for k in range(n_z):
                        for i in range(n_x):
                            col_indices_0.append(k * n_x * n_y + i)
                            col_indices_1.append(k * n_x * n_y + n_x + i)
                            col_indices_2.append(
                                k * n_x * n_y + 2 * n_x + i
                                if n_y > 2
                                else k * n_x * n_y + n_x + i
                            )

                    col_indices_0 = np.array(col_indices_0)
                    col_indices_1 = np.array(col_indices_1)
                    col_indices_2 = np.array(col_indices_2)

                    # Quadratic extrapolation coefficients
                    c0 = (2 * dx1 + dx2) / (dx1 * (dx1 + dx2))
                    c1 = -(dx1 + dx2) / (dx1 * dx2)
                    c2 = dx1 / (dx2 * (dx1 + dx2))

                    vals_0 = np.ones(n_x * n_z) * c0
                    vals_1 = np.ones(n_x * n_z) * c1
                    vals_2 = np.ones(n_x * n_z) * c2

                    sub_matrix = csr_matrix(
                        (
                            np.hstack([vals_0, vals_1, vals_2]),
                            (
                                np.hstack([row_indices, row_indices, row_indices]),
                                np.hstack(
                                    [col_indices_0, col_indices_1, col_indices_2]
                                ),
                            ),
                        ),
                        shape=(n_x * n_z, n_x * n_y * n_z),
                    )
                    additive = pybamm.Scalar(0)
                else:
                    raise NotImplementedError(
                        f"Order {extrap_order_gradient} not implemented for front gradient"
                    )

            elif primary_side == "back":
                if extrap_order_gradient == "linear":
                    dxNm1 = dxNm1_y
                    row_indices = np.arange(0, n_x * n_z)
                    col_indices_Nm1 = []
                    col_indices_N = []
                    for k in range(n_z):
                        for i in range(n_x):
                            col_indices_Nm1.append(k * n_x * n_y + (n_y - 2) * n_x + i)
                            col_indices_N.append(k * n_x * n_y + (n_y - 1) * n_x + i)
                    col_indices_Nm1 = np.array(col_indices_Nm1)
                    col_indices_N = np.array(col_indices_N)
                    vals_Nm1 = np.ones(n_x * n_z) * (-1 / dxNm1)
                    vals_N = np.ones(n_x * n_z) * (1 / dxNm1)
                    sub_matrix = csr_matrix(
                        (
                            np.hstack([vals_Nm1, vals_N]),
                            (
                                np.hstack([row_indices, row_indices]),
                                np.hstack([col_indices_Nm1, col_indices_N]),
                            ),
                        ),
                        shape=(n_x * n_z, n_x * n_y * n_z),
                    )
                    additive = pybamm.Scalar(0)
                elif extrap_order_gradient == "quadratic":
                    dxNm1 = dxNm1_y
                    dxNm2 = (
                        submesh.d_nodes_y[-2] if len(submesh.d_nodes_y) > 1 else dxNm1_y
                    )
                    row_indices = np.arange(0, n_x * n_z)
                    col_indices_Nm2 = []
                    col_indices_Nm1 = []
                    col_indices_N = []

                    for k in range(n_z):
                        for i in range(n_x):
                            col_indices_Nm2.append(
                                k * n_x * n_y + (n_y - 3) * n_x + i
                                if n_y > 2
                                else k * n_x * n_y + (n_y - 2) * n_x + i
                            )
                            col_indices_Nm1.append(k * n_x * n_y + (n_y - 2) * n_x + i)
                            col_indices_N.append(k * n_x * n_y + (n_y - 1) * n_x + i)

                    col_indices_Nm2 = np.array(col_indices_Nm2)
                    col_indices_Nm1 = np.array(col_indices_Nm1)
                    col_indices_N = np.array(col_indices_N)

                    # Quadratic extrapolation coefficients for back boundary
                    c0 = dxNm1 / (dxNm2 * (dxNm1 + dxNm2))
                    c1 = -(dxNm1 + dxNm2) / (dxNm1 * dxNm2)
                    c2 = (2 * dxNm1 + dxNm2) / (dxNm1 * (dxNm1 + dxNm2))

                    vals_Nm2 = np.ones(n_x * n_z) * c0
                    vals_Nm1 = np.ones(n_x * n_z) * c1
                    vals_N = np.ones(n_x * n_z) * c2

                    sub_matrix = csr_matrix(
                        (
                            np.hstack([vals_Nm2, vals_Nm1, vals_N]),
                            (
                                np.hstack([row_indices, row_indices, row_indices]),
                                np.hstack(
                                    [col_indices_Nm2, col_indices_Nm1, col_indices_N]
                                ),
                            ),
                        ),
                        shape=(n_x * n_z, n_x * n_y * n_z),
                    )
                    additive = pybamm.Scalar(0)
                else:
                    raise NotImplementedError(
                        f"Order {extrap_order_gradient} not implemented for back gradient"
                    )

            elif primary_side == "bottom":
                if extrap_order_gradient == "linear":
                    dx1 = dx1_z
                    row_indices = np.arange(0, n_x * n_y)
                    col_indices_0 = np.arange(0, n_x * n_y)
                    col_indices_1 = np.arange(n_x * n_y, 2 * n_x * n_y)
                    vals_0 = np.ones(n_x * n_y) * (-1 / dx1)
                    vals_1 = np.ones(n_x * n_y) * (1 / dx1)
                    sub_matrix = csr_matrix(
                        (
                            np.hstack([vals_0, vals_1]),
                            (
                                np.hstack([row_indices, row_indices]),
                                np.hstack([col_indices_0, col_indices_1]),
                            ),
                        ),
                        shape=(n_x * n_y, n_x * n_y * n_z),
                    )
                    additive = pybamm.Scalar(0)
                elif extrap_order_gradient == "quadratic":
                    dx1 = dx1_z
                    dx2 = submesh.d_nodes_z[1] if len(submesh.d_nodes_z) > 1 else dx1_z
                    row_indices = np.arange(0, n_x * n_y)
                    col_indices_0 = np.arange(0, n_x * n_y)
                    col_indices_1 = np.arange(n_x * n_y, 2 * n_x * n_y)
                    col_indices_2 = (
                        np.arange(2 * n_x * n_y, 3 * n_x * n_y)
                        if n_z > 2
                        else np.arange(n_x * n_y, 2 * n_x * n_y)
                    )

                    # Quadratic extrapolation coefficients
                    c0 = (2 * dx1 + dx2) / (dx1 * (dx1 + dx2))
                    c1 = -(dx1 + dx2) / (dx1 * dx2)
                    c2 = dx1 / (dx2 * (dx1 + dx2))

                    vals_0 = np.ones(n_x * n_y) * c0
                    vals_1 = np.ones(n_x * n_y) * c1
                    vals_2 = np.ones(n_x * n_y) * c2

                    sub_matrix = csr_matrix(
                        (
                            np.hstack([vals_0, vals_1, vals_2]),
                            (
                                np.hstack([row_indices, row_indices, row_indices]),
                                np.hstack(
                                    [col_indices_0, col_indices_1, col_indices_2]
                                ),
                            ),
                        ),
                        shape=(n_x * n_y, n_x * n_y * n_z),
                    )
                    additive = pybamm.Scalar(0)
                else:
                    raise NotImplementedError(
                        f"Order {extrap_order_gradient} not implemented for bottom gradient"
                    )

            elif primary_side == "top":
                if extrap_order_gradient == "linear":
                    dxNm1 = dxNm1_z
                    row_indices = np.arange(0, n_x * n_y)
                    col_indices_Nm1 = np.arange(
                        (n_z - 2) * n_x * n_y, (n_z - 1) * n_x * n_y
                    )
                    col_indices_N = np.arange((n_z - 1) * n_x * n_y, n_z * n_x * n_y)
                    vals_Nm1 = np.ones(n_x * n_y) * (-1 / dxNm1)
                    vals_N = np.ones(n_x * n_y) * (1 / dxNm1)
                    sub_matrix = csr_matrix(
                        (
                            np.hstack([vals_Nm1, vals_N]),
                            (
                                np.hstack([row_indices, row_indices]),
                                np.hstack([col_indices_Nm1, col_indices_N]),
                            ),
                        ),
                        shape=(n_x * n_y, n_x * n_y * n_z),
                    )
                    additive = pybamm.Scalar(0)
                elif extrap_order_gradient == "quadratic":
                    dxNm1 = dxNm1_z
                    dxNm2 = (
                        submesh.d_nodes_z[-2] if len(submesh.d_nodes_z) > 1 else dxNm1_z
                    )
                    row_indices = np.arange(0, n_x * n_y)
                    col_indices_Nm2 = (
                        np.arange((n_z - 3) * n_x * n_y, (n_z - 2) * n_x * n_y)
                        if n_z > 2
                        else np.arange((n_z - 2) * n_x * n_y, (n_z - 1) * n_x * n_y)
                    )
                    col_indices_Nm1 = np.arange(
                        (n_z - 2) * n_x * n_y, (n_z - 1) * n_x * n_y
                    )
                    col_indices_N = np.arange((n_z - 1) * n_x * n_y, n_z * n_x * n_y)

                    # Quadratic extrapolation coefficients for top boundary
                    c0 = dxNm1 / (dxNm2 * (dxNm1 + dxNm2))
                    c1 = -(dxNm1 + dxNm2) / (dxNm1 * dxNm2)
                    c2 = (2 * dxNm1 + dxNm2) / (dxNm1 * (dxNm1 + dxNm2))

                    vals_Nm2 = np.ones(n_x * n_y) * c0
                    vals_Nm1 = np.ones(n_x * n_y) * c1
                    vals_N = np.ones(n_x * n_y) * c2

                    sub_matrix = csr_matrix(
                        (
                            np.hstack([vals_Nm2, vals_Nm1, vals_N]),
                            (
                                np.hstack([row_indices, row_indices, row_indices]),
                                np.hstack(
                                    [col_indices_Nm2, col_indices_Nm1, col_indices_N]
                                ),
                            ),
                        ),
                        shape=(n_x * n_y, n_x * n_y * n_z),
                    )
                    additive = pybamm.Scalar(0)
                else:
                    raise NotImplementedError(
                        f"Order {extrap_order_gradient} not implemented for top gradient"
                    )

            else:
                raise NotImplementedError(
                    f"Gradient boundary condition for {primary_side} not implemented"
                )

        else:
            raise ValueError(f"Unknown symbol type: {type(symbol)}")

        # Fallback if sub_matrix is still None or empty
        if sub_matrix is None or sub_matrix.nnz == 0:
            # Create simple extraction matrix based on primary_side
            if primary_side in ["left", "right"]:
                shape = (n_y * n_z, n_x * n_y * n_z)
                rows = np.arange(n_y * n_z)
                if primary_side == "left":
                    cols = np.arange(0, n_x * n_y * n_z, n_x)
                else:  # right
                    cols = np.arange(n_x - 1, n_x * n_y * n_z, n_x)

            elif primary_side in ["front", "back"]:
                shape = (n_x * n_z, n_x * n_y * n_z)
                rows = np.arange(n_x * n_z)
                cols = []
                for k in range(n_z):
                    for i in range(n_x):
                        if primary_side == "front":
                            cols.append(k * n_x * n_y + i)
                        else:  # back
                            cols.append(k * n_x * n_y + (n_y - 1) * n_x + i)
                cols = np.array(cols)

            else:  # bottom, top
                shape = (n_x * n_y, n_x * n_y * n_z)
                rows = np.arange(n_x * n_y)
                if primary_side == "bottom":
                    cols = np.arange(n_x * n_y)
                else:  # top
                    cols = np.arange((n_z - 1) * n_x * n_y, n_z * n_x * n_y)

            vals = np.ones(len(rows))
            sub_matrix = csr_matrix((vals, (rows, cols)), shape=shape)

        # Create final matrix
        matrix = csr_matrix(kron(eye(repeats), sub_matrix))
        matrix = pybamm.Matrix(matrix)
        boundary_value = matrix @ discretised_child
        boundary_value.copy_domains(symbol)

        additive.copy_domains(symbol)
        boundary_value += additive

        return boundary_value

    def evaluate_at(self, symbol, discretised_child, position):
        """
        Evaluate a symbol at a specific position in 3D space.

        Parameters
        ----------
        symbol: :class:`pybamm.Symbol`
            The boundary value or flux symbol.
        discretised_child : :class:`pybamm.StateVector`
            The discretised variable from which to calculate the value.
        position : list or array
            The point in 3D space at which to evaluate the symbol.

        Returns
        -------
        :class:`pybamm.MatrixMultiplication`
            The variable representing the value at the given point.
        """
        submesh = self.mesh[symbol.domain]
        x_pos, y_pos, z_pos = position

        # Find closest indices
        x_idx = np.argmin(np.abs(submesh.nodes_x - x_pos))
        y_idx = np.argmin(np.abs(submesh.nodes_y - y_pos))
        z_idx = np.argmin(np.abs(submesh.nodes_z - z_pos))

        # Convert to linear index
        linear_idx = (
            z_idx * submesh.npts_x * submesh.npts_y + y_idx * submesh.npts_x + x_idx
        )

        # Create selection matrix
        selection_matrix = csr_matrix(
            ([1], ([0], [linear_idx])), shape=(1, submesh.npts)
        )

        return pybamm.Matrix(selection_matrix) @ discretised_child

    def process_binary_operators(self, bin_op, left, right, disc_left, disc_right):
        """Discretise binary operators in model equations for 3D finite volume.
        Performs appropriate averaging of diffusivities if one of the children is a
        gradient operator, so that discretised sizes match up. For this averaging we
        use the harmonic mean [1].

        [1] Recktenwald, Gerald. "The control-volume finite-difference approximation to
        the diffusion equation." (2012).

        Parameters
        ----------
        bin_op : :class:`pybamm.BinaryOperator`
            Binary operator to discretise
        left : :class:`pybamm.Symbol`
            The left child of `bin_op`
        right : :class:`pybamm.Symbol`
            The right child of `bin_op`
        disc_left : :class:`pybamm.Symbol`
            The discretised left child of `bin_op`
        disc_right : :class:`pybamm.Symbol`
            The discretised right child of `bin_op`
        Returns
        -------
        :class:`pybamm.BinaryOperator`
            Discretised binary operator
        """

        # Post-processing to make sure discretised dimensions match
        left_evaluates_on_edges = left.evaluates_on_edges("primary")
        right_evaluates_on_edges = right.evaluates_on_edges("primary")

        # Handle 3D VectorField cases
        if hasattr(disc_left, "x_field") and hasattr(disc_right, "x_field"):
            if right_evaluates_on_edges and not left_evaluates_on_edges:
                if isinstance(right, pybamm.Gradient):
                    method = "harmonic"
                else:
                    method = "arithmetic"

                disc_left_x = self.node_to_edge(
                    disc_left.x_field, method=method, direction="x"
                )
                disc_left_y = self.node_to_edge(
                    disc_left.y_field, method=method, direction="y"
                )
                disc_left_z = self.node_to_edge(
                    disc_left.z_field, method=method, direction="z"
                )
                disc_left = pybamm.VectorField3D(disc_left_x, disc_left_y, disc_left_z)

            elif left_evaluates_on_edges and not right_evaluates_on_edges:
                if isinstance(left, pybamm.Gradient):
                    method = "harmonic"
                else:
                    method = "arithmetic"

                disc_right_x = self.node_to_edge(
                    disc_right.x_field, method=method, direction="x"
                )
                disc_right_y = self.node_to_edge(
                    disc_right.y_field, method=method, direction="y"
                )
                disc_right_z = self.node_to_edge(
                    disc_right.z_field, method=method, direction="z"
                )
                disc_right = pybamm.VectorField3D(
                    disc_right_x, disc_right_y, disc_right_z
                )

            x_field = pybamm.simplify_if_constant(
                bin_op.create_copy([disc_left.x_field, disc_right])
            )
            y_field = pybamm.simplify_if_constant(
                bin_op.create_copy([disc_left.y_field, disc_right])
            )
            z_field = pybamm.simplify_if_constant(
                bin_op.create_copy([disc_left.z_field, disc_right])
            )
            return pybamm.VectorField3D(x_field, y_field, z_field)

        elif hasattr(disc_left, "x_field") and not hasattr(disc_right, "x_field"):
            if left_evaluates_on_edges and not right_evaluates_on_edges:
                if isinstance(left, pybamm.Gradient):
                    method = "harmonic"
                else:
                    method = "arithmetic"

                disc_right_x = self.node_to_edge(
                    disc_right, method=method, direction="x"
                )
                disc_right_y = self.node_to_edge(
                    disc_right, method=method, direction="y"
                )
                disc_right_z = self.node_to_edge(
                    disc_right, method=method, direction="z"
                )
                disc_right_vector = pybamm.VectorField3D(
                    disc_right_x, disc_right_y, disc_right_z
                )

                x_field = pybamm.simplify_if_constant(
                    bin_op.create_copy([disc_left.x_field, disc_right_vector.x_field])
                )
                y_field = pybamm.simplify_if_constant(
                    bin_op.create_copy([disc_left.y_field, disc_right_vector.y_field])
                )
                z_field = pybamm.simplify_if_constant(
                    bin_op.create_copy([disc_left.z_field, disc_right_vector.z_field])
                )
            else:
                x_field = pybamm.simplify_if_constant(
                    bin_op.create_copy([disc_left.x_field, disc_right])
                )
                y_field = pybamm.simplify_if_constant(
                    bin_op.create_copy([disc_left.y_field, disc_right])
                )
                z_field = pybamm.simplify_if_constant(
                    bin_op.create_copy([disc_left.z_field, disc_right])
                )
            return pybamm.VectorField3D(x_field, y_field, z_field)

        elif not hasattr(disc_left, "x_field") and hasattr(disc_right, "x_field"):
            if right_evaluates_on_edges and not left_evaluates_on_edges:
                if isinstance(right, pybamm.Gradient):
                    method = "harmonic"
                else:
                    method = "arithmetic"

                disc_left_x = self.node_to_edge(disc_left, method=method, direction="x")
                disc_left_y = self.node_to_edge(disc_left, method=method, direction="y")
                disc_left_z = self.node_to_edge(disc_left, method=method, direction="z")
                disc_left_vector = pybamm.VectorField3D(
                    disc_left_x, disc_left_y, disc_left_z
                )

                x_field = pybamm.simplify_if_constant(
                    bin_op.create_copy([disc_left_vector.x_field, disc_right.x_field])
                )
                y_field = pybamm.simplify_if_constant(
                    bin_op.create_copy([disc_left_vector.y_field, disc_right.y_field])
                )
                z_field = pybamm.simplify_if_constant(
                    bin_op.create_copy([disc_left_vector.z_field, disc_right.z_field])
                )
            else:
                x_field = pybamm.simplify_if_constant(
                    bin_op.create_copy([disc_left, disc_right.x_field])
                )
                y_field = pybamm.simplify_if_constant(
                    bin_op.create_copy([disc_left, disc_right.y_field])
                )
                z_field = pybamm.simplify_if_constant(
                    bin_op.create_copy([disc_left, disc_right.z_field])
                )
            return pybamm.VectorField3D(x_field, y_field, z_field)

        # If neither child evaluates on edges, or both children have gradients,
        # no need to do any averaging
        elif left_evaluates_on_edges == right_evaluates_on_edges:
            pass
        # If only left child evaluates on edges, map right child onto edges
        # using the harmonic mean if the left child is a gradient (i.e. this
        # binary operator represents a flux)
        elif left_evaluates_on_edges and not right_evaluates_on_edges:
            if isinstance(left, pybamm.Gradient):
                method = "harmonic"
            else:
                method = "arithmetic"
            disc_right = self.node_to_edge(disc_right, method=method)

        # If only right child evaluates on edges, map left child onto edges
        # using the harmonic mean if the right child is a gradient (i.e. this
        # binary operator represents a flux)
        elif right_evaluates_on_edges and not left_evaluates_on_edges:
            if isinstance(right, pybamm.Gradient):
                method = "harmonic"
            else:
                method = "arithmetic"
            disc_left = self.node_to_edge(disc_left, method=method)

        # Return new binary operator with appropriate class
        out = pybamm.simplify_if_constant(bin_op.create_copy([disc_left, disc_right]))

        return out

    def concatenation(self, disc_children):
        """
        Discrete concatenation for 3D finite-volume.

        Any child vector of length n_edges (i.e. defined on edges) is
        first averaged to nodes via self.edge_to_node(). All others
        must then be node-valued (length n_nodes). Finally we
        call domain_concatenation() which handles the x→y→z flattening.
        """
        for idx, child in enumerate(disc_children):
            submesh = self.mesh[child.domain]
            repeats = self._get_auxiliary_domain_repeats(child.domains)

            # total number of cell centers (nodes) in 3D = npts_x * npts_y * npts_z
            n_nodes = submesh.npts_x * submesh.npts_y * submesh.npts_z * repeats

            # total number of edges: sum over x , y and z directed edges
            n_ex = (submesh.npts_x - 1) * submesh.npts_y * submesh.npts_z
            n_ey = submesh.npts_x * (submesh.npts_y - 1) * submesh.npts_z
            n_ez = submesh.npts_x * submesh.npts_y * (submesh.npts_z - 1)
            n_edges = (n_ex + n_ey + n_ez) * repeats

            if child.size == n_edges:
                # average and replace in place
                disc_children[idx] = self.edge_to_node(child)
            elif child.size != n_nodes:
                raise pybamm.ShapeError(
                    f"In 3D concatenation expected child size to be n_nodes={n_nodes} "
                    f"or n_edges={n_edges}, but got {child.size}"
                )

        return pybamm.domain_concatenation(disc_children, self.mesh)

    def edge_to_node(self, discretised_symbol, method="arithmetic", direction="x"):
        """
        Convert a discretised symbol evaluated on the cell edges to a discretised symbol
        evaluated on the cell nodes.
        See :meth:`pybamm.FiniteVolume.shift`
        """
        return self.shift(discretised_symbol, "edge to node", method, direction)

    def node_to_edge(self, discretised_symbol, method="arithmetic", direction="x"):
        """
        Convert a discretised symbol evaluated on the cell nodes to a discretised symbol
        evaluated on the cell edges.
        See :meth:`pybamm.FiniteVolume.shift`
        """
        return self.shift(discretised_symbol, "node to edge", method, direction)

    def shift(self, discretised_symbol, shift_key, method, direction="x"):
        """
        Convert a discretised symbol evaluated at edges/nodes, to a discretised symbol
        evaluated at nodes/edges. Can be the arithmetic mean or the harmonic mean.

        Note: when computing fluxes at cell edges it is better to take the
        harmonic mean based on [1].

        [1] Recktenwald, Gerald. "The control-volume finite-difference approximation to
        the diffusion equation." (2012).

        Parameters
        ----------
        discretised_symbol : :class:`pybamm.Symbol`
            Symbol to be averaged. When evaluated, this symbol returns either a scalar
            or an array of shape (n,) or (n+1,), where n is the number of points in the
            mesh for the symbol's domain (n = self.mesh[symbol.domain].npts)
        shift_key : str
            Whether to shift from nodes to edges ("node to edge"), or from edges to
            nodes ("edge to node")
        method : str
            Whether to use the "arithmetic" or "harmonic" mean
        direction : str
            Direction along which to shift ("x", "y", or "z")

        Returns
        -------
        :class:`pybamm.Symbol`
            Averaged symbol. When evaluated, this returns either a scalar or an array of
            shape (n+1,) (if `shift_key = "node to edge"`) or (n,) (if
            `shift_key = "edge to node"`)
        """

        def arithmetic_mean(array, direction):
            """Calculate the arithmetic mean of an array using matrix multiplication"""
            submesh = self.mesh[array.domain]

            n_x, n_y, n_z = submesh.npts_x, submesh.npts_y, submesh.npts_z

            if shift_key == "node to edge":
                if direction == "x":
                    sub_matrix_1d = diags([0.5, 0.5], [0, 1], shape=(n_x - 1, n_x))
                    sub_matrix = block_diag([sub_matrix_1d] * (n_y * n_z))

                elif direction == "y":
                    # For y-direction: we need to connect adjacent nodes in y for each x,z position
                    # Total y-edges: n_x * (n_y-1) * n_z
                    # Each edge connects two nodes that are n_x positions apart in the flattened array

                    n_edges_y = n_x * (n_y - 1) * n_z
                    n_nodes = n_x * n_y * n_z

                    data = []
                    rows = []
                    cols = []

                    edge_idx = 0
                    for k in range(n_z):
                        for j in range(n_y - 1):
                            for i in range(n_x):
                                node1 = k * (n_x * n_y) + j * n_x + i
                                node2 = k * (n_x * n_y) + (j + 1) * n_x + i

                                rows.extend([edge_idx, edge_idx])
                                cols.extend([node1, node2])
                                data.extend([0.5, 0.5])

                                edge_idx += 1

                    sub_matrix = csr_matrix(
                        (data, (rows, cols)), shape=(n_edges_y, n_nodes)
                    )

                elif direction == "z":
                    n_edges_z = n_x * n_y * (n_z - 1)
                    n_nodes = n_x * n_y * n_z

                    data = []
                    rows = []
                    cols = []

                    edge_idx = 0
                    for k in range(n_z - 1):
                        for j in range(n_y):
                            for i in range(n_x):
                                node1 = k * (n_x * n_y) + j * n_x + i
                                node2 = (k + 1) * (n_x * n_y) + j * n_x + i

                                rows.extend([edge_idx, edge_idx])
                                cols.extend([node1, node2])
                                data.extend([0.5, 0.5])

                                edge_idx += 1

                    sub_matrix = csr_matrix(
                        (data, (rows, cols)), shape=(n_edges_z, n_nodes)
                    )

                else:
                    raise ValueError(f"direction '{direction}' not recognised")

            elif shift_key == "edge to node":
                if direction == "x":
                    data = []
                    rows = []
                    cols = []

                    row_idx = 0
                    col_idx = 0

                    for _k in range(n_z):
                        for _j in range(n_y):
                            rows.append(row_idx)
                            cols.append(col_idx)
                            data.append(1.0)
                            row_idx += 1

                            for i in range(1, n_x - 1):
                                rows.extend([row_idx, row_idx])
                                cols.extend([col_idx + i - 1, col_idx + i])
                                data.extend([0.5, 0.5])
                                row_idx += 1

                            rows.append(row_idx)
                            cols.append(col_idx + n_x - 2)
                            data.append(1.0)
                            row_idx += 1

                            col_idx += n_x - 1

                    sub_matrix = csr_matrix(
                        (data, (rows, cols)),
                        shape=(n_x * n_y * n_z, (n_x - 1) * n_y * n_z),
                    )

                elif direction == "y":
                    n_edges_y = n_x * (n_y - 1) * n_z
                    n_nodes = n_x * n_y * n_z

                    data = []
                    rows = []
                    cols = []

                    node_idx = 0

                    for k in range(n_z):
                        for j in range(n_y):
                            for i in range(n_x):
                                if j == 0:
                                    edge_idx = k * n_x * (n_y - 1) + 0 * n_x + i
                                    rows.append(node_idx)
                                    cols.append(edge_idx)
                                    data.append(1.0)
                                elif j == n_y - 1:
                                    edge_idx = k * n_x * (n_y - 1) + (n_y - 2) * n_x + i
                                    rows.append(node_idx)
                                    cols.append(edge_idx)
                                    data.append(1.0)
                                else:
                                    edge_left = k * n_x * (n_y - 1) + (j - 1) * n_x + i
                                    edge_right = k * n_x * (n_y - 1) + j * n_x + i
                                    rows.extend([node_idx, node_idx])
                                    cols.extend([edge_left, edge_right])
                                    data.extend([0.5, 0.5])

                                node_idx += 1
                                print(f"Expected: nodes={n_nodes}, edges_y={n_edges_y}")

                    sub_matrix = csr_matrix(
                        (data, (rows, cols)), shape=(n_nodes, n_edges_y)
                    )

                elif direction == "z":
                    n_edges_z = n_x * n_y * (n_z - 1)
                    n_nodes = n_x * n_y * n_z

                    data = []
                    rows = []
                    cols = []

                    node_idx = 0

                    for k in range(n_z):
                        for j in range(n_y):
                            for i in range(n_x):
                                if k == 0:
                                    edge_idx = 0 * (n_x * n_y) + j * n_x + i
                                    rows.append(node_idx)
                                    cols.append(edge_idx)
                                    data.append(1.0)
                                elif k == n_z - 1:
                                    edge_idx = (n_z - 2) * (n_x * n_y) + j * n_x + i
                                    rows.append(node_idx)
                                    cols.append(edge_idx)
                                    data.append(1.0)
                                else:
                                    edge_lower = (k - 1) * (n_x * n_y) + j * n_x + i
                                    edge_upper = k * (n_x * n_y) + j * n_x + i
                                    rows.extend([node_idx, node_idx])
                                    cols.extend([edge_lower, edge_upper])
                                    data.extend([0.5, 0.5])

                                node_idx += 1

                    sub_matrix = csr_matrix(
                        (data, (rows, cols)), shape=(n_nodes, n_edges_z)
                    )

            else:
                raise ValueError(f"shift key '{shift_key}' not recognised")

            second_dim_repeats = self._get_auxiliary_domain_repeats(array.domains)
            matrix = csr_matrix(kron(eye(second_dim_repeats), sub_matrix))
            print(f"Matrix shape: {sub_matrix.shape}")
            print(f"After repeats: {matrix.shape}")

            return pybamm.Matrix(matrix) @ array

        def harmonic_mean(array, direction):
            """Calculate the harmonic mean of an array using matrix multiplication"""
            submesh = self.mesh[array.domain]
            n_x, n_y, n_z = submesh.npts_x, submesh.npts_y, submesh.npts_z
            repeats = self._get_auxiliary_domain_repeats(array.domains)

            if shift_key == "node to edge":
                if direction == "x":
                    n_edges_x = n_x - 1
                    D1_1d = lil_matrix((n_edges_x, n_x))
                    D2_1d = lil_matrix((n_edges_x, n_x))

                    for i in range(n_edges_x):
                        D1_1d[i, i] = 1.0  # Left node
                        D2_1d[i, i + 1] = 1.0  # Right node

                    D1_1d = D1_1d.tocsr()
                    D2_1d = D2_1d.tocsr()

                    M1 = kron(eye(n_z), kron(eye(n_y), D1_1d))
                    M2 = kron(eye(n_z), kron(eye(n_y), D2_1d))

                    D1 = pybamm.Matrix(M1) @ array
                    D2 = pybamm.Matrix(M2) @ array

                    dx = submesh.d_edges_x
                    beta_1d = dx[:-2] / (dx[1:-1] + dx[:-2])

                    n_edges_total = (n_x - 1) * n_y * n_z

                    beta_full = np.zeros(n_edges_total)

                    for k in range(n_z):
                        for j in range(n_y):
                            for i in range(n_x - 1):
                                edge_idx = k * n_y * (n_x - 1) + j * (n_x - 1) + i
                                if i > 0 and i < n_x - 2:
                                    beta_full[edge_idx] = beta_1d[i - 1]
                                else:
                                    beta_full[edge_idx] = 0.5

                    beta_pybamm = pybamm.Array(beta_full.reshape(-1, 1))

                    denominator = beta_pybamm * D2 + (1 - beta_pybamm) * D1

                    epsilon = 1e-12
                    denominator = denominator + epsilon

                    return D1 * D2 / denominator

                elif direction == "y":
                    D1_1d_y = hstack([eye(n_y - 1), csr_matrix((n_y - 1, 1))])
                    D2_1d_y = hstack([csr_matrix((n_y - 1, 1)), eye(n_y - 1)])

                    sub_D1_y = block_diag([D1_1d_y] * n_x)
                    sub_D2_y = block_diag([D2_1d_y] * n_x)

                    M1 = kron(eye(n_z), sub_D1_y)
                    M2 = kron(eye(n_z), sub_D2_y)

                    D1 = pybamm.Matrix(kron(eye(repeats), M1)) @ array
                    D2 = pybamm.Matrix(kron(eye(repeats), M2)) @ array

                    dy = submesh.d_edges_y
                    beta_1d = dy[:-2] / (dy[1:-1] + dy[:-2])

                    n_edges_total = n_x * (n_y - 1) * n_z
                    beta_full = np.zeros(n_edges_total)

                    for k in range(n_z):
                        for i in range(n_x):
                            for j in range(n_y - 1):
                                edge_idx = k * n_x * (n_y - 1) + i * (n_y - 1) + j
                                if j > 0 and j < n_y - 2:
                                    beta_full[edge_idx] = beta_1d[j - 1]
                                else:
                                    beta_full[edge_idx] = 0.5

                    beta_pybamm = pybamm.Array(beta_full.reshape(-1, 1))
                    denominator = beta_pybamm * D2 + (1 - beta_pybamm) * D1

                    epsilon = 1e-12
                    denominator = denominator + epsilon

                    return D1 * D2 / denominator

                elif direction == "z":
                    D1_1d_z = hstack([eye(n_z - 1), csr_matrix((n_z - 1, 1))])
                    D2_1d_z = hstack([csr_matrix((n_z - 1, 1)), eye(n_z - 1)])
                    sub_D1_z = block_diag([D1_1d_z] * (n_x * n_y))
                    sub_D2_z = block_diag([D2_1d_z] * (n_x * n_y))
                    M1 = sub_D1_z
                    M2 = sub_D2_z
                    D1 = pybamm.Matrix(kron(eye(repeats), M1)) @ array
                    D2 = pybamm.Matrix(kron(eye(repeats), M2)) @ array

                    dz = submesh.d_edges_z
                    beta_1d = dz[:-2] / (dz[1:-1] + dz[:-2])

                    n_edges_total = n_x * n_y * (n_z - 1)
                    beta_full = np.zeros(n_edges_total)

                    for i in range(n_x):
                        for j in range(n_y):
                            for k in range(n_z - 1):
                                edge_idx = (i * n_y + j) * (n_z - 1) + k
                                if k > 0 and k < n_z - 2:
                                    beta_full[edge_idx] = beta_1d[k - 1]
                                else:
                                    beta_full[edge_idx] = 0.5

                    beta_pybamm = pybamm.Array(beta_full.reshape(-1, 1))
                    denominator = beta_pybamm * D2 + (1 - beta_pybamm) * D1

                    epsilon = 1e-12
                    denominator = denominator + epsilon

                    return D1 * D2 / denominator
                else:
                    return arithmetic_mean(array, direction)

            elif shift_key == "edge to node":
                if direction == "x":
                    n_edges_x = n_x - 1

                    stencil_1d = lil_matrix((n_x, n_edges_x))

                    stencil_1d[0, 0] = 1.0

                    for i in range(1, n_x - 1):
                        stencil_1d[i, i - 1] = 0.5
                        stencil_1d[i, i] = 0.5

                    stencil_1d[n_x - 1, n_edges_x - 1] = 1.0

                    stencil_1d = stencil_1d.tocsr()

                    E = kron(eye(n_z), kron(eye(n_y), stencil_1d))

                    return pybamm.Matrix(E) @ array

                elif direction == "y":
                    n_edges_y = n_y - 1

                    stencil_2d = lil_matrix((n_x * n_y, n_x * n_edges_y))

                    for i in range(n_x):
                        edge_start = i * n_edges_y
                        node_start = i * n_y
                        stencil_2d[node_start, edge_start] = 1.0

                        for j in range(1, n_y - 1):
                            node_idx = node_start + j
                            edge_left = edge_start + j - 1
                            edge_right = edge_start + j
                            stencil_2d[node_idx, edge_left] = 0.5
                            stencil_2d[node_idx, edge_right] = 0.5

                        stencil_2d[node_start + n_y - 1, edge_start + n_edges_y - 1] = (
                            1.0
                        )

                    stencil_2d = stencil_2d.tocsr()

                    E = kron(eye(n_z), stencil_2d)
                    E = csr_matrix(kron(eye(repeats), E))

                    return pybamm.Matrix(E) @ array

                elif direction == "z":
                    n_edges_z = n_z - 1

                    stencil_3d = lil_matrix((n_x * n_y * n_z, n_x * n_y * n_edges_z))

                    for i in range(n_x):
                        for j in range(n_y):
                            xy_idx = i * n_y + j

                            node_start = xy_idx * n_z
                            edge_start = xy_idx * n_edges_z
                            stencil_3d[node_start, edge_start] = 1.0

                            for k in range(1, n_z - 1):
                                node_idx = node_start + k
                                edge_left = edge_start + k - 1
                                edge_right = edge_start + k
                                stencil_3d[node_idx, edge_left] = 0.5
                                stencil_3d[node_idx, edge_right] = 0.5

                            stencil_3d[
                                node_start + n_z - 1, edge_start + n_edges_z - 1
                            ] = 1.0

                    stencil_3d = stencil_3d.tocsr()

                    E = csr_matrix(kron(eye(repeats), stencil_3d))

                    return pybamm.Matrix(E) @ array

            else:
                raise ValueError(f"shift key '{shift_key}' not recognised")

        # Helper methods for harmonic mean
        def _exterior_stencil(n):
            left = csr_matrix(([1.5, -0.5], ([0, 0], [0, 1])), shape=(1, n))
            right = csr_matrix(([-0.5, 1.5], ([0, 0], [n - 2, n - 1])), shape=(1, n))
            return left, right

        def _pick_D1_D2(n):
            D1 = hstack([eye(n - 1), csr_matrix((n - 1, 1))])
            D2 = hstack([csr_matrix((n - 1, 1)), eye(n - 1)])
            return D1, D2

        self._exterior_stencil = _exterior_stencil
        self._pick_D1_D2 = _pick_D1_D2

        if discretised_symbol.size == 1:
            out = discretised_symbol
        elif method == "arithmetic":
            out = arithmetic_mean(discretised_symbol, direction)
        elif method == "harmonic":
            out = harmonic_mean(discretised_symbol, direction)
        else:
            raise ValueError(f"method '{method}' not recognised")
        return out

    def upwind_or_downwind(
        self,
        symbol,
        discretised_symbol,
        bcs,
        x_direction=None,
        y_direction=None,
        z_direction=None,
    ):
        """
        3D upwind/downwind: for each axis, either insert a Dirichlet ghost layer
        on the specified face (if direction is "upwind"/"downwind") or else average
        from nodes to edges (if direction is None).

        Parameters
        ----------
        symbol : :class:`pybamm.SpatialVariable`
            The variable to be discretised.
        discretised_symbol : :class:`pybamm.Vector`
            The already discretised symbol (no ghost cells).
        bcs : dict
            A dictionary mapping `symbol` → {face_name: (value, "Dirichlet"/"Neumann"), …}.
            Valid face names: "left", "right", "front", "back", "bottom", "top".
        x_direction : {None, "upwind", "downwind"}, default None
            If "upwind", insert a left‐face Dirichlet ghost; if "downwind", insert a right‐face.
            If None, perform node→edge averaging in x.
        y_direction : {None, "upwind", "downwind"}, default None
            If "upwind", insert a front‐face Dirichlet ghost; if "downwind", insert a back‐face.
            If None, perform node→edge averaging in y.
        z_direction : {None, "upwind", "downwind"}, default None
            If "upwind", insert a bottom‐face Dirichlet ghost; if "downwind", insert a top‐face.
            If None, perform node→edge averaging in z.

        Returns
        -------
        :class:`pybamm.VectorField3D`
            A 3D vector field whose x, y, z components have been upwinded (i.e. ghost‐cell
            Dirichlet if requested, or else averaged from nodes to edges).
        """
        if symbol not in bcs:
            raise pybamm.ModelError(f"No boundary conditions defined for {symbol!r}.")

        # Determine which face (if any) to use in each axis
        if x_direction == "upwind":
            x_face = "left"
        elif x_direction == "downwind":
            x_face = "right"
        elif x_direction is None:
            x_face = None
        else:
            raise ValueError(
                f"x_direction must be 'upwind', 'downwind', or None, not '{x_direction}'"
            )

        if y_direction == "upwind":
            y_face = "front"
        elif y_direction == "downwind":
            y_face = "back"
        elif y_direction is None:
            y_face = None
        else:
            raise ValueError(
                f"y_direction must be 'upwind', 'downwind', or None, not '{y_direction}'"
            )

        if z_direction == "upwind":
            z_face = "bottom"
        elif z_direction == "downwind":
            z_face = "top"
        elif z_direction is None:
            z_face = None
        else:
            raise ValueError(
                f"z_direction must be 'upwind', 'downwind', or None, not '{z_direction}'"
            )

        # X-component
        if x_face is not None:
            if x_face not in bcs[symbol]:
                raise pybamm.ModelError(
                    f"No BC provided for face '{x_face}' of {symbol!r}."
                )
            val, bc_type = bcs[symbol][x_face]
            if bc_type != "Dirichlet":
                raise pybamm.ModelError(
                    f"Dirichlet BC required on '{x_face}' for x upwind/downwind."
                )
            bc_subset = {x_face: (val, bc_type)}
            out_x, _ = self.add_ghost_nodes(symbol, discretised_symbol, bc_subset)
        else:
            out_x = self.node_to_edge(discretised_symbol, direction="x")

        # Y-component
        if y_face is not None:
            if y_face not in bcs[symbol]:
                raise pybamm.ModelError(
                    f"No BC provided for face '{y_face}' of {symbol!r}."
                )
            val, bc_type = bcs[symbol][y_face]
            if bc_type != "Dirichlet":
                raise pybamm.ModelError(
                    f"Dirichlet BC required on '{y_face}' for y upwind/downwind."
                )
            bc_subset = {y_face: (val, bc_type)}
            out_y, _ = self.add_ghost_nodes(symbol, discretised_symbol, bc_subset)
        else:
            out_y = self.node_to_edge(discretised_symbol, direction="y")

        # Z-component
        if z_face is not None:
            if z_face not in bcs[symbol]:
                raise pybamm.ModelError(
                    f"No BC provided for face '{z_face}' of {symbol!r}."
                )
            val, bc_type = bcs[symbol][z_face]
            if bc_type != "Dirichlet":
                raise pybamm.ModelError(
                    f"Dirichlet BC required on '{z_face}' for z upwind/downwind."
                )
            bc_subset = {z_face: (val, bc_type)}
            out_z, _ = self.add_ghost_nodes(symbol, discretised_symbol, bc_subset)
        else:
            out_z = self.node_to_edge(discretised_symbol, direction="z")

        return pybamm.VectorField3D(out_x, out_y, out_z)
