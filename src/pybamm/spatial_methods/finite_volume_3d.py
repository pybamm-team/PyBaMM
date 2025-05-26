#
# Finite Volume discretisation class
#
import pybamm

from scipy.sparse import (
    diags,
    eye,
    kron,
    csr_matrix,
    vstack,
    hstack,
    block_diag,
)
import numpy as np


def _scatter_rows(rows, values, length):
    """Return a 1D numpy array of size `length` with `values[i]` at
    position `rows[i]` and zeros elsewhere."""
    out = np.zeros((length,))
    out[rows] = values[rows]  # both are same length
    return out


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
            x = X.flatten()
            y = Y.flatten()
            z = Z.flatten()
        else:
            X, Y, Z = np.meshgrid(
                symbol_mesh.nodes_x,
                symbol_mesh.nodes_y,
                symbol_mesh.nodes_z,
                indexing="ij",
            )
            x = X.flatten()
            y = Y.flatten()
            z = Z.flatten()
        if symbol_direction == "x":
            entries = np.tile(x, repeats)
        elif symbol_direction == "y":
            entries = np.tile(y, repeats)
        elif symbol_direction == "z":
            entries = np.tile(z, repeats)
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

        if symbol in boundary_conditions:
            bcs = {
                key: boundary_conditions[symbol][key]
                for key in relevant_bcs
                if key in boundary_conditions[symbol]
            }
            if any(bc[1] == "Dirichlet" for bc in bcs.values()):
                # add ghost nodes and update domain
                discretised_symbol, domain = self.add_ghost_nodes(
                    symbol, discretised_symbol, bcs
                )

        gradient_matrix = self.gradient_matrix(domain, symbol.domains, direction)
        grad = gradient_matrix @ discretised_symbol
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
        """Matrix-vector multiplication to implement the gradient operator in 3D."""
        # Compute gradients in all three directions
        grad_x = self._gradient(symbol, discretised_symbol, boundary_conditions, "x")
        grad_y = self._gradient(symbol, discretised_symbol, boundary_conditions, "y")
        grad_z = self._gradient(symbol, discretised_symbol, boundary_conditions, "z")

        # Return as a 3D vector field
        grad = pybamm.VectorField3D(grad_x, grad_y, grad_z)
        return grad

    def gradient_matrix(self, domain, domains, direction):
        """
        Gradient matrix for finite volumes in 3D.
        """
        submesh = self.mesh[domain]

        # Create matrix using submesh
        n_x = submesh.npts_x
        n_y = submesh.npts_y
        n_z = submesh.npts_z

        e_x = 1 / submesh.d_nodes_x
        e_y = 1 / submesh.d_nodes_y
        e_z = 1 / submesh.d_nodes_z

        if direction == "x":
            # Gradient in x-direction
            if submesh.coord_sys == "cylindrical polar":
                r_nodes = submesh.nodes_x
                r_weights = r_nodes
                sub_matrix = diags(
                    [-e_x * r_weights[:-1], e_x * r_weights[1:]],
                    [0, 1],
                    shape=(n_x - 1, n_x),
                )
                sub_matrix = block_diag((sub_matrix,) * (n_y * n_z))
            elif submesh.coord_sys == "spherical polar":
                r_nodes = submesh.nodes_x
                e_x = 1 / submesh.d_nodes_x
                r_weights = r_nodes**2  # radial weighting
                sub_matrix = diags(
                    [-e_x * r_weights[:-1], e_x * r_weights[1:]],
                    [0, 1],
                    shape=(n_x - 1, n_x),
                )
                sub_matrix = block_diag((sub_matrix,) * (n_y * n_z))
            elif submesh.coord_sys == "spiral":
                spiral_metric = self.compute_spiral_metric(submesh)
                sub_matrix = diags(
                    [-e_x * spiral_metric[:-1], e_x * spiral_metric[1:]],
                    [0, 1],
                    shape=(n_x - 1, n_x),
                )
                sub_matrix = block_diag((sub_matrix,) * (n_y * n_z))
            else:
                sub_matrix = diags([-e_x, e_x], [0, 1], shape=(n_x - 1, n_x))
                sub_matrix = block_diag((sub_matrix,) * (n_y * n_z))

        elif direction == "y":
            # Gradient in y-direction
            if submesh.coord_sys == "cylindrical polar":
                r_nodes = submesh.nodes_x
                r_weights = r_nodes
                e_y_repeated = np.repeat(e_y, n_x)
                sub_matrix = diags(
                    [-e_y_repeated, e_y_repeated],
                    [0, n_x],
                    shape=(n_x * (n_y - 1), n_x * n_y),
                )
                # Repeat for each z-plane
                sub_matrix = block_diag((sub_matrix,) * n_z)
            elif submesh.coord_sys == "spherical polar":
                r_nodes = submesh.nodes_x
                r_weights = r_nodes**2
                e_y_repeated = np.repeat(e_y, n_x)
                sub_matrix = diags(
                    [-e_y_repeated, e_y_repeated],
                    [0, n_x],
                    shape=(n_x * (n_y - 1), n_x * n_y),
                )
                # Repeat for each z-plane
                sub_matrix = block_diag((sub_matrix,) * n_z)
            elif submesh.coord_sys == "spiral":
                spiral_metric = self.compute_spiral_metric(submesh)
                e_y_repeated = np.repeat(e_y * spiral_metric[:-1], n_x)
                sub_matrix = diags(
                    [-e_y_repeated, e_y_repeated],
                    [0, n_x],
                    shape=(n_x * (n_y - 1), n_x * n_y),
                )
                # Repeat for each z-plane
                sub_matrix = block_diag((sub_matrix,) * n_z)
            else:
                e_y_repeated = np.repeat(e_y, n_x)
                sub_matrix = diags(
                    [-e_y_repeated, e_y_repeated],
                    [0, n_x],
                    shape=(n_x * (n_y - 1), n_x * n_y),
                )
                # Repeat for each z-plane
                sub_matrix = block_diag((sub_matrix,) * n_z)

        elif direction == "z":
            # Gradient in z-direction
            if submesh.coord_sys == "cylindrical polar":
                r_nodes = submesh.nodes_x
                r_weights = r_nodes
                e_z_repeated = np.repeat(e_z, n_x * n_y)
                sub_matrix = diags(
                    [-e_z_repeated, e_z_repeated],
                    [0, n_x * n_y],
                    shape=(n_x * n_y * (n_z - 1), n_x * n_y * n_z),
                )
            elif submesh.coord_sys == "spherical polar":
                r_nodes = submesh.nodes_x
                r_weights = r_nodes**2
                e_z_repeated = np.repeat(e_z, n_x * n_y)
                sub_matrix = diags(
                    [-e_z_repeated, e_z_repeated],
                    [0, n_x * n_y],
                    shape=(n_x * n_y * (n_z - 1), n_x * n_y * n_z),
                )
            elif submesh.coord_sys == "spiral":
                spiral_metric = self.compute_spiral_metric(submesh)
                e_z_repeated = np.repeat(e_z * spiral_metric[:-1], n_x * n_y)
                sub_matrix = diags(
                    [-e_z_repeated, e_z_repeated],
                    [0, n_x * n_y],
                    shape=(n_x * n_y * (n_z - 1), n_x * n_y * n_z),
                )
            else:
                e_z_repeated = np.repeat(e_z, n_x * n_y)
                sub_matrix = diags(
                    [-e_z_repeated, e_z_repeated],
                    [0, n_x * n_y],
                    shape=(n_x * n_y * (n_z - 1), n_x * n_y * n_z),
                )

        # number of repeats for auxiliary domains
        second_dim_repeats = self._get_auxiliary_domain_repeats(domains)

        # generate full matrix from the submatrix
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
        """
        submesh = self.mesh[domains["primary"]]
        n_x = submesh.npts_x
        n_y = submesh.npts_y
        n_z = submesh.npts_z

        e_x = 1 / submesh.d_edges_x
        e_y = 1 / submesh.d_edges_y
        e_z = 1 / submesh.d_edges_z

        if direction == "x":
            # Divergence in x-direction
            if submesh.coord_sys == "cylindrical polar":
                r_nodes = submesh.nodes_x
                r_weights = r_nodes
                sub_matrix = diags(
                    [-e_x * r_weights[:-1], e_x * r_weights[1:]],
                    [0, 1],
                    shape=(n_x, n_x + 1),
                )
                sub_matrix = block_diag((sub_matrix,) * (n_y * n_z))
            elif submesh.coord_sys == "spherical polar":
                r_nodes = submesh.nodes_x
                r_weights = r_nodes**2
                sub_matrix = diags(
                    [-e_x * r_weights[:-1], e_x * r_weights[1:]],
                    [0, 1],
                    shape=(n_x, n_x + 1),
                )
                sub_matrix = block_diag((sub_matrix,) * (n_y * n_z))
            elif submesh.coord_sys == "spiral":
                spiral_metric = self.compute_spiral_metric(submesh)
                sub_matrix = diags(
                    [-e_x * spiral_metric[:-1], e_x * spiral_metric[1:]],
                    [0, 1],
                    shape=(n_x, n_x + 1),
                )
                sub_matrix = block_diag((sub_matrix,) * (n_y * n_z))
            else:
                sub_matrix = diags([-e_x, e_x], [0, 1], shape=(n_x, n_x + 1))
                sub_matrix = block_diag((sub_matrix,) * (n_y * n_z))

        elif direction == "y":
            # Divergence in y-direction
            if submesh.coord_sys == "cylindrical polar":
                r_nodes = submesh.nodes_x
                r_weights = r_nodes
                e_y_repeated = np.repeat(e_y, n_x)
                sub_matrix = diags(
                    [-e_y_repeated, e_y_repeated],
                    [0, n_x],
                    shape=(n_x, n_x * (n_y + 1)),
                )
                sub_matrix = block_diag((sub_matrix,) * n_z)
            elif submesh.coord_sys == "spherical polar":
                r_nodes = submesh.nodes_x
                r_weights = r_nodes**2
                e_y_repeated = np.repeat(e_y, n_x)
                sub_matrix = diags(
                    [-e_y_repeated, e_y_repeated],
                    [0, n_x],
                    shape=(n_x, n_x * (n_y + 1)),
                )
                sub_matrix = block_diag((sub_matrix,) * n_z)
            elif submesh.coord_sys == "spiral":
                spiral_metric = self.compute_spiral_metric(submesh)
                e_y_repeated = np.repeat(e_y * spiral_metric[:-1], n_x)
                sub_matrix = diags(
                    [-e_y_repeated, e_y_repeated],
                    [0, n_x],
                    shape=(n_x, n_x * (n_y + 1)),
                )
                sub_matrix = block_diag((sub_matrix,) * n_z)
            else:
                e_y_repeated = np.repeat(e_y, n_x + 1)
                sub_matrix = diags(
                    [-e_y_repeated, e_y_repeated],
                    [0, n_x],
                    shape=(n_x * n_y, n_x * (n_y + 1)),
                )
                sub_matrix = block_diag((sub_matrix,) * n_z)

        elif direction == "z":
            # Divergence in z-direction
            if submesh.coord_sys == "cylindrical polar":
                r_nodes = submesh.nodes_x
                r_weights = r_nodes
                e_z_repeated = np.repeat(e_z, n_x * n_y)
                sub_matrix = diags(
                    [-e_z_repeated, e_z_repeated],
                    [0, n_x * n_y],
                    shape=(n_x * n_y, n_x * n_y * (n_z + 1)),
                )
            elif submesh.coord_sys == "spherical polar":
                r_nodes = submesh.nodes_x
                r_weights = r_nodes**2
                e_z_repeated = np.repeat(e_z, n_x * n_y)
                sub_matrix = diags(
                    [-e_z_repeated, e_z_repeated],
                    [0, n_x * n_y],
                    shape=(n_x * n_y, n_x * n_y * (n_z + 1)),
                )
            elif submesh.coord_sys == "spiral":
                spiral_metric = self.compute_spiral_metric(submesh)
                e_z_repeated = np.repeat(e_z * spiral_metric[:-1], n_x * n_y)
                sub_matrix = diags(
                    [-e_z_repeated, e_z_repeated],
                    [0, n_x * n_y],
                    shape=(n_x * n_y, n_x * n_y * (n_z + 1)),
                )
            else:
                e_z_repeated = np.repeat(e_z, n_x * n_y + 1)
                sub_matrix = diags(
                    [-e_z_repeated, e_z_repeated],
                    [0, n_x * n_y],
                    shape=(n_x * n_y * n_z, n_x * n_y * (n_z + 1)),
                )

        return pybamm.Matrix(sub_matrix)

    def integral(
        self, child, discretised_child, integration_dimension, integration_variable
    ):
        """3D integration operator."""
        integration_matrix = self.definite_integral_matrix(
            child,
            integration_dimension=integration_dimension,
            integration_variable=integration_variable,
        )

        # Handle multiple integration variables (e.g., integrating over x, y, and z)
        if len(integration_variable) > 1:
            for _i, var in enumerate(integration_variable[1:], 1):
                direction = var.direction
                one_dimensional_matrix = self.one_dimensional_integral_matrix(
                    child, direction
                )
                # Fix dimension matching issue
                if integration_matrix.shape[1] != one_dimensional_matrix.shape[0]:
                    # Adjust the matrix dimensions to match
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
                    weights = np.sin(theta_nodes)  # sin(θ) weighting for spherical
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
        indefinite_integral_matrix = self.indefinite_integral_matrix_nodes(
            child.domains, direction
        )
        return indefinite_integral_matrix @ discretised_child

    def indefinite_integral_matrix_edges(self, domains, direction):
        """
        Matrix for indefinite integral from edges to nodes in 3D.
        """
        submesh = self.mesh[domains["primary"]]
        n_x = submesh.npts_x
        n_y = submesh.npts_y
        n_z = submesh.npts_z

        if direction == "x":
            # Cumulative sum in x-direction
            d_edges = submesh.d_edges_x
            sub_matrix = np.tril(np.ones((n_x, n_x + 1)))
            sub_matrix = sub_matrix @ diags(d_edges, shape=(n_x + 1, n_x + 1))
            sub_matrix = block_diag((sub_matrix,) * (n_y * n_z))

        elif direction == "y":
            # Cumulative sum in y-direction
            d_edges = submesh.d_edges_y
            sub_matrix = np.tril(np.ones((n_y, n_y + 1)))
            sub_matrix = sub_matrix @ diags(d_edges, shape=(n_y + 1, n_y + 1))
            # Create block structure for 3D
            sub_matrix = kron(eye(n_z), kron(sub_matrix, eye(n_x)))

        elif direction == "z":
            # Cumulative sum in z-direction
            d_edges = submesh.d_edges_z
            sub_matrix = np.tril(np.ones((n_z, n_z + 1)))
            sub_matrix = sub_matrix @ diags(d_edges, shape=(n_z + 1, n_z + 1))
            # Create block structure for 3D
            sub_matrix = kron(sub_matrix, eye(n_x * n_y))

        # repeat matrix for secondary dimensions
        second_dim_repeats = self._get_auxiliary_domain_repeats(domains)
        matrix = csr_matrix(kron(eye(second_dim_repeats), sub_matrix))
        return pybamm.Matrix(matrix)

    def indefinite_integral_matrix_nodes(self, domains, direction):
        """
        Matrix for indefinite integral from nodes to nodes in 3D.
        """
        submesh = self.mesh[domains["primary"]]
        n_x = submesh.npts_x
        n_y = submesh.npts_y
        n_z = submesh.npts_z

        if direction == "x":
            # Cumulative sum in x-direction
            d_edges = submesh.d_edges_x
            sub_matrix = np.tril(np.ones((n_x, n_x)))
            sub_matrix = sub_matrix @ diags(d_edges, shape=(n_x, n_x))
            sub_matrix = block_diag((sub_matrix,) * (n_y * n_z))

        elif direction == "y":
            # Cumulative sum in y-direction
            d_edges = submesh.d_edges_y
            sub_matrix = np.tril(np.ones((n_y, n_y)))
            sub_matrix = sub_matrix @ diags(d_edges, shape=(n_y, n_y))
            # Create block structure for 3D
            sub_matrix = kron(eye(n_z), kron(sub_matrix, eye(n_x)))

        elif direction == "z":
            # Cumulative sum in z-direction
            d_edges = submesh.d_edges_z
            sub_matrix = np.tril(np.ones((n_z, n_z)))
            sub_matrix = sub_matrix @ diags(d_edges, shape=(n_z, n_z))
            # Create block structure for 3D
            sub_matrix = kron(sub_matrix, eye(n_x * n_y))

        # repeat matrix for secondary dimensions
        second_dim_repeats = self._get_auxiliary_domain_repeats(domains)
        matrix = csr_matrix(kron(eye(second_dim_repeats), sub_matrix))
        return pybamm.Matrix(matrix)

    def delta_function(self, symbol, discretised_symbol):
        """
        Delta function implementation for 3D.

        Parameters
        ----------
        symbol : :class:`pybamm.Symbol`
            The delta function symbol.
        discretised_symbol : :class:`pybamm.StateVector`
            The discretised variable.

        Returns
        -------
        :class:`pybamm.Vector`
            The delta function vector.
        """
        position = symbol.child
        submesh = self.mesh[symbol.domain]

        # Evaluate position
        pos_val = position.evaluate() if hasattr(position, "evaluate") else position
        x_pos, y_pos, z_pos = pos_val

        # Find closest indices
        x_idx = np.argmin(np.abs(submesh.nodes_x - x_pos))
        y_idx = np.argmin(np.abs(submesh.nodes_y - y_pos))
        z_idx = np.argmin(np.abs(submesh.nodes_z - z_pos))

        # Convert to linear index
        linear_idx = (
            z_idx * submesh.npts_x * submesh.npts_y + y_idx * submesh.npts_x + x_idx
        )

        # Create delta vector
        delta_vec = np.zeros(submesh.npts)
        delta_vec[linear_idx] = 1.0 / (
            submesh.d_edges_x[x_idx]
            * submesh.d_edges_y[y_idx]
            * submesh.d_edges_z[z_idx]
        )

        return pybamm.Vector(delta_vec, domains=symbol.domains)

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

        # Create matrices to extract boundary values
        # Assuming connection is in x direction (can be extended for other directions)
        left_sub_matrix = np.zeros((1, left_npts))
        # Extract rightmost face of left domain
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
        # Extract leftmost face of right domain
        for k in range(right_npts_z):
            for j in range(right_npts_y):
                idx = k * right_npts_x * right_npts_y + j * right_npts_x + 0
                right_sub_matrix[0, idx] = 1

        right_matrix = pybamm.Matrix(
            csr_matrix(kron(eye(second_dim_repeats), right_sub_matrix))
        )

        # Calculate finite volume derivative
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
        3D version of add_ghost_nodes.  Builds ghost‐layers on any of the six faces
        for Dirichlet BCs in x (left/right), y (front/back), z (bottom/top).
        Neumann faces are handled elsewhere in the gradient routine.
        """
        domain = symbol.domain
        submesh = self.mesh[domain]

        nx, ny, nz = submesh.npts_x, submesh.npts_y, submesh.npts_z
        # n = nx * ny * nz
        repeats = self._get_auxiliary_domain_repeats(symbol.domains)

        # Pull out BC values & types
        (xl_val, xl_type) = bcs.get("left", (None, None))
        (xr_val, xr_type) = bcs.get("right", (None, None))
        (yf_val, yf_type) = bcs.get("front", (None, None))
        (yb_val, yb_type) = bcs.get("back", (None, None))
        (zb_val, zb_type) = bcs.get("bottom", (None, None))
        (zt_val, zt_type) = bcs.get("top", (None, None))

        ng_x = (1 if xl_type == "Dirichlet" else 0) + (
            1 if xr_type == "Dirichlet" else 0
        )
        ng_y = (1 if yf_type == "Dirichlet" else 0) + (
            1 if yb_type == "Dirichlet" else 0
        )
        ng_z = (1 if zb_type == "Dirichlet" else 0) + (
            1 if zt_type == "Dirichlet" else 0
        )
        # new total points in each direction
        NX = nx + ng_x
        NY = ny + ng_y
        NZ = nz + ng_z
        N = NX * NY * NZ

        # build the sparse re indexing matrix M of shape (N, n) that
        # takes the old n vector and injects it into the middle of size
        # leaving rows for the new ghost layers.
        #
        Ix_rows = []
        Ix_cols = []
        Ix_data = []
        # if left ghost, that sits at index 0..0, then the real 0..nx-1 go at offset ng_left
        x_offset = 1 if xl_type == "Dirichlet" else 0
        for i in range(nx):
            Ix_rows.append(i + x_offset)
            Ix_cols.append(i)
            Ix_data.append(1)
        Ix = csr_matrix((Ix_data, (Ix_rows, Ix_cols)), shape=(NX, nx))

        Iy_rows = []
        Iy_cols = []
        Iy_data = []
        y_offset = 1 if yf_type == "Dirichlet" else 0
        for j in range(ny):
            Iy_rows.append(j + y_offset)
            Iy_cols.append(j)
            Iy_data.append(1)
        Iy = csr_matrix((Iy_data, (Iy_rows, Iy_cols)), shape=(NY, ny))

        Iz_rows = []
        Iz_cols = []
        Iz_data = []
        z_offset = 1 if zb_type == "Dirichlet" else 0
        for k in range(nz):
            Iz_rows.append(k + z_offset)
            Iz_cols.append(k)
            Iz_data.append(1)
        Iz = csr_matrix((Iz_data, (Iz_rows, Iz_cols)), shape=(NZ, nz))

        # the full 3D injection is Kron(Iz, Kron(Iy, Ix)), of shape (N, n):
        M = kron(Iz, kron(Iy, Ix))

        bc_vec = pybamm.Vector(np.zeros((N * repeats,)))

        def face_constant(val, face):
            # build a vector of length N that has exactly the ghost layer index
            # for that face picked out, times (2*val), and zero elsewhere.
            if not val.evaluates_to_number():
                # symbolic
                c = 2 * val
                return c * pybamm.Vector(np.ones((N * repeats,)))
            else:
                return 2 * float(val) * np.ones((N * repeats,))

        # LEFT
        if xl_type == "Dirichlet":
            # those ghosts sit at x-index = 0, for all y,z
            rows = []
            for kk in range(NZ):
                for jj in range(NY):
                    idx = (kk * NY + jj) * NX + 0
                    rows.append(idx)
            c = face_constant(xl_val, "left")
            bc_vec += pybamm.Vector(_scatter_rows(rows, c, N * repeats))
        # RIGHT
        if xr_type == "Dirichlet":
            rows = []
            for kk in range(NZ):
                for jj in range(NY):
                    idx = (kk * NY + jj) * NX + (NX - 1)
                    rows.append(idx)
            c = face_constant(xr_val, "right")
            bc_vec += pybamm.Vector(_scatter_rows(rows, c, N * repeats))
        # FRONT (y-min)
        if yf_type == "Dirichlet":
            rows = []
            for kk in range(NZ):
                for ii in range(NX):
                    idx = kk * (NY * NX) + 0 * NX + ii
                    rows.append(idx)
            c = face_constant(yf_val, "front")
            bc_vec += pybamm.Vector(_scatter_rows(rows, c, N * repeats))
        # BACK (y-max)
        if yb_type == "Dirichlet":
            rows = []
            for kk in range(NZ):
                for ii in range(NX):
                    idx = kk * (NY * NX) + (NY - 1) * NX + ii
                    rows.append(idx)
            c = face_constant(yb_val, "back")
            bc_vec += pybamm.Vector(_scatter_rows(rows, c, N * repeats))
        # BOTTOM (z-min)
        if zb_type == "Dirichlet":
            rows = []
            for jj in range(NY):
                for ii in range(NX):
                    idx = 0 * (NY * NX) + jj * NX + ii
                    rows.append(idx)
            c = face_constant(zb_val, "bottom")
            bc_vec += pybamm.Vector(_scatter_rows(rows, c, N * repeats))
        # TOP (z-max)
        if zt_type == "Dirichlet":
            rows = []
            for jj in range(NY):
                for ii in range(NX):
                    idx = (NZ - 1) * (NY * NX) + jj * NX + ii
                    rows.append(idx)
            c = face_constant(zt_val, "top")
            bc_vec += pybamm.Vector(_scatter_rows(rows, c, N * repeats))

        new_symbol = pybamm.Matrix(M) @ discretised_symbol + bc_vec

        new_symbol.copy_domains(discretised_symbol)
        return new_symbol

    def add_neumann_values(self, symbol, discretised_gradient, bcs, domain):
        """
        3D version of add_neumann_values.  Any Neumann BC on one of the six faces
        contributes a known flux into the gradient vector; Dirichlet BCs were handled
        earlier by ghost‐nodes.
        """
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

        if xlt == "Neumann" and float(xlv) != 0:
            rows = []
            for k in range(nz):
                for j in range(ny):
                    # row index in flattened (nnx*ny*nz) vector
                    idx = k * (nny * nnx) + j * nnx + 0
                    rows.append(idx)
            val = float(xlv) if xlv.evaluates_to_number() else xlv
            bc_vec += pybamm.Vector(
                scatter(val * np.ones(len(rows)), rows, Mx.shape[0] * repeats)
            )

        if xrt == "Neumann" and float(xrv) != 0:
            rows = []
            for k in range(nz):
                for j in range(ny):
                    idx = k * (nny * nnx) + j * nnx + (nnx - 1)
                    rows.append(idx)
            val = float(xrv) if xrv.evaluates_to_number() else xrv
            bc_vec += pybamm.Vector(
                scatter(val * np.ones(len(rows)), rows, Mx.shape[0] * repeats)
            )

        if yft == "Neumann" and float(yfv) != 0:
            rows = []
            for k in range(nz):
                for i in range(
                    nx - 1 + (1 if xlt == "Neumann" or xrt == "Neumann" else 0)
                ):
                    idx = k * (nny * nnx) + 0 * nnx + i
                    rows.append(idx)
            val = float(yfv) if yfv.evaluates_to_number() else yfv
            bc_vec += pybamm.Vector(
                scatter(val * np.ones(len(rows)), rows, Mx.shape[0] * repeats)
            )

        if ybt == "Neumann" and float(ybv) != 0:
            rows = []
            for k in range(nz):
                for i in range(
                    nx - 1 + (1 if xlt == "Neumann" or xrt == "Neumann" else 0)
                ):
                    idx = k * (nny * nnx) + (nny - 1) * nnx + i
                    rows.append(idx)
            val = float(ybv) if ybv.evaluates_to_number() else ybv
            bc_vec += pybamm.Vector(
                scatter(val * np.ones(len(rows)), rows, Mx.shape[0] * repeats)
            )

        if zbt == "Neumann" and float(zbv) != 0:
            rows = []
            for j in range(ny + (1 if yft == "Neumann" or ybt == "Neumann" else 0)):
                for i in range(
                    nx - 1 + (1 if xlt == "Neumann" or xrt == "Neumann" else 0)
                ):
                    idx = 0 * (nny * nnx) + j * nnx + i
                    rows.append(idx)
            val = float(zbv) if zbv.evaluates_to_number() else zbv
            bc_vec += pybamm.Vector(
                scatter(val * np.ones(len(rows)), rows, Mx.shape[0] * repeats)
            )

        if ztt == "Neumann" and float(ztv) != 0:
            rows = []
            for j in range(ny + (1 if yft == "Neumann" or ybt == "Neumann" else 0)):
                for i in range(
                    nx - 1 + (1 if xlt == "Neumann" or xrt == "Neumann" else 0)
                ):
                    idx = (nnz - 1) * (nny * nnx) + j * nnx + i
                    rows.append(idx)
            val = float(ztv) if ztv.evaluates_to_number() else ztv
            bc_vec += pybamm.Vector(
                scatter(val * np.ones(len(rows)), rows, Mx.shape[0] * repeats)
            )

        # finally assemble
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
                # Apply secondary boundary condition by matrix multiplication
                if secondary_side == "front":
                    if use_bcs and pybamm.has_bc_of_form(
                        child, secondary_side, bcs, "Neumann"
                    ):
                        dx0 = dx0_y
                        additive_secondary = -dx0 * bcs[child][secondary_side][0]
                        # Create selection matrix for front edge
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
                        additive_secondary = pybamm.Scalar(0)

                    sub_matrix = secondary_matrix @ sub_matrix
                    if "additive_secondary" in locals():
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
                        additive_secondary = pybamm.Scalar(0)

                    sub_matrix = secondary_matrix @ sub_matrix
                    if "additive_secondary" in locals():
                        additive += additive_secondary

                elif secondary_side == "bottom":
                    if use_bcs and pybamm.has_bc_of_form(
                        child, secondary_side, bcs, "Neumann"
                    ):
                        dx0 = dx0_z
                        additive_secondary = -dx0 * bcs[child][secondary_side][0]
                        # Create selection matrix for bottom edge
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
                        additive_secondary = pybamm.Scalar(0)

                    sub_matrix = secondary_matrix @ sub_matrix
                    if "additive_secondary" in locals():
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
                        additive_secondary = pybamm.Scalar(0)

                    sub_matrix = secondary_matrix @ sub_matrix
                    if "additive_secondary" in locals():
                        additive += additive_secondary

                # Handle tertiary sides (compound boundaries like corners)
                if tertiary_side is not None:
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
                            additive_tertiary = pybamm.Scalar(0)

                        sub_matrix = tertiary_matrix @ sub_matrix
                        if "additive_tertiary" in locals():
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
                            additive_tertiary = pybamm.Scalar(0)

                        sub_matrix = tertiary_matrix @ sub_matrix
                        if "additive_tertiary" in locals():
                            additive += additive_tertiary

        elif isinstance(symbol, pybamm.BoundaryGradient):
            # Handle Neumann BCs first
            if use_bcs and pybamm.has_bc_of_form(child, primary_side, bcs, "Neumann"):
                if primary_side in ["left", "right"]:
                    sub_matrix = csr_matrix((n_y * n_z, n_x * n_y * n_z))
                elif primary_side in ["front", "back"]:
                    sub_matrix = csr_matrix((n_x * n_z, n_x * n_y * n_z))
                elif primary_side in ["bottom", "top"]:
                    sub_matrix = csr_matrix((n_x * n_y, n_x * n_y * n_z))
                additive = bcs[child][primary_side][0]

            # Handle gradient extrapolation cases
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

        matrix = csr_matrix(kron(eye(repeats), sub_matrix))

        # Return boundary value with proper domain
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
                bin_op.create_copy([disc_left.x_field, disc_right.x_field])
            )
            y_field = pybamm.simplify_if_constant(
                bin_op.create_copy([disc_left.y_field, disc_right.y_field])
            )
            z_field = pybamm.simplify_if_constant(
                bin_op.create_copy([disc_left.z_field, disc_right.z_field])
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

    def edge_to_node(self, discretised_symbol, method="arithmetic"):
        """
        Convert a discretised symbol evaluated on the cell edges to a discretised symbol
        evaluated on the cell nodes.
        See :meth:`pybamm.FiniteVolume.shift`
        """
        return self.shift(discretised_symbol, "edge to node", method)

    def node_to_edge(self, discretised_symbol, method="arithmetic", direction="lr"):
        """
        Convert a discretised symbol evaluated on the cell nodes to a discretised symbol
        evaluated on the cell edges.
        See :meth:`pybamm.FiniteVolume.shift`
        """
        return self.shift(discretised_symbol, "node to edge", method, direction)

    def shift(self, discretised_symbol, shift_key, method, direction="lr"):
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

        Returns
        -------
        :class:`pybamm.Symbol`
            Averaged symbol. When evaluated, this returns either a scalar or an array of
            shape (n+1,) (if `shift_key = "node to edge"`) or (n,) (if
            `shift_key = "edge to node"`)
        """

        def arithmetic_mean(self, array, direction):
            """
            Calculate the arithmetic mean (node→edge) in 3D along 'x', 'y' or 'z'.

            Parameters
            ----------
            array : :class:`pybamm.Symbol`
                Node valued vector of length (n_x * n_y * n_z * repeats)
            direction : {"x", "y", "z"}
                Axis along which to compute face values.

            Returns
            -------
            :class:`pybamm.Symbol`
                Matrix @ array, length=(n_edges_axis * other_dims * repeats)
            """
            submesh = self.mesh[array.domain]
            n_x, n_y, n_z = submesh.npts_x, submesh.npts_y, submesh.npts_z
            repeats = self._get_auxiliary_domain_repeats(array.domains)

            def build_1d(n):
                # left boundary
                top = csr_matrix(([1.5, -0.5], ([0, 0], [0, 1])), shape=(1, n))
                # interior
                center = diags([0.5, 0.5], [0, 1], shape=(n - 1, n))
                # right boundary
                bottom = csr_matrix(
                    ([-0.5, 1.5], ([0, 0], [n - 2, n - 1])), shape=(1, n)
                )
                return vstack([top, center, bottom])

            # Pick axis and lift to 3D
            if direction == "x":
                S = build_1d(n_x)
                M = kron(eye(n_z), kron(eye(n_y), S))

            elif direction == "y":
                S = build_1d(n_y)  #
                M = kron(eye(n_z), kron(S, eye(n_x)))

            elif direction == "z":
                S = build_1d(n_z)
                M = kron(S, kron(eye(n_y), eye(n_x)))

            else:
                raise ValueError(f"Unknown direction '{direction}' for arithmetic_mean")

            M_full = csr_matrix(kron(eye(repeats), M))

            return pybamm.Matrix(M_full) @ array

        def harmonic_mean(self, array, direction):
            """
            3D node→edge harmonic mean along 'x', 'y', or 'z'.
            """
            submesh = self.mesh[array.domain]
            repeats = self._get_auxiliary_domain_repeats(array.domains)

            # Unpack mesh sizes
            n_x, n_y, n_z = submesh.npts_x, submesh.npts_y, submesh.npts_z

            def exterior_stencil(n):
                left = csr_matrix(([1.5, -0.5], ([0, 0], [0, 1])), shape=(1, n))
                right = csr_matrix(
                    ([-0.5, 1.5], ([0, 0], [n - 2, n - 1])), shape=(1, n)
                )
                return left, right

            def pick_D1_D2(n):
                D1 = hstack([eye(n - 1), csr_matrix((n - 1, 1))])
                D2 = hstack([csr_matrix((n - 1, 1)), eye(n - 1)])
                return D1, D2

            if direction == "x":
                left, right = exterior_stencil(n_x)
                interior = csr_matrix((n_x - 1, n_x))
                stencil_1d = vstack([left, interior, right])
                E = kron(eye(n_z), kron(eye(n_y), stencil_1d))

                D1_1d, D2_1d = pick_D1_D2(n_x)
                M1 = kron(eye(n_z), kron(eye(n_y), D1_1d))
                M2 = kron(eye(n_z), kron(eye(n_y), D2_1d))

                D1 = pybamm.Matrix(M1) @ array
                D2 = pybamm.Matrix(M2) @ array

                dx = submesh.d_edges_x
                beta = dx[:-2] / (dx[1:-1] + dx[:-2])

                beta_full = np.repeat(beta[:, None], n_y * n_z * repeats, axis=1)
                beta = pybamm.Array(beta_full.flatten()[:, None])

            elif direction == "y":
                left, right = exterior_stencil(n_y)
                interior = csr_matrix((n_y - 1, n_y))
                stencil_1d = vstack([left, interior, right])
                E = kron(eye(n_z), kron(stencil_1d, eye(n_x)))

                D1_1d, D2_1d = pick_D1_D2(n_y)
                M1 = kron(eye(n_z), kron(D1_1d, eye(n_x)))
                M2 = kron(eye(n_z), kron(D2_1d, eye(n_x)))

                D1 = pybamm.Matrix(M1) @ array
                D2 = pybamm.Matrix(M2) @ array

                dx = submesh.d_edges_y
                beta = dx[:-2] / (dx[1:-1] + dx[:-2])
                beta_full = np.repeat(beta[:, None], n_x * n_z * repeats, axis=1)
                beta = pybamm.Array(beta_full.flatten()[:, None])

            elif direction == "z":
                left, right = exterior_stencil(n_z)
                interior = csr_matrix((n_z - 1, n_z))
                stencil_1d = vstack([left, interior, right])
                E = kron(stencil_1d, kron(eye(n_y), eye(n_x)))

                D1_1d, D2_1d = pick_D1_D2(n_z)
                M1 = kron(D1_1d, kron(eye(n_y), eye(n_x)))
                M2 = kron(D2_1d, kron(eye(n_y), eye(n_x)))

                D1 = pybamm.Matrix(M1) @ array
                D2 = pybamm.Matrix(M2) @ array

                dx = submesh.d_edges_z
                beta = dx[:-2] / (dx[1:-1] + dx[:-2])
                beta_full = np.repeat(beta[:, None], n_x * n_y * repeats, axis=1)
                beta = pybamm.Array(beta_full.flatten()[:, None])

            else:
                raise ValueError(f"Unknown direction '{direction}'")

            # Interior harmonic mean
            D_eff = D1 * D2 / (beta * D2 + (1 - beta) * D1)

            out = pybamm.Matrix(E) @ array + D_eff

            # Expand over any auxiliary repeats
            return out

    def upwind_or_downwind(
        self, symbol, discretised_symbol, bcs, x_direction, y_direction, z_direction
    ):
        """
        3D upwind/downwind: only adds ghost nodes on the specified x, y and z faces.

        Parameters
        ----------
        symbol : :class:`pybamm.SpatialVariable`
            The variable to be discretised
        discretised_symbol : :class:`pybamm.Vector`
            The already discretised symbol (no ghost cells)
        bcs : dict
            Keys are faces ("left","right","front","back","bottom","top"),
            values are (value, "Dirichlet"|"Neumann") tuples.
        x_direction : {"upwind","downwind"}
            'upwind'→use left Dirichlet BC; 'downwind'→use right
        y_direction : {"upwind","downwind"}
            'upwind'→use front Dirichlet BC; 'downwind'→use back
        z_direction : {"upwind","downwind"}
            'upwind'→use bottom Dirichlet BC; 'downwind'→use top
        """
        # Map each logical up/downwind onto the correct geometric face:
        face_map = {
            "x": {"upwind": "left", "downwind": "right"},
            "y": {"upwind": "front", "downwind": "back"},
            "z": {"upwind": "bottom", "downwind": "top"},
        }
        try:
            x_face = face_map["x"][x_direction]
            y_face = face_map["y"][y_direction]
            z_face = face_map["z"][z_direction]
        except KeyError as err:
            raise ValueError(f"Invalid direction flag: {err}") from err

        # Check that each required face has a Dirichlet BC
        for face in (x_face, y_face, z_face):
            if face not in bcs:
                raise pybamm.ModelError(
                    f"No BC provided for {face!r}, but needed for upwind/downwind"
                )
            _, bc_type = bcs[face]
            if bc_type != "Dirichlet":
                raise pybamm.ModelError(
                    f"Must have Dirichlet BC on {face!r} for up/downwind"
                )

        # Build a minimal subset of BCs
        bc_subset = {face: bcs[face] for face in (x_face, y_face, z_face)}

        # Delegate to add_ghost_nodes (which will now insert ghost layers
        # on exactly those faces)
        upwinded, _ = self.add_ghost_nodes(symbol, discretised_symbol, bc_subset)
        return upwinded
