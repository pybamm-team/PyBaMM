#
# Finite Volume discretisation class
#
import pybamm

from scipy.sparse import (
    diags,
    spdiags,
    eye,
    kron,
    csr_matrix,
    vstack,
    hstack,
    coo_matrix,
    block_diag,
)
import numpy as np


class FiniteVolume2D(pybamm.SpatialMethod):
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
        if symbol_mesh.dimension != 2:
            raise ValueError(f"Spatial variable {symbol} is not in 2D")
        repeats = self._get_auxiliary_domain_repeats(symbol.domains)
        # Vector should be size npts_lr x npts_tb
        # Do LR first, then TB
        if symbol.evaluates_on_edges("primary"):
            LR, TB = np.meshgrid(symbol_mesh.edges_lr, symbol_mesh.edges_tb)
            lr = LR.flatten()
            tb = TB.flatten()
        else:
            LR, TB = np.meshgrid(symbol_mesh.nodes_lr, symbol_mesh.nodes_tb)
            lr = LR.flatten()
            tb = TB.flatten()
        if symbol_direction == "lr":
            entries = np.tile(lr, repeats)
        elif symbol_direction == "tb":
            entries = np.tile(tb, repeats)
        return pybamm.Vector(entries, domains=symbol.domains)

    def _gradient(self, symbol, discretised_symbol, boundary_conditions, direction):
        """
        Gradient with a specific direction (lr or tb)
        """
        domain = symbol.domain

        # Add Dirichlet boundary conditions, if defined
        if direction == "lr":
            relevant_bcs = ["left", "right"]
        elif direction == "tb":
            relevant_bcs = ["top", "bottom"]
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
        """Matrix-vector multiplication to implement the gradient operator.
        See :meth:`pybamm.SpatialMethod.gradient`
        """
        # Multiply by gradient matrix
        grad_lr = self._gradient(symbol, discretised_symbol, boundary_conditions, "lr")
        grad_tb = self._gradient(symbol, discretised_symbol, boundary_conditions, "tb")
        grad = pybamm.VectorField(grad_lr, grad_tb)
        return grad

    def gradient_matrix(self, domain, domains, direction):
        """
        Gradient matrix for finite volumes in the appropriate domain.
        Equivalent to grad(y) = (y[1:] - y[:-1])/dx

        Parameters
        ----------
        domains : list
            The domain in which to compute the gradient matrix, including ghost nodes

        Returns
        -------
        :class:`pybamm.Matrix`
            The (sparse) finite volume gradient matrix for the domain
        """
        # Create appropriate submesh by combining submeshes in primary domain
        submesh = self.mesh[domain]

        # Create matrix using submesh
        n_lr = submesh.npts_lr
        n_tb = submesh.npts_tb
        e_lr = 1 / submesh.d_nodes_lr
        e_tb = 1 / submesh.d_nodes_tb
        if direction == "lr":
            sub_matrix = diags([-e_lr, e_lr], [0, 1], shape=((n_lr - 1), n_lr))
            sub_matrix = block_diag((sub_matrix,) * n_tb)
        elif direction == "tb":
            e_tb = np.repeat(e_tb, n_lr)
            sub_matrix = diags(
                [-e_tb, e_tb], [0, n_lr], shape=(n_lr * (n_tb - 1), n_lr * n_tb)
            )

        # number of repeats
        second_dim_repeats = self._get_auxiliary_domain_repeats(domains)

        # generate full matrix from the submatrix
        # Convert to csr_matrix so that we can take the index (row-slicing), which is
        # not supported by the default kron format
        # Note that this makes column-slicing inefficient, but this should not be an
        # issue
        matrix = csr_matrix(kron(eye(second_dim_repeats), sub_matrix))
        return pybamm.Matrix(matrix)

    def divergence(self, symbol, discretised_symbol, boundary_conditions):
        """Matrix-vector multiplication to implement the divergence operator.
        See :meth:`pybamm.SpatialMethod.divergence`
        """

        divergence_matrix_lr = self.divergence_matrix(symbol.domains, "lr")
        divergence_matrix_tb = self.divergence_matrix(symbol.domains, "tb")

        grad_lr = discretised_symbol.lr_field
        grad_tb = discretised_symbol.tb_field

        div_lr = divergence_matrix_lr @ grad_lr
        div_tb = divergence_matrix_tb @ grad_tb

        out = div_lr + div_tb

        return out

    def divergence_matrix(self, domains, direction):
        """
        Divergence with a specific direction (lr or tb)
        """
        submesh = self.mesh[domains["primary"]]
        n_lr = submesh.npts_lr
        n_tb = submesh.npts_tb
        e_lr = 1 / submesh.d_edges_lr
        e_tb = 1 / submesh.d_edges_tb
        if direction == "lr":
            sub_matrix = diags([-e_lr, e_lr], [0, 1], shape=(n_lr, n_lr + 1))
            sub_matrix = block_diag((sub_matrix,) * n_tb)
        elif direction == "tb":
            e_tb = np.repeat(e_tb, n_lr + 1)
            sub_matrix = diags(
                [-e_tb, e_tb], [0, n_lr], shape=(n_lr * n_tb, n_lr * (n_tb + 1))
            )
        return pybamm.Matrix(sub_matrix)

    def laplacian(self, symbol, discretised_symbol, boundary_conditions):
        """
        Laplacian operator, implemented as div(grad(.))
        See :meth:`pybamm.SpatialMethod.laplacian`
        """
        grad = self.gradient(symbol, discretised_symbol, boundary_conditions)
        return self.divergence(grad, grad, boundary_conditions)

    def definite_integral_matrix(
        self, child, vector_type="row", integration_dimension="primary"
    ):
        raise NotImplementedError

    def indefinite_integral(self, child, discretised_child, direction):
        raise NotImplementedError

    def indefinite_integral_matrix_edges(self, domains, direction):
        raise NotImplementedError

    def indefinite_integral_matrix_nodes(self, domains, direction):
        raise NotImplementedError

    def delta_function(self, symbol, discretised_symbol):
        raise NotImplementedError

    def internal_neumann_condition(
        self, left_symbol_disc, right_symbol_disc, left_mesh, right_mesh
    ):
        """
        A method to find the internal Neumann conditions between two symbols
        on adjacent subdomains.

        Parameters
        ----------
        left_symbol_disc : :class:`pybamm.Symbol`
            The discretised symbol on the left subdomain
        right_symbol_disc : :class:`pybamm.Symbol`
            The discretised symbol on the right subdomain
        left_mesh : list
            The mesh on the left subdomain
        right_mesh : list
            The mesh on the right subdomain
        """

        left_npts = left_mesh.npts
        left_npts_lr = left_mesh.npts_lr
        right_npts = right_mesh.npts
        right_npts_lr = right_mesh.npts_lr

        second_dim_repeats = self._get_auxiliary_domain_repeats(
            left_symbol_disc.domains
        )

        if second_dim_repeats != self._get_auxiliary_domain_repeats(
            right_symbol_disc.domains
        ):
            raise pybamm.DomainError(
                """Number of secondary points in subdomains do not match"""
            )

        left_sub_matrix = np.zeros((1, left_npts))
        left_sub_matrix[0][left_npts_lr - 1 : left_npts_lr : left_npts] = 1

        left_matrix = pybamm.Matrix(
            csr_matrix(kron(eye(second_dim_repeats), left_sub_matrix))
        )

        right_sub_matrix = np.zeros((1, right_npts))
        right_sub_matrix[0][0:right_npts_lr:right_npts] = 1
        right_matrix = pybamm.Matrix(
            csr_matrix(kron(eye(second_dim_repeats), right_sub_matrix))
        )

        # Finite volume derivative
        # Remove domains to avoid clash
        right_mesh_x = right_mesh.nodes_lr[0]
        left_mesh_x = left_mesh.nodes_lr[-1]
        dx = right_mesh_x - left_mesh_x
        dy_r = (right_matrix / dx) @ right_symbol_disc
        dy_r.clear_domains()
        dy_l = (left_matrix / dx) @ left_symbol_disc
        dy_l.clear_domains()

        return dy_r - dy_l

    def add_ghost_nodes(self, symbol, discretised_symbol, bcs):
        """
        Add ghost nodes to a symbol.

        For Dirichlet bcs, for a boundary condition "y = a at the left-hand boundary",
        we concatenate a ghost node to the start of the vector y with value "2*a - y1"
        where y1 is the value of the first node.
        Similarly for the right-hand boundary condition and top and bottom boundaries.

        For Neumann bcs no ghost nodes are added. Instead, the exact value provided
        by the boundary condition is used at the cell edge when calculating the
        gradient (see :meth:`pybamm.FiniteVolume2D.add_neumann_values`).

        Parameters
        ----------
        symbol : :class:`pybamm.SpatialVariable`
            The variable to be discretised
        discretised_symbol : :class:`pybamm.Vector`
            Contains the discretised variable
        bcs : dict of tuples (:class:`pybamm.Scalar`, str)
            Dictionary (with keys "left", "right", "top", "bottom") of boundary conditions. Each
            boundary condition consists of a value and a flag indicating its type
            (e.g. "Dirichlet")

        Returns
        -------
        :class:`pybamm.Symbol`
            `Matrix @ discretised_symbol + bcs_vector`. When evaluated, this gives the
            discretised_symbol, with appropriate ghost nodes concatenated at each end.

        """
        domain = symbol.domain
        submesh = self.mesh[domain]

        # Prepare sizes and empty bcs_vector
        n_lr = submesh.npts_lr
        n_tb = submesh.npts_tb
        n = submesh.npts
        second_dim_repeats = self._get_auxiliary_domain_repeats(symbol.domains)

        # Catch if no boundary conditions are defined
        if (
            "left" not in bcs.keys()
            and "right" not in bcs.keys()
            and "top" not in bcs.keys()
            and "bottom" not in bcs.keys()
        ):
            raise ValueError(f"No boundary conditions have been provided for {symbol}")

        # Allow to only pass one boundary condition (for upwind/downwind)
        lbc_value, lbc_type = bcs.get("left", (None, None))
        rbc_value, rbc_type = bcs.get("right", (None, None))
        tbc_value, tbc_type = bcs.get("top", (None, None))
        bbc_value, bbc_type = bcs.get("bottom", (None, None))

        # Add ghost node(s) to domain where necessary and count number of
        # Dirichlet boundary conditions
        # [left, top, n, bottom, right]
        n_bcs = 0

        if tbc_type == "Dirichlet" and bbc_type != "Dirichlet":
            if isinstance(domain, list) or isinstance(domain, tuple):
                domain = [(d + "_top ghost cell", d) for d in domain]
            else:
                domain = [domain + "_top ghost cell", domain]
            n_bcs += 1
        elif tbc_type != "Dirichlet" and bbc_type == "Dirichlet":
            if isinstance(domain, list) or isinstance(domain, tuple):
                domain = [(d, d + "_bottom ghost cell") for d in domain]
            else:
                domain = [domain, domain + "_bottom ghost cell"]
            n_bcs += 1
        elif tbc_type == "Dirichlet" and bbc_type == "Dirichlet":
            if isinstance(domain, list) or isinstance(domain, tuple):
                domain = [
                    (d + "_top ghost cell", d, d + "_bottom ghost cell") for d in domain
                ]
            else:
                domain = [
                    domain + "_top ghost cell",
                    domain,
                    domain + "_bottom ghost cell",
                ]
            n_bcs += 2

        if lbc_type == "Dirichlet":
            domain = [domain[0] + "_left ghost cell", *domain]
            n_bcs += 1
        if rbc_type == "Dirichlet":
            domain = [*domain, domain[-1] + "_right ghost cell"]
            n_bcs += 1

        # Calculate values for ghost nodes for any Dirichlet boundary conditions
        if lbc_type == "Dirichlet":
            # Create matrix to extract the leftmost column of values
            lbc_sub_matrix = coo_matrix(([1], ([0], [0])), shape=(n_lr + n_bcs, 1))
            lbc_matrix = csr_matrix(kron(eye(second_dim_repeats), lbc_sub_matrix))
            lbc_matrix = vstack(
                [
                    lbc_matrix,
                ]
                * n_tb
            )
            if lbc_value.evaluates_to_number():
                left_ghost_constant = (
                    2 * lbc_value * pybamm.Vector(np.ones(second_dim_repeats))
                )
            else:
                left_ghost_constant = 2 * lbc_value

            lbc_vector = pybamm.Matrix(lbc_matrix) @ left_ghost_constant
        elif lbc_type == "Neumann":
            lbc_vector = pybamm.Vector(
                np.zeros((n_lr + n_bcs) * second_dim_repeats * n_tb)
            )
        else:
            lbc_vector = pybamm.Vector(
                np.zeros((n_tb + n_bcs) * second_dim_repeats * n_lr)
            )

        # Calculate values for ghost nodes for any Dirichlet boundary conditions
        if rbc_type == "Dirichlet":
            # Create matrix to extract the leftmost column of values
            rbc_sub_matrix = coo_matrix(
                ([1], ([n_lr + n_bcs - 1], [0])), shape=(n_lr + n_bcs, 1)
            )
            rbc_matrix = csr_matrix(kron(eye(second_dim_repeats), rbc_sub_matrix))
            rbc_matrix = vstack(
                [
                    rbc_matrix,
                ]
                * n_tb
            )
            if rbc_value.evaluates_to_number():
                right_ghost_constant = (
                    2 * rbc_value * pybamm.Vector(np.ones(second_dim_repeats))
                )
            else:
                right_ghost_constant = 2 * rbc_value
            rbc_vector = pybamm.Matrix(rbc_matrix) @ right_ghost_constant
        elif rbc_type == "Neumann":
            rbc_vector = pybamm.Vector(
                np.zeros((n_lr + n_bcs) * second_dim_repeats * n_tb)
            )
        else:
            rbc_vector = pybamm.Vector(
                np.zeros((n_tb + n_bcs) * second_dim_repeats * n_lr)
            )

        # Calculate values for ghost nodes for any Dirichlet boundary conditions
        if tbc_type == "Dirichlet":
            # Create matrix to extract the leftmost column of values
            row_indices = np.arange(0, n_lr)
            col_indices = np.zeros(len(row_indices))
            vals = np.ones(len(row_indices))
            tbc_sub_matrix = coo_matrix(
                (vals, (row_indices, col_indices)), shape=((n_tb + n_bcs) * n_lr, 1)
            )
            tbc_matrix = csr_matrix(kron(eye(second_dim_repeats), tbc_sub_matrix))

            if tbc_value.evaluates_to_number():
                top_ghost_constant = (
                    2 * tbc_value * pybamm.Vector(np.ones(second_dim_repeats))
                )
            else:
                top_ghost_constant = 2 * tbc_value
            tbc_vector = pybamm.Matrix(tbc_matrix) @ top_ghost_constant
        elif tbc_type == "Neumann":
            tbc_vector = pybamm.Vector(
                np.zeros((n_tb + n_bcs) * second_dim_repeats * n_lr)
            )
        else:
            tbc_vector = pybamm.Vector(
                np.zeros((n_lr + n_bcs) * second_dim_repeats * n_tb)
            )

        # Calculate values for ghost nodes for any Dirichlet boundary conditions
        if bbc_type == "Dirichlet":
            # Create matrix to extract the leftmost column of values
            row_indices = np.arange(
                (n_lr * (n_tb + n_bcs)) - n_lr, n_lr * (n_tb + n_bcs)
            )
            col_indices = np.zeros(len(row_indices))
            vals = np.ones(len(row_indices))
            bbc_sub_matrix = coo_matrix(
                (vals, (row_indices, col_indices)), shape=((n_tb + n_bcs) * n_lr, 1)
            )
            bbc_matrix = csr_matrix(kron(eye(second_dim_repeats), bbc_sub_matrix))

            if bbc_value.evaluates_to_number():
                bottom_ghost_constant = (
                    2 * bbc_value * pybamm.Vector(np.ones(second_dim_repeats))
                )
            else:
                bottom_ghost_constant = 2 * bbc_value
            bbc_vector = pybamm.Matrix(bbc_matrix) @ bottom_ghost_constant
        elif bbc_type == "Neumann":
            bbc_vector = pybamm.Vector(
                np.zeros((n_tb + n_bcs) * second_dim_repeats * n_lr)
            )
        else:
            bbc_vector = pybamm.Vector(
                np.zeros((n_lr + n_bcs) * second_dim_repeats * n_tb)
            )

        bcs_vector = lbc_vector + rbc_vector + tbc_vector + bbc_vector
        # Need to match the domain. E.g. in the case of the boundary condition
        # on the particle, the gradient has domain particle but the bcs_vector
        # has domain electrode, since it is a function of the macroscopic variables
        bcs_vector.copy_domains(discretised_symbol)

        # Make matrix to calculate ghost nodes
        # coo_matrix takes inputs (data, (row, col)) and puts data[i] at the point
        # (row[i], col[i]) for each index of data.
        if lbc_type == "Dirichlet":
            left_ghost_vector = coo_matrix(([-1], ([0], [0])), shape=(1, n_lr))
        else:
            left_ghost_vector = None
        if rbc_type == "Dirichlet":
            right_ghost_vector = coo_matrix(([-1], ([0], [n_lr - 1])), shape=(1, n_lr))
        else:
            right_ghost_vector = None

        if tbc_type == "Dirichlet":
            row_indices = np.arange(0, n_lr)
            col_indices = np.arange(0, n_lr)
            top_ghost_vector = coo_matrix(
                (-np.ones(n_lr), (row_indices, col_indices)), shape=(n_lr, n)
            )
        else:
            top_ghost_vector = None
        if bbc_type == "Dirichlet":
            row_indices = np.arange(0, n_lr)
            col_indices = np.arange(n - n_lr, n)
            bottom_ghost_vector = coo_matrix(
                (-np.ones(n_lr), (row_indices, col_indices)), shape=(n_lr, n)
            )
        else:
            bottom_ghost_vector = None

        if lbc_type == "Dirichlet" or rbc_type == "Dirichlet":
            sub_matrix = vstack([left_ghost_vector, eye(n_lr), right_ghost_vector])
            sub_matrix = block_diag((sub_matrix,) * n_tb)
        else:
            sub_matrix = vstack(
                [
                    left_ghost_vector,
                    top_ghost_vector,
                    eye(n),
                    bottom_ghost_vector,
                    right_ghost_vector,
                ]
            )

        # repeat matrix for secondary dimensions
        # Convert to csr_matrix so that we can take the index (row-slicing), which is
        # not supported by the default kron format
        # Note that this makes column-slicing inefficient, but this should not be an
        # issue
        matrix = csr_matrix(kron(eye(second_dim_repeats), sub_matrix))

        new_symbol = pybamm.Matrix(matrix) @ discretised_symbol + bcs_vector

        return new_symbol, domain

    def add_neumann_values(self, symbol, discretised_gradient, bcs, domain):
        """
        Add the known values of the gradient from Neumann boundary conditions to
        the discretised gradient.

        Dirichlet bcs are implemented using ghost nodes, see
        :meth:`pybamm.FiniteVolume.add_ghost_nodes`.

        Parameters
        ----------
        symbol : :class:`pybamm.SpatialVariable`
            The variable to be discretised
        discretised_gradient : :class:`pybamm.Vector`
            Contains the discretised gradient of symbol
        bcs : dict of tuples (:class:`pybamm.Scalar`, str)
            Dictionary (with keys "left" and "right") of boundary conditions. Each
            boundary condition consists of a value and a flag indicating its type
            (e.g. "Dirichlet")
        domain : list of strings
            The domain of the gradient of the symbol (may include ghost nodes)

        Returns
        -------
        :class:`pybamm.Symbol`
            `Matrix @ discretised_gradient + bcs_vector`. When evaluated, this gives the
            discretised_gradient, with the values of the Neumann boundary conditions
            concatenated at each end (if given).

        """
        # get relevant grid points
        submesh = self.mesh[domain]

        # Prepare sizes and empty bcs_vector
        n_lr = submesh.npts_lr
        n_tb = submesh.npts_tb
        second_dim_repeats = self._get_auxiliary_domain_repeats(symbol.domains)

        lbc_value, lbc_type = bcs.get("left", (None, None))
        rbc_value, rbc_type = bcs.get("right", (None, None))
        tbc_value, tbc_type = bcs.get("top", (None, None))
        bbc_value, bbc_type = bcs.get("bottom", (None, None))

        # Count number of Neumann boundary conditions
        n_bcs = 0
        if lbc_type == "Neumann":
            n_bcs += 1
        if rbc_type == "Neumann":
            n_bcs += 1
        if tbc_type == "Neumann":
            n_bcs += 1
        if bbc_type == "Neumann":
            n_bcs += 1

        # Add any values from Neumann boundary conditions to the bcs vector
        if lbc_type == "Neumann" and lbc_value != 0:
            lbc_sub_matrix = coo_matrix(([1], ([0], [0])), shape=(n_lr - 1 + n_bcs, 1))
            lbc_matrix = csr_matrix(kron(eye(second_dim_repeats), lbc_sub_matrix))
            lbc_matrix = vstack([lbc_matrix] * n_tb)
            if lbc_value.evaluates_to_number():
                left_bc = lbc_value * pybamm.Vector(np.ones(second_dim_repeats))
            else:
                left_bc = lbc_value
            lbc_vector = pybamm.Matrix(lbc_matrix) @ left_bc
        elif lbc_type == "Dirichlet" or (lbc_type == "Neumann" and lbc_value == 0):
            lbc_vector = pybamm.Vector(
                np.zeros((n_lr - 1 + n_bcs) * second_dim_repeats * n_tb)
            )
        elif lbc_type is None and rbc_type is None:
            lbc_vector = pybamm.Vector(
                np.zeros((n_tb - 1 + n_bcs) * second_dim_repeats * n_lr)
            )
        else:
            raise ValueError(
                f"boundary condition must be Dirichlet or Neumann, not '{rbc_type}'"
            )

        if rbc_type == "Neumann" and rbc_value != 0:
            rbc_sub_matrix = coo_matrix(
                ([1], ([n_lr + n_bcs - 2], [0])), shape=(n_lr - 1 + n_bcs, 1)
            )
            rbc_matrix = csr_matrix(kron(eye(second_dim_repeats), rbc_sub_matrix))
            rbc_matrix = vstack([rbc_matrix] * n_tb)
            if rbc_value.evaluates_to_number():
                right_bc = rbc_value * pybamm.Vector(np.ones(second_dim_repeats))
            else:
                right_bc = rbc_value
            rbc_vector = pybamm.Matrix(rbc_matrix) @ right_bc
        elif rbc_type == "Dirichlet" or (rbc_type == "Neumann" and rbc_value == 0):
            rbc_vector = pybamm.Vector(
                np.zeros((n_lr - 1 + n_bcs) * second_dim_repeats * n_tb)
            )
        elif rbc_type is None and lbc_type is None:
            rbc_vector = pybamm.Vector(
                np.zeros((n_tb - 1 + n_bcs) * second_dim_repeats * n_lr)
            )
        else:
            raise ValueError(
                f"boundary condition must be Dirichlet or Neumann, not '{rbc_type}'"
            )

        # Add any values from Neumann boundary conditions to the bcs vector
        if tbc_type == "Neumann" and tbc_value != 0:
            row_indices = np.arange(0, n_lr)
            col_indices = np.zeros(len(row_indices))
            vals = np.ones(len(row_indices))
            tbc_sub_matrix = coo_matrix(
                (vals, (row_indices, col_indices)), shape=((n_tb - 1 + n_bcs) * n_lr, 1)
            )
            tbc_matrix = csr_matrix(kron(eye(second_dim_repeats), tbc_sub_matrix))
            if tbc_value.evaluates_to_number():
                top_bc = tbc_value * pybamm.Vector(np.ones(second_dim_repeats))
            else:
                top_bc = tbc_value
            tbc_vector = pybamm.Matrix(tbc_matrix) @ top_bc
        elif tbc_type == "Dirichlet" or (tbc_type == "Neumann" and tbc_value == 0):
            tbc_vector = pybamm.Vector(
                np.zeros((n_tb - 1 + n_bcs) * second_dim_repeats * n_lr)
            )
        elif tbc_type is None and bbc_type is None:
            tbc_vector = pybamm.Vector(
                np.zeros((n_lr - 1 + n_bcs) * second_dim_repeats * n_tb)
            )
        else:
            raise ValueError(
                f"boundary condition must be Dirichlet or Neumann, not '{tbc_type}'"
            )

        # Add any values from Neumann boundary conditions to the bcs vector
        if bbc_type == "Neumann" and bbc_value != 0:
            row_indices = np.arange(
                (n_lr * (n_tb - 1 + n_bcs)) - n_lr, n_lr * (n_tb - 1 + n_bcs)
            )
            col_indices = np.zeros(len(row_indices))
            vals = np.ones(len(row_indices))
            bbc_sub_matrix = coo_matrix(
                (vals, (row_indices, col_indices)), shape=((n_tb - 1 + n_bcs) * n_lr, 1)
            )
            bbc_matrix = csr_matrix(kron(eye(second_dim_repeats), bbc_sub_matrix))
            if bbc_value.evaluates_to_number():
                bottom_bc = bbc_value * pybamm.Vector(np.ones(second_dim_repeats))
            else:
                bottom_bc = bbc_value
            bbc_vector = pybamm.Matrix(bbc_matrix) @ bottom_bc
        elif bbc_type == "Dirichlet" or (bbc_type == "Neumann" and bbc_value == 0):
            bbc_vector = pybamm.Vector(
                np.zeros((n_tb - 1 + n_bcs) * second_dim_repeats * n_lr)
            )
        elif bbc_type is None and tbc_type is None:
            bbc_vector = pybamm.Vector(
                np.zeros((n_lr - 1 + n_bcs) * second_dim_repeats * n_tb)
            )
        else:
            raise ValueError(
                f"boundary condition must be Dirichlet or Neumann, not '{bbc_type}'"
            )

        bcs_vector = lbc_vector + rbc_vector + tbc_vector + bbc_vector
        # Need to match the domain. E.g. in the case of the boundary condition
        # on the particle, the gradient has domain particle but the bcs_vector
        # has domain electrode, since it is a function of the macroscopic variables
        bcs_vector.copy_domains(discretised_gradient)

        # Make matrix which makes "gaps" in the the discretised gradient into
        # which the known Neumann values will be added. E.g. in 1D if the left
        # boundary condition is Dirichlet and the right Neumann, this matrix will
        # act to append a zero to the end of the discretised gradient
        if lbc_type == "Neumann":
            left_vector = csr_matrix((1, n_lr - 1))
        else:
            left_vector = None
        if rbc_type == "Neumann":
            right_vector = csr_matrix((1, n_lr - 1))
        else:
            right_vector = None
        if tbc_type == "Neumann":
            top_vector = csr_matrix((n_lr, (n_tb - 1) * n_lr))
        else:
            top_vector = None
        if bbc_type == "Neumann":
            bottom_vector = csr_matrix((n_lr, (n_tb - 1) * n_lr))
        else:
            bottom_vector = None

        if lbc_type == "Neumann" or rbc_type == "Neumann":
            sub_matrix = vstack([left_vector, eye(n_lr - 1), right_vector])
            sub_matrix = block_diag((sub_matrix,) * n_tb)
        elif tbc_type == "Neumann" or bbc_type == "Neumann":
            sub_matrix = vstack([top_vector, eye((n_tb - 1) * n_lr), bottom_vector])

        # repeat matrix for secondary dimensions
        # Convert to csr_matrix so that we can take the index (row-slicing), which is
        # not supported by the default kron format
        # Note that this makes column-slicing inefficient, but this should not be an
        # issue
        matrix = csr_matrix(kron(eye(second_dim_repeats), sub_matrix))

        new_gradient = pybamm.Matrix(matrix) @ discretised_gradient + bcs_vector

        return new_gradient

    def boundary_value_or_flux(self, symbol, discretised_child, bcs=None):
        """
        Uses extrapolation to get the boundary value or flux of a variable in the
        Finite Volume Method.

        See :meth:`pybamm.SpatialMethod.boundary_value`
        """
        # Find the number of submeshes
        submesh = self.mesh[discretised_child.domain]

        if "-" in symbol.side:
            side_first, side_second = symbol.side.split("-")
            if ("top" in side_first or "bottom" in side_first) and (
                "right" in side_second or "left" in side_second
            ):
                side_first, side_second = side_second, side_first
        else:
            side_first = symbol.side
            side_second = None

        repeats = self._get_auxiliary_domain_repeats(discretised_child.domains)

        if bcs is None:
            bcs = {}

        extrap_order_gradient = self.options["extrapolation"]["order"]["gradient"]
        extrap_order_value = self.options["extrapolation"]["order"]["value"]
        use_bcs = self.options["extrapolation"]["use bcs"]

        n_lr = submesh.npts_lr
        n_tb = submesh.npts_tb

        nodes_lr = submesh.nodes_lr
        edges_lr = submesh.edges_lr
        nodes_tb = submesh.nodes_tb
        edges_tb = submesh.edges_tb

        dx0_lr = nodes_lr[0] - edges_lr[0]
        dx1_lr = submesh.d_nodes_lr[0]
        # dx2_lr = submesh.d_nodes_lr[1]

        dxN_lr = edges_lr[-1] - nodes_lr[-1]
        dxNm1_lr = submesh.d_nodes_lr[-1]
        # dxNm2_lr = submesh.d_nodes_lr[-2]

        dx0_tb = nodes_tb[0] - edges_tb[0]
        dx1_tb = submesh.d_nodes_tb[0]
        # dx2_tb = submesh.d_nodes_tb[1]

        dxN_tb = edges_tb[-1] - nodes_tb[-1]
        dxNm1_tb = submesh.d_nodes_tb[-1]
        # dxNm2_tb = submesh.d_nodes_tb[-2]

        child = symbol.child

        # Create submatrix to compute boundary values or fluxes
        # Derivation of extrapolation formula can be found at:
        # https://github.com/Scottmar93/extrapolation-coefficents/tree/master
        if isinstance(symbol, pybamm.BoundaryValue):
            if use_bcs and pybamm.has_bc_of_form(child, side_first, bcs, "Dirichlet"):
                if side_first == "left" or side_first == "right":
                    # just use the value from the bc: f(x*)
                    sub_matrix = csr_matrix((n_tb, n_tb * n_lr))
                    additive = bcs[child][side_first][0]
                elif side_first == "top" or side_first == "bottom":
                    sub_matrix = csr_matrix((n_lr, n_lr * n_tb))
                    additive = bcs[child][side_first][0]
                else:
                    raise ValueError(
                        "side_first must be 'left', 'right', 'top', or 'bottom'"
                    )

            elif side_first == "left":
                if extrap_order_value == "linear":
                    # to find value at x* use formula:
                    # f(x*) = f_1 - (dx0 / dx1) (f_2 - f_1)

                    if use_bcs and pybamm.has_bc_of_form(
                        child, side_first, bcs, "Neumann"
                    ):
                        dx0 = dx0_lr
                        row_indices = np.arange(0, n_tb)
                        col_indices_0 = np.arange(0, n_tb * n_lr, n_lr)
                        vals_0 = np.ones(n_tb)
                        sub_matrix = csr_matrix(
                            (
                                vals_0,
                                (
                                    row_indices,
                                    col_indices_0,
                                ),
                            ),
                            shape=(n_tb, n_tb * n_lr),
                        )
                        additive = -dx0 * bcs[child][side_first][0]

                    else:
                        dx0 = dx0_lr
                        dx1 = dx1_lr
                        row_indices = np.arange(0, n_tb)
                        col_indices_0 = np.arange(0, n_tb * n_lr, n_lr)
                        col_indices_1 = col_indices_0 + 1
                        vals_0 = np.ones(n_tb) * (1 + (dx0 / dx1))
                        vals_1 = np.ones(n_tb) * (-(dx0 / dx1))
                        sub_matrix = csr_matrix(
                            (
                                np.hstack([vals_0, vals_1]),
                                (
                                    np.hstack([row_indices, row_indices]),
                                    np.hstack([col_indices_0, col_indices_1]),
                                ),
                            ),
                            shape=(n_tb, n_tb * n_lr),
                        )
                        additive = pybamm.Scalar(0)

                elif extrap_order_value == "quadratic":
                    if use_bcs and pybamm.has_bc_of_form(
                        child, symbol.side, bcs, "Neumann"
                    ):
                        raise NotImplementedError

                    else:
                        raise NotImplementedError

                else:
                    raise NotImplementedError

            elif side_first == "right":
                if extrap_order_value == "linear":
                    if use_bcs and pybamm.has_bc_of_form(
                        child, side_first, bcs, "Neumann"
                    ):
                        dxN = dxN_lr
                        row_indices = np.arange(0, n_tb)
                        col_indices_N = np.arange(n_lr - 1, n_lr * n_tb, n_lr)
                        vals_N = np.ones(n_tb)
                        sub_matrix = csr_matrix(
                            (
                                vals_0,
                                (
                                    row_indices,
                                    col_indices_N,
                                ),
                            ),
                            shape=(n_tb, n_tb * n_lr),
                        )
                        additive = dxN * bcs[child][side_first][0]

                    else:
                        # to find value at x* use formula:
                        # f(x*) = f_N - (dxN / dxNm1) (f_N - f_Nm1)
                        dxN = dxN_lr
                        dxNm1 = dxNm1_lr
                        row_indices = np.arange(0, n_tb)
                        col_indices_Nm1 = np.arange(n_lr - 2, n_lr * n_tb, n_lr)
                        col_indices_N = col_indices_Nm1 + 1
                        vals_Nm1 = np.ones(n_tb) * (-(dxN / dxNm1))
                        vals_N = np.ones(n_tb) * (1 + (dxN / dxNm1))
                        sub_matrix = csr_matrix(
                            (
                                np.hstack([vals_Nm1, vals_N]),
                                (
                                    np.hstack([row_indices, row_indices]),
                                    np.hstack([col_indices_Nm1, col_indices_N]),
                                ),
                            ),
                            shape=(n_tb, n_tb * n_lr),
                        )
                        additive = pybamm.Scalar(0)
                elif extrap_order_value == "quadratic":
                    if use_bcs and pybamm.has_bc_of_form(
                        child, symbol.side, bcs, "Neumann"
                    ):
                        raise NotImplementedError
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError

            elif side_first == "top":
                if extrap_order_value == "linear":
                    if use_bcs and pybamm.has_bc_of_form(
                        child, side_first, bcs, "Neumann"
                    ):
                        dx0 = dx0_tb
                        first_val = np.ones(n_lr)
                        sub_matrix = spdiags(
                            first_val,
                            [
                                0,
                            ],
                            n_lr,
                            n_lr * n_tb,
                        )
                        additive = -dx0 * bcs[child][side_first][0]
                    else:
                        dx0 = dx0_tb
                        dx1 = dx1_tb
                        first_val = (1 + (dx0 / dx1)) * np.ones(n_lr)
                        second_val = -(dx0 / dx1) * np.ones(n_lr)
                        sub_matrix = spdiags(
                            [first_val, second_val], [0, n_lr], n_lr, n_lr * n_tb
                        )
                        additive = pybamm.Scalar(0)

                elif extrap_order_value == "quadratic":
                    if use_bcs and pybamm.has_bc_of_form(
                        child, symbol.side, bcs, "Neumann"
                    ):
                        raise NotImplementedError
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError

            elif side_first == "bottom":
                if extrap_order_value == "linear":
                    if use_bcs and pybamm.has_bc_of_form(
                        child, side_first, bcs, "Neumann"
                    ):
                        dxNm1 = dxNm1_tb
                        dxN = dxN_tb
                        val_N = np.ones(n_lr)
                        sub_matrix = spdiags(
                            val_N,
                            [
                                (n_tb - 1) * n_lr,
                            ],
                            n_lr,
                            n_lr * n_tb,
                        )
                        additive = dxN * bcs[child][side_first][0]
                    else:
                        dx0 = dxNm1_tb
                        dx1 = dxN_tb
                        first_val = -(dx0 / dx1) * np.ones(n_lr)
                        second_val = (1 + (dx0 / dx1)) * np.ones(n_lr)
                        sub_matrix = spdiags(
                            [first_val, second_val],
                            [(n_tb - 2) * n_lr, (n_tb - 1) * n_lr],
                            n_lr,
                            n_lr * n_tb,
                        )
                        additive = pybamm.Scalar(0)

                elif extrap_order_value == "quadratic":
                    if use_bcs and pybamm.has_bc_of_form(
                        child, symbol.side, bcs, "Neumann"
                    ):
                        raise NotImplementedError
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError

            if side_second == "top":
                if use_bcs and pybamm.has_bc_of_form(
                    child, side_second, bcs, "Neumann"
                ):
                    dx0 = dx0_tb
                    additive = -dx0 * bcs[child][side_second][0]
                    sub_matrix_second = csr_matrix(
                        (
                            [
                                1,
                            ],
                            (
                                [
                                    0,
                                ],
                                [
                                    0,
                                ],
                            ),
                        ),
                        shape=(1, n_tb),
                    )
                    sub_matrix = sub_matrix_second @ sub_matrix
                else:
                    dx0 = dx0_tb
                    dx1 = dx1_tb
                    row_indices = [0, 0]
                    col_indices = [0, 1]
                    vals = [1 + (dx0 / dx1), -(dx0 / dx1)]
                    sub_matrix_second = csr_matrix(
                        (vals, (row_indices, col_indices)), shape=(1, n_tb)
                    )
                    sub_matrix = sub_matrix_second @ sub_matrix
            elif side_second == "bottom":
                if use_bcs and pybamm.has_bc_of_form(
                    child, side_second, bcs, "Neumann"
                ):
                    dxN = dxN_tb
                    additive = dxN * bcs[child][side_second][0]
                    sub_matrix_second = csr_matrix(
                        (
                            [
                                1,
                            ],
                            (
                                [
                                    0,
                                ],
                                [
                                    n_tb - 1,
                                ],
                            ),
                        ),
                        shape=(1, n_tb),
                    )
                    sub_matrix = sub_matrix_second @ sub_matrix
                else:
                    dxN = dxN_tb
                    dxNm1 = dxNm1_tb
                    row_indices = [0, 0]
                    col_indices = [n_tb - 2, n_tb - 1]
                    vals = [-(dxN / dxNm1), 1 + (dxN / dxNm1)]
                    sub_matrix_second = csr_matrix(
                        (vals, (row_indices, col_indices)), shape=(1, n_tb)
                    )
                    sub_matrix = sub_matrix_second @ sub_matrix
            elif side_second is None:
                pass
            else:
                raise ValueError("side_second must be 'top' or 'bottom'")

        elif isinstance(symbol, pybamm.BoundaryGradient):
            if use_bcs and pybamm.has_bc_of_form(child, side_first, bcs, "Neumann"):
                if side_first == "left" or side_first == "right":
                    # just use the value from the bc: f(x*)
                    sub_matrix = csr_matrix((n_tb, n_tb * n_lr))
                    additive = bcs[child][side_first][0]
                elif side_first == "top" or side_first == "bottom":
                    sub_matrix = csr_matrix((n_lr, n_lr * n_tb))
                    additive = bcs[child][side_first][0]
                else:
                    raise ValueError(
                        "side_first must be 'left', 'right', 'top', or 'bottom'"
                    )

            elif side_first == "left":
                if extrap_order_gradient == "linear":
                    # f'(x*) = (f_2 - f_1) / dx1
                    dx1 = dx1_lr
                    row_indices = np.arange(0, n_tb)
                    col_indices_0 = np.arange(0, n_tb * n_lr, n_lr)
                    col_indices_1 = col_indices_0 + 1
                    vals_0 = np.ones(n_tb) * (-1 / dx1)
                    vals_1 = np.ones(n_tb) * (1 / dx1)
                    sub_matrix = csr_matrix(
                        (
                            np.hstack([vals_0, vals_1]),
                            (
                                np.hstack([row_indices, row_indices]),
                                np.hstack([col_indices_0, col_indices_1]),
                            ),
                        ),
                        shape=(n_tb, n_tb * n_lr),
                    )
                    additive = pybamm.Scalar(0)

                elif extrap_order_gradient == "quadratic":
                    raise NotImplementedError

                else:
                    raise NotImplementedError

            elif side_first == "right":
                if extrap_order_gradient == "linear":
                    # use formula:
                    # f'(x*) = (f_N - f_Nm1) / dxNm1
                    dxN = dxN_lr
                    dxNm1 = dxNm1_lr
                    row_indices = np.arange(0, n_tb)
                    col_indices_Nm1 = np.arange(n_lr - 2, n_lr * n_tb, n_lr)
                    col_indices_N = col_indices_Nm1 + 1
                    vals_Nm1 = np.ones(n_tb) * (-1 / dxNm1)
                    vals_N = np.ones(n_tb) * (1 / dxNm1)
                    sub_matrix = csr_matrix(
                        (
                            np.hstack([vals_Nm1, vals_N]),
                            (
                                np.hstack([row_indices, row_indices]),
                                np.hstack([col_indices_Nm1, col_indices_N]),
                            ),
                        ),
                        shape=(n_tb, n_tb * n_lr),
                    )
                    additive = pybamm.Scalar(0)

                elif extrap_order_gradient == "quadratic":
                    raise NotImplementedError
                else:
                    raise NotImplementedError

            elif side_first == "top":
                dx1 = dx1_tb
                first_val = (-1 / dx1) * np.ones(n_lr)
                second_val = (1 / dx1) * np.ones(n_lr)
                sub_matrix = spdiags(
                    [first_val, second_val], [0, n_lr], n_lr, n_lr * n_tb
                )
                additive = pybamm.Scalar(0)

            elif side_first == "bottom":
                dx0 = dxNm1_tb
                dx1 = dxN_tb
                first_val = -(dx0 / dx1) * np.ones(n_lr)
                second_val = (1 + (dx0 / dx1)) * np.ones(n_lr)
                sub_matrix = spdiags(
                    [first_val, second_val],
                    [(n_tb - 2) * n_lr, (n_tb - 1) * n_lr],
                    n_lr,
                    n_lr * n_tb,
                )
                additive = pybamm.Scalar(0)

            if side_second == "top":
                dx0 = dx0_tb
                dx1 = dx1_tb
                row_indices = [0, 0]
                col_indices = [0, 1]
                vals = [-1 / dx1, 1 / dx1]
                sub_matrix_second = csr_matrix(
                    (vals, (row_indices, col_indices)), shape=(1, n_tb)
                )
                sub_matrix = sub_matrix_second @ sub_matrix
            elif side_second == "bottom":
                dxN = dxN_tb
                dxNm1 = dxNm1_tb
                row_indices = [0, 0]
                col_indices = [n_tb - 2, n_tb - 1]
                vals = [-1 / dxNm1, 1 / dxNm1]
                sub_matrix_second = csr_matrix(
                    (vals, (row_indices, col_indices)), shape=(1, n_tb)
                )
                sub_matrix = sub_matrix_second @ sub_matrix
            elif side_second is None:
                pass
            else:
                raise ValueError("side_second must be 'top' or 'bottom'")
        # Generate full matrix from the submatrix
        # Convert to csr_matrix so that we can take the index (row-slicing), which is
        # not supported by the default kron format
        # Note that this makes column-slicing inefficient, but this should not be an
        # issue
        matrix = csr_matrix(kron(eye(repeats), sub_matrix))

        # Return boundary value with domain given by symbol
        matrix = pybamm.Matrix(matrix)
        boundary_value = matrix @ discretised_child
        boundary_value.copy_domains(symbol)

        additive.copy_domains(symbol)
        boundary_value += additive

        return boundary_value

    def evaluate_at(self, symbol, discretised_child, position):
        """
        Returns the symbol evaluated at a given position in space.

        Parameters
        ----------
        symbol: :class:`pybamm.Symbol`
            The boundary value or flux symbol
        discretised_child : :class:`pybamm.StateVector`
            The discretised variable from which to calculate the boundary value
        position : :class:`pybamm.Scalar`
            The point in one-dimensional space at which to evaluate the symbol.

        Returns
        -------
        :class:`pybamm.MatrixMultiplication`
            The variable representing the value at the given point.
        """
        raise NotImplementedError

    def process_binary_operators(self, bin_op, left, right, disc_left, disc_right):
        """Discretise binary operators in model equations.  Performs appropriate
        averaging of diffusivities if one of the children is a gradient operator, so
        that discretised sizes match up. For this averaging we use the harmonic
        mean [1].

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
        """Discretise binary operators in model equations.  Performs appropriate
        averaging of diffusivities if one of the children is a gradient operator, so
        that discretised sizes match up. For this averaging we use the harmonic
        mean [1].

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

        # This could be cleaned up a bit, but it works for now.
        if hasattr(disc_left, "lr_field") and hasattr(disc_right, "lr_field"):
            if right_evaluates_on_edges and not left_evaluates_on_edges:
                if isinstance(right, pybamm.Gradient):
                    method = "harmonic"
                    disc_left_lr = self.node_to_edge(
                        disc_left.lr_field, method=method, direction="lr"
                    )
                    disc_left_tb = self.node_to_edge(
                        disc_left.tb_field, method=method, direction="tb"
                    )
                    disc_left = pybamm.VectorField(disc_left_lr, disc_left_tb)
                else:
                    method = "arithmetic"
                    disc_left_lr = self.node_to_edge(
                        disc_left.lr_field, method=method, direction="lr"
                    )
                    disc_left_tb = self.node_to_edge(
                        disc_left.tb_field, method=method, direction="tb"
                    )
                    disc_left = pybamm.VectorField(disc_left_lr, disc_left_tb)
            elif left_evaluates_on_edges and not right_evaluates_on_edges:
                if isinstance(left, pybamm.Gradient):
                    method = "harmonic"
                    disc_right_lr = self.node_to_edge(
                        disc_right.lr_field, method=method, direction="lr"
                    )
                    disc_right_tb = self.node_to_edge(
                        disc_right.tb_field, method=method, direction="tb"
                    )
                    disc_right = pybamm.VectorField(disc_right_lr, disc_right_tb)
                else:
                    method = "arithmetic"
            # both are vector fields, so we need make a new vector field.
            lr_field = pybamm.simplify_if_constant(
                bin_op.create_copy([disc_left.lr_field, disc_right.lr_field])
            )
            tb_field = pybamm.simplify_if_constant(
                bin_op.create_copy([disc_left.tb_field, disc_right.tb_field])
            )
            return pybamm.VectorField(lr_field, tb_field)
        elif hasattr(disc_left, "lr_field") and not hasattr(disc_right, "lr_field"):
            # one is a vector field, so we need to make a new vector field.
            if left_evaluates_on_edges and not right_evaluates_on_edges:
                if isinstance(left, pybamm.Gradient):
                    method = "harmonic"
                    disc_right_lr = self.node_to_edge(
                        disc_right, method=method, direction="lr"
                    )
                    disc_right_tb = self.node_to_edge(
                        disc_right, method=method, direction="tb"
                    )
                    disc_right = pybamm.VectorField(disc_right_lr, disc_right_tb)
                else:
                    method = "arithmetic"
            lr_field = pybamm.simplify_if_constant(
                bin_op.create_copy([disc_left.lr_field, disc_right])
            )
            tb_field = pybamm.simplify_if_constant(
                bin_op.create_copy([disc_left.tb_field, disc_right])
            )
            return pybamm.VectorField(lr_field, tb_field)
        elif not hasattr(disc_left, "lr_field") and hasattr(disc_right, "lr_field"):
            # one is a vector field, so we need to make a new vector field.
            if right_evaluates_on_edges and not left_evaluates_on_edges:
                if isinstance(right, pybamm.Gradient):
                    method = "harmonic"
                    disc_left_lr = self.node_to_edge(
                        disc_left, method=method, direction="lr"
                    )
                    disc_left_tb = self.node_to_edge(
                        disc_left, method=method, direction="tb"
                    )
                    disc_left = pybamm.VectorField(disc_left_lr, disc_left_tb)
                else:
                    method = "arithmetic"
            lr_field = pybamm.simplify_if_constant(
                bin_op.create_copy([disc_left, disc_right.lr_field])
            )
            tb_field = pybamm.simplify_if_constant(
                bin_op.create_copy([disc_left, disc_right.tb_field])
            )
            return pybamm.VectorField(lr_field, tb_field)
        else:
            pass
        # inner product takes fluxes from edges to nodes
        if isinstance(bin_op, pybamm.Inner):
            if left_evaluates_on_edges:
                disc_left = self.edge_to_node(disc_left)
            if right_evaluates_on_edges:
                disc_right = self.edge_to_node(disc_right)

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
                disc_right_lr = self.node_to_edge(
                    disc_right, method=method, direction="lr"
                )
                disc_right_tb = self.node_to_edge(
                    disc_right, method=method, direction="tb"
                )
                disc_right = pybamm.VectorField(disc_right_lr, disc_right_tb)
            else:
                method = "arithmetic"

        # If only right child evaluates on edges, map left child onto edges
        # using the harmonic mean if the right child is a gradient (i.e. this
        # binary operator represents a flux)
        elif right_evaluates_on_edges and not left_evaluates_on_edges:
            if isinstance(right, pybamm.Gradient):
                method = "harmonic"
                disc_left_lr = self.node_to_edge(
                    disc_left, method=method, direction="lr"
                )
                disc_left_tb = self.node_to_edge(
                    disc_left, method=method, direction="tb"
                )
                disc_left = pybamm.VectorField(disc_left_lr, disc_left_tb)
            else:
                method = "arithmetic"

        # Return new binary operator with appropriate class
        out = pybamm.simplify_if_constant(bin_op.create_copy([disc_left, disc_right]))

        return out

    def concatenation(self, disc_children):
        """Discrete concatenation, taking `edge_to_node` for children that evaluate on
        edges.
        See :meth:`pybamm.SpatialMethod.concatenation`
        """
        for idx, child in enumerate(disc_children):
            submesh = self.mesh[child.domain]
            repeats = self._get_auxiliary_domain_repeats(child.domains)
            n_nodes = len(submesh.nodes_lr) * len(submesh.nodes_tb) * repeats
            n_edges = len(submesh.edges_lr) * len(submesh.edges_tb) * repeats
            child_size = child.size
            if child_size != n_nodes:
                # Average any children that evaluate on the edges (size n_edges) to
                # evaluate on nodes instead, so that concatenation works properly
                if child_size == n_edges:
                    disc_children[idx] = self.edge_to_node(child)
                else:
                    raise pybamm.ShapeError(
                        "child must have size n_nodes (number of nodes in the mesh) "
                        "or n_edges (number of edges in the mesh)"
                    )
        # EXPERIMENTAL: Need to reorder things for 2D
        if not all(isinstance(child, pybamm.StateVector) for child in disc_children):
            # All will have the same number of points in the tb direction, so we just need to get the lr points
            lr_mesh_points = [
                self.mesh[child.domain[0]].npts_lr for child in disc_children
            ]
            tb_mesh_points = self.mesh[disc_children[0].domain[0]].npts_tb
            num_children = len(disc_children)
            eyes = [np.eye(lr_mesh_points[i]) for i in range(num_children)]
            num_blocks = num_children * tb_mesh_points
            block_mat = [
                [
                    np.zeros((lr_mesh_points_this, lr_mesh_points_this))
                    for lr_mesh_points_this in lr_mesh_points
                    for _ in range(tb_mesh_points)
                ]
                for _ in range(num_blocks)
            ]
            i = 0
            for tb_idx in range(tb_mesh_points):
                for child_idx in range(num_children):
                    if isinstance(disc_children[child_idx], pybamm.StateVector):
                        block_mat[i][tb_idx + child_idx * tb_mesh_points] = eyes[
                            child_idx
                        ]
                        i += 1
                    else:
                        block_mat[i][tb_idx + child_idx * tb_mesh_points] = eyes[
                            child_idx
                        ]
                        i += 1
            block_mat = csr_matrix(np.block(block_mat))
            block_mat = kron(eye(repeats), block_mat)
            matrix = pybamm.Matrix(block_mat)
            repeats = self._get_auxiliary_domain_repeats(disc_children[0].domains)
            return matrix @ pybamm.domain_concatenation(disc_children, self.mesh)
        else:
            print("hi")
            return pybamm.domain_concatenation(disc_children, self.mesh)

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

        def arithmetic_mean(array, direction):
            """Calculate the arithmetic mean of an array using matrix multiplication"""
            # Create appropriate submesh by combining submeshes in domain
            submesh = self.mesh[array.domain]

            # Create 1D matrix using submesh
            n_lr = submesh.npts_lr
            n_tb = submesh.npts_tb

            if shift_key == "node to edge":
                if direction == "lr":
                    sub_matrix_left = csr_matrix(
                        ([1.5, -0.5], ([0, 0], [0, 1])), shape=(1, n_lr)
                    )
                    sub_matrix_center = diags(
                        [0.5, 0.5], [0, 1], shape=(n_lr - 1, n_lr)
                    )
                    sub_matrix_right = csr_matrix(
                        ([-0.5, 1.5], ([0, 0], [n_lr - 2, n_lr - 1])), shape=(1, n_lr)
                    )
                    sub_matrix = vstack(
                        [sub_matrix_left, sub_matrix_center, sub_matrix_right]
                    )
                    sub_matrix = block_diag((sub_matrix,) * n_tb)
                elif direction == "tb":
                    # Matrix to compute values at the exterior edges
                    one_fives_top = np.ones(n_lr) * 1.5
                    neg_zero_fives_top = np.ones(n_lr) * -0.5
                    rows = np.arange(0, n_lr)
                    cols_first = np.arange(0, n_lr)
                    cols_second = np.arange(n_lr, 2 * n_lr)
                    data = np.hstack([one_fives_top, neg_zero_fives_top])
                    cols = np.hstack([cols_first, cols_second])
                    rows = np.hstack([rows, rows])
                    sub_matrix_top = csr_matrix(
                        (data, (rows, cols)), shape=(n_lr, n_lr * n_tb)
                    )
                    cols_first = np.arange((n_tb - 2) * (n_lr), (n_tb - 1) * (n_lr))
                    cols_second = np.arange((n_tb - 1) * (n_lr), (n_tb) * (n_lr))
                    data = np.hstack([neg_zero_fives_top, one_fives_top])
                    cols = np.hstack([cols_first, cols_second])
                    sub_matrix_bottom = csr_matrix(
                        (data, (rows, cols)), shape=(n_lr, n_lr * n_tb)
                    )
                    data = np.ones((n_tb - 1) * n_lr) * 0.5
                    data = np.hstack([data, data])
                    rows = np.arange(0, (n_tb - 1) * n_lr)
                    rows = np.hstack([rows, rows])
                    cols_first = np.arange(0, (n_tb - 1) * n_lr)
                    cols_second = np.arange(n_lr, n_lr * n_tb)
                    cols = np.hstack([cols_first, cols_second])
                    sub_matrix_center = csr_matrix(
                        (data, (rows, cols)), shape=(n_lr * (n_tb - 1), n_lr * n_tb)
                    )
                    sub_matrix = vstack(
                        [
                            sub_matrix_top,
                            sub_matrix_center,
                            sub_matrix_bottom,
                        ]
                    )

            elif shift_key == "edge to node":
                raise NotImplementedError
            else:
                raise ValueError(f"shift key '{shift_key}' not recognised")
            # Second dimension length
            second_dim_repeats = self._get_auxiliary_domain_repeats(
                discretised_symbol.domains
            )

            # Generate full matrix from the submatrix
            # Convert to csr_matrix so that we can take the index (row-slicing), which
            # is not supported by the default kron format
            # Note that this makes column-slicing inefficient, but this should not be an
            # issue
            matrix = csr_matrix(kron(eye(second_dim_repeats), sub_matrix))

            return pybamm.Matrix(matrix) @ array

        def harmonic_mean(array, direction):
            """
            Calculate the harmonic mean of an array using matrix multiplication.
            The harmonic mean is computed as

            .. math::
                D_{eff} = \\frac{D_1  D_2}{\\beta D_2 + (1 - \\beta) D_1},

            where

            .. math::
                \\beta = \\frac{\\Delta x_1}{\\Delta x_2 + \\Delta x_1}

            accounts for the difference in the control volume widths. This is the
            definiton from [1], which is the same as that in [2] but with slightly
            different notation.

            [1] Torchio, M et al. "LIONSIMBA: A Matlab Framework Based on a Finite
            Volume Model Suitable for Li-Ion Battery Design, Simulation, and Control."
            (2016).
            [2] Recktenwald, Gerald. "The control-volume finite-difference
            approximation to the diffusion equation." (2012).
            """
            # Create appropriate submesh by combining submeshes in domain
            submesh = self.mesh[array.domain]

            # Get second dimension length for use later
            second_dim_repeats = self._get_auxiliary_domain_repeats(
                discretised_symbol.domains
            )

            # Create 1D matrix using submesh
            n = submesh.npts
            n_lr = submesh.npts_lr
            n_tb = submesh.npts_tb

            if shift_key == "node to edge":
                if direction == "lr":
                    # Matrix to compute values at the exterior edges
                    edges_sub_matrix_left = csr_matrix(
                        ([1.5, -0.5], ([0, 0], [0, 1])), shape=(1, n_lr)
                    )
                    edges_sub_matrix_center = csr_matrix((n_lr - 1, n_lr))
                    edges_sub_matrix_right = csr_matrix(
                        ([-0.5, 1.5], ([0, 0], [n_lr - 2, n_lr - 1])), shape=(1, n_lr)
                    )
                    edges_sub_matrix = vstack(
                        [
                            edges_sub_matrix_left,
                            edges_sub_matrix_center,
                            edges_sub_matrix_right,
                        ]
                    )
                    edges_sub_matrix = block_diag((edges_sub_matrix,) * n_tb)

                    # Generate full matrix from the submatrix
                    # Convert to csr_matrix so that we can take the index (row-slicing),
                    # which is not supported by the default kron format
                    # Note that this makes column-slicing inefficient, but this should
                    # not be an issue
                    edges_matrix = csr_matrix(
                        kron(eye(second_dim_repeats), edges_sub_matrix)
                    )

                    # Matrix to extract the node values running from the first node
                    # to the penultimate node in the primary dimension (D_1 in the
                    # definiton of the harmonic mean)
                    sub_matrix_D1 = hstack([eye(n_lr - 1), csr_matrix((n_lr - 1, 1))])
                    sub_matrix_D1 = block_diag((sub_matrix_D1,) * n_tb)
                    matrix_D1 = csr_matrix(kron(eye(second_dim_repeats), sub_matrix_D1))
                    D1 = pybamm.Matrix(matrix_D1) @ array

                    # Matrix to extract the node values running from the second node
                    # to the final node in the primary dimension  (D_2 in the
                    # definiton of the harmonic mean)
                    sub_matrix_D2 = hstack([csr_matrix((n_lr - 1, 1)), eye(n_lr - 1)])
                    sub_matrix_D2 = block_diag((sub_matrix_D2,) * n_tb)
                    matrix_D2 = csr_matrix(kron(eye(second_dim_repeats), sub_matrix_D2))
                    D2 = pybamm.Matrix(matrix_D2) @ array

                    # Compute weight beta
                    dx = submesh.d_edges_lr
                    sub_beta = (dx[:-1] / (dx[1:] + dx[:-1]))[:, np.newaxis]
                    sub_beta = np.repeat(sub_beta, n_tb, axis=0)
                    beta = pybamm.Array(
                        np.kron(np.ones((second_dim_repeats, 1)), sub_beta)
                    )

                    # dx_real = dx * length, therefore, beta is unchanged
                    # Compute harmonic mean on internal edges
                    D_eff = D1 * D2 / (D2 * beta + D1 * (1 - beta))

                    # Matrix to pad zeros at the beginning and end of the array where
                    # the exterior edge values will be added
                    sub_matrix = vstack(
                        [
                            csr_matrix((1, n_lr - 1)),
                            eye(n_lr - 1),
                            csr_matrix((1, n_lr - 1)),
                        ]
                    )
                    sub_matrix = block_diag((sub_matrix,) * n_tb)

                    # Generate full matrix from the submatrix
                    # Convert to csr_matrix so that we can take the index (row-slicing),
                    # which is not supported by the default kron format
                    # Note that this makes column-slicing inefficient, but this should
                    # not be an issue
                    matrix = csr_matrix(kron(eye(second_dim_repeats), sub_matrix))

                    return (
                        pybamm.Matrix(edges_matrix) @ array
                        + pybamm.Matrix(matrix) @ D_eff
                    )
                elif direction == "tb":
                    # Matrix to compute values at the exterior edges
                    # Matrix to compute values at the exterior edges
                    one_fives_top = np.ones(n_lr) * 1.5
                    neg_zero_fives_top = np.ones(n_lr) * -0.5
                    rows = np.arange(0, n_lr)
                    cols_first = np.arange(0, n_lr)
                    cols_second = np.arange(n_lr, 2 * n_lr)
                    data = np.hstack([one_fives_top, neg_zero_fives_top])
                    cols = np.hstack([cols_first, cols_second])
                    rows = np.hstack([rows, rows])
                    edges_sub_matrix_top = csr_matrix(
                        (data, (rows, cols)), shape=(n_lr, n_lr * n_tb)
                    )
                    cols_first = np.arange((n_tb - 2) * (n_lr), (n_tb - 1) * (n_lr))
                    cols_second = np.arange((n_tb - 1) * (n_lr), (n_tb) * (n_lr))
                    data = np.hstack([neg_zero_fives_top, one_fives_top])
                    cols = np.hstack([cols_first, cols_second])
                    edges_sub_matrix_bottom = csr_matrix(
                        (data, (rows, cols)), shape=(n_lr, n_lr * n_tb)
                    )
                    edges_sub_matrix_center = csr_matrix(
                        ((n_tb - 1) * n_lr, n_tb * n_lr)
                    )
                    edges_sub_matrix = vstack(
                        [
                            edges_sub_matrix_top,
                            edges_sub_matrix_center,
                            edges_sub_matrix_bottom,
                        ]
                    )

                    # Generate full matrix from the submatrix
                    # Convert to csr_matrix so that we can take the index (row-slicing),
                    # which is not supported by the default kron format
                    # Note that this makes column-slicing inefficient, but this should
                    # not be an issue
                    edges_matrix = csr_matrix(
                        kron(eye(second_dim_repeats), edges_sub_matrix)
                    )

                    # Matrix to extract the node values running from the first node
                    # to the penultimate node in the primary dimension (D_1 in the
                    # definiton of the harmonic mean)
                    sub_matrix_D1 = hstack(
                        [eye(n_lr * (n_tb - 1)), csr_matrix((n_lr * (n_tb - 1), n_lr))]
                    )
                    matrix_D1 = csr_matrix(kron(eye(second_dim_repeats), sub_matrix_D1))
                    D1 = pybamm.Matrix(matrix_D1) @ array

                    # Matrix to extract the node values running from the second node
                    # to the final node in the primary dimension  (D_2 in the
                    # definiton of the harmonic mean)
                    sub_matrix_D2 = hstack(
                        [csr_matrix((n_lr * (n_tb - 1), n_lr)), eye(n_lr * (n_tb - 1))]
                    )
                    matrix_D2 = csr_matrix(kron(eye(second_dim_repeats), sub_matrix_D2))
                    D2 = pybamm.Matrix(matrix_D2) @ array

                    # Compute weight beta
                    dx = submesh.d_edges_tb
                    sub_beta = (dx[:-1] / (dx[1:] + dx[:-1]))[:, np.newaxis]
                    sub_beta = np.repeat(sub_beta, n_lr, axis=0)
                    beta = pybamm.Array(
                        np.kron(np.ones((second_dim_repeats, 1)), sub_beta)
                    )

                    # dx_real = dx * length, therefore, beta is unchanged
                    # Compute harmonic mean on internal edges
                    # Note: add small number to denominator to regularise D_eff
                    D_eff = D1 * D2 / (D2 * beta + D1 * (1 - beta))

                    # Matrix to pad zeros at the beginning and end of the array where
                    # the exterior edge values will be added
                    sub_matrix = vstack(
                        [
                            csr_matrix((n_lr, n_lr * (n_tb - 1))),
                            eye(n_lr * (n_tb - 1)),
                            csr_matrix((n_lr, n_lr * (n_tb - 1))),
                        ]
                    )

                    # Generate full matrix from the submatrix
                    # Convert to csr_matrix so that we can take the index (row-slicing),
                    # which is not supported by the default kron format
                    # Note that this makes column-slicing inefficient, but this should
                    # not be an issue
                    matrix = csr_matrix(kron(eye(second_dim_repeats), sub_matrix))

                    return (
                        pybamm.Matrix(edges_matrix) @ array
                        + pybamm.Matrix(matrix) @ D_eff
                    )

            elif shift_key == "edge to node":
                # Matrix to extract the edge values running from the first edge
                # to the penultimate edge in the primary dimension (D_1 in the
                # definiton of the harmonic mean)
                raise NotImplementedError
                sub_matrix_D1 = hstack([eye(n), csr_matrix((n, 1))])
                matrix_D1 = csr_matrix(kron(eye(second_dim_repeats), sub_matrix_D1))
                D1 = pybamm.Matrix(matrix_D1) @ array

                # Matrix to extract the edge values running from the second edge
                # to the final edge in the primary dimension  (D_2 in the
                # definiton of the harmonic mean)
                sub_matrix_D2 = hstack([csr_matrix((n, 1)), eye(n)])
                matrix_D2 = csr_matrix(kron(eye(second_dim_repeats), sub_matrix_D2))
                D2 = pybamm.Matrix(matrix_D2) @ array

                # Compute weight beta
                dx0 = submesh.nodes[0] - submesh.edges[0]  # first edge to node
                dxN = submesh.edges[-1] - submesh.nodes[-1]  # last node to edge
                dx = np.concatenate(([dx0], submesh.d_nodes, [dxN]))
                sub_beta = (dx[:-1] / (dx[1:] + dx[:-1]))[:, np.newaxis]
                beta = pybamm.Array(np.kron(np.ones((second_dim_repeats, 1)), sub_beta))

                # Compute harmonic mean on nodes
                # Note: add small number to denominator to regularise D_eff
                D_eff = D1 * D2 / (D2 * beta + D1 * (1 - beta) + 1e-16)

                return D_eff

            else:
                raise ValueError(f"shift key '{shift_key}' not recognised")

        # If discretised_symbol evaluates to number there is no need to average
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
        self, symbol, discretised_symbol, bcs, lr_direction, tb_direction
    ):
        """
        Implement an upwinding operator. Currently, this requires the symbol to have
        a Dirichlet boundary condition on the left side or top side (for upwinding) or right side
        or bottom side (for downwinding).

        Parameters
        ----------
        symbol : :class:`pybamm.SpatialVariable`
            The variable to be discretised
        discretised_gradient : :class:`pybamm.Vector`
            Contains the discretised gradient of symbol
        bcs : dict of tuples (:class:`pybamm.Scalar`, str)
            Dictionary (with keys "left" and "right") of boundary conditions. Each
            boundary condition consists of a value and a flag indicating its type
            (e.g. "Dirichlet")
        lr_direction : str
            Direction in which to apply the operator (upwind or downwind) in the lr direction
        tb_direction : str
            Direction in which to apply the operator (upwind or downwind) in the tb direction
        """
        if symbol not in bcs:
            raise pybamm.ModelError(
                f"Boundary conditions must be provided for {lr_direction}ing '{symbol}' and {tb_direction}ing '{symbol}'"
            )

        if lr_direction == "upwind":
            bc_side = "left"
        elif lr_direction == "downwind":
            bc_side = "right"
        else:
            raise ValueError(f"direction '{lr_direction}' not recognised")

        if tb_direction == "upwind":
            bc_side = "top"
        elif tb_direction == "downwind":
            bc_side = "bottom"
        else:
            raise ValueError(f"direction '{tb_direction}' not recognised")

        if bcs[symbol][bc_side][1] != "Dirichlet":
            raise pybamm.ModelError(
                "Dirichlet boundary conditions must be provided for "
                f"{lr_direction}ing '{symbol}' and {tb_direction}ing '{symbol}'"
            )

        # Extract only the relevant boundary condition as the model might have both
        bc_subset = {bc_side: bcs[symbol][bc_side]}
        symbol_out, _ = self.add_ghost_nodes(symbol, discretised_symbol, bc_subset)
        return symbol_out
