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
        grad = pybamm.Concatenation(
            grad_lr, grad_tb, check_domain=False, concat_fun=np.vstack
        )
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

        grad_lr, grad_tb = discretised_symbol.orphans
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
        raise NotImplementedError

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
        raise NotImplementedError

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
        base_domain = domain[0]
        if tbc_type == "Dirichlet":
            domain = [base_domain + "_top ghost cell", *domain]
            n_bcs += 1
        if lbc_type == "Dirichlet":
            domain = [base_domain + "_left ghost cell", *domain]
            n_bcs += 1
        if bbc_type == "Dirichlet":
            domain = [*domain, base_domain + "_bottom ghost cell"]
            n_bcs += 1
        if rbc_type == "Dirichlet":
            domain = [*domain, base_domain + "_right ghost cell"]
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
        raise NotImplementedError

    def boundary_value_or_flux(self, symbol, discretised_child, bcs=None):
        """
        Uses extrapolation to get the boundary value or flux of a variable in the
        Finite Volume Method.

        See :meth:`pybamm.SpatialMethod.boundary_value`
        """
        raise NotImplementedError

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
        """Discrete concatenation, taking `edge_to_node` for children that evaluate on
        edges.
        See :meth:`pybamm.SpatialMethod.concatenation`
        """
        raise NotImplementedError

    def edge_to_node(self, discretised_symbol, method="arithmetic"):
        """
        Convert a discretised symbol evaluated on the cell edges to a discretised symbol
        evaluated on the cell nodes.
        See :meth:`pybamm.FiniteVolume.shift`
        """
        raise NotImplementedError

    def node_to_edge(self, discretised_symbol, method="arithmetic"):
        """
        Convert a discretised symbol evaluated on the cell nodes to a discretised symbol
        evaluated on the cell edges.
        See :meth:`pybamm.FiniteVolume.shift`
        """
        raise NotImplementedError

    def shift(self, discretised_symbol, shift_key, method):
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
        raise NotImplementedError

    def upwind_or_downwind(self, symbol, discretised_symbol, bcs, direction):
        """
        Implement an upwinding operator. Currently, this requires the symbol to have
        a Dirichlet boundary condition on the left side (for upwinding) or right side
        (for downwinding).

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
        direction : str
            Direction in which to apply the operator (upwind or downwind)
        """
        raise NotImplementedError
