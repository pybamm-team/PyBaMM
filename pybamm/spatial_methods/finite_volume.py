#
# Finite Volume discretisation class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

from scipy.sparse import diags, eye, kron, csr_matrix, vstack
import autograd.numpy as np
from autograd.builtins import isinstance


class FiniteVolume(pybamm.SpatialMethod):
    """
    A class which implements the steps specific to the finite volume method during
    discretisation.

    For broadcast and mass_matrix, we follow the default behaviour from SpatialMethod.

    Parameters
    ----------
    mesh : :class:`pybamm.Mesh`
        Contains all the submeshes for discretisation

    **Extends:"": :class:`pybamm.SpatialMethod`
    """

    def __init__(self, mesh):
        super().__init__(mesh)
        # add npts_for_broadcast to mesh domains for this particular discretisation
        for dom in mesh.keys():
            for i in range(len(mesh[dom])):
                mesh[dom][i].npts_for_broadcast = mesh[dom][i].npts

    def spatial_variable(self, symbol):
        """
        Creates a discretised spatial variable compatible with
        the FiniteVolume method.

        Parameters
        -----------
        symbol : :class:`pybamm.SpatialVariable`
            The spatial variable to be discretised.

        Returns
        -------
        :class:`pybamm.Vector`
            Contains the discretised spatial variable
        """
        # for finite volume we use the cell centres
        if symbol.name in ["x_n", "x_s", "x_p", "r_n", "r_p", "x", "r"]:
            symbol_mesh = self.mesh.combine_submeshes(*symbol.domain)
            return pybamm.Vector(symbol_mesh[0].nodes, domain=symbol.domain)
        else:
            raise NotImplementedError("3D meshes not yet implemented")

    def gradient(self, symbol, discretised_symbol, boundary_conditions):
        """Matrix-vector multiplication to implement the gradient operator.
        See :meth:`pybamm.SpatialMethod.gradient`
        """
        # Check that boundary condition keys are hashes (ids)
        for key in boundary_conditions.keys():
            assert isinstance(key, int), TypeError(
                "boundary condition keys should be hashes, not {}".format(type(key))
            )
        # Discretise symbol
        domain = symbol.domain
        # Add Dirichlet boundary conditions, if defined
        if symbol.id in boundary_conditions:
            bcs = boundary_conditions[symbol.id]
            # get boundary conditions and edit domain
            if "left" in bcs.keys():
                lbc = bcs["left"]
                domain = [domain[0] + "_left ghost cell"] + domain
            else:
                lbc = None
            if "right" in bcs.keys():
                rbc = bcs["right"]
                domain = domain + [domain[-1] + "_right ghost cell"]
            else:
                rbc = None
            # add ghost nodes
            discretised_symbol = self.add_ghost_nodes(
                symbol, discretised_symbol, lbc, rbc
            )

        # note in 1D spherical grad and normal grad are the same
        gradient_matrix = self.gradient_matrix(domain)

        # set ghost cells
        gradient_matrix.has_left_ghost_cell = discretised_symbol.has_left_ghost_cell
        gradient_matrix.has_right_ghost_cell = discretised_symbol.has_right_ghost_cell

        return gradient_matrix @ discretised_symbol

    def gradient_matrix(self, domain):
        """
        Gradient matrix for finite volumes in the appropriate domain.
        Equivalent to grad(y) = (y[1:] - y[:-1])/dx

        Parameters
        ----------
        domain : list
            The domain(s) in which to compute the gradient matrix

        Returns
        -------
        :class:`pybamm.Matrix`
            The (sparse) finite volume gradient matrix for the domain
        """
        # Create appropriate submesh by combining submeshes in domain
        submesh_list = self.mesh.combine_submeshes(*domain)

        # can just use 1st entry of list to obtain the point etc
        submesh = submesh_list[0]

        # Create 1D matrix using submesh
        n = submesh.npts
        e = 1 / submesh.d_nodes
        sub_matrix = diags([-e, e], [0, 1], shape=(n - 1, n))

        # second dim length
        second_dim_len = len(submesh_list)

        # generate full matrix from the submatrix
        matrix = kron(eye(second_dim_len), sub_matrix)

        return pybamm.Matrix(matrix)

    def divergence(self, symbol, discretised_symbol, boundary_conditions):
        """Matrix-vector multiplication to implement the divergence operator.
        See :meth:`pybamm.SpatialMethod.divergence`
        """
        # Check that boundary condition keys are hashes (ids)
        for key in boundary_conditions.keys():
            assert isinstance(key, int), TypeError(
                "boundary condition keys should be hashes, not {}".format(type(key))
            )

        domain = symbol.domain
        submesh_list = self.mesh.combine_submeshes(*domain)

        # create a bc vector of length equal to the number variables
        # (only has non zero entries for neumann bcs)
        prim_dim = submesh_list[0].npts
        second_dim = len(submesh_list)
        total_pts = prim_dim * second_dim

        # Add Neumann boundary conditions if defined
        if symbol.id in boundary_conditions:

            # get boundary conditions
            bcs = boundary_conditions[symbol.id]
            if set(bcs.keys()) == set(["left", "right"]):
                # neumann on both sides
                lbc = bcs["left"]
                rbc = bcs["right"]
                # now we must create a matrix of size (npts * (npts -1) )
                # this is a different size to the one created when we have
                # flux boundary conditions so need a flag
                divergence_matrix = self.divergence_matrix(
                    domain, bc_type="neumann_neumann"
                )
                # only need interior edges (for spherical neumann_neumann)
                edges = submesh_list[0].edges[1:-1]
            elif set(bcs.keys()) == set(["left"]):
                # neumann on left, dirichlet on right
                lbc = bcs["left"]
                rbc = pybamm.Scalar(0)
                # divergence matrix and edges
                divergence_matrix = self.divergence_matrix(
                    domain, bc_type="neumann_dirichlet"
                )
                edges = submesh_list[0].edges[1:]
            elif set(bcs.keys()) == set(["right"]):
                # neumann on right, dirichlet on left
                lbc = pybamm.Scalar(0)
                rbc = bcs["right"]
                # divergence matrix and edges
                divergence_matrix = self.divergence_matrix(
                    domain, bc_type="dirichlet_neumann"
                )
                edges = submesh_list[0].edges[:-1]

            # taking divergence removes ghost cells
            discretised_symbol.has_left_ghost_cell = False
            discretised_symbol.has_right_ghost_cell = False

            # doing via loop so that it is easier to implement x varing bcs
            bcs_symbol = pybamm.Vector(np.array([]))  # empty vector
            for i in range(len(submesh_list)):

                if lbc.evaluates_to_number():
                    lbc_i = lbc
                else:
                    lbc_i = pybamm.Index(lbc, i)
                if rbc.evaluates_to_number():
                    rbc_i = rbc
                else:
                    rbc_i = pybamm.Index(rbc, i)
                # only the interior equations:
                interior = pybamm.Vector(np.zeros(prim_dim - 2))
                left = -lbc_i / pybamm.Vector(np.array([submesh_list[i].d_edges[0]]))
                right = rbc_i / pybamm.Vector(np.array([submesh_list[i].d_edges[-1]]))
                bcs_symbol = pybamm.NumpyConcatenation(
                    bcs_symbol, left, interior, right
                )

        else:
            divergence_matrix = self.divergence_matrix(
                domain, bc_type="dirichlet_dirichlet"
            )
            bcs_vec = np.zeros(total_pts)
            bcs_symbol = pybamm.Vector(bcs_vec)
            # need all edges for spherical dirichlet
            edges = submesh_list[0].edges

        # check for particle domain
        if submesh_list[0].coord_sys == "spherical polar":

            # create np.array of repeated submesh[0].nodes
            r_numpy = np.kron(np.ones(second_dim), submesh_list[0].nodes)
            r_edges_numpy = np.kron(np.ones(second_dim), edges)

            r = pybamm.Vector(r_numpy)
            r_edges = pybamm.Vector(r_edges_numpy)

            # for clarity, we are implicitly multiplying the the lbc by r^2=0
            # and the rbc by r^2=1. But lbc is 0 so we don't need to do
            # any r_edges^2 operations on bcs_symbol
            out = (1 / (r ** 2)) * (
                divergence_matrix @ ((r_edges ** 2) * discretised_symbol) + bcs_symbol
            )
        else:
            out = divergence_matrix @ discretised_symbol + bcs_symbol

        return out

    def divergence_matrix(self, domain, bc_type="dirichlet_dirichlet"):
        """
        Divergence matrix for finite volumes in the appropriate domain.
        Equivalent to div(N) = (N[1:] - N[:-1])/dx

        Parameters
        ----------
        domain : list
            The domain(s) in which to compute the divergence matrix
        bc_type : str
            What type of boundary condition to apply. Affects the size of the resulting
            matrix

        Returns
        -------
        :class:`pybamm.Matrix`
            The (sparse) finite volume divergence matrix for the domain
        """
        # Create appropriate submesh by combining submeshes in domain
        submesh_list = self.mesh.combine_submeshes(*domain)

        # can just use 1st entry of list to obtain the point etc
        submesh = submesh_list[0]
        e = 1 / submesh.d_edges

        # Create matrix using submesh
        n = submesh.npts + 1
        if bc_type == "dirichlet_dirichlet":
            sub_matrix = diags([-e, e], [0, 1], shape=(n - 1, n))
        elif bc_type == "dirichlet_neumann":
            # we don't have to act on right bc flux which is now in the bc vector
            sub_matrix = diags([-e, e], [0, 1], shape=(n - 1, n - 1))
        elif bc_type == "neumann_dirichlet":
            # we don't have to act on left bc flux which is now in the bc vector
            sub_matrix = diags([-e[1:], e], [-1, 0], shape=(n - 1, n - 1))
        elif bc_type == "neumann_neumann":
            # we don't have to act on bc fluxes which are now in the bc vector
            sub_matrix = diags([-e[1:], e], [-1, 0], shape=(n - 1, n - 2))
        else:
            raise NotImplementedError(
                "Can only process Neumann or Dirichlet boundary conditions"
            )

        # repeat matrix for each node in secondary dimensions
        second_dim_len = len(submesh_list)
        # generate full matrix from the submatrix
        matrix = kron(eye(second_dim_len), sub_matrix)
        return pybamm.Matrix(matrix)

    def integral(self, domain, symbol, discretised_symbol):
        """Vector-vector dot product to implement the integral operator.
        See :meth:`pybamm.BaseDiscretisation.integral`
        """
        # Calculate integration vector
        integration_vector = self.definite_integral_vector(domain)

        # Check for spherical domains
        submesh_list = self.mesh.combine_submeshes(*symbol.domain)
        if submesh_list[0].coord_sys == "spherical polar":
            second_dim = len(submesh_list)
            r_numpy = np.kron(np.ones(second_dim), submesh_list[0].nodes)
            r = pybamm.Vector(r_numpy)
            out = 4 * np.pi ** 2 * integration_vector @ (discretised_symbol * r)
        else:
            out = integration_vector @ discretised_symbol
        out.domain = []
        return out

    def definite_integral_vector(self, domain):
        """
        Vector for finite-volume implementation of the definite integral

        .. math::
            I = \\int_{a}^{b}\\!f(s)\\,ds

        for where :math:`a` and :math:`b` are the left-hand and right-hand boundaries of
        the domain respectively

        Parameters
        ----------
        domain : list
            The domain(s) of integration

        Returns
        -------
        :class:`pybamm.Vector`
            The finite volume integral vector for the domain
        """
        # Create appropriate submesh by combining submeshes in domain
        submesh_list = self.mesh.combine_submeshes(*domain)

        # Create vector of ones using submesh
        vector = np.array([])
        for submesh in submesh_list:
            vector = np.append(vector, submesh.d_edges * np.ones_like(submesh.nodes))

        return pybamm.Vector(vector)

    def indefinite_integral(self, domain, symbol, discretised_symbol):
        """Implementation of the indefinite integral operator. The
        input discretised symbol must be defined on the internal mesh edges.
        See :meth:`pybamm.BaseDiscretisation.indefinite_integral`
        """

        if not symbol.has_gradient_and_not_divergence():
            raise pybamm.ModelError(
                "Symbol to be integrated must be valid on the mesh edges"
            )

        # Calculate integration matrix
        integration_matrix = self.indefinite_integral_matrix(domain)

        # Don't need to check for spherical domains as spherical polars
        # only change the diveregence (symbols here have grad and no div)
        out = integration_matrix @ discretised_symbol

        out.domain = domain

        return out

    def indefinite_integral_matrix(self, domain):
        """
        Matrix for finite-volume implementation of the indefinite integral

        .. math::
            F = \\int\\!f(u)\\,du


        Parameters
        ----------
        domain : list
            The domain(s) of integration

        Returns
        -------
        :class:`pybamm.Vector`
            The finite volume integral vector for the domain
        """

        # Create appropriate submesh by combining submeshes in domain
        submesh_list = self.mesh.combine_submeshes(*domain)
        submesh = submesh_list[0]
        n = submesh.npts
        sec_pts = len(submesh_list)

        # note we have added a row of zeros at top for F(0) = 0
        du_n = submesh.d_nodes
        du_entries = [du_n] * (n - 1)
        offset = -np.arange(1, n, 1)
        sub_matrix = diags(du_entries, offset, shape=(n, n - 1))
        matrix = kron(eye(sec_pts), sub_matrix)

        return pybamm.Matrix(matrix)

    def add_ghost_nodes(self, symbol, discretised_symbol, lbc=None, rbc=None):
        """
        Add Dirichlet boundary conditions via ghost nodes.

        For a boundary condition "y = a at the left-hand boundary",
        we concatenate a ghost node to the start of the vector y with value "2*a - y1"
        where y1 is the value of the first node.
        Similarly for the right-hand boundary condition.

        Currently, Dirichlet boundary conditions can only be applied on state
        variables (e.g. concentration, temperature), and not on expressions.
        To access the value of the first node (y1), we create a "first_node" object
        which is a StateVector whose y_slice is the start of the y_slice of
        discretised_symbol.
        Similarly, the last node is a StateVector whose y_slice is the end of the
        y_slice of discretised_symbol

        Parameters
        ----------
        discretised_symbol : :class:`pybamm.StateVector` (size n)
            The discretised variable (a state vector) to which to add ghost nodes
        lbc : :class:`pybamm.Scalar`
            Dirichlet boundary condition on the left-hand side. Default is None.
        rbc : :class:`pybamm.Scalar`
            Dirichlet boundary condition on the right-hand side. Default is None.

        Returns
        -------
        :class:`pybamm.Concatenation` (size n+1 or n+2)
            Concatenation of the variable (a state vector) and ghost nodes

        """
        if isinstance(discretised_symbol, pybamm.StateVector):
            y_slice_start = discretised_symbol.y_slice.start
            y_slice_stop = discretised_symbol.y_slice.stop
        elif isinstance(discretised_symbol, pybamm.Concatenation):
            y_slice_start = discretised_symbol.children[0].y_slice.start
            y_slice_stop = discretised_symbol.children[-1].y_slice.stop
        else:
            raise TypeError(
                """
                discretised_symbol must be a StateVector or Concatenation, not '{}'
                """.format(
                    type(discretised_symbol)
                )
            )
        y = np.arange(y_slice_start, y_slice_stop)

        # reshape y_slices into more helpful form
        submesh_list = self.mesh.combine_submeshes(*symbol.domain)
        if isinstance(submesh_list[0].npts, list):
            NotImplementedError("Can only take in 1D primary directions")

        size = [len(submesh_list), submesh_list[0].npts]
        y = np.reshape(y, size)
        y_left = y[:, 0]
        y_right = y[:, -1]

        new_discretised_symbol = pybamm.Vector(np.array([]))  # starts empty

        for i in range(len(submesh_list)):
            y_slice_start = y_left[i]
            y_slice_stop = y_right[i]

            # left ghost cell
            first_node = pybamm.StateVector(slice(y_slice_start, y_slice_start + 1))

            # middle symbol
            sub_disc_symbol = pybamm.StateVector(slice(y_slice_start, y_slice_stop + 1))

            # right ghost cell
            last_node = pybamm.StateVector(slice(y_slice_stop, y_slice_stop + 1))

            if lbc is not None and rbc is not None:
                if lbc.evaluates_to_number():
                    lbc_i = lbc
                else:
                    lbc_i = pybamm.Index(lbc, i)
                if rbc.evaluates_to_number():
                    rbc_i = rbc
                else:
                    rbc_i = pybamm.Index(rbc, i)

                left_ghost_cell = 2 * lbc_i - first_node
                right_ghost_cell = 2 * rbc_i - last_node
                # concatenate and flag ghost cells
                concatenated_sub_disc_symbol = pybamm.NumpyConcatenation(
                    left_ghost_cell, sub_disc_symbol, right_ghost_cell
                )
                new_discretised_symbol = pybamm.NumpyConcatenation(
                    new_discretised_symbol, concatenated_sub_disc_symbol
                )
                new_discretised_symbol.has_left_ghost_cell = True
                new_discretised_symbol.has_right_ghost_cell = True
            elif lbc is not None:
                if lbc.evaluates_to_number():
                    lbc_i = lbc
                else:
                    lbc_i = pybamm.Index(lbc, i)

                # left ghost cell only
                left_ghost_cell = 2 * lbc_i - first_node
                # concatenate and flag ghost cells
                concatenated_sub_disc_symbol = pybamm.NumpyConcatenation(
                    left_ghost_cell, sub_disc_symbol
                )
                new_discretised_symbol = pybamm.NumpyConcatenation(
                    new_discretised_symbol, concatenated_sub_disc_symbol
                )
                new_discretised_symbol.has_left_ghost_cell = True
            elif rbc is not None:
                if rbc.evaluates_to_number():
                    rbc_i = rbc
                else:
                    rbc_i = pybamm.Index(rbc, i)
                # right ghost cell only
                right_ghost_cell = 2 * rbc_i - last_node
                # concatenate and flag ghost cells
                concatenated_sub_disc_symbol = pybamm.NumpyConcatenation(
                    sub_disc_symbol, right_ghost_cell
                )
                new_discretised_symbol = pybamm.NumpyConcatenation(
                    new_discretised_symbol, concatenated_sub_disc_symbol
                )
                new_discretised_symbol.has_right_ghost_cell = True
            else:
                raise ValueError("at least one boundary condition must be provided")

        return new_discretised_symbol

    def boundary_value(self, symbol, discretised_symbol, side):
        """
        Uses linear extrapolation to get the boundary value of a variable in the
        Finite Volume Method.

        See :meth:`pybamm.SpatialMethod.boundary_value`
        """

        # Find the number of submeshes
        submesh_list = self.mesh.combine_submeshes(*symbol.domain)
        if isinstance(submesh_list[0].npts, list):
            NotImplementedError("Can only take in 1D primary directions")

        prim_pts = submesh_list[0].npts
        sec_pts = len(submesh_list)

        # Create submatrix to compute boundary values
        if side == "left":
            sub_matrix = csr_matrix(
                ([1.5, -0.5], ([0, 0], [0, 1])), shape=(1, prim_pts)
            )
        elif side == "right":
            sub_matrix = csr_matrix(
                ([-0.5, 1.5], ([0, 0], [prim_pts - 2, prim_pts - 1])),
                shape=(1, prim_pts),
            )

        # Generate full matrix from the submatrix
        matrix = kron(eye(sec_pts), sub_matrix)

        # Return boundary value with domain removed
        boundary_value = pybamm.Matrix(matrix) @ discretised_symbol
        boundary_value.domain = []

        return boundary_value

    def process_binary_operators(self, bin_op, left, right, disc_left, disc_right):
        """Discretise binary operators in model equations.  Performs appropriate
        averaging of diffusivities if one of the children is a gradient operator, so
        that discretised sizes match up.

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
        # If neither child has gradients, or both children have gradients
        # no need to do any averaging
        if (
            left.has_gradient_and_not_divergence()
            == right.has_gradient_and_not_divergence()
        ):
            pass
        # If only left child has gradient, compute diffusivity for right child
        elif (
            left.has_gradient_and_not_divergence()
            and not right.has_gradient_and_not_divergence()
        ):
            # Extrapolate at either end depending on the ghost cells (from gradient)
            extrapolate_left = any(
                [x.has_left_ghost_cell for x in disc_left.pre_order()]
            )
            extrapolate_right = any(
                [x.has_right_ghost_cell for x in disc_left.pre_order()]
            )
            disc_right = self.compute_diffusivity(
                disc_right, extrapolate_left, extrapolate_right
            )
        # If only right child has gradient, compute diffusivity for left child
        elif (
            right.has_gradient_and_not_divergence()
            and not left.has_gradient_and_not_divergence()
        ):
            # Extrapolate at either end depending on the ghost cells (from gradient)
            extrapolate_left = any(
                [x.has_left_ghost_cell for x in disc_right.pre_order()]
            )
            extrapolate_right = any(
                [x.has_right_ghost_cell for x in disc_right.pre_order()]
            )
            disc_left = self.compute_diffusivity(
                disc_left, extrapolate_left, extrapolate_right
            )
        # Return new binary operator with appropriate class
        return bin_op.__class__(disc_left, disc_right)

    def compute_diffusivity(
        self, discretised_symbol, extrapolate_left=False, extrapolate_right=False
    ):
        """
        Compute the diffusivity at cell edges, based on the diffusivity at cell nodes.
        For now we just take the arithemtic mean, though it may be better to take the
        harmonic mean based on [1].

        [1] Recktenwald, Gerald. "The control-volume finite-difference approximation to
        the diffusion equation." (2012).

        Parameters
        ----------
        discretised_symbol : :class:`pybamm.Symbol`
            Symbol to be averaged. When evaluated, this symbol returns either a scalar
            or an array of shape (n,), where n is the number of points in the mesh for
            the symbol's domain (n = self.mesh[symbol.domain].npts)
        extrapolate_left : boolean
            Whether to extrapolate one node to the left when computing the
            diffusivity, to account for ghost cells. Default is False
        extrapolate_right : boolean
            Whether to extrapolate one node to the right when computing the
            diffusivity, to account for ghost cells. Default is False

        Returns
        -------
        :class:`pybamm.Function`
            Averaged symbol. When evaluated, this returns either a scalar or an array of
            shape (n-1,) as appropriate.
        """

        def arithmetic_mean(array):
            """Calculate the arithemetic mean of an array using matrix multiplication"""
            # Create appropriate submesh by combining submeshes in domain
            submesh_list = self.mesh.combine_submeshes(*array.domain)

            # Can just use 1st entry of list to obtain the point etc
            submesh = submesh_list[0]

            # Create 1D matrix using submesh
            n = submesh.npts
            sub_matrix = diags([0.5, 0.5], [0, 1], shape=(n - 1, n))

            if extrapolate_left:
                sub_matrix_left = csr_matrix(
                    ([1.5, -0.5], ([0, 0], [0, 1])), shape=(1, n)
                )
                sub_matrix = vstack([sub_matrix_left, sub_matrix])
            if extrapolate_right:
                sub_matrix_right = csr_matrix(
                    ([-0.5, 1.5], ([0, 0], [n - 2, n - 1])), shape=(1, n)
                )
                sub_matrix = vstack([sub_matrix, sub_matrix_right])

            # Second dimension length
            second_dim_len = len(submesh_list)

            # Generate full matrix from the submatrix
            matrix = kron(eye(second_dim_len), sub_matrix)

            return pybamm.Matrix(matrix) @ array

        # If discretised_symbol evaluates to number there is no need to average
        # NOTE: Doing this check every time might be slow?
        if discretised_symbol.evaluates_to_number():
            return discretised_symbol
        else:
            return arithmetic_mean(discretised_symbol)
