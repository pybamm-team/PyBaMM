#
# Finite Volume discretisation class
#
import pybamm

from scipy.sparse import diags, eye, kron, csr_matrix, vstack, hstack, lil_matrix
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
        # Discretise symbol
        domain = symbol.domain

        # Add boundary conditions, if defined
        if symbol.id in boundary_conditions:
            bcs = boundary_conditions[symbol.id]
            # add ghost nodes
            discretised_symbol = self.add_ghost_nodes(symbol, discretised_symbol, bcs)
            # edit domain
            domain = (
                [domain[0] + "_left ghost cell"]
                + domain
                + [domain[-1] + "_right ghost cell"]
            )

        # note in 1D spherical grad and normal grad are the same
        gradient_matrix = self.gradient_matrix(domain)

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
        domain = symbol.domain
        submesh_list = self.mesh.combine_submeshes(*domain)

        divergence_matrix = self.divergence_matrix(domain)

        # check for particle domain
        if submesh_list[0].coord_sys == "spherical polar":
            second_dim = len(submesh_list)
            edges = submesh_list[0].edges

            # create np.array of repeated submesh[0].nodes
            r_numpy = np.kron(np.ones(second_dim), submesh_list[0].nodes)
            r_edges_numpy = np.kron(np.ones(second_dim), edges)

            r = pybamm.Vector(r_numpy)
            r_edges = pybamm.Vector(r_edges_numpy)

            out = (1 / (r ** 2)) * (
                divergence_matrix @ ((r_edges ** 2) * discretised_symbol)
            )
        else:
            out = divergence_matrix @ discretised_symbol

        return out

    def divergence_matrix(self, domain):
        """
        Divergence matrix for finite volumes in the appropriate domain.
        Equivalent to div(N) = (N[1:] - N[:-1])/dx

        Parameters
        ----------
        domain : list
            The domain(s) in which to compute the divergence matrix

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
        sub_matrix = diags([-e, e], [0, 1], shape=(n - 1, n))

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

        return pybamm.Matrix(vector[np.newaxis, :])

    def indefinite_integral(self, domain, symbol, discretised_symbol):
        """Implementation of the indefinite integral operator. The
        input discretised symbol must be defined on the internal mesh edges.
        See :meth:`pybamm.BaseDiscretisation.indefinite_integral`
        """

        if not symbol.evaluates_on_edges():
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
            F(x) = \\int_0^x\\!f(u)\\,du

        The indefinite integral must satisfy the following conditions:

        - :math:`F(0) = 0`
        - :math:`f(x) = \\frac{dF}{dx}`

        or, in discrete form,

        - `BoundaryValue(F, "left") = 0`, i.e. :math:`3*F_0 - F_1 = 0`
        - :math:`f_{i+1/2} = (F_{i+1} - F_i) / dx_{i+1/2}`

        Hence we must have

        - :math:`F_0 = du_{1/2} * f_{1/2} / 2`
        - :math:`F_{i+1} = F_i + du * f_{i+1/2}`

        Note that :math:`f_{-1/2}` and :math:`f_{n+1/2}` are included in the discrete
        integrand vector `f`, so we add a column of zeros at each end of the
        indefinite integral matrix to ignore these.

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

        du_n = submesh.d_nodes
        du_entries = [du_n] * (n - 1)
        offset = -np.arange(1, n, 1)
        main_integral_matrix = diags(du_entries, offset, shape=(n, n - 1))
        bc_offset_matrix = lil_matrix((n, n - 1))
        bc_offset_matrix[:, 0] = du_n[0] / 2
        sub_matrix = main_integral_matrix + bc_offset_matrix
        # add a column of zeros at each end
        zero_col = csr_matrix((n, 1))
        sub_matrix = hstack([zero_col, sub_matrix, zero_col])
        matrix = kron(eye(sec_pts), sub_matrix)

        return pybamm.Matrix(matrix)

    def add_ghost_nodes(self, symbol, discretised_symbol, bcs):
        """
        Add boundary conditions via ghost nodes.

        For Dirichlet bcs, for a boundary condition "y = a at the left-hand boundary",
        we concatenate a ghost node to the start of the vector y with value "2*a - y1"
        where y1 is the value of the first node.
        Similarly for the right-hand boundary condition.

        For Dirichlet bcs, for a boundary condition "y = a at the left-hand boundary",
        we concatenate a ghost node to the start of the vector y with value "2*a - y1"
        where y1 is the value of the first node.
        Similarly for the right-hand boundary condition.

        For Neumann bcs, for a boundary condition "dy/dx = b at the left-hand boundary",
        we concatenate a ghost node to the start of the vector y with value "b*h + y1"
        where y1 is the value of the first node and h is the mesh size.
        Similarly for the right-hand boundary condition.

        Currently, boundary conditions can only be applied on state
        variables (e.g. concentration, temperature) or concatenations, and not on
        arbitrary expressions.
        To access the value of the first node (y1), we create a "first_node" object
        which is a StateVector whose y_slice is the start of the y_slice of
        discretised_symbol.
        Similarly, the last node is a StateVector whose y_slice is the end of the
        y_slice of discretised_symbol

        Parameters
        ----------
        discretised_symbol : :class:`pybamm.StateVector` (size n)
            The discretised variable (a state vector) to which to add ghost nodes
        bcs : dict of tuples (:class:`pybamm.Scalar`, str)
            Dictionary (with keys "left" and "right") of boundary conditions. Each
            boundary condition consists of a value and a flag indicating its type
            (e.g. "Dirichlet")

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

        lbc_value, lbc_type = bcs["left"]
        rbc_value, rbc_type = bcs["right"]

        for i in range(len(submesh_list)):
            y_slice_start = y_left[i]
            y_slice_stop = y_right[i]

            # left ghost cell
            first_node = pybamm.StateVector(slice(y_slice_start, y_slice_start + 1))

            # middle symbol
            sub_disc_symbol = pybamm.StateVector(slice(y_slice_start, y_slice_stop + 1))

            # right ghost cell
            last_node = pybamm.StateVector(slice(y_slice_stop, y_slice_stop + 1))

            if lbc_value.evaluates_to_number():
                lbc_i = lbc_value
            else:
                lbc_i = pybamm.Index(lbc_value, i)
            if rbc_value.evaluates_to_number():
                rbc_i = rbc_value
            else:
                rbc_i = pybamm.Index(rbc_value, i)

            if lbc_type == "Dirichlet":
                left_ghost_cell = 2 * lbc_i - first_node
            elif lbc_type == "Neumann":
                dx = 2 * (submesh_list[0].nodes[0] - submesh_list[0].edges[0])
                left_ghost_cell = first_node - dx * lbc_i
            else:
                raise ValueError(
                    "boundary condition must be Dirichlet or Neumann, not '{}'".format(
                        lbc_type
                    )
                )
            if rbc_type == "Dirichlet":
                right_ghost_cell = 2 * rbc_i - last_node
            elif rbc_type == "Neumann":
                dx = 2 * (submesh_list[0].edges[-1] - submesh_list[0].nodes[-1])
                right_ghost_cell = last_node + dx * rbc_i
            else:
                raise ValueError(
                    "boundary condition must be Dirichlet or Neumann, not '{}'".format(
                        rbc_type
                    )
                )
            # concatenate
            concatenated_sub_disc_symbol = pybamm.NumpyConcatenation(
                left_ghost_cell, sub_disc_symbol, right_ghost_cell
            )
            new_discretised_symbol = pybamm.NumpyConcatenation(
                new_discretised_symbol, concatenated_sub_disc_symbol
            )

        # Keep same domain
        new_discretised_symbol.domain = discretised_symbol.domain

        return new_discretised_symbol

    def boundary_value_or_flux(self, symbol, discretised_child):
        """
        Uses linear extrapolation to get the boundary value or flux of a variable in the
        Finite Volume Method.

        See :meth:`pybamm.SpatialMethod.boundary_value`
        """

        # Find the number of submeshes
        submesh_list = self.mesh.combine_submeshes(*discretised_child.domain)
        if isinstance(submesh_list[0].npts, list):
            NotImplementedError("Can only take in 1D primary directions")

        prim_pts = submesh_list[0].npts
        sec_pts = len(submesh_list)

        # Create submatrix to compute boundary values or fluxes
        if isinstance(symbol, pybamm.BoundaryValue):
            if symbol.side == "left":
                sub_matrix = csr_matrix(
                    ([1.5, -0.5], ([0, 0], [0, 1])), shape=(1, prim_pts)
                )
            elif symbol.side == "right":
                sub_matrix = csr_matrix(
                    ([-0.5, 1.5], ([0, 0], [prim_pts - 2, prim_pts - 1])),
                    shape=(1, prim_pts),
                )
        elif isinstance(symbol, pybamm.BoundaryFlux):
            if symbol.side == "left":
                dx = submesh_list[0].d_nodes[0]
                sub_matrix = (1 / dx) * csr_matrix(
                    ([-1, 1], ([0, 0], [0, 1])), shape=(1, prim_pts)
                )
            elif symbol.side == "right":
                dx = submesh_list[0].d_nodes[-1]
                sub_matrix = (1 / dx) * csr_matrix(
                    ([-1, 1], ([0, 0], [prim_pts - 2, prim_pts - 1])),
                    shape=(1, prim_pts),
                )

        # Generate full matrix from the submatrix
        matrix = kron(eye(sec_pts), sub_matrix)

        # Return boundary value with domain removed
        boundary_value = pybamm.Matrix(matrix) @ discretised_child

        if sec_pts > 1:
            if discretised_child.domain == ["negative particle"]:
                boundary_value.domain = ["negative electrode"]
            elif discretised_child.domain == ["positive particle"]:
                boundary_value.domain = ["positive electrode"]
        else:
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
        left_evaluates_on_edges = left.evaluates_on_edges()
        right_evaluates_on_edges = right.evaluates_on_edges()
        # If neither child evaluates on edges, or both children have gradients,
        # no need to do any averaging
        if left_evaluates_on_edges == right_evaluates_on_edges:
            pass
        # If only left child evaluates on edges, compute diffusivity for right child
        elif left_evaluates_on_edges and not right_evaluates_on_edges:
            disc_right = self.compute_diffusivity(disc_right)
        # If only right child evaluates on edges, compute diffusivity for left child
        elif right_evaluates_on_edges and not left_evaluates_on_edges:
            disc_left = self.compute_diffusivity(disc_left)
        # Return new binary operator with appropriate class
        return bin_op.__class__(disc_left, disc_right)

    def compute_diffusivity(self, discretised_symbol):
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

        Returns
        -------
        :class:`pybamm.Function`
            Averaged symbol. When evaluated, this returns either a scalar or an array of
            shape (n+1,) as appropriate.
        """

        def arithmetic_mean(array):
            """Calculate the arithemetic mean of an array using matrix multiplication"""
            # Create appropriate submesh by combining submeshes in domain
            submesh_list = self.mesh.combine_submeshes(*array.domain)

            # Can just use 1st entry of list to obtain the point etc
            submesh = submesh_list[0]

            # Create 1D matrix using submesh
            n = submesh.npts

            sub_matrix_left = csr_matrix(([1.5, -0.5], ([0, 0], [0, 1])), shape=(1, n))
            sub_matrix_center = diags([0.5, 0.5], [0, 1], shape=(n - 1, n))
            sub_matrix_right = csr_matrix(
                ([-0.5, 1.5], ([0, 0], [n - 2, n - 1])), shape=(1, n)
            )
            sub_matrix = vstack([sub_matrix_left, sub_matrix_center, sub_matrix_right])

            # Second dimension length
            second_dim_len = len(submesh_list)

            # Generate full matrix from the submatrix
            matrix = kron(eye(second_dim_len), sub_matrix)

            return pybamm.Matrix(matrix) @ array

        # If discretised_symbol evaluates to number there is no need to average
        if discretised_symbol.evaluates_to_number():
            return discretised_symbol
        else:
            return arithmetic_mean(discretised_symbol)
