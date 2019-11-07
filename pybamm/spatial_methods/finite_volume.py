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
    lil_matrix,
    coo_matrix,
)
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

        # there is no way to set this at the moment
        self.extrapolation = "quadratic"

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
        symbol_mesh = self.mesh.combine_submeshes(*symbol.domain)
        return pybamm.Vector(symbol_mesh[0].nodes, domain=symbol.domain)

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

        out = gradient_matrix @ discretised_symbol
        return out

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
        # Convert to csr_matrix so that we can take the index (row-slicing), which is
        # not supported by the default kron format
        # Note that this makes column-slicing inefficient, but this should not be an
        # issue
        matrix = csr_matrix(kron(eye(second_dim_len), sub_matrix))

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
        # Convert to csr_matrix so that we can take the index (row-slicing), which is
        # not supported by the default kron format
        # Note that this makes column-slicing inefficient, but this should not be an
        # issue
        matrix = csr_matrix(kron(eye(second_dim_len), sub_matrix))
        return pybamm.Matrix(matrix)

    def laplacian(self, symbol, discretised_symbol, boundary_conditions):
        """
        Laplacian operator, implemented as div(grad(.))
        See :meth:`pybamm.SpatialMethod.laplacian`
        """
        grad = self.gradient(symbol, discretised_symbol, boundary_conditions)
        return self.divergence(grad, grad, boundary_conditions)

    def integral(self, child, discretised_child):
        """Vector-vector dot product to implement the integral operator. """
        # Calculate integration vector
        integration_vector = self.definite_integral_matrix(child.domain)

        # Check for spherical domains
        submesh_list = self.mesh.combine_submeshes(*child.domain)
        if submesh_list[0].coord_sys == "spherical polar":
            second_dim = len(submesh_list)
            r_numpy = np.kron(np.ones(second_dim), submesh_list[0].nodes)
            r = pybamm.Vector(r_numpy)
            out = 4 * np.pi ** 2 * integration_vector @ (discretised_child * r)
        else:
            out = integration_vector @ discretised_child

        return out

    def definite_integral_matrix(self, domain, vector_type="row"):
        """
        Matrix for finite-volume implementation of the definite integral in the
        primary dimension

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
        :class:`pybamm.Matrix`
            The finite volume integral matrix for the domain
        vector_type : str, optional
            Whether to return a row or column vector in the primary dimension
            (default is row)
        """
        # Create appropriate submesh by combining submeshes in domain
        submesh_list = self.mesh.combine_submeshes(*domain)

        # Create vector of ones for primary domain submesh
        submesh = submesh_list[0]
        vector = submesh.d_edges * np.ones_like(submesh.nodes)

        if vector_type == "row":
            vector = vector[np.newaxis, :]
        elif vector_type == "column":
            vector = vector[:, np.newaxis]

        # repeat matrix for each node in secondary dimensions
        second_dim_len = len(submesh_list)
        # generate full matrix from the submatrix
        # Convert to csr_matrix so that we can take the index (row-slicing), which is
        # not supported by the default kron format
        # Note that this makes column-slicing inefficient, but this should not be an
        # issue
        matrix = csr_matrix(kron(eye(second_dim_len), vector))
        return pybamm.Matrix(matrix)

    def indefinite_integral(self, child, discretised_child):
        """Implementation of the indefinite integral operator. """

        # Different integral matrix depending on whether the integrand evaluates on
        # edges or nodes
        if child.evaluates_on_edges():
            integration_matrix = self.indefinite_integral_matrix_edges(child.domain)
        else:
            integration_matrix = self.indefinite_integral_matrix_nodes(child.domain)

        # Don't need to check for spherical domains as spherical polars
        # only change the diveregence (childs here have grad and no div)
        out = integration_matrix @ discretised_child

        out.domain = child.domain
        out.auxiliary_domains = child.auxiliary_domains

        return out

    def indefinite_integral_matrix_edges(self, domain):
        """
        Matrix for finite-volume implementation of the indefinite integral where the
        integrand is evaluated on mesh edges

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
        :class:`pybamm.Matrix`
            The finite volume integral matrix for the domain
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
        # Convert to csr_matrix so that we can take the index (row-slicing), which is
        # not supported by the default kron format
        # Note that this makes column-slicing inefficient, but this should not be an
        # issue
        matrix = csr_matrix(kron(eye(sec_pts), sub_matrix))

        return pybamm.Matrix(matrix)

    def delta_function(self, symbol, discretised_symbol):
        """
        Delta function. Implemented as a vector whose only non-zero element is the
        first (if symbol.side = "left") or last (if symbol.side = "right"), with
        appropriate value so that the integral of the delta function across the whole
        domain is the same as the integral of the discretised symbol across the whole
        domain.

        See :meth:`pybamm.SpatialMethod.delta_function`
        """
        # Find the number of submeshes
        submesh_list = self.mesh.combine_submeshes(*symbol.domain)

        prim_pts = submesh_list[0].npts
        sec_pts = len(submesh_list)

        # Create submatrix to compute delta function as a flux
        if symbol.side == "left":
            dx = submesh_list[0].d_nodes[0]
            sub_matrix = csr_matrix(([1], ([0], [0])), shape=(prim_pts, 1))
        elif symbol.side == "right":
            dx = submesh_list[0].d_nodes[-1]
            sub_matrix = csr_matrix(([1], ([prim_pts - 1], [0])), shape=(prim_pts, 1))

        # Calculate domain width, to make sure that the integral of the delta function
        # is the same as the integral of the child
        domain_width = submesh_list[0].edges[-1] - submesh_list[0].edges[0]
        # Generate full matrix from the submatrix
        # Convert to csr_matrix so that we can take the index (row-slicing), which is
        # not supported by the default kron format
        # Note that this makes column-slicing inefficient, but this should not be an
        # issue
        matrix = kron(eye(sec_pts), sub_matrix).toarray()

        # Return delta function, keep domains
        delta_fn = pybamm.Matrix(domain_width / dx * matrix) * discretised_symbol
        delta_fn.domain = symbol.domain
        delta_fn.auxiliary_domains = symbol.auxiliary_domains

        return delta_fn

    def internal_neumann_condition(
        self, left_symbol_disc, right_symbol_disc, left_mesh, right_mesh
    ):
        """
        A method to find the internal neumann conditions between two symbols
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

        left_npts = left_mesh[0].npts
        right_npts = right_mesh[0].npts

        sec_pts = len(left_mesh)

        if sec_pts != len(right_mesh):
            raise pybamm.DomainError(
                """Number of secondary points in subdomains do not match"""
            )

        left_sub_matrix = np.zeros((1, left_npts))
        left_sub_matrix[0][left_npts - 1] = 1
        left_matrix = pybamm.Matrix(csr_matrix(kron(eye(sec_pts), left_sub_matrix)))

        right_sub_matrix = np.zeros((1, right_npts))
        right_sub_matrix[0][0] = 1
        right_matrix = pybamm.Matrix(csr_matrix(kron(eye(sec_pts), right_sub_matrix)))

        # Remove domains to avoid clash
        left_domain = left_symbol_disc.domain
        right_domain = right_symbol_disc.domain
        left_symbol_disc.domain = []
        right_symbol_disc.domain = []

        # Finite volume derivative
        dy = right_matrix @ right_symbol_disc - left_matrix @ left_symbol_disc
        dx = right_mesh[0].nodes[0] - left_mesh[0].nodes[-1]

        # Change domains back
        left_symbol_disc.domain = left_domain
        right_symbol_disc.domain = right_domain

        return dy / dx

    def indefinite_integral_matrix_nodes(self, domain):
        """
        Matrix for finite-volume implementation of the indefinite integral where the
        integrand is evaluated on mesh nodes.
        This is just a straightforward cumulative sum of the integrand

        Parameters
        ----------
        domain : list
            The domain(s) of integration

        Returns
        -------
        :class:`pybamm.Matrix`
            The finite volume integral matrix for the domain
        """

        # Create appropriate submesh by combining submeshes in domain
        submesh_list = self.mesh.combine_submeshes(*domain)
        submesh = submesh_list[0]
        n = submesh.npts
        sec_pts = len(submesh_list)

        du_n = submesh.d_edges
        du_entries = [du_n] * (n)
        offset = -np.arange(1, n + 1, 1)
        sub_matrix = diags(du_entries, offset, shape=(n + 1, n))
        # Convert to csr_matrix so that we can take the index (row-slicing), which is
        # not supported by the default kron format
        # Note that this makes column-slicing inefficient, but this should not be an
        # issue
        matrix = csr_matrix(kron(eye(sec_pts), sub_matrix))

        return pybamm.Matrix(matrix)

    def add_ghost_nodes(self, symbol, discretised_symbol, bcs):
        """
        Add ghost nodes to a symbol.

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

        Parameters
        ----------
        domain : list of strings
            The domain of the symbol for which to add ghost nodes
        bcs : dict of tuples (:class:`pybamm.Scalar`, str)
            Dictionary (with keys "left" and "right") of boundary conditions. Each
            boundary condition consists of a value and a flag indicating its type
            (e.g. "Dirichlet")

        Returns
        -------
        :class:`pybamm.Symbol` (shape (n+2, n))
            `Matrix @ discretised_symbol + bcs_vector`. When evaluated, this gives the
            discretised_symbol, with appropriate ghost nodes concatenated at each end.

        """
        # get relevant grid points
        submesh_list = self.mesh.combine_submeshes(*symbol.domain)

        # Prepare sizes and empty bcs_vector
        n = submesh_list[0].npts
        sec_pts = len(submesh_list)

        bcs_vector = pybamm.Vector(np.array([]))  # starts empty

        lbc_value, lbc_type = bcs["left"]
        rbc_value, rbc_type = bcs["right"]

        for i in range(sec_pts):
            if lbc_value.evaluates_to_number():
                lbc_i = lbc_value
            else:
                lbc_i = lbc_value[i]
            if rbc_value.evaluates_to_number():
                rbc_i = rbc_value
            else:
                rbc_i = rbc_value[i]
            if lbc_type == "Dirichlet":
                left_ghost_constant = 2 * lbc_i
            elif lbc_type == "Neumann":
                dx = 2 * (submesh_list[0].nodes[0] - submesh_list[0].edges[0])
                left_ghost_constant = -dx * lbc_i
            else:
                raise ValueError(
                    "boundary condition must be Dirichlet or Neumann, not '{}'".format(
                        lbc_type
                    )
                )
            if rbc_type == "Dirichlet":
                right_ghost_constant = 2 * rbc_i
            elif rbc_type == "Neumann":
                dx = 2 * (submesh_list[0].edges[-1] - submesh_list[0].nodes[-1])
                right_ghost_constant = dx * rbc_i
            else:
                raise ValueError(
                    "boundary condition must be Dirichlet or Neumann, not '{}'".format(
                        rbc_type
                    )
                )
            # concatenate
            bcs_vector = pybamm.NumpyConcatenation(
                bcs_vector,
                left_ghost_constant,
                pybamm.Vector(np.zeros(n)),
                right_ghost_constant,
            )

        # Make matrix to calculate ghost nodes
        bc_factors = {"Dirichlet": -1, "Neumann": 1}
        left_factor = bc_factors[lbc_type]
        right_factor = bc_factors[rbc_type]
        # coo_matrix takes inputs (data, (row, col)) and puts data[i] at the point
        # (row[i], col[i]) for each index of data.
        left_ghost_vector = coo_matrix(([left_factor], ([0], [0])), shape=(1, n))
        right_ghost_vector = coo_matrix(([right_factor], ([0], [n - 1])), shape=(1, n))
        sub_matrix = vstack([left_ghost_vector, eye(n), right_ghost_vector])

        # repeat matrix for secondary dimensions
        # Convert to csr_matrix so that we can take the index (row-slicing), which is
        # not supported by the default kron format
        # Note that this makes column-slicing inefficient, but this should not be an
        # issue
        matrix = csr_matrix(kron(eye(sec_pts), sub_matrix))

        return pybamm.Matrix(matrix) @ discretised_symbol + bcs_vector

    def boundary_value_or_flux(self, symbol, discretised_child, bcs=None):
        """
        Uses extrapolation to get the boundary value or flux of a variable in the
        Finite Volume Method.

        See :meth:`pybamm.SpatialMethod.boundary_value`
        """

        # Find the number of submeshes
        submesh_list = self.mesh.combine_submeshes(*discretised_child.domain)

        prim_pts = submesh_list[0].npts
        sec_pts = len(submesh_list)

        if not bcs:
            bcs = {}

        # Create submatrix to compute boundary values or fluxes
        if isinstance(symbol, pybamm.BoundaryValue):

            # Derivation of extrapolation formula can be found at:
            # https://github.com/Scottmar93/extrapolation-coefficents/tree/master
            nodes = submesh_list[0].nodes
            edges = submesh_list[0].edges

            dx0 = nodes[0] - edges[0]
            dx1 = submesh_list[0].d_nodes[0]
            dx2 = submesh_list[0].d_nodes[1]

            dxN = edges[-1] - nodes[-1]
            dxNm1 = submesh_list[0].d_nodes[-1]
            dxNm2 = submesh_list[0].d_nodes[-2]

            child = symbol.child

            if symbol.side == "left":

                if self.extrapolation == "linear":
                    # to find value at x* use formula:
                    # f(x*) = f_1 - (dx0 / dx1) (f_2 - f_1)

                    if pybamm.has_bc_condition_of_form(
                        child, symbol.side, bcs, "Neumann"
                    ):
                        sub_matrix = csr_matrix(([1], ([0], [0])), shape=(1, prim_pts),)

                        additive = -dx0 * bcs[child.id][symbol.side][0]

                    else:
                        sub_matrix = csr_matrix(
                            ([1 + (dx0 / dx1), -(dx0 / dx1)], ([0, 0], [0, 1])),
                            shape=(1, prim_pts),
                        )
                        additive = pybamm.Scalar(0)

                elif self.extrapolation == "quadratic":

                    if pybamm.has_bc_condition_of_form(
                        child, symbol.side, bcs, "Neumann"
                    ):
                        a = (dx0 + dx1) ** 2 / (dx1 * (2 * dx0 + dx1))
                        b = -(dx0 ** 2) / (2 * dx0 * dx1 + dx1 ** 2)
                        alpha = -(dx0 * (dx0 + dx1)) / (2 * dx0 + dx1)

                        sub_matrix = csr_matrix(
                            ([a, b], ([0, 0], [0, 1])), shape=(1, prim_pts),
                        )
                        additive = alpha * bcs[child.id][symbol.side][0]

                    else:
                        a = (dx0 + dx1) * (dx0 + dx1 + dx2) / (dx1 * (dx1 + dx2))
                        b = -dx0 * (dx0 + dx1 + dx2) / (dx1 * dx2)
                        c = dx0 * (dx0 + dx1) / (dx2 * (dx1 + dx2))

                        sub_matrix = csr_matrix(
                            ([a, b, c], ([0, 0, 0], [0, 1, 2])), shape=(1, prim_pts),
                        )

                        additive = pybamm.Scalar(0)
                else:
                    raise NotImplementedError

            elif symbol.side == "right":

                if self.extrapolation == "linear":

                    if pybamm.has_bc_condition_of_form(
                        child, symbol.side, bcs, "Neumann"
                    ):
                        # use formula:
                        # f(x*) = fN + dxN * f'(x*)
                        sub_matrix = csr_matrix(
                            ([1], ([0], [prim_pts - 1]),), shape=(1, prim_pts),
                        )
                        additive = dxN * bcs[child.id][symbol.side][0]

                    elif pybamm.has_bc_condition_of_form(
                        child, symbol.side, bcs, "Dirichlet"
                    ):
                        # just use the value from the bc: f(x*)
                        sub_matrix = csr_matrix((1, prim_pts))
                        additive = bcs[child.id][symbol.side][0]

                    else:
                        # to find value at x* use formula:
                        # f(x*) = f_N - (dxN / dxNm1) (f_N - f_Nm1)
                        sub_matrix = csr_matrix(
                            (
                                [-(dxN / dxNm1), 1 + (dxN / dxNm1)],
                                ([0, 0], [prim_pts - 2, prim_pts - 1]),
                            ),
                            shape=(1, prim_pts),
                        )
                        additive = pybamm.Scalar(0)
                elif self.extrapolation == "quadratic":

                    if pybamm.has_bc_condition_of_form(
                        child, symbol.side, bcs, "Neumann"
                    ):
                        a = (dxN + dxNm1) ** 2 / (dxNm1 * (2 * dxN + dxNm1))
                        b = -(dxN ** 2) / (2 * dxN * dxNm1 + dxNm1 ** 2)
                        alpha = dxN * (dxN + dxNm1) / (2 * dxN + dxNm1)
                        sub_matrix = csr_matrix(
                            ([b, a], ([0, 0], [prim_pts - 2, prim_pts - 1]),),
                            shape=(1, prim_pts),
                        )

                        additive = alpha * bcs[child.id][symbol.side][0]

                    else:
                        a = (
                            (dxN + dxNm1)
                            * (dxN + dxNm1 + dxNm2)
                            / (dxNm1 * (dxNm1 + dxNm2))
                        )
                        b = -dxN * (dxN + dxNm1 + dxNm2) / (dxNm1 * dxNm2)
                        c = dxN * (dxN + dxNm1) / (dxNm2 * (dxNm1 + dxNm2))

                        sub_matrix = csr_matrix(
                            (
                                [c, b, a],
                                ([0, 0, 0], [prim_pts - 3, prim_pts - 2, prim_pts - 1]),
                            ),
                            shape=(1, prim_pts),
                        )
                        additive = pybamm.Scalar(0)
                else:
                    raise NotImplementedError

        elif isinstance(symbol, pybamm.BoundaryGradient):
            if symbol.side == "left":

                if self.extrapolation == "linear":

                    if pybamm.has_bc_condition_of_form(
                        child, symbol.side, bcs, "Neumann"
                    ):
                        # just use the value from the bc: f'(x*)
                        sub_matrix = csr_matrix((1, prim_pts))
                        additive = bcs[child.id][symbol.side][0]
                    else:
                        # use formula:
                        # f'(x*) = (f_2 - f_1) / dx1
                        sub_matrix = (1 / dx1) * csr_matrix(
                            ([-1, 1], ([0, 0], [0, 1])), shape=(1, prim_pts)
                        )
                        additive = pybamm.Scalar(0)
                elif self.extrapolation == "quadratic":

                    a = -(2 * dx0 + 2 * dx1 + dx2) / (dx1 ** 2 + dx1 * dx2)
                    b = (2 * dx0 + dx1 + dx2) / (dx1 * dx2)
                    c = -(2 * dx0 + dx1) / (dx1 * dx2 + dx2 ** 2)

                    sub_matrix = csr_matrix(
                        ([a, b, c], ([0, 0, 0], [0, 1, 2])), shape=(1, prim_pts)
                    )
                    additive = pybamm.Scalar(0)
                else:
                    raise NotImplementedError

            elif symbol.side == "right":

                if self.extrapolation == "linear":
                    if pybamm.has_bc_condition_of_form(
                        child, symbol.side, bcs, "Neumann"
                    ):
                        # just use the value from the bc: f'(x*)
                        sub_matrix = csr_matrix((1, prim_pts))
                        additive = bcs[child.id][symbol.side][0]
                    else:
                        # use formula:
                        # f'(x*) = (f_N - f_Nm1) / dxNm1
                        sub_matrix = (1 / dxNm1) * csr_matrix(
                            ([-1, 1], ([0, 0], [prim_pts - 2, prim_pts - 1])),
                            shape=(1, prim_pts),
                        )
                        additive = pybamm.Scalar(0)

                elif self.extrapolation == "quadratic":
                    a = (2 * dxN + 2 * dxNm1 + dxNm2) / (dxNm1 ** 2 + dxNm1 * dxNm2)
                    b = -(2 * dxN + dxNm1 + dxNm2) / (dxNm1 * dxNm2)
                    c = (2 * dxN + dxNm1) / (dxNm1 * dxNm2 + dxNm2 ** 2)

                    sub_matrix = csr_matrix(
                        (
                            [c, b, a],
                            ([0, 0, 0], [prim_pts - 3, prim_pts - 2, prim_pts - 1]),
                        ),
                        shape=(1, prim_pts),
                    )
                    additive = pybamm.Scalar(0)
                else:
                    raise NotImplementedError

        # Generate full matrix from the submatrix
        # Convert to csr_matrix so that we can take the index (row-slicing), which is
        # not supported by the default kron format
        # Note that this makes column-slicing inefficient, but this should not be an
        # issue
        matrix = csr_matrix(kron(eye(sec_pts), sub_matrix))

        # Return boundary value with domain given by symbol
        boundary_value = pybamm.Matrix(matrix) @ discretised_child
        boundary_value.domain = symbol.domain
        boundary_value.auxiliary_domains = symbol.auxiliary_domains

        additive.domain = symbol.domain
        additive.auxiliary_domains = symbol.auxiliary_domains
        boundary_value += additive

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
        elif left_evaluates_on_edges and not right_evaluates_on_edges:
            disc_right = self.node_to_edge(disc_right)
        # If only right child evaluates on edges, map left child onto edges
        elif right_evaluates_on_edges and not left_evaluates_on_edges:
            disc_left = self.node_to_edge(disc_left)
        # Return new binary operator with appropriate class
        out = bin_op.__class__(disc_left, disc_right)
        return out

    def concatenation(self, disc_children):
        """Discrete concatenation, taking `edge_to_node` for children that evaluate on
        edges.
        See :meth:`pybamm.SpatialMethod.concatenation`
        """
        for idx, child in enumerate(disc_children):
            n_nodes = sum(
                len(mesh.nodes) for mesh in self.mesh.combine_submeshes(*child.domain)
            )
            n_edges = sum(
                len(mesh.edges) for mesh in self.mesh.combine_submeshes(*child.domain)
            )
            child_size = child.size
            if child_size != n_nodes:
                # Average any children that evaluate on the edges (size n_edges) to
                # evaluate on nodes instead, so that concatenation works properly
                if child_size == n_edges:
                    disc_children[idx] = self.edge_to_node(child)
                else:
                    raise pybamm.ShapeError(
                        """
                        child must have size n_nodes (number of nodes in the mesh)
                        or n_edges (number of edges in the mesh)
                        """
                    )
        return pybamm.DomainConcatenation(disc_children, self.mesh)

    def edge_to_node(self, discretised_symbol):
        """
        Convert a discretised symbol evaluated on the cell edges to a discretised symbol
        evaluated on the cell nodes.
        See :meth:`pybamm.FiniteVolume.shift`
        """
        return self.shift(discretised_symbol, "edge to node")

    def node_to_edge(self, discretised_symbol):
        """
        Convert a discretised symbol evaluated on the cell nodes to a discretised symbol
        evaluated on the cell edges.
        See :meth:`pybamm.FiniteVolume.shift`
        """
        return self.shift(discretised_symbol, "node to edge")

    def shift(self, discretised_symbol, shift_key):
        """
        Convert a discretised symbol evaluated at edges/nodes, to a discretised symbol
        evaluated at nodes/edges.
        For now we just take the arithemtic mean, though it may be better to take the
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

        Returns
        -------
        :class:`pybamm.Symbol`
            Averaged symbol. When evaluated, this returns either a scalar or an array of
            shape (n+1,) (if `shift_key = "node to edge"`) or (n,) (if
            `shift_key = "edge to node"`)
        """

        def arithmetic_mean(array):
            """Calculate the arithmetic mean of an array using matrix multiplication"""
            # Create appropriate submesh by combining submeshes in domain
            submesh_list = self.mesh.combine_submeshes(*array.domain)

            # Can just use 1st entry of list to obtain the point etc
            submesh = submesh_list[0]

            # Create 1D matrix using submesh
            n = submesh.npts

            if shift_key == "node to edge":
                sub_matrix_left = csr_matrix(
                    ([1.5, -0.5], ([0, 0], [0, 1])), shape=(1, n)
                )
                sub_matrix_center = diags([0.5, 0.5], [0, 1], shape=(n - 1, n))
                sub_matrix_right = csr_matrix(
                    ([-0.5, 1.5], ([0, 0], [n - 2, n - 1])), shape=(1, n)
                )
                sub_matrix = vstack(
                    [sub_matrix_left, sub_matrix_center, sub_matrix_right]
                )
            elif shift_key == "edge to node":
                sub_matrix = diags([0.5, 0.5], [0, 1], shape=(n, n + 1))
            else:
                raise ValueError("shift key '{}' not recognised".format(shift_key))
            # Second dimension length
            second_dim_len = len(submesh_list)

            # Generate full matrix from the submatrix
            # Convert to csr_matrix so that we can take the index (row-slicing), which
            # is not supported by the default kron format
            # Note that this makes column-slicing inefficient, but this should not be an
            # issue
            matrix = csr_matrix(kron(eye(second_dim_len), sub_matrix))

            return pybamm.Matrix(matrix) @ array

        # If discretised_symbol evaluates to number there is no need to average
        if discretised_symbol.evaluates_to_number():
            out = discretised_symbol
        else:
            out = arithmetic_mean(discretised_symbol)

        return out
