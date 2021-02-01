import pybamm

import numpy as np

from scipy.sparse import diags, eye, kron, csr_matrix, lil_matrix, coo_matrix, vstack


class SpectralVolume(pybamm.FiniteVolume):
    """
    A class which implements the steps specific to the Spectral Volume
    discretisation. It is implemented in such a way that it is very
    similar to FiniteVolume; that comes at the cost that it is only
    compatible with the SpectralVolume1DSubMesh (which is a certain
    subdivision of any 1D mesh, so it shouldn't be a problem).

    For broadcast and mass_matrix, we follow the default behaviour from
    SpatialMethod. For spatial_variable, preprocess_external_variables,
    divergence, divergence_matrix, laplacian, integral,
    definite_integral_matrix, indefinite_integral,
    indefinite_integral_matrix, indefinite_integral_matrix_nodes,
    indefinite_integral_matrix_edges, delta_function
    we follow the behaviour from FiniteVolume. This is possible since
    the node values are integral averages with Spectral Volume, just
    as with Finite Volume. delta_function assigns the integral value
    to a CV instead of a SV this way, but that doesn't matter too much.
    Additional methods that are inherited by FiniteVolume which
    technically are not suitable for Spectral Volume are
    boundary_value_or_flux, process_binary_operators, concatenation,
    node_to_edge, edge_to_node and shift. While node_to_edge (as well as
    boundary_value_or_flux and process_binary_operators)
    could utilize the reconstruction approach of Spectral Volume, the
    inverse edge_to_node would still have to fall back to the Finite
    Volume behaviour. So these are simply inherited for consistency.
    boundary_value_or_flux might not benefit from the reconstruction
    approach at all, as it seems to only preprocess symbols.

    Parameters
    ----------
    mesh : :class:`pybamm.Mesh`
        Contains all the submeshes for discretisation

    **Extends:"": :class:`pybamm.FiniteVolume`
    """

    def __init__(self, options=None, order=2):
        self.order = order
        super().__init__(options)
        pybamm.citations.register("Wang2002")

    def chebyshev_collocation_points(self, noe, a=-1.0, b=1.0):
        """
        Calculates Chebyshev collocation points in descending order.

        Parameters
        ----------
        noe: integer
            The number of the collocation points. "number of edges"
        a: float
            Left end of the interval on which the Chebyshev collocation
            points are constructed. Default is -1.
        b: float
            Right end of the interval on which the Chebyshev collocation
            points are constructed. Default is 1.

        Returns
        -------
        :class:`numpy.array`
        Chebyshev collocation points on [a,b].
        """

        return a + 0.5 * (b - a) * (
            1
            + np.sin(
                np.pi
                * np.array([(noe - 1 - 2 * i) / (2 * noe - 2) for i in range(noe)])
            )
        )

    def cv_boundary_reconstruction_sub_matrix(self):
        """
        Coefficients for reconstruction of a function through averages.
        The resulting matrix is scale-invariant [2]_.

        Parameters
        ----------

        Returns
        -------

        References
        ----------
        .. [2] Z. J. Wang.
               “Spectral (Finite) Volume Method for Conservation Laws
               on Unstructured Grids”.
               Journal of Computational Physics,
               178:210–251, 2002
        """

        # While Spectral Volume in general may use any point
        # distribution for CVs, the Chebyshev nodes are the most stable.
        # The differentiation matrices are only implemented for those.
        edges = np.flip(self.chebyshev_collocation_points(self.order + 1))

        # Nomenclature in the reference:
        # c[j,l] are the coefficients from the reference.
        # The index of the CV boundaries j ranges from 0 to self.order.
        # The index of the CVs themselves l ranges from 1 to self.order.
        # l ranges from 0 to self.order - 1 here.
        c = np.empty([self.order + 1, self.order])
        # h[l] are the lengths of the CVs.
        h = [edges[i + 1] - edges[i] for i in range(self.order)]

        # Optimised derivative of the "Lagrange polynomial denominator".
        # It is equivalent to d_omega_d_x(x) at x = x_{j+1/2}.
        def d_omega_d_x(j):
            return np.prod(
                edges[j] - edges,
                where=[True] * j + [False] + [True] * (len(edges) - 1 - j),
            )

        for j in range(self.order + 1):
            for ell in range(self.order):
                c[j, ell] = h[ell] * np.sum(
                    [
                        1.0
                        / d_omega_d_x(r)
                        * np.sum(
                            [
                                np.prod(
                                    edges[j] - edges,
                                    where=[
                                        q != r and q != m for q in range(self.order + 1)
                                    ],
                                )
                                for m in range(self.order + 1)
                            ],
                            where=[m != r for m in range(self.order + 1)],
                        )
                        for r in range(ell + 1, self.order + 1)
                    ]
                )

        return c

    def cv_boundary_reconstruction_matrix(self, domain, auxiliary_domains):
        """
        "Broadcasts" the basic edge value reconstruction matrix to the
        actual shape of the discretised symbols. Note that the product
        of this and a discretised symbol is a vector which represents
        duplicate values for all inner SV edges. These are the
        reconstructed values from both sides.

        Parameters
        ----------
        domain : list
            The domain(s) in which to compute the gradient matrix
        auxiliary_domains : dict
            The auxiliary domains in which to compute the gradient
            matrix

        Returns
        -------
        :class:`pybamm.Matrix`
            The (sparse) CV reconstruction matrix for the domain
        """
        # Create appropriate submesh by combining submeshes in domain
        submesh = self.mesh.combine_submeshes(*domain)

        # Obtain the basic reconstruction matrix.
        recon_sub_matrix = self.cv_boundary_reconstruction_sub_matrix()

        # Create 1D matrix using submesh
        # n is the number of SVs, submesh.npts is the number of CVs
        n = submesh.npts // self.order
        sub_matrix = csr_matrix(kron(eye(n), recon_sub_matrix))

        # number of repeats
        second_dim_repeats = self._get_auxiliary_domain_repeats(auxiliary_domains)

        # generate full matrix from the submatrix
        # Convert to csr_matrix so that we can take the index
        # (row-slicing), which is not supported by the default kron
        # format. Note that this makes column-slicing inefficient,
        # but this should not be an issue.
        matrix = csr_matrix(kron(eye(second_dim_repeats), sub_matrix))

        return pybamm.Matrix(matrix)

    def chebyshev_differentiation_matrices(self, noe, dod):
        """
        Chebyshev differentiation matrices [1]_.

        Parameters
        ----------
        noe: integer
            The number of the collocation points. "number of edges"
        dod: integer
            The maximum order of differentiation for which a
            differentiation matrix shall be calculated. Note that it has
            to be smaller than 'noe'. "degrees of differentiation"

        Returns
        -------
        list(:class:`numpy.array`)
            The differentiation matrices in ascending order of
            differentiation order. With exact arithmetic, the diff.
            matrix of order p would just be the pth matrix power of
            the diff. matrix of order 1. This method computes the higher
            orders in a more numerically stable way.

        References
        ----------
        .. [1] Richard Baltensperger and Manfred R. Trummer.
               “Spectral Differencing With A Twist”.
               Society for Industrial and Applied Mathematics,
               24(5):1465–1487, 2003
        """
        if dod >= noe:
            raise ValueError(
                "Too many degrees of differentiation. At most "
                + str(noe - 1)
                + " are possible for "
                + str(noe)
                + " edges."
            )

        edges = self.chebyshev_collocation_points(noe)

        # These matrices tend to be dense, thus numpy arrays are used.
        prefactors = np.array(
            [[(i - j + 1) % 2 - (i - j) % 2 for j in range(noe)] for i in range(noe)]
        )
        prefactors = (prefactors * np.array([2] + [1 for i in range(noe - 2)] + [2])).T
        prefactors = prefactors * np.array([0.5] + [1 for i in range(noe - 2)] + [0.5])

        inverse_difference = np.array(
            [
                [1.0 / (edges[i] - edges[j]) for j in range(i)]
                + [0.0]
                + [1.0 / (edges[i] - edges[j]) for j in range(i + 1, noe)]
                for i in range(noe)
            ]
        )

        differentiation_matrices = []
        # This matrix changes in each of the following iterations.
        temp_diff = np.eye(noe)

        # The calculation here makes extensive use of the element-wise
        # multiplication of numpy.arrays. The * are intentionally not @!
        for p in range(dod):
            temp = (prefactors.T * np.diag(temp_diff)).T - temp_diff
            temp_diff = (p + 1) * inverse_difference * temp
            # Negative sum trick: the rows of the exact matrices sum to
            # zero. The diagonal gets less accurate with this, but the
            # approximation of the differential will be better overall.
            for i in range(noe):
                temp_diff[i, i] = -np.sum(np.delete(temp_diff[i], i))
            differentiation_matrices.append(temp_diff.copy())

        return differentiation_matrices

    def gradient(self, symbol, discretised_symbol, boundary_conditions):
        """Matrix-vector multiplication to implement the gradient
        operator. See :meth:`pybamm.SpatialMethod.gradient`
        """
        # Discretise symbol
        domain = symbol.domain

        # Reconstruct edge values from node values.
        reconstructed_symbol = (
            self.cv_boundary_reconstruction_matrix(domain, symbol.auxiliary_domains)
            @ discretised_symbol
        )

        # Add Dirichlet boundary conditions, if defined
        if symbol.id in boundary_conditions:
            bcs = boundary_conditions[symbol.id]
            if any(bc[1] == "Dirichlet" for bc in bcs.values()):
                # add ghost nodes and update domain
                reconstructed_symbol = self.replace_dirichlet_values(
                    symbol, reconstructed_symbol, bcs
                )

        # note in 1D spherical grad and normal grad are the same
        gradient_matrix = self.gradient_matrix(domain, symbol.auxiliary_domains)
        penalty_matrix = self.penalty_matrix(domain, symbol.auxiliary_domains)

        # Multiply by gradient matrix
        out = (
            gradient_matrix @ reconstructed_symbol + penalty_matrix @ discretised_symbol
        )

        # Add Neumann boundary conditions, if defined
        if symbol.id in boundary_conditions:
            bcs = boundary_conditions[symbol.id]
            if any(bc[1] == "Neumann" for bc in bcs.values()):
                out = self.replace_neumann_values(symbol, out, bcs)

        return out

    def gradient_matrix(self, domain, auxiliary_domains):
        """
        Gradient matrix for Spectral Volume in the appropriate domain.
        Note that it contains the averaging of the duplicate SV edge
        gradient values, such that the product of it and a reconstructed
        discretised symbol simply represents CV edge values.
        On its own, it only works on non-concatenated domains, since
        only then the boundary conditions ensure correct behaviour.
        More generally, it only works if gradients are a result of
        boundary conditions rather than continuity conditions.
        For example, two adjacent SVs with gradient zero in each of them
        but with different variable values will have zero gradient
        between them. This is fixed with "penalty_matrix".

        Parameters
        ----------
        domain : list
            The domain(s) in which to compute the gradient matrix
        auxiliary_domains : dict
            The auxiliary domains in which to compute the gradient
            matrix

        Returns
        -------
        :class:`pybamm.Matrix`
            The (sparse) Spectral Volume gradient matrix for the domain
        """
        # Create appropriate submesh by combining submeshes in domain
        submesh = self.mesh.combine_submeshes(*domain)

        # Obtain the Chebyshev differentiation matrix.
        # Flip it, since it is defined for the Chebyshev
        # collocation points in descending order.
        chebdiff = np.flip(
            self.chebyshev_differentiation_matrices(self.order + 1, 1)[0]
        )

        # Create 1D matrix using submesh
        # submesh.npts is the number of CVs and n the number of SVs
        n = submesh.npts // self.order
        d = self.order
        # Compute the lengths of the Spectral Volumes.
        d_sv_edges = np.array(
            [
                np.sum(submesh.d_edges[d * i : d * i + d])
                for i in range(len(submesh.d_edges) // d)
            ]
        )
        # The 2 scales from [-1,1] (Chebyshev default) to [0,1].
        # e = 2 / submesh.d_sv_edges
        e = 2 / d_sv_edges
        # This factor scales the contribution of the reconstructed
        # gradient to the finite difference at the SV edges.
        # 0.0 is the value that makes it work with the "penalty_matrix".
        # 0.5 is the value that makes it work without it, but remember,
        # that effectively removes any implicit continuity conditions.
        f = 0.0
        # Here, the differentials are scaled to the SV.
        sub_matrix_raw = csr_matrix(kron(diags(e), chebdiff))
        if n == 1:
            sub_matrix = sub_matrix_raw
        else:
            sub_matrix = lil_matrix((n * d + 1, n * (d + 1)))
            sub_matrix[:d, : d + 1] = sub_matrix_raw[:d, : d + 1]
            sub_matrix[d, : d + 1] = f * sub_matrix_raw[d, : d + 1]
            # for loop of shame (optimisation potential via vectorisation)
            for i in range(1, n - 1):
                sub_matrix[i * d, i * (d + 1) : (i + 1) * (d + 1)] = (
                    f * sub_matrix_raw[i * (d + 1), i * (d + 1) : (i + 1) * (d + 1)]
                )
                sub_matrix[
                    i * d + 1 : (i + 1) * d, i * (d + 1) : (i + 1) * (d + 1)
                ] = sub_matrix_raw[
                    i * (d + 1) + 1 : (i + 1) * (d + 1) - 1,
                    i * (d + 1) : (i + 1) * (d + 1),
                ]
                sub_matrix[(i + 1) * d, i * (d + 1) : (i + 1) * (d + 1)] = (
                    f * sub_matrix_raw[i * (d + 1) + d, i * (d + 1) : (i + 1) * (d + 1)]
                )
            sub_matrix[-d - 1, -d - 1 :] = f * sub_matrix_raw[-d - 1, -d - 1 :]
            sub_matrix[-d:, -d - 1 :] = sub_matrix_raw[-d:, -d - 1 :]

        # number of repeats
        second_dim_repeats = self._get_auxiliary_domain_repeats(auxiliary_domains)

        # generate full matrix from the submatrix
        # Convert to csr_matrix so that we can take the index
        # (row-slicing), which is not supported by the default kron
        # format. Note that this makes column-slicing inefficient,
        # but this should not be an issue.
        matrix = csr_matrix(kron(eye(second_dim_repeats), sub_matrix))

        return pybamm.Matrix(matrix)

    def penalty_matrix(self, domain, auxiliary_domains):
        """
        Penalty matrix for Spectral Volume in the appropriate domain.
        This works the same as the "gradient_matrix" of FiniteVolume
        does, just between SVs and not between CVs. Think of it as a
        continuity penalty.

        Parameters
        ----------
        domain : list
            The domain(s) in which to compute the gradient matrix
        auxiliary_domains : dict
            The auxiliary domains in which to compute the gradient
            matrix

        Returns
        -------
        :class:`pybamm.Matrix`
            The (sparse) Spectral Volume penalty matrix for the domain
        """
        # Create appropriate submesh by combining submeshes in domain
        submesh = self.mesh.combine_submeshes(*domain)

        # Create 1D matrix using submesh
        n = submesh.npts
        d = self.order
        e = np.zeros(n - 1)
        e[d - 1 :: d] = 1 / submesh.d_nodes[d - 1 :: d]
        sub_matrix = vstack(
            [np.zeros(n), diags([-e, e], [0, 1], shape=(n - 1, n)), np.zeros(n)]
        )

        # number of repeats
        second_dim_repeats = self._get_auxiliary_domain_repeats(auxiliary_domains)

        # generate full matrix from the submatrix
        # Convert to csr_matrix so that we can take the index
        # (row-slicing), which is not supported by the default kron
        # format. Note that this makes column-slicing inefficient, but
        # this should not be an issue.
        matrix = csr_matrix(kron(eye(second_dim_repeats), sub_matrix))

        return pybamm.Matrix(matrix)

    # def spectral_volume_internal_neumann_condition(
    #    self, left_symbol_disc, right_symbol_disc, left_mesh, right_mesh
    # ):
    #    """
    #    A method to find the internal neumann conditions between two
    #    symbols on adjacent subdomains. This method is never called,
    #    it's just here to show how a reconstructed gradient-based
    #    internal neumann_condition would look like.
    #    Parameters
    #    ----------
    #    left_symbol_disc : :class:`pybamm.Symbol`
    #        The discretised symbol on the left subdomain
    #    right_symbol_disc : :class:`pybamm.Symbol`
    #        The discretised symbol on the right subdomain
    #    left_mesh : list
    #        The mesh on the left subdomain
    #    right_mesh : list
    #        The mesh on the right subdomain
    #    """
    #
    #    second_dim_repeats = self._get_auxiliary_domain_repeats(
    #        left_symbol_disc.domains
    #    )
    #
    #    if second_dim_repeats != self._get_auxiliary_domain_repeats(
    #        right_symbol_disc.domains
    #    ):
    #        raise pybamm.DomainError(
    #            "Number of secondary points in subdomains do not match"
    #        )
    #
    #    # Use the Spectral Volume reconstruction and differentiation.
    #    left_reconstruction_matrix = self.cv_boundary_reconstruction_matrix(
    #        left_symbol_disc.domain,
    #        left_symbol_disc.auxiliary_domains
    #    )
    #    left_gradient_matrix = self.gradient_matrix(
    #        left_symbol_disc.domain,
    #        left_symbol_disc.auxiliary_domains
    #    ).entries[-1]
    #    left_matrix = left_gradient_matrix @ left_reconstruction_matrix
    #
    #    right_reconstruction_matrix = self.cv_boundary_reconstruction_matrix(
    #        right_symbol_disc.domain,
    #        right_symbol_disc.auxiliary_domains
    #    )
    #    right_gradient_matrix = self.gradient_matrix(
    #        right_symbol_disc.domain,
    #        right_symbol_disc.auxiliary_domains
    #    ).entries[0]
    #    right_matrix = right_gradient_matrix @ right_reconstruction_matrix
    #
    #    # Remove domains to avoid clash
    #    left_domain = left_symbol_disc.domain
    #    right_domain = right_symbol_disc.domain
    #    left_auxiliary_domains = left_symbol_disc.auxiliary_domains
    #    right_auxiliary_domains = right_symbol_disc.auxiliary_domains
    #    left_symbol_disc.clear_domains()
    #    right_symbol_disc.clear_domains()
    #
    #    # Spectral Volume derivative (i.e., the mean of the two
    #    # reconstructed gradients from each side)
    #    # Note that this is the version without "penalty_matrix".
    #    dy_dx = 0.5 * (right_matrix @ right_symbol_disc
    #                   + left_matrix @ left_symbol_disc)
    #
    #    # Change domains back
    #    left_symbol_disc.domain = left_domain
    #    right_symbol_disc.domain = right_domain
    #    left_symbol_disc.auxiliary_domains = left_auxiliary_domains
    #    right_symbol_disc.auxiliary_domains = right_auxiliary_domains
    #
    #    return dy_dx

    def replace_dirichlet_values(self, symbol, discretised_symbol, bcs):
        """
        Replace the reconstructed value at Dirichlet boundaries with the
        boundary condition.

        Parameters
        ----------
        symbol : :class:`pybamm.SpatialVariable`
            The variable to be discretised
        discretised_symbol : :class:`pybamm.Vector`
            Contains the discretised variable
        bcs : dict of tuples (:class:`pybamm.Scalar`, str)
            Dictionary (with keys "left" and "right") of boundary
            conditions. Each boundary condition consists of a value and
            a flag indicating its type (e.g. "Dirichlet")

        Returns
        -------
        :class:`pybamm.Symbol`
            `Matrix @ discretised_symbol + bcs_vector`. When evaluated,
            this gives the discretised_symbol, with its boundary values
            replaced by the Dirichlet boundary conditions.
        """
        # get relevant grid points
        domain = symbol.domain
        submesh = self.mesh.combine_submeshes(*domain)

        # Prepare sizes
        n = (submesh.npts // self.order) * (self.order + 1)
        second_dim_repeats = self._get_auxiliary_domain_repeats(symbol.domains)

        lbc_value, lbc_type = bcs["left"]
        rbc_value, rbc_type = bcs["right"]

        # write boundary values into vectors of according shape
        if lbc_type == "Dirichlet":
            lbc_sub_matrix = coo_matrix(([1], ([0], [0])), shape=(n, 1))
            lbc_matrix = csr_matrix(kron(eye(second_dim_repeats), lbc_sub_matrix))
            if lbc_value.evaluates_to_number():
                left_bc = lbc_value * pybamm.Vector(np.ones(second_dim_repeats))
            else:
                left_bc = lbc_value
            lbc_vector = pybamm.Matrix(lbc_matrix) @ left_bc
        elif lbc_type == "Neumann":
            lbc_vector = pybamm.Vector(np.zeros(n * second_dim_repeats))
        else:
            raise ValueError(
                "boundary condition must be Dirichlet or Neumann, "
                "not '{}'".format(lbc_type)
            )

        if rbc_type == "Dirichlet":
            rbc_sub_matrix = coo_matrix(([1], ([n - 1], [0])), shape=(n, 1))
            rbc_matrix = csr_matrix(kron(eye(second_dim_repeats), rbc_sub_matrix))
            if rbc_value.evaluates_to_number():
                right_bc = rbc_value * pybamm.Vector(np.ones(second_dim_repeats))
            else:
                right_bc = rbc_value
            rbc_vector = pybamm.Matrix(rbc_matrix) @ right_bc
        elif rbc_type == "Neumann":
            rbc_vector = pybamm.Vector(np.zeros(n * second_dim_repeats))
        else:
            raise ValueError(
                "boundary condition must be Dirichlet or Neumann, "
                "not '{}'".format(rbc_type)
            )

        bcs_vector = lbc_vector + rbc_vector
        # Need to match the domain. E.g. in the case of the boundary
        # condition on the particle, the gradient has domain particle
        # but the bcs_vector has domain electrode, since it is a
        # function of the macroscopic variables
        bcs_vector.copy_domains(discretised_symbol)

        # Make matrix which makes "gaps" at the boundaries into which
        # the known Dirichlet values will be added. If the boundary
        # condition is not Dirichlet, it acts as identity.
        sub_matrix = diags(
            [int(lbc_type != "Dirichlet")]
            + [1 for i in range(n - 2)]
            + [int(rbc_type != "Dirichlet")]
        )

        # repeat matrix for secondary dimensions
        # Convert to csr_matrix so that we can take the index
        # (row-slicing), which is not supported by the default kron
        # format. Note that this makes column-slicing inefficient, but
        # this should not be an issue.
        matrix = csr_matrix(kron(eye(second_dim_repeats), sub_matrix))

        new_symbol = pybamm.Matrix(matrix) @ discretised_symbol + bcs_vector

        return new_symbol

    def replace_neumann_values(self, symbol, discretised_gradient, bcs):
        """
        Replace the known values of the gradient from Neumann boundary
        conditions into the discretised gradient.

        Parameters
        ----------
        symbol : :class:`pybamm.SpatialVariable`
            The variable to be discretised
        discretised_gradient : :class:`pybamm.Vector`
            Contains the discretised gradient of symbol
        bcs : dict of tuples (:class:`pybamm.Scalar`, str)
            Dictionary (with keys "left" and "right") of boundary
            conditions. Each boundary condition consists of a value and
            a flag indicating its type (e.g. "Dirichlet")

        Returns
        -------
        :class:`pybamm.Symbol`
            `Matrix @ discretised_gradient + bcs_vector`. When
            evaluated, this gives the discretised_gradient, with its
            boundary values replaced by the Neumann boundary conditions.
        """
        # get relevant grid points
        domain = symbol.domain
        submesh = self.mesh.combine_submeshes(*domain)

        # Prepare sizes
        n = submesh.npts + 1
        second_dim_repeats = self._get_auxiliary_domain_repeats(symbol.domains)

        lbc_value, lbc_type = bcs["left"]
        rbc_value, rbc_type = bcs["right"]

        # Add any values from Neumann boundary conditions to the bcs vector
        if lbc_type == "Neumann":
            lbc_sub_matrix = coo_matrix(([1], ([0], [0])), shape=(n, 1))
            lbc_matrix = csr_matrix(kron(eye(second_dim_repeats), lbc_sub_matrix))
            if lbc_value.evaluates_to_number():
                left_bc = lbc_value * pybamm.Vector(np.ones(second_dim_repeats))
            else:
                left_bc = lbc_value
            lbc_vector = pybamm.Matrix(lbc_matrix) @ left_bc
        elif lbc_type == "Dirichlet":
            lbc_vector = pybamm.Vector(np.zeros(n * second_dim_repeats))
        else:
            raise ValueError(
                "boundary condition must be Dirichlet or Neumann, "
                "not '{}'".format(lbc_type)
            )

        if rbc_type == "Neumann":
            rbc_sub_matrix = coo_matrix(([1], ([n - 1], [0])), shape=(n, 1))
            rbc_matrix = csr_matrix(kron(eye(second_dim_repeats), rbc_sub_matrix))
            if rbc_value.evaluates_to_number():
                right_bc = rbc_value * pybamm.Vector(np.ones(second_dim_repeats))
            else:
                right_bc = rbc_value
            rbc_vector = pybamm.Matrix(rbc_matrix) @ right_bc
        elif rbc_type == "Dirichlet":
            rbc_vector = pybamm.Vector(np.zeros(n * second_dim_repeats))
        else:
            raise ValueError(
                "boundary condition must be Dirichlet or Neumann, "
                "not '{}'".format(rbc_type)
            )

        bcs_vector = lbc_vector + rbc_vector
        # Need to match the domain. E.g. in the case of the boundary
        # condition on the particle, the gradient has domain particle
        # but the bcs_vector has domain electrode, since it is a
        # function of the macroscopic variables
        bcs_vector.copy_domains(discretised_gradient)

        # Make matrix which makes "gaps" at the boundaries into which
        # the known Neumann values will be added. If the boundary
        # condition is not Neumann, it acts as identity.
        sub_matrix = diags(
            [int(lbc_type != "Neumann")]
            + [1 for i in range(n - 2)]
            + [int(rbc_type != "Neumann")]
        )

        # repeat matrix for secondary dimensions
        # Convert to csr_matrix so that we can take the index
        # (row-slicing), which is not supported by the default kron
        # format. Note that this makes column-slicing inefficient, but
        # this should not be an issue.
        matrix = csr_matrix(kron(eye(second_dim_repeats), sub_matrix))

        new_gradient = pybamm.Matrix(matrix) @ discretised_gradient + bcs_vector

        return new_gradient
