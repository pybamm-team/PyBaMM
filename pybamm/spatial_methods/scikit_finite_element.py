#
# Finite Element discretisation class which uses scikit-fem
#
import pybamm

from scipy.sparse import csr_matrix
import autograd.numpy as np

if not pybamm.have_scikit_fem():
    import skfem


class ScikitFiniteElement(pybamm.SpatialMethod):
    """
    A class which implements the steps specific to the finite element method during
    discretisation. The class uses scikit-fem to discretise the problem to obtain
    the mass and stifnness matrices. At present, this class is only used for
    solving the Poisson problem -grad^2 u = f in the y-z plane (i.e. not the
    through-cell direction).

    For broadcast we follow the default behaviour from SpatialMethod.

    Parameters
    ----------
    mesh : :class:`pybamm.Mesh`
        Contains all the submeshes for discretisation

    **Extends:"": :class:`pybamm.SpatialMethod`
    """

    def __init__(self, mesh):
        if pybamm.have_scikit_fem() is None:
            raise ImportError("scikit-fem is not installed")

        super().__init__(mesh)
        # add npts_for_broadcast to mesh domains for this particular discretisation
        for dom in mesh.keys():
            for i in range(len(mesh[dom])):
                mesh[dom][i].npts_for_broadcast = mesh[dom][i].npts

    def spatial_variable(self, symbol):
        """
        Creates a discretised spatial variable compatible with
        the FiniteElement method.

        Parameters
        -----------
        symbol : :class:`pybamm.SpatialVariable`
            The spatial variable to be discretised.

        Returns
        -------
        :class:`pybamm.Vector`
            Contains the discretised spatial variable
        """
        symbol_mesh = self.mesh
        if symbol.name == "y":
            vector = pybamm.Vector(
                symbol_mesh["current collector"][0].edges["y"], domain=symbol.domain
            )
        elif symbol.name == "z":
            vector = pybamm.Vector(
                symbol_mesh["current collector"][0].edges["z"], domain=symbol.domain
            )
        else:
            raise pybamm.GeometryError(
                "Spatial variable must be 'y' or 'z' not {}".format(symbol.name)
            )
        return vector

    def gradient(self, symbol, discretised_symbol, boundary_conditions):
        """Matrix-vector multiplication to implement the gradient operator.
        See :meth:`pybamm.SpatialMethod.gradient`
        """
        raise NotImplementedError

    def divergence(self, symbol, discretised_symbol, boundary_conditions):
        """Matrix-vector multiplication to implement the divergence operator.
        See :meth:`pybamm.SpatialMethod.divergence`
        """
        raise NotImplementedError

    def laplacian(self, symbol, discretised_symbol, boundary_conditions):
        """Matrix-vector multiplication to implement the laplacian operator.

        Parameters
        ----------
        symbol: :class:`pybamm.Symbol`
            The symbol that we will take the laplacian of.
        discretised_symbol: :class:`pybamm.Symbol`
            The discretised symbol of the correct size
        boundary_conditions : dict
            The boundary conditions of the model
            ({symbol.id: {"left": left bc, "right": right bc}})

        Returns
        -------
        :class: `pybamm.Array`
            Contains the result of acting the discretised gradient on
            the child discretised_symbol
        """
        domain = symbol.domain[0]
        mesh = self.mesh[domain][0]

        stiffness_matrix = self.stiffness_matrix(symbol, boundary_conditions)

        # get boundary conditions and type, here lbc: negative tab, rbc: positive tab
        lbc_value, lbc_type = boundary_conditions[symbol.id]["left"]
        rbc_value, rbc_type = boundary_conditions[symbol.id]["right"]
        # boundary load vector is adjusted to account for boundary conditions below
        boundary_load = pybamm.Vector(np.zeros(mesh.npts))

        # assemble boundary load if Neumann boundary conditions
        if "Neumann" in [lbc_type, rbc_type]:
            # make form for unit load over the boundary
            @skfem.linear_form
            def unit_bc_load_form(v, dv, w):
                return v

        if lbc_type == "Neumann":
            # assemble unit load over tab
            lbc_load = skfem.asm(unit_bc_load_form, mesh.negative_tab_basis)
            # value multiplied by weights
            boundary_load = boundary_load + lbc_value * pybamm.Vector(lbc_load)
        elif lbc_type == "Dirichlet":
            # set Dirichlet value at facets corresponding to tab
            lbc_load = np.zeros(mesh.npts)
            lbc_load[mesh.negative_tab_dofs] = 1
            boundary_load = boundary_load - lbc_value * pybamm.Vector(lbc_load)
        else:
            raise ValueError(
                "boundary condition must be Dirichlet or Neumann, not '{}'".format(
                    lbc_type
                )
            )

        if rbc_type == "Neumann":
            # assemble unit load over tab
            rbc_load = skfem.asm(unit_bc_load_form, mesh.positive_tab_basis)
            # value multiplied by weights
            boundary_load = boundary_load + rbc_value * pybamm.Vector(rbc_load)
        elif rbc_type == "Dirichlet":
            # set Dirichlet value at facets corresponding to tab
            rbc_load = np.zeros(mesh.npts)
            rbc_load[mesh.positive_tab_dofs] = 1
            boundary_load = boundary_load - rbc_value * pybamm.Vector(rbc_load)
        else:
            raise ValueError(
                "boundary condition must be Dirichlet or Neumann, not '{}'".format(
                    rbc_type
                )
            )

        return -stiffness_matrix @ discretised_symbol + boundary_load

    def stiffness_matrix(self, symbol, boundary_conditions):
        """
        Laplacian (stiffness) matrix for finite elements in the appropriate domain.

        Parameters
        ----------
        symbol: :class:`pybamm.Symbol`
            The symbol for which we want to calculate the laplacian matrix
        boundary_conditions : dict
            The boundary conditions of the model
            ({symbol.id: {"left": left bc, "right": right bc}})

        Returns
        -------
        :class:`pybamm.Matrix`
            The (sparse) finite element stiffness matrix for the domain
        """
        # get primary domain mesh
        domain = symbol.domain[0]
        mesh = self.mesh[domain][0]

        # make form for the stiffness
        @skfem.bilinear_form
        def stiffness_form(u, du, v, dv, w):
            return sum(du * dv)

        # assemble the stifnness matrix
        stiffness = skfem.asm(stiffness_form, mesh.basis)

        # get boundary conditions and type, here lbc: negative tab, rbc: positive tab
        _, lbc_type = boundary_conditions[symbol.id]["left"]
        _, rbc_type = boundary_conditions[symbol.id]["right"]

        # adjust matrix for Dirichlet boundary conditions
        if lbc_type == "Dirichlet":
            self.bc_apply(stiffness, mesh.negative_tab_dofs)
        if rbc_type == "Dirichlet":
            self.bc_apply(stiffness, mesh.positive_tab_dofs)

        return pybamm.Matrix(stiffness)

    def integral(self, child, discretised_child):
        """Vector-vector dot product to implement the integral operator.
        See :meth:`pybamm.SpatialMethod.integral`
        """
        # Calculate integration vector
        integration_vector = self.definite_integral_vector(child.domain[0])

        out = integration_vector @ discretised_child

        return out

    def definite_integral_vector(self, domain, vector_type="row"):
        """
        Vector for finite-element implementation of the definite integral over
        the entire domain

        .. math::
            I = \\int_{\Omega}\\!f(s)\\,dx

        for where :math:`\Omega` is the domain.

        Parameters
        ----------
        domain : list
            The domain(s) of integration
        vector_type : str, optional
            Whether to return a row or column vector (defualt is row)

        Returns
        -------
        :class:`pybamm.Matrix`
            The finite element integral vector for the domain
        """
        # get primary domain mesh
        if isinstance(domain, list):
            domain = domain[0]
        mesh = self.mesh[domain][0]

        # make form for the integral
        @skfem.linear_form
        def integral_form(v, dv, w):
            return v

        # assemble
        vector = skfem.asm(integral_form, mesh.basis)

        if vector_type == "row":
            return pybamm.Matrix(vector[np.newaxis, :])
        elif vector_type == "column":
            return pybamm.Matrix(vector[:, np.newaxis])

    def indefinite_integral(self, child, discretised_child):
        """Implementation of the indefinite integral operator. The
        input discretised child must be defined on the internal mesh edges.
        See :meth:`pybamm.SpatialMethod.indefinite_integral`
        """
        raise NotImplementedError

    def boundary_integral(self, child, discretised_child, region=None):
        """Implementation of the boundary integral operator.
        See :meth:`pybamm.SpatialMethod.boundary_integral`
        """
        # Calculate integration vector
        integration_vector = self.boundary_integral_vector(
            child.domain[0], region=region
        )

        out = integration_vector @ discretised_child
        out.domain = []
        return out

    def boundary_integral_vector(self, domain, region=None):
        """A node in the expression tree representing an integral operator over the
        boundary of a domain

        .. math::
            I = \\int_{\\partial a}\\!f(u)\\,du,

        where :math:`\\partial a` is the boundary of the domain, and
        :math:`u\\in\\text{domain boundary}`.

        Parameters
        ----------
        domain : list
            The domain(s) of the variable in the integrand
        region : str, optional
            The region of the boundary over which to integrate. If region is None
            (default) the integration is carried out over the entire boundary. If
            region is `negative tab` or `positive tab` then the integration is only
            carried out over the appropriate part of the boundary corresponding to
            the tab.

        Returns
        -------
        :class:`pybamm.Matrix`
            The finite element integral vector for the domain
        """
        # get primary domain mesh
        if isinstance(domain, list):
            domain = domain[0]
        mesh = self.mesh[domain][0]

        # make form for the boundary integral
        @skfem.linear_form
        def integral_form(v, dv, w):
            return v

        if region is None:
            # assemble over all facets
            integration_vector = skfem.asm(integral_form, mesh.facet_basis)
        elif region == "negative tab":
            # assemble over negative tab facets
            integration_vector = skfem.asm(integral_form, mesh.negative_tab_basis)
        elif region == "positive tab":
            # assemble over positive tab facets
            integration_vector = skfem.asm(integral_form, mesh.positive_tab_basis)

        return pybamm.Matrix(integration_vector[np.newaxis, :])

    def boundary_value_or_flux(self, symbol, discretised_child):
        """
        Returns the average value of the symbol over the negative tab ("left")
        or the positive tab ("right") in the Finite Element Method.

        Overwrites the default :meth:`pybamm.SpatialMethod.boundary_value`
        """

        # Return average value on the negative tab for "left" and positive tab
        # for "right"
        if isinstance(symbol, pybamm.BoundaryValue):
            # get integration_vector
            if symbol.side == "left":
                region = "negative tab"
            elif symbol.side == "right":
                region = "positive tab"
            domain = symbol.children[0].domain[0]
            integration_vector = self.boundary_integral_vector(domain, region=region)

            # divide integration weights by (numerical) tab width to give average value
            boundary_val_vector = integration_vector / (
                integration_vector @ pybamm.Vector(np.ones(integration_vector.shape[1]))
            )

        elif isinstance(symbol, pybamm.BoundaryGradient):
            raise NotImplementedError

        # Return boundary value with domain given by symbol
        boundary_value = boundary_val_vector @ discretised_child

        boundary_value.domain = symbol.domain

        return boundary_value

    def mass_matrix(self, symbol, boundary_conditions):
        """
        Calculates the mass matrix for the finite element method.

        Parameters
        ----------
        symbol: :class:`pybamm.Variable`
            The variable corresponding to the equation for which we are
            calculating the mass matrix.
        boundary_conditions : dict
            The boundary conditions of the model
            ({symbol.id: {"left": left bc, "right": right bc}})

        Returns
        -------
        :class:`pybamm.Matrix`
            The (sparse) mass matrix for the spatial method.
        """
        # get primary domain mesh
        domain = symbol.domain[0]
        mesh = self.mesh[domain][0]

        # create form for mass
        @skfem.bilinear_form
        def mass_form(u, du, v, dv, w):
            return u * v

        # assemble mass matrix
        mass = skfem.asm(mass_form, mesh.basis)

        # get boundary conditions and type, here lbc: negative tab, rbc: positive tab
        if symbol.id in boundary_conditions:
            _, lbc_type = boundary_conditions[symbol.id]["left"]
            _, rbc_type = boundary_conditions[symbol.id]["right"]

            if lbc_type == "Dirichlet":
                # set source terms to zero on boundary by zeroing out mass matrix
                self.bc_apply(mass, mesh.negative_tab_dofs, zero=True)
            if rbc_type == "Dirichlet":
                # set source terms to zero on boundary by zeroing out mass matrix
                self.bc_apply(mass, mesh.positive_tab_dofs, zero=True)

        return pybamm.Matrix(mass)

    def bc_apply(self, M, boundary, zero=False):
        """
        Adjusts the assemled finite element matrices to account for boundary conditons.

        Parameters
        ----------
        M: :class:`scipy.sparse.coo_matrix`
            The assemled finite element matrix to adjust.
        boundary: :class:`numpy.array`
            Array of the indicies which correspond to the boundary.
        zero: bool, optional
            If True, the rows of M given by the indicies in boundary are set to zero.
            If False, the diagonal element is set to one. Defualt is False.
        """
        row = np.arange(0, np.size(boundary))
        if zero:
            data = np.zeros_like(boundary)
        else:
            data = np.ones_like(boundary)
        bc_matrix = csr_matrix(
            (data, (row, boundary)), shape=(np.size(boundary), np.shape(M)[1])
        )
        M[boundary] = bc_matrix
