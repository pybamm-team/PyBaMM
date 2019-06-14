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
        return pybamm.Vector(symbol_mesh[0].npts, domain=symbol.domain)

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
        boundary_load = pybamm.Vector(np.zeros(mesh.npts))

        # assemble boundary load if Neumann boundary conditions
        if "Neumann" in [lbc_type, rbc_type]:
            # make form for unit load at the boundary
            @skfem.linear_form
            def unit_bc_load_form(v, dv, w):
                return v
            # assemble form
            unit_load = skfem.asm(unit_bc_load_form, mesh.facet_basis)

        if lbc_type == "Neumann":
            # value multiplied by weights
            lbc_load = np.zeros(mesh.npts)
            lbc_load[mesh.negative_tab] = unit_load[mesh.negative_tab]
            boundary_load = boundary_load + lbc_value * pybamm.Vector(lbc_load)
        elif lbc_type == "Dirichlet":
            lbc_load = np.zeros(mesh.npts)
            lbc_load[mesh.negative_tab] = 1
            boundary_load = boundary_load + lbc_value * pybamm.Vector(lbc_load)
        else:
            raise ValueError(
                "boundary condition must be Dirichlet or Neumann, not '{}'".format(
                    lbc_type
                )
            )

        if rbc_type == "Neumann":
            # value multiplied by weights
            rbc_load = np.zeros(mesh.npts)
            rbc_load[mesh.positive_tab] = unit_load[mesh.positive_tab]
            boundary_load = boundary_load + rbc_value * pybamm.Vector(rbc_load)
        elif rbc_type == "Dirichlet":
            rbc_load = np.zeros(mesh.npts)
            rbc_load[mesh.positive_tab] = 1
            boundary_load = boundary_load + rbc_value * pybamm.Vector(rbc_load)
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
            self.bc_apply(stiffness, mesh.negative_tab)
        if rbc_type == "Dirichlet":
            self.bc_apply(stiffness, mesh.positive_tab)

        return pybamm.Matrix(stiffness)

    def integral(self, domain, symbol, discretised_symbol):
        """Vector-vector dot product to implement the integral operator.
        See :meth:`pybamm.SpatialMethod.integral`
        """

        # Calculate integration vector
        integration_vector = self.definite_integral_vector(domain[0])

        out = integration_vector @ discretised_symbol
        out.domain = []
        return out

    def definite_integral_vector(self, domain):
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

        Returns
        -------
        :class:`pybamm.Vector`
            The finite volume integral vector for the domain
        """
        # get primary domain mesh
        mesh = self.mesh[domain][0]

        # make form for the integral
        @skfem.linear_form
        def integral_form(v, dv, w):
            return v

        # assemble
        vector = skfem.asm(integral_form, mesh.basis)
        return pybamm.Matrix(vector[np.newaxis, :])

    def indefinite_integral(self, domain, symbol, discretised_symbol):
        """Implementation of the indefinite integral operator. The
        input discretised symbol must be defined on the internal mesh edges.
        See :meth:`pybamm.SpatialMethod.indefinite_integral`
        """
        raise NotImplementedError

    def boundary_value_or_flux(self, symbol, discretised_child):
        """
        Uses linear extrapolation to get the boundary value or flux of a variable in the
        Finite Element Method.

        See :meth:`pybamm.SpatialMethod.boundary_value`
        """
        raise NotImplementedError

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
        _, lbc_type = boundary_conditions[symbol.id]["left"]
        _, rbc_type = boundary_conditions[symbol.id]["right"]

        if lbc_type == "Dirichlet":
            # set source terms to zero on boundary by zeroing out mass matrix
            self.bc_apply(mass, mesh.negative_tab, zero=True)
        if rbc_type == "Dirichlet":
            # set source terms to zero on boundary by zeroing out mass matrix
            self.bc_apply(mass, mesh.positive_tab, zero=True)

        return pybamm.Matrix(mass)

    def bc_apply(self, M, boundary, zero=False):
        """
        Adjusts the assemled finite element matrices to account for bpundary conditons.

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
