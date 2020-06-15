#
# Finite Element discretisation class which uses scikit-fem
#
import pybamm

from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import inv
import numpy as np
import skfem


class ScikitFiniteElement(pybamm.SpatialMethod):
    """
    A class which implements the steps specific to the finite element method during
    discretisation. The class uses scikit-fem to discretise the problem to obtain
    the mass and stiffness matrices. At present, this class is only used for
    solving the Poisson problem -grad^2 u = f in the y-z plane (i.e. not the
    through-cell direction).

    For broadcast we follow the default behaviour from SpatialMethod.

    Parameters
    ----------
    mesh : :class:`pybamm.Mesh`
        Contains all the submeshes for discretisation

    **Extends:"": :class:`pybamm.SpatialMethod`
    """

    def __init__(self, options=None):
        super().__init__(options)
        pybamm.citations.register("scikit-fem")

    def build(self, mesh):
        super().build(mesh)
        # add npts_for_broadcast to mesh domains for this particular discretisation
        for dom in mesh.keys():
            mesh[dom].npts_for_broadcast_to_nodes = mesh[dom].npts

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
                symbol_mesh["current collector"].coordinates[0, :][:, np.newaxis]
            )
        elif symbol.name == "z":
            vector = pybamm.Vector(
                symbol_mesh["current collector"].coordinates[1, :][:, np.newaxis]
            )
        else:
            raise pybamm.GeometryError(
                "Spatial variable must be 'y' or 'z' not {}".format(symbol.name)
            )
        return vector

    def gradient(self, symbol, discretised_symbol, boundary_conditions):
        """Matrix-vector multiplication to implement the gradient operator. The
        gradient w of the function u is approximated by the finite element method
        using the same function space as u, i.e. we solve w = grad(u), which
        corresponds to the weak form w*v*dx = grad(u)*v*dx, where v is a suitable
        test function.

        Parameters
        ----------
        symbol: :class:`pybamm.Symbol`
            The symbol that we will take the laplacian of.
        discretised_symbol: :class:`pybamm.Symbol`
            The discretised symbol of the correct size
        boundary_conditions : dict
            The boundary conditions of the model
            ({symbol.id: {"negative tab": neg. tab bc, "positive tab": pos. tab bc}})

        Returns
        -------
        :class: `pybamm.Concatenation`
            A concatenation that contains the result of acting the discretised
            gradient on the child discretised_symbol. The first column corresponds
            to the y-component of the gradient and the second column corresponds
            to the z component of the gradient.
        """
        domain = symbol.domain[0]
        mesh = self.mesh[domain]

        # get gradient matrix
        grad_y_matrix, grad_z_matrix = self.gradient_matrix(symbol, boundary_conditions)

        # assemble mass matrix (there is no need to zero out entries here, since
        # boundary conditions are already accounted for in the governing pde
        # for the symbol we are taking the gradient of. we just want to get the
        # correct weights)
        @skfem.bilinear_form
        def mass_form(u, du, v, dv, w):
            return u * v

        mass = skfem.asm(mass_form, mesh.basis)
        # we need the inverse
        mass_inv = pybamm.Matrix(inv(csc_matrix(mass)))

        # compute gradient
        grad_y = mass_inv @ (grad_y_matrix @ discretised_symbol)
        grad_z = mass_inv @ (grad_z_matrix @ discretised_symbol)

        # create concatenation
        grad = pybamm.Concatenation(
            grad_y, grad_z, check_domain=False, concat_fun=np.hstack
        )
        grad.domain = domain

        return grad

    def gradient_squared(self, symbol, discretised_symbol, boundary_conditions):
        """Multiplication to implement the inner product of the gradient operator
        with itself. See :meth:`pybamm.SpatialMethod.gradient_squared`
        """
        grad = self.gradient(symbol, discretised_symbol, boundary_conditions)
        grad_y, grad_z = grad.orphans
        return grad_y ** 2 + grad_z ** 2

    def gradient_matrix(self, symbol, boundary_conditions):
        """
        Gradient matrix for finite elements in the appropriate domain.

        Parameters
        ----------
        symbol: :class:`pybamm.Symbol`
            The symbol for which we want to calculate the gradient matrix
        boundary_conditions : dict
            The boundary conditions of the model
            ({symbol.id: {"negative tab": neg. tab bc, "positive tab": pos. tab bc}})

        Returns
        -------
        :class:`pybamm.Matrix`
            The (sparse) finite element gradient matrix for the domain
        """
        # get primary domain mesh
        domain = symbol.domain[0]
        mesh = self.mesh[domain]

        # make form for the gradient in the y direction
        @skfem.bilinear_form
        def gradient_dy(u, du, v, dv, w):
            return du[0] * v[0]

        # make form for the gradient in the z direction
        @skfem.bilinear_form
        def gradient_dz(u, du, v, dv, w):
            return du[1] * v[1]

        # assemble the matrices
        grad_y = skfem.asm(gradient_dy, mesh.basis)
        grad_z = skfem.asm(gradient_dz, mesh.basis)

        return pybamm.Matrix(grad_y), pybamm.Matrix(grad_z)

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
            ({symbol.id: {"negative tab": neg. tab bc, "positive tab": pos. tab bc}})

        Returns
        -------
        :class: `pybamm.Array`
            Contains the result of acting the discretised gradient on
            the child discretised_symbol
        """
        domain = symbol.domain[0]
        mesh = self.mesh[domain]

        stiffness_matrix = self.stiffness_matrix(symbol, boundary_conditions)

        # get boundary conditions and type
        neg_bc_value, neg_bc_type = boundary_conditions[symbol.id]["negative tab"]
        pos_bc_value, pos_bc_type = boundary_conditions[symbol.id]["positive tab"]
        # boundary load vector is adjusted to account for boundary conditions below
        boundary_load = pybamm.Vector(np.zeros(mesh.npts))

        # assemble boundary load if Neumann boundary conditions
        if "Neumann" in [neg_bc_type, pos_bc_type]:
            # make form for unit load over the boundary
            @skfem.linear_form
            def unit_bc_load_form(v, dv, w):
                return v

        if neg_bc_type == "Neumann":
            # assemble unit load over tab
            neg_bc_load = skfem.asm(unit_bc_load_form, mesh.negative_tab_basis)
            # value multiplied by weights
            boundary_load = boundary_load + neg_bc_value * pybamm.Vector(neg_bc_load)
        elif neg_bc_type == "Dirichlet":
            # set Dirichlet value at facets corresponding to tab
            neg_bc_load = np.zeros(mesh.npts)
            neg_bc_load[mesh.negative_tab_dofs] = 1
            boundary_load = boundary_load + neg_bc_value * pybamm.Vector(neg_bc_load)
        else:
            raise ValueError(
                "boundary condition must be Dirichlet or Neumann, not '{}'".format(
                    neg_bc_type
                )
            )

        if pos_bc_type == "Neumann":
            # assemble unit load over tab
            pos_bc_load = skfem.asm(unit_bc_load_form, mesh.positive_tab_basis)
            # value multiplied by weights
            boundary_load = boundary_load + pos_bc_value * pybamm.Vector(pos_bc_load)
        elif pos_bc_type == "Dirichlet":
            # set Dirichlet value at facets corresponding to tab
            pos_bc_load = np.zeros(mesh.npts)
            pos_bc_load[mesh.positive_tab_dofs] = 1
            boundary_load = boundary_load + pos_bc_value * pybamm.Vector(pos_bc_load)
        else:
            raise ValueError(
                "boundary condition must be Dirichlet or Neumann, not '{}'".format(
                    pos_bc_type
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
            ({symbol.id: {"negative tab": neg. tab bc, "positive tab": pos. tab bc}})

        Returns
        -------
        :class:`pybamm.Matrix`
            The (sparse) finite element stiffness matrix for the domain
        """
        # get primary domain mesh
        domain = symbol.domain[0]
        mesh = self.mesh[domain]

        # make form for the stiffness
        @skfem.bilinear_form
        def stiffness_form(u, du, v, dv, w):
            return sum(du * dv)

        # assemble the stifnness matrix
        stiffness = skfem.asm(stiffness_form, mesh.basis)

        # get boundary conditions and type
        try:
            _, neg_bc_type = boundary_conditions[symbol.id]["negative tab"]
            _, pos_bc_type = boundary_conditions[symbol.id]["positive tab"]
        except KeyError:
            raise pybamm.ModelError(
                "No boundary conditions provided for symbol `{}``".format(symbol)
            )

        # adjust matrix for Dirichlet boundary conditions
        if neg_bc_type == "Dirichlet":
            self.bc_apply(stiffness, mesh.negative_tab_dofs)
        if pos_bc_type == "Dirichlet":
            self.bc_apply(stiffness, mesh.positive_tab_dofs)

        return pybamm.Matrix(stiffness)

    def integral(self, child, discretised_child):
        """Vector-vector dot product to implement the integral operator.
        See :meth:`pybamm.SpatialMethod.integral`
        """
        # Calculate integration vector
        integration_vector = self.definite_integral_matrix(child.domains)

        out = integration_vector @ discretised_child

        return out

    def definite_integral_matrix(self, domains, vector_type="row"):
        """
        Matrix for finite-element implementation of the definite integral over
        the entire domain

        .. math::
            I = \\int_{\Omega}\\!f(s)\\,dx

        for where :math:`\Omega` is the domain.

        Parameters
        ----------
        domains : dict
            The domain(s) of integration
        vector_type : str, optional
            Whether to return a row or column vector (default is row)

        Returns
        -------
        :class:`pybamm.Matrix`
            The finite element integral vector for the domain
        """
        # get primary domain mesh
        domain = domains["primary"]
        if isinstance(domain, list):
            domain = domain[0]
        mesh = self.mesh[domain]

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

    def boundary_integral(self, child, discretised_child, region):
        """Implementation of the boundary integral operator.
        See :meth:`pybamm.SpatialMethod.boundary_integral`
        """
        # Calculate integration vector
        integration_vector = self.boundary_integral_vector(
            child.domain[0], region=region
        )

        out = integration_vector @ discretised_child
        out.clear_domains()
        return out

    def boundary_integral_vector(self, domain, region):
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
        region : str
            The region of the boundary over which to integrate. If region is
            `entire` the integration is carried out over the entire boundary. If
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
        mesh = self.mesh[domain]

        # make form for the boundary integral
        @skfem.linear_form
        def integral_form(v, dv, w):
            return v

        if region == "entire":
            # assemble over all facets
            integration_vector = skfem.asm(integral_form, mesh.facet_basis)
        elif region == "negative tab":
            # assemble over negative tab facets
            integration_vector = skfem.asm(integral_form, mesh.negative_tab_basis)
        elif region == "positive tab":
            # assemble over positive tab facets
            integration_vector = skfem.asm(integral_form, mesh.positive_tab_basis)

        return pybamm.Matrix(integration_vector[np.newaxis, :])

    def boundary_value_or_flux(self, symbol, discretised_child, bcs=None):
        """
        Returns the average value of the symbol over the negative tab ("negative tab")
        or the positive tab ("positive tab") in the Finite Element Method.

        Overwrites the default :meth:`pybamm.SpatialMethod.boundary_value`
        """

        # Return average value on the negative tab for "negative tab" and positive tab
        # for "positive tab"
        if isinstance(symbol, pybamm.BoundaryValue):
            # get integration_vector
            if symbol.side == "negative tab":
                region = "negative tab"
            elif symbol.side == "positive tab":
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
            ({symbol.id: {"negative tab": neg. tab bc, "positive tab": pos. tab bc}})

        Returns
        -------
        :class:`pybamm.Matrix`
            The (sparse) mass matrix for the spatial method.
        """
        return self.assemble_mass_form(symbol, boundary_conditions)

    def boundary_mass_matrix(self, symbol, boundary_conditions):
        """
        Calculates the mass matrix for the finite element method assembled
        over the boundary.

        Parameters
        ----------
        symbol: :class:`pybamm.Variable`
            The variable corresponding to the equation for which we are
            calculating the mass matrix.
        boundary_conditions : dict
            The boundary conditions of the model
            ({symbol.id: {"negative tab": neg. tab bc, "positive tab": pos. tab bc}})

        Returns
        -------
        :class:`pybamm.Matrix`
            The (sparse) mass matrix for the spatial method.
        """
        return self.assemble_mass_form(symbol, boundary_conditions, region="boundary")

    def assemble_mass_form(self, symbol, boundary_conditions, region="interior"):
        """
        Assembles the form of the finite element mass matrix over the domain
        interior or boundary.

        Parameters
        ----------
        symbol: :class:`pybamm.Variable`
            The variable corresponding to the equation for which we are
            calculating the mass matrix.
        boundary_conditions : dict
            The boundary conditions of the model
            ({symbol.id: {"negative tab": neg. tab bc, "positive tab": pos. tab bc}})
        region: str, optional
            The domain over which to assemble the mass matrix form. Can be "interior"
            (default) or "boundary".

        Returns
        -------
        :class:`pybamm.Matrix`
            The (sparse) mass matrix for the spatial method.
        """
        # get primary domain mesh
        domain = symbol.domain[0]
        mesh = self.mesh[domain]

        # create form for mass
        @skfem.bilinear_form
        def mass_form(u, du, v, dv, w):
            return u * v

        # assemble mass matrix
        if region == "interior":
            mass = skfem.asm(mass_form, mesh.basis)
        if region == "boundary":
            mass = skfem.asm(mass_form, mesh.facet_basis)

        # get boundary conditions and type
        if symbol.id in boundary_conditions:
            _, neg_bc_type = boundary_conditions[symbol.id]["negative tab"]
            _, pos_bc_type = boundary_conditions[symbol.id]["positive tab"]

            if neg_bc_type == "Dirichlet":
                # set source terms to zero on boundary by zeroing out mass matrix
                self.bc_apply(mass, mesh.negative_tab_dofs, zero=True)
            if pos_bc_type == "Dirichlet":
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
            If False, the diagonal element is set to one. default is False.
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
