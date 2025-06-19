import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv

import pybamm
from pybamm.util import import_optional_dependency


class ScikitFiniteElement3D(pybamm.SpatialMethod):
    """
    A class which implements the steps specific to the 3D finite element method
    during discretisation using scikit-fem.

    Parameters
    ----------
    options : dict-like, optional
        A dictionary of options to be passed to the spatial method.
    """

    def __init__(self, options=None):
        super().__init__(options)
        pybamm.citations.register("Gustafsson2020")

    def build(self, mesh):
        super().build(mesh)
        for dom in mesh.keys():
            mesh[dom].npts_for_broadcast_to_nodes = mesh[dom].npts

    def spatial_variable(self, symbol):
        """
        Creates a discretised spatial variable for x, y, or z.

        Parameters
        ----------
        symbol : :class:`pybamm.SpatialVariable`
            The spatial variable to discretise (must be 'x', 'y', or 'z')

        Returns
        -------
        :class:`pybamm.Vector`
            The discretised spatial variable as a vector of nodal values
        """
        domain = symbol.domain[0]
        mesh = self.mesh[domain]

        if symbol.name == "x":
            entries = mesh.nodes[:, 0][:, np.newaxis]
        elif symbol.name == "y":
            entries = mesh.nodes[:, 1][:, np.newaxis]
        elif symbol.name == "z":
            entries = mesh.nodes[:, 2][:, np.newaxis]
        else:
            raise pybamm.GeometryError(
                f"Spatial variable must be 'x', 'y' or 'z', not {symbol.name}"
            )  # pragma: no cover
        return pybamm.Vector(entries, domains=symbol.domains)

    def gradient(self, symbol, discretised_symbol, boundary_conditions):
        """
        Matrix-vector multiplication to implement the 3D gradient operator.
        Returns a Concatenation of [grad_x, grad_y, grad_z] similar to 2D approach.

        Parameters
        ----------
        symbol: :class:`pybamm.Symbol`
            The symbol that we will take the gradient of.
        discretised_symbol: :class:`pybamm.Symbol`
            The discretised symbol of the correct size
        boundary_conditions : dict
            The boundary conditions of the model

        Returns
        -------
        :class:`pybamm.Concatenation`
            The 3D gradient as concatenation of x, y, z components
        """
        skfem = import_optional_dependency("skfem")
        domain = symbol.domain[0]
        mesh = self.mesh[domain]

        grad_x_matrix, grad_y_matrix, grad_z_matrix = self.gradient_matrix(
            symbol, boundary_conditions
        )

        @skfem.BilinearForm
        def mass_form(u, v, w):
            return u * v

        mass = skfem.asm(mass_form, mesh.basis)
        mass_inv = pybamm.Matrix(inv(csc_matrix(mass)))

        grad_x = mass_inv @ (grad_x_matrix @ discretised_symbol)
        grad_y = mass_inv @ (grad_y_matrix @ discretised_symbol)
        grad_z = mass_inv @ (grad_z_matrix @ discretised_symbol)

        # Create concatenation
        grad = pybamm.Concatenation(
            grad_x, grad_y, grad_z, check_domain=False, concat_fun=np.hstack
        )
        grad.copy_domains(symbol)

        return grad

    def gradient_squared(self, symbol, discretised_symbol, boundary_conditions):
        """
        Multiplication to implement the inner product of the gradient operator
        with itself in 3D.
        """
        grad = self.gradient(symbol, discretised_symbol, boundary_conditions)
        grad_x, grad_y, grad_z = grad.orphans
        return grad_x**2 + grad_y**2 + grad_z**2

    def gradient_matrix(self, symbol, boundary_conditions):
        """
        Gradient matrices for finite elements in 3D.

        Parameters
        ----------
        symbol: :class:`pybamm.Symbol`
            The symbol for which we want to calculate the gradient matrix
        boundary_conditions : dict
            The boundary conditions of the model

        Returns
        -------
        tuple of :class:`pybamm.Matrix`
            The (sparse) finite element gradient matrices for x, y, z directions
        """
        skfem = import_optional_dependency("skfem")
        domain = symbol.domain[0]
        mesh = self.mesh[domain]

        @skfem.BilinearForm
        def gradient_dx(u, v, w):
            return u.grad[0] * v

        @skfem.BilinearForm
        def gradient_dy(u, v, w):
            return u.grad[1] * v

        @skfem.BilinearForm
        def gradient_dz(u, v, w):
            return u.grad[2] * v

        grad_x = skfem.asm(gradient_dx, mesh.basis)
        grad_y = skfem.asm(gradient_dy, mesh.basis)
        grad_z = skfem.asm(gradient_dz, mesh.basis)

        return pybamm.Matrix(grad_x), pybamm.Matrix(grad_y), pybamm.Matrix(grad_z)

    def laplacian(self, symbol, discretised_symbol, boundary_conditions):
        """
        Matrix-vector multiplication to implement the 3D Laplacian operator.
        This should be called only after boundary conditions have been properly
        discretised by the PyBaMM discretisation process.

        Parameters
        ----------
        symbol: :class:`pybamm.Symbol`
            The symbol for which we want to calculate the Laplacian
        discretised_symbol: :class:`pybamm.Symbol`
            The discretised symbol of the correct size
        boundary_conditions : dict
            The boundary conditions of the model

        Returns
        -------
        :class:`pybamm.Symbol`
            The Laplacian of the symbol
        """
        stiffness_matrix = self.stiffness_matrix(symbol, boundary_conditions)
        boundary_load = self.laplacian_boundary_load(symbol, boundary_conditions)

        return -stiffness_matrix @ discretised_symbol + boundary_load

    def divergence(self, symbol, discretised_symbol, boundary_conditions):
        """
        Matrix-vector multiplication to implement the 3D divergence operator.
        Expects discretised_symbol to be a Concatenation of [Fx, Fy, Fz].

        Parameters
        ----------
        symbol: :class:`pybamm.Symbol`
            The symbol representing the divergence operation
        discretised_symbol: :class:`pybamm.Concatenation`
            The discretised vector field as concatenation [Fx, Fy, Fz]
        boundary_conditions : dict
            The boundary conditions of the model

        Returns
        -------
        :class:`pybamm.Symbol`
            The divergence of the vector field
        """
        domain_key = symbol.domain[0]

        if (
            hasattr(discretised_symbol, "orphans")
            and len(discretised_symbol.orphans) == 3
        ):
            Fx, Fy, Fz = discretised_symbol.orphans
        else:
            raise ValueError(
                "divergence expects a concatenation of 3 vector components"
            )  # pragma: no cover

        Fx_var = pybamm.Variable("Fx", domain=[domain_key])
        grad_M = self.gradient_matrix(Fx_var, boundary_conditions)

        skfem = import_optional_dependency("skfem")
        mesh = self.mesh[domain_key]

        @skfem.BilinearForm
        def mass_form(u, v, w):
            return u * v

        mass = skfem.asm(mass_form, mesh.basis)
        mass_inv = pybamm.Matrix(inv(csc_matrix(mass)))

        grad_x_T = pybamm.Matrix(grad_M[0].entries)
        grad_y_T = pybamm.Matrix(grad_M[1].entries)
        grad_z_T = pybamm.Matrix(grad_M[2].entries)

        div_x = mass_inv @ (grad_x_T @ Fx)
        div_y = mass_inv @ (grad_y_T @ Fy)
        div_z = mass_inv @ (grad_z_T @ Fz)

        for var_symbol, bcs in boundary_conditions.items():
            if var_symbol.domain and var_symbol.domain[0] == domain_key:
                for name, (_, bc_type) in bcs.items():
                    if bc_type == "Dirichlet":
                        boundary_dofs = getattr(mesh, f"{name}_dofs", None)
                        if boundary_dofs is not None:
                            self.bc_apply(mass, boundary_dofs, zero=True)

        div_result = div_x + div_y + div_z
        div_result.clear_domains()
        return div_result

    def stiffness_matrix(self, symbol, boundary_conditions):
        """
        Laplacian (stiffness) matrix for finite elements in 3D.

        Parameters
        ----------
        symbol: :class:`pybamm.Symbol`
            The symbol for which we want to calculate the stiffness matrix
        boundary_conditions : dict
            The boundary conditions of the model

        Returns
        -------
        :class:`pybamm.Matrix`
            The (sparse) finite element stiffness matrix
        """
        skfem = import_optional_dependency("skfem")
        domain = symbol.domain[0]
        mesh = self.mesh[domain]

        @skfem.BilinearForm
        def stiffness_form(u, v, w):
            return sum(u.grad * v.grad)

        stiffness = skfem.asm(stiffness_form, mesh.basis)

        if symbol in boundary_conditions:
            bcs = boundary_conditions[symbol]
            for name, (_, bc_type) in bcs.items():
                if bc_type == "Dirichlet":
                    boundary_dofs = getattr(mesh, f"{name}_dofs", None)
                    if boundary_dofs is not None and len(boundary_dofs) > 0:
                        self.bc_apply(stiffness, boundary_dofs)

        return pybamm.Matrix(stiffness)

    def laplacian_boundary_load(self, symbol_for_laplacian, boundary_conditions_dict):
        """
        Assembles the boundary load vector for the 3D Laplacian.

        Parameters
        ----------
        symbol: :class:`pybamm.Symbol`
            The symbol for which we want to calculate the boundary load
        boundary_conditions : dict
            The boundary conditions of the model

        Returns
        -------
        :class:`pybamm.Vector`
            The boundary load vector
        """
        skfem = import_optional_dependency("skfem")
        domain = symbol_for_laplacian.domain[0]
        fem_mesh = self.mesh[domain]

        current_boundary_load_symbol = pybamm.Vector(
            np.zeros((fem_mesh.npts, 1)), domains=symbol_for_laplacian.domains
        )

        bcs_for_symbol = boundary_conditions_dict.get(symbol_for_laplacian, {})

        unit_bc_load_form = None
        if any(bc_type == "Neumann" for _, (_, bc_type) in bcs_for_symbol.items()):

            @skfem.LinearForm
            def _unit_bc_load_form(v, w):
                return v

            unit_bc_load_form = _unit_bc_load_form
        if symbol_for_laplacian in boundary_conditions_dict:
            for name, (bc_value_symbol, bc_type) in bcs_for_symbol.items():
                term_contribution = None
                if bc_type == "Neumann":
                    numeric_coeffs = skfem.asm(
                        unit_bc_load_form, getattr(fem_mesh, f"{name}_basis")
                    )
                    if numeric_coeffs.ndim == 1:
                        numeric_coeffs = numeric_coeffs[:, np.newaxis]
                    term_contribution = bc_value_symbol * pybamm.Vector(numeric_coeffs)
                    current_boundary_load_symbol += term_contribution
                elif bc_type == "Dirichlet":
                    boundary_dofs = getattr(fem_mesh, f"{name}_dofs")
                    numeric_mask = np.zeros((fem_mesh.npts, 1))
                    numeric_mask[boundary_dofs, 0] = 1.0
                    term_contribution = bc_value_symbol * pybamm.Vector(numeric_mask)
                    current_boundary_load_symbol += term_contribution

        return current_boundary_load_symbol

    def integral(
        self, child, discretised_child, integration_dimension, integration_variable
    ):
        """
        Vector-vector dot product to implement the 3D integral operator.

        Parameters
        ----------
        child : :class:`pybamm.Symbol`
            The symbol being integrated
        discretised_child : :class:`pybamm.Symbol`
            The discretised symbol being integrated (vector of nodal values)
        integration_dimension : str
            The dimension over which to integrate (e.g. "primary" for volume)
        integration_variable : :class:`pybamm.SpatialVariable`
            The variable(s) of integration

        Returns
        -------
        :class:`pybamm.Symbol`
            The result of the integration (a scalar value).
        """
        integration_vector = self.definite_integral_matrix(child)
        out = integration_vector @ discretised_child
        return out

    def definite_integral_matrix(self, child, vector_type="row"):
        """
        Matrix for definite integral *vector* (vector of ones).
        The child is used to determine the domain and size.
        'integration_dimension' and 'integration_variable' are kept for signature
        compatibility but are not strictly used for this method's typical behavior.

        Parameters
        ----------
        child : :class:`pybamm.Symbol`
            The symbol whose domain/mesh determines the size.
        vector_type : str, optional
            "row" or "column" for the shape of the resulting vector of ones.
        integration_dimension : str, optional
            (Not directly used for vector of ones)
        integration_variable : :class:`pybamm.SpatialVariable`, optional
            (Not directly used for vector of ones)

        Returns
        -------
        :class:`pybamm.Matrix`
            A matrix representing a vector of ones (either row or column).
        """
        skfem = import_optional_dependency("skfem")
        # get primary domain mesh
        domain = child.domain[0]
        mesh = self.mesh[domain]

        # make form for the integral
        @skfem.LinearForm
        def integral_form(v, w):
            return v

        vector = skfem.asm(integral_form, mesh.basis)

        if vector_type == "row":
            return pybamm.Matrix(vector[np.newaxis, :])
        elif vector_type == "column":  # pragma: no cover
            return pybamm.Matrix(vector[:, np.newaxis])

    def indefinite_integral(self, child, discretised_child, direction):
        """
        Implementation of the indefinite integral operator in 3D.
        """
        raise NotImplementedError(
            "Indefinite integral not implemented for 3D finite elements"
        )

    def boundary_integral(self, child, discretised_child, region):
        """
        Implementation of the 3D boundary integral operator.

        Parameters
        ----------
        child : :class:`pybamm.Symbol`
            The symbol being integrated
        discretised_child : :class:`pybamm.Symbol`
            The discretised symbol being integrated
        region : str
            The boundary region over which to integrate

        Returns
        -------
        :class:`pybamm.Symbol`
            The result of the boundary integration
        """
        integration_vector = self.boundary_integral_vector(child.domain, region=region)
        out = integration_vector @ discretised_child
        out.clear_domains()
        return out

    def boundary_integral_vector(self, domain, region):
        """
        A vector representing an integral operator over the boundary.

        Parameters
        ----------
        domain : list
            The domain(s) of the variable in the integrand
        region : str
            The region of the boundary over which to integrate

        Returns
        -------
        :class:`pybamm.Matrix`
            The finite element boundary integral vector
        """
        skfem = import_optional_dependency("skfem")
        mesh = self.mesh[domain[0]]

        @skfem.LinearForm
        def integral_form(v, w):
            return v

        if region == "entire":
            integration_vector = skfem.asm(integral_form, mesh.facet_basis)
        elif hasattr(mesh, f"{region}_basis"):
            integration_vector = skfem.asm(
                integral_form, getattr(mesh, f"{region}_basis")
            )

        return pybamm.Matrix(integration_vector[np.newaxis, :])

    def boundary_value_or_flux(self, symbol, discretised_child, bcs=None):
        """
        Returns the boundary value or flux of a variable in 3D.

        Parameters
        ----------
        symbol : :class:`pybamm.Symbol`
            The boundary symbol
        discretised_child : :class:`pybamm.Symbol`
            The discretised variable
        bcs : dict, optional
            Boundary conditions

        Returns
        -------
        :class:`pybamm.Symbol`
            The boundary value or flux
        """
        domain = symbol.children[0].domain
        region = symbol.side

        if isinstance(symbol, pybamm.BoundaryValue):
            integration_vector = self.boundary_integral_vector(domain, region=region)
            boundary_val_vector = integration_vector / (
                integration_vector @ pybamm.Vector(np.ones(integration_vector.shape[1]))
            )
            boundary_value = boundary_val_vector @ discretised_child
            boundary_value.copy_domains(symbol)
            return boundary_value

        elif isinstance(symbol, pybamm.BoundaryGradient):
            raise NotImplementedError(
                "Boundary gradient not implemented for 3D finite elements"
            )  # pragma: no cover

    def mass_matrix(self, symbol, boundary_conditions):
        """
        Calculates the mass matrix for the 3D finite element method.

        Parameters
        ----------
        symbol: :class:`pybamm.Variable`
            The variable corresponding to the equation
        boundary_conditions : dict
            The boundary conditions of the model

        Returns
        -------
        :class:`pybamm.Matrix`
            The (sparse) mass matrix for the spatial method.
        """
        return self.assemble_mass_form(symbol, boundary_conditions)

    def boundary_mass_matrix(self, symbol, boundary_conditions):
        """
        Calculates the mass matrix assembled over the 3D boundary.

        Parameters
        ----------
        symbol: :class:`pybamm.Variable`
            The variable corresponding to the equation
        boundary_conditions : dict
            The boundary conditions of the model

        Returns
        -------
        :class:`pybamm.Matrix`
            The (sparse) boundary mass matrix
        """
        return self.assemble_mass_form(symbol, boundary_conditions, region="boundary")

    def assemble_mass_form(self, symbol, boundary_conditions, region="interior"):
        """
        Assembles the form of the 3D finite element mass matrix.

        Parameters
        ----------
        symbol: :class:`pybamm.Variable`
            The variable corresponding to the equation
        boundary_conditions : dict
            The boundary conditions of the model
        region: str, optional
            The domain over which to assemble

        Returns
        -------
        :class:`pybamm.Matrix`
            The (sparse) mass matrix
        """
        skfem = import_optional_dependency("skfem")
        domain = symbol.domain[0]
        mesh = self.mesh[domain]

        @skfem.BilinearForm
        def mass_form(u, v, w):
            return u * v

        if region == "interior":
            mass = skfem.asm(mass_form, mesh.basis)
        elif region == "boundary":  # pragma: no cover
            mass = skfem.asm(mass_form, mesh.facet_basis)

        if symbol in boundary_conditions:
            bcs = boundary_conditions[symbol]
            for name, (_, bc_type) in bcs.items():
                if bc_type == "Dirichlet":
                    boundary_dofs = getattr(mesh, f"{name}_dofs", None)
                    if boundary_dofs is not None and len(boundary_dofs) > 0:
                        self.bc_apply(mass, boundary_dofs, zero=True)
        return pybamm.Matrix(mass)

    def bc_apply(self, M, boundary_dofs, zero=False):
        """
        Adjusts matrices for 3D boundary conditions.

        Parameters
        ----------
        M : scipy.sparse matrix
            Matrix to modify
        boundary_dofs : array_like
            Boundary degrees of freedom
        zero : bool, optional
            Whether to zero the diagonal entries
        """
        M_lil = M.tolil()

        for dof in boundary_dofs:
            M_lil[dof, :] = 0
            if not zero:
                M_lil[dof, dof] = 1.0

        M_csr = M_lil.tocsr()
        M.data = M_csr.data
        M.indices = M_csr.indices
        M.indptr = M_csr.indptr
        M._shape = M_csr._shape
