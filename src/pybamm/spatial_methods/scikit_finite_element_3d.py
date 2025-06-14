import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
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
        """Creates a discretised spatial variable for x, y, or z."""
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
            )
        return pybamm.Vector(entries, domains=symbol.domains)

    def gradient(self, symbol, discretised_symbol, boundary_conditions):
        """
        Matrix-vector multiplication to implement the 3D gradient operator.

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
        :class:`pybamm.VectorField3D`
            The 3D gradient vector field
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

        try:
            mass_inv = pybamm.Matrix(inv(csc_matrix(mass)))
        except Exception:
            return pybamm.VectorField3D(
                pybamm.Vector(np.full(mesh.npts, np.nan)),
                pybamm.Vector(np.full(mesh.npts, np.nan)),
                pybamm.Vector(np.full(mesh.npts, np.nan)),
            )

        grad_x = mass_inv @ (grad_x_matrix @ discretised_symbol)
        grad_y = mass_inv @ (grad_y_matrix @ discretised_symbol)
        grad_z = mass_inv @ (grad_z_matrix @ discretised_symbol)
        grad_field = pybamm.VectorField3D(grad_x, grad_y, grad_z)
        grad_field.evaluates_on_edges = lambda _: True

        return grad_field

    def gradient_squared(self, symbol, discretised_symbol, boundary_conditions):
        """
        Multiplication to implement the inner product of the gradient operator
        with itself in 3D.
        """
        grad = self.gradient(symbol, discretised_symbol, boundary_conditions)
        grad_x = grad.x_field
        grad_y = grad.y_field
        grad_z = grad.z_field
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

    def divergence(self, symbol, discretised_symbol, boundary_conditions):
        """
        Matrix-vector multiplication to implement the 3D divergence operator.

        Parameters
        ----------
        symbol: :class:`pybamm.Symbol`
            The symbol representing the divergence operation
        discretised_symbol: :class:`pybamm.VectorField3D`
            The discretised vector field
        boundary_conditions : dict
            The boundary conditions of the model

        Returns
        -------
        :class:`pybamm.Symbol`
            The divergence of the vector field
        """

        try:
            domain_key = discretised_symbol.domain[0]
        except (AttributeError, IndexError):
            domain_key = symbol.domain[0]
        mesh = self.mesh[domain_key]

        dummy_scalar_for_grad_ops = pybamm.Variable("dummy_scalar", domain=domain_key)
        grad_x_op, grad_y_op, grad_z_op = self.gradient_matrix(
            dummy_scalar_for_grad_ops, {}
        )

        grad_x_T = pybamm.Matrix(grad_x_op.entries.T)
        grad_y_T = pybamm.Matrix(grad_y_op.entries.T)
        grad_z_T = pybamm.Matrix(grad_z_op.entries.T)

        Fx = discretised_symbol.x_field
        Fy = discretised_symbol.y_field
        Fz = discretised_symbol.z_field

        rhs_divergence = -(grad_x_T @ Fx + grad_y_T @ Fy + grad_z_T @ Fz)

        mass_mat_raw = self.mass_matrix(dummy_scalar_for_grad_ops, {})
        mass_entries_sparse = csc_matrix(mass_mat_raw.entries)

        if mass_entries_sparse.shape[0] == 0:
            return pybamm.Scalar(0) * Fx

        try:
            mass_inv_sparse = inv(mass_entries_sparse)
            mass_inv = pybamm.Matrix(mass_inv_sparse)
        except Exception:
            return pybamm.Vector(np.full(mesh.npts, np.nan))

        div_op_val = mass_inv @ rhs_divergence
        div_op_val.clear_domains()
        return div_op_val

    def laplacian(self, symbol, discretised_symbol, boundary_conditions):
        """
        Matrix-vector multiplication to implement the 3D Laplacian operator.

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
        stiffness_mat_raw = self.stiffness_matrix(symbol, {})
        boundary_load_neumann_raw = self.laplacian_boundary_load(
            symbol, boundary_conditions
        )
        mass_mat_raw = self.mass_matrix(symbol, {})

        mass_entries_sparse = csc_matrix(mass_mat_raw.entries)

        if mass_entries_sparse.shape[0] == 0:
            return pybamm.Scalar(0) * discretised_symbol

        try:
            mass_inv_sparse = inv(mass_entries_sparse)
            mass_inv = pybamm.Matrix(mass_inv_sparse)
        except Exception:
            return pybamm.Vector(
                np.full(
                    discretised_symbol.shape[0]
                    if hasattr(discretised_symbol, "shape")
                    else mass_entries_sparse.shape[0],
                    np.nan,
                )
            )

        lap_op_val = mass_inv @ (
            -stiffness_mat_raw @ discretised_symbol + boundary_load_neumann_raw
        )
        lap_op_val.clear_domains()
        return lap_op_val

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
        return pybamm.Matrix(stiffness)

    def laplacian_boundary_load(self, symbol, boundary_conditions):
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
        domain = symbol.domain[0]
        mesh = self.mesh[domain]

        boundary_load = np.zeros(mesh.npts)
        bcs = boundary_conditions.get(symbol, {})

        @skfem.LinearForm
        def unit_bc_load_form(v, w):
            return v

        for name, (bc_value, bc_type) in bcs.items():
            if bc_type == "Neumann":
                if hasattr(mesh, f"{name}_basis"):
                    boundary_basis = getattr(mesh, f"{name}_basis")
                    bc_contrib = skfem.asm(unit_bc_load_form, boundary_basis)
                    if hasattr(bc_value, "evaluate"):
                        bc_val = bc_value.evaluate()
                    else:
                        bc_val = float(bc_value)
                    boundary_load += bc_val * bc_contrib

        return pybamm.Vector(boundary_load)

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
            The discretised symbol being integrated
        integration_dimension : str
            The dimension over which to integrate
        integration_variable : :class:`pybamm.SpatialVariable`
            The variable of integration

        Returns
        -------
        :class:`pybamm.Symbol`
            The result of the integration
        """
        integration_matrix = self.definite_integral_matrix(
            child, integration_dimension, integration_variable
        )
        return integration_matrix @ discretised_child

    def definite_integral_matrix(
        self, child, integration_dimension="primary", integration_variable=None
    ):
        """
        Matrix for finite-element implementation of the definite integral.

        Parameters
        ----------
        child : :class:`pybamm.Symbol`
            The symbol being integrated
        integration_dimension : str, optional
            The dimension over which to integrate
        integration_variable : :class:`pybamm.SpatialVariable`, optional
            The variable of integration

        Returns
        -------
        :class:`pybamm.Matrix`
            The finite element integral matrix
        """
        skfem = import_optional_dependency("skfem")
        domain = (
            child.domain[0] if hasattr(child, "domain") else child.domains["primary"][0]
        )
        mesh = self.mesh[domain]

        @skfem.LinearForm
        def integral_form(v, w):
            return v

        vector = skfem.asm(integral_form, mesh.basis)
        return pybamm.Matrix(vector[np.newaxis, :])

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
        else:
            raise ValueError(f"Unknown boundary region: {region}")

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
        if isinstance(symbol, pybamm.BoundaryValue):
            domain = symbol.children[0].domain
            region = symbol.side
            mesh = self.mesh[domain[0]]

            if hasattr(mesh, f"{region}_dofs"):
                boundary_dofs = getattr(mesh, f"{region}_dofs")
                boundary_matrix = csr_matrix(
                    (
                        np.ones(len(boundary_dofs)),
                        (range(len(boundary_dofs)), boundary_dofs),
                    ),
                    shape=(len(boundary_dofs), mesh.npts),
                )
                boundary_values = pybamm.Matrix(boundary_matrix) @ discretised_child

                avg_matrix = np.ones((1, len(boundary_dofs))) / len(boundary_dofs)
                boundary_value = pybamm.Matrix(avg_matrix) @ boundary_values
                boundary_value.copy_domains(symbol)
                return boundary_value
            else:
                integration_vector = self.boundary_integral_vector(
                    domain, region=region
                )
                boundary_val_vector = integration_vector / (
                    integration_vector
                    @ pybamm.Vector(np.ones(integration_vector.shape[1]))
                )
                boundary_value = boundary_val_vector @ discretised_child
                boundary_value.copy_domains(symbol)
                return boundary_value

        elif isinstance(symbol, pybamm.BoundaryGradient):
            domain = symbol.children[0].domain
            region = symbol.side

            laplacian_of_child = self.laplacian(
                symbol.children[0], discretised_child, bcs
            )
            stiffness = self.stiffness_matrix(symbol.children[0], bcs)
            mass = self.mass_matrix(symbol.children[0], bcs)

            flux_vector = mass @ laplacian_of_child + stiffness @ discretised_child
            integration_vector = self.boundary_integral_vector(domain, region=region)

            total_flux_on_boundary = integration_vector @ flux_vector
            boundary_area_vector = self.boundary_integral_vector(domain, region=region)
            total_area = boundary_area_vector @ pybamm.Vector(np.ones(mesh.npts))

            avg_flux = total_flux_on_boundary / total_area
            avg_flux.copy_domains(symbol)
            return avg_flux
        else:
            raise TypeError("symbol must be BoundaryValue or BoundaryGradient")

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
        elif region == "boundary":
            mass = skfem.asm(mass_form, mesh.facet_basis)

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
            else:
                M_lil[dof, dof] = 0.0

        M_csr = M_lil.tocsr()
        M.data = M_csr.data
        M.indices = M_csr.indices
        M.indptr = M_csr.indptr
        M._shape = M_csr._shape
