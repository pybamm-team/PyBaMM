#
# Finite Element discretisation class which uses fenics
#
import pybamm

from scipy.sparse import csr_matrix
import autograd.numpy as np
from autograd.builtins import isinstance

import importlib

dolfin_spec = importlib.util.find_spec("dolfin")
if dolfin_spec is not None:
    dolfin = importlib.util.module_from_spec(dolfin_spec)
    dolfin_spec.loader.exec_module(dolfin)


class FiniteElementFenics(pybamm.SpatialMethod):
    """
    A class which implements the steps specific to the finite element method during
    discretisation. The class uses fenics to discretise the problem to obtain
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
        # only implemented in y-z plane
        if symbol.name in ["y", "z"]:
            symbol_mesh = self.mesh.combine_submeshes(*symbol.domain)
            return pybamm.Vector(symbol_mesh[0].nodes, domain=symbol.domain)
        else:
            raise NotImplementedError(
                "FiniteElementFenics only implemented in the y-z plane"
            )

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
            The symbol that we will take the gradient of.
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
        domain = symbol.domain

        # TO DO: finish fenics mesh class
        mesh = self.mesh

        stiffness_matrix = self.stiffness_matrix(domain)

        # get boundary conditions and type, here lbc: negative tab, rbc: positive tab
        lbc_value, lbc_type = boundary_conditions[symbol.id]["left"]
        rbc_value, rbc_type = boundary_conditions[symbol.id]["right"]
        boundary_load = pybamm.Vector(np.zeros(mesh.N_dofs))

        if lbc_type == "Neumann":
            # make form for the boundary conditions
            lbc_form = dolfin.Constant(1) * mesh.TestFunction * self.ds(1)
            lbc_load = lbc_value * pybamm.Vector(
                dolfin.assemble(lbc_form).get_local()[:]
            )
            boundary_load = boundary_load + lbc_load
        elif lbc_type == "Dirichlet":
            raise NotImplementedError("Dirichlet boundary conditons not implemented")
        else:
            raise ValueError(
                "boundary condition must be Dirichlet or Neumann, not '{}'".format(
                    lbc_type
                )
            )

        if rbc_type == "Neumann":
            # make form for the boundary conditions
            rbc_form = dolfin.Constant(1) * mesh.TestFunction * self.ds(2)
            rbc_load = rbc_value * pybamm.Vector(
                dolfin.assemble(rbc_form).get_local()[:]
            )
            boundary_load = boundary_load + rbc_load
        elif rbc_type == "Dirichlet":
            raise NotImplementedError("Dirichlet boundary conditons not implemented")
        else:
            raise ValueError(
                "boundary condition must be Dirichlet or Neumann, not '{}'".format(
                    rbc_type
                )
            )

        return stiffness_matrix @ discretised_symbol + boundary_load

    def stiffness_matrix(self, domain):
        """
        Laplacian (stiffness) matrix for finite elements in the appropriate domain.

        Parameters
        ----------
        domain : list
            The domain(s) in which to compute the gradient matrix

        Returns
        -------
        :class:`pybamm.Matrix`
            The (sparse) finite element stiffness matrix for the domain
        """
        # TO DO: finish fenics mesh class
        mesh = self.mesh

        # make form for the stiffness
        stiffness_form = (
            dolfin.inner(
                dolfin.grad(mesh.TrialFunction), dolfin.grad(mesh.TestFunction)
            )
            * mesh.dx
        )

        # assemble the stifnness matrix
        stiffness = csr_matrix(dolfin.assemble(stiffness_form).array())

        return pybamm.Matrix(stiffness)

    def integral(self, domain, symbol, discretised_symbol):
        """Vector-vector dot product to implement the integral operator.
        See :meth:`pybamm.BaseDiscretisation.integral`
        """
        # Calculate integration vector
        integration_vector = self.definite_integral_vector()

        out = integration_vector @ discretised_symbol
        out.domain = []
        return out

    def definite_integral_vector(self):
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
        mesh = self.mesh
        vector = dolfin.assemble(mesh.TrialFunction * mesh.dx)
        return pybamm.Vector(vector)

    def indefinite_integral(self, domain, symbol, discretised_symbol):
        """Implementation of the indefinite integral operator. The
        input discretised symbol must be defined on the internal mesh edges.
        See :meth:`pybamm.BaseDiscretisation.indefinite_integral`
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
        # TO DO: finish fenics mesh class
        mesh = self.mesh

        # create form for mass
        mass_form = mesh.TrialFunction * mesh.TestFunction * mesh.dx

        # assemble mass matrix
        mass = csr_matrix(dolfin.assemble(mass_form).array())

        # TO DO: implement Dirichlet BCs
        # get boundary conditions and type, here lbc: negative tab, rbc: positive tab
        lbc_value, lbc_type = boundary_conditions[symbol.id]["left"]
        rbc_value, rbc_type = boundary_conditions[symbol.id]["right"]

        if lbc_type == "Dirichlet":
            raise NotImplementedError("Dirichlet boundary conditons not implemented")

        if rbc_type == "Dirichlet":
            raise NotImplementedError("Dirichlet boundary conditons not implemented")

        return pybamm.Matrix(mass)

    def source(self, symbol, discretised_symbol, boundary_conditions):
        """
        Calculates weak form of source terms in the finite element method.

        Parameters
        ----------
        symbol: :class:`pybamm.Variable`
            The variable corresponding to the equation for which we are
            calculating the mass matrix.
        discretised_symbol: :class:`pybamm.Symbol`
            The discretised symbol of the correct size
        boundary_conditions : dict
            The boundary conditions of the model
            ({symbol.id: {"left": left bc, "right": right bc}})

        Returns
        -------
        :class: `pybamm.Array`
            Contains the result of acting the mass matrix on
            the child discretised_symbol
        """
        mass_matrix = self.mass_matrix(symbol, boundary_conditions)

        return mass_matrix @ discretised_symbol

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
        raise NotImplementedError
