#
# A general spatial method class
#
import pybamm
from scipy.sparse import eye, kron


class SpatialMethod:
    """
    A general spatial methods class, with default (trivial) behaviour for broadcast,
    mass_matrix and compute_diffusivity.
    All spatial methods will follow the general form of SpatialMethod in
    that they contain a method for broadcasting variables onto a mesh,
    a gradient operator, and a diverence operator.

    Parameters
    ----------
    mesh : :class: `pybamm.Mesh` (or subclass)
        Contains all the submeshes for discretisation
    """

    def __init__(self, mesh):
        self._mesh = mesh

    @property
    def mesh(self):
        return self._mesh

    def spatial_variable(self, symbol):
        """
        Convert a :class:`pybamm.SpatialVariable` node to a linear algebra object that
        can be evaluated (e.g. a :class:`pybamm.Vector`).

        Parameters
        -----------
        symbol : :class:`pybamm.SpatialVariable`
            The spatial variable to be discretised.

        Returns
        -------
        :class:`pybamm.Vector`
            Contains the discretised spatial variable
        """
        raise NotImplementedError

    def broadcast(self, symbol, domain):
        """
        Broadcast symbol to a specified domain. To do this, calls
        :class:`pybamm.NumpyBroadcast`

        Parameters
        ----------
        symbol : :class:`pybamm.Symbol`
            The symbol to be broadcasted
        domain : iterable of strings
            The domain to broadcast to

        Returns
        -------
        broadcasted_symbol: class: `pybamm.Symbol`
            The discretised symbol of the correct size for the spatial method
        """
        # Default behaviour: use NumpyBroadcast
        return pybamm.NumpyBroadcast(symbol, domain, self.mesh)

    def gradient(self, symbol, discretised_symbol, boundary_conditions):
        """
        Implements the gradient for a spatial method.

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
        raise NotImplementedError

    def divergence(self, symbol, discretised_symbol, boundary_conditions):
        """
        Implements the divergence for a spatial method.

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
            Contains the result of acting the discretised divergence on
            the child discretised_symbol
        """
        raise NotImplementedError

    def integral(self, domain, symbol, discretised_symbol):
        """
        Implements the integral for a spatial method.

        Parameters
        ----------
        domain: iterable of strings
            The domain in which to integrate
        symbol: :class:`pybamm.Symbol`
            The symbol to which is being integrated
        discretised_symbol: :class:`pybamm.Symbol`
            The discretised symbol of the correct size

        Returns
        -------
        :class: `pybamm.Array`
            Contains the result of acting the discretised integral on
            the child discretised_symbol
        """
        raise NotImplementedError

    def indefinite_integral(self, domain, symbol, discretised_symbol):
        """
        Implements the indefinite integral for a spatial method.

        Parameters
        ----------
        domain: iterable of strings
            The domain in which to integrate
        symbol: :class:`pybamm.Symbol`
            The symbol to which is being integrated
        discretised_symbol: :class:`pybamm.Symbol`
            The discretised symbol of the correct size

        Returns
        -------
        :class: `pybamm.Array`
            Contains the result of acting the discretised indefinite integral on
            the child discretised_symbol
        """
        raise NotImplementedError

    def boundary_value(self, discretised_symbol):
        """
        Returns the surface value using the approriate expression for the
        spatial method.

        Parameters
        -----------
        discretised_symbol : :class:`pybamm.StateVector`
            The discretised variable (a state vector) from which to calculate
            the surface value.

        Returns
        -------
        :class:`pybamm.Variable`
            The variable representing the surface value.
        """
        raise NotImplementedError

    def mass_matrix(self, symbol, boundary_conditions):
        """
        Calculates the mass matrix for a spatial method.

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
        # NOTE: for different spatial methods the matrix may need to be adjusted
        # to account for Dirichlet boundary conditions. Here, we just have the default
        # behaviour that the mass matrix is the identity.

        # Create appropriate submesh by combining submeshes in domain
        submesh = self.mesh.combine_submeshes(*symbol.domain)

        # Get number of points in primary dimension
        n = submesh[0].npts

        # Create mass matrix for primary dimension
        prim_mass = eye(n)

        # Get number of points in secondary dimension
        sec_pts = len(submesh)

        mass = kron(eye(sec_pts), prim_mass)
        return pybamm.Matrix(mass)

    def compute_diffusivity(
        self, symbol, extrapolate_left=None, extrapolate_right=None
    ):
        """Compute the diffusivity at edges of cells.
        Could interpret this as: find diffusivity as
        off grid locations
        """
        # Default behaviour (identity operator): return symbol
        return symbol
