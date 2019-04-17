#
# A general spatial method class
#


class SpatialMethod:
    """
    A general spatial methods class.
    All spatial methods will follow the general form of SpatialMethod in
    that they contain a method for broadcasting variables onto a mesh,
    a gradient operator, and a diverence operator.

    Parameters
    ----------
    mesh : :class: `pybamm.Mesh`
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
        Broadcast symbol to a specified domain.

        Parameters
        ----------
        symbol : :class:`pybamm.Symbol`
            The symbol to be broadcasted
        domain : iterable of strings
            The domain to broadcast to

        Returns
        -------
        broadcasted_symbol: class: `pybamm.Array`
            The discretised symbol of the correct size for
            the spatial method
        """
        raise NotImplementedError

    def gradient(self, symbol, discretised_symbol, boundary_conditions):
        """
        Implements the gradient for a spatial method.

        Parameters
        ----------
        symbol: :class:`pybamm.Symbol`
            The symbol that we will take the gradient of.
        discretised_symbol: :class:`pybamm.Array`
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
        discretised_symbol: :class:`pybamm.Array`
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
        discretised_symbol: :class:`pybamm.Array`
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
        discretised_symbol: :class:`pybamm.Array`
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
        raise NotImplementedError

    # We could possibly move the following outside of SpatialMethod
    # depending on the requirements of the FiniteVolume

    def compute_diffusivity(self, extrapolate_left=False, extrapolate_right=False):
        """Compute the diffusivity at edges of cells.
        Could interpret this as: find diffusivity as
        off grid locations
        """
        raise NotImplementedError
