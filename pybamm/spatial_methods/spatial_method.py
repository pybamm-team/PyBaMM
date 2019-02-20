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
        raise NotImplementedError

    def broadcast(self, symbol, domain):
        """
        Broadcast symbol to a specified domain.

        Parameters
        ----------
        symbol : :class:`pybamm.Symbol`
            The symbol to be broadcasted
        domain : iterable of string
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
        broadcasted_symbol: class: pybamm.Array
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

    def divergence(self, symbol, broadcasted_symbol, boundary_conditions):
        """
        Implements the divergence for a spatial method.

        Parameters
        ----------
        symbol: :class:`pybamm.Symbol`
            The symbol that we will take the gradient of.
        broadcasted_symbol: class: pybamm.Array
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

    # We could possibly move the following outside of SpatialMethod
    # depending on the requirements of the FiniteVolume

    def compute_diffusivity(self):
        """Compute the diffusivity at edges of cells.
        Could interpret this as: find diffusivity as
        off grid locations
        """
        raise NotImplementedError
