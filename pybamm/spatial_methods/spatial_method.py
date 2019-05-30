#
# A general spatial method class
#
import pybamm
import numpy as np
from scipy.sparse import eye, kron, coo_matrix


class SpatialMethod:
    """
    A general spatial methods class, with default (trivial) behaviour for broadcast,
    mass_matrix and compute_diffusivity.
    All spatial methods will follow the general form of SpatialMethod in
    that they contain a method for broadcasting variables onto a mesh,
    a gradient operator, and a diverence operator.

    Parameters
    ----------
    mesh : :class: `pybamm.Mesh`
        Contains all the submeshes for discretisation
    """

    def __init__(self, mesh):
        # add npts_for_broadcast to mesh domains for this particular discretisation
        for dom in mesh.keys():
            for i in range(len(mesh[dom])):
                mesh[dom][i].npts_for_broadcast = mesh[dom][i].npts
        self._mesh = mesh

    @property
    def mesh(self):
        return self._mesh

    def spatial_variable(self, symbol):
        """
        Convert a :class:`pybamm.SpatialVariable` node to a linear algebra object that
        can be evaluated (here, a :class:`pybamm.Vector` on the nodes).

        Parameters
        -----------
        symbol : :class:`pybamm.SpatialVariable`
            The spatial variable to be discretised.

        Returns
        -------
        :class:`pybamm.Vector`
            Contains the discretised spatial variable
        """
        symbol_mesh = self.mesh.combine_submeshes(*symbol.domain)
        return pybamm.Vector(symbol_mesh[0].nodes, domain=symbol.domain)

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
        broadcasted_symbol: class: `pybamm.Symbol`
            The discretised symbol of the correct size for the spatial method
        """
        vector_size_1D = sum(self.mesh[dom][0].npts_for_broadcast for dom in domain)
        vector_size_2D = sum(
            subdom.npts_for_broadcast for dom in domain for subdom in self.mesh[dom]
        )

        if symbol.domain == ["current collector"]:
            out = pybamm.Outer(
                symbol, pybamm.Vector(np.ones(vector_size_1D), domain=domain)
            )
        else:
            out = symbol * pybamm.Vector(np.ones(vector_size_2D), domain=domain)
        return out

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

    def boundary_value_or_flux(self, symbol, discretised_child):
        """
        Returns the boundary value or flux using the approriate expression for the
        spatial method. To do this, we create a sparse vector 'bv_vector' that extracts
        either the first (for side="left") or last (for side="right") point from
        'discretised_child'.

        Parameters
        -----------
        symbol: :class:`pybamm.Symbol`
            The boundary value or flux symbol
        discretised_child : :class:`pybamm.StateVector`
            The discretised variable from which to calculate the boundary value

        Returns
        -------
        :class:`pybamm.Variable`
            The variable representing the surface value.
        """
        if any(len(self.mesh[dom]) > 1 for dom in discretised_child.domain):
            raise NotImplementedError("Cannot process 2D symbol in base spatial method")
        if isinstance(symbol, pybamm.BoundaryFlux):
            raise TypeError("Cannot process BoundaryFlux in base spatial method")
        n = sum(self.mesh[dom][0].npts for dom in discretised_child.domain)
        if symbol.side == "left":
            # coo_matrix takes inputs (data, (row, col)) and puts data[i] at the point
            # (row[i], col[i]) for each index of data. Here we just want a single point
            # with value 1 at (0,0).
            left_vector = coo_matrix(([1], ([0], [0])), shape=(1, n))
            bv_vector = pybamm.Matrix(left_vector)
        elif symbol.side == "right":
            # as above, but now we want a single point with value 1 at (0, n-1)
            right_vector = coo_matrix(([1], ([0], [n - 1])), shape=(1, n))
            bv_vector = pybamm.Matrix(right_vector)

        try:
            out = bv_vector @ discretised_child
        except pybamm.ShapeError:
            import ipdb

            ipdb.set_trace()
        # boundary value removes domain
        out.domain = []
        return out

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

    def process_binary_operators(self, bin_op, left, right, disc_left, disc_right):
        """Discretise binary operators in model equations. Default behaviour is to
        return a new binary operator with the discretised children.

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
        return bin_op.__class__(disc_left, disc_right)
