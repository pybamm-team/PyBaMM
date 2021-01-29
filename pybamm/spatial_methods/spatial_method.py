#
# A general spatial method class
#
import pybamm
import numpy as np
from scipy.sparse import eye, kron, coo_matrix, csr_matrix, vstack


class SpatialMethod:
    """
    A general spatial methods class, with default (trivial) behaviour for some spatial
    operations.
    All spatial methods will follow the general form of SpatialMethod in
    that they contain a method for broadcasting variables onto a mesh,
    a gradient operator, and a divergence operator.

    Parameters
    ----------
    mesh : :class: `pybamm.Mesh`
        Contains all the submeshes for discretisation
    """

    def __init__(self, options=None):

        self.options = {"extrapolation": {"order": "linear", "use bcs": False}}

        # update double-layered dict
        if options:
            for opt, val in options.items():
                if isinstance(val, dict):
                    self.options[opt].update(val)
                else:
                    self.options[opt] = val

        self._mesh = None

    def build(self, mesh):
        # add npts_for_broadcast to mesh domains for this particular discretisation
        for dom in mesh.keys():
            mesh[dom].npts_for_broadcast_to_nodes = mesh[dom].npts
        self._mesh = mesh

    def _get_auxiliary_domain_repeats(self, auxiliary_domains, tertiary_only=False):
        """
        Helper method to read the 'auxiliary_domain' meshes
        """
        if tertiary_only is False and "secondary" in auxiliary_domains:
            sec_mesh_npts = self.mesh.combine_submeshes(
                *auxiliary_domains["secondary"]
            ).npts
        else:
            sec_mesh_npts = 1
        if "tertiary" in auxiliary_domains:
            tert_mesh_npts = self.mesh.combine_submeshes(
                *auxiliary_domains["tertiary"]
            ).npts
        else:
            tert_mesh_npts = 1
        return sec_mesh_npts * tert_mesh_npts

    @property
    def mesh(self):
        return self._mesh

    def spatial_variable(self, symbol):
        """
        Convert a :class:`pybamm.SpatialVariable` node to a linear algebra object that
        can be evaluated (here, a :class:`pybamm.Vector` on either the nodes or the
        edges).

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
        repeats = self._get_auxiliary_domain_repeats(symbol.auxiliary_domains)
        if symbol.evaluates_on_edges("primary"):
            entries = np.tile(symbol_mesh.edges, repeats)
        else:
            entries = np.tile(symbol_mesh.nodes, repeats)
        return pybamm.Vector(
            entries, domain=symbol.domain, auxiliary_domains=symbol.auxiliary_domains
        )

    def broadcast(self, symbol, domain, auxiliary_domains, broadcast_type):
        """
        Broadcast symbol to a specified domain.

        Parameters
        ----------
        symbol : :class:`pybamm.Symbol`
            The symbol to be broadcasted
        domain : iterable of strings
            The domain to broadcast to
        auxiliary_domains : dict of strings
            The auxiliary domains for broadcasting
        broadcast_type : str
            The type of broadcast: 'primary to node', 'primary to edges', 'secondary to
            nodes', 'secondary to edges', 'full to nodes' or 'full to edges'

        Returns
        -------
        broadcasted_symbol: class: `pybamm.Symbol`
            The discretised symbol of the correct size for the spatial method
        """

        primary_domain_size = sum(
            self.mesh[dom].npts_for_broadcast_to_nodes for dom in domain
        )
        secondary_domain_size = self._get_auxiliary_domain_repeats(auxiliary_domains)
        full_domain_size = primary_domain_size * secondary_domain_size
        if broadcast_type.endswith("to edges"):
            # add one point to each domain for broadcasting to edges
            primary_domain_size += 1
            full_domain_size = primary_domain_size * secondary_domain_size
            secondary_domain_size += 1

        if broadcast_type.startswith("primary"):
            # Make copies of the child stacked on top of each other
            sub_vector = np.ones((primary_domain_size, 1))
            if symbol.shape_for_testing == ():
                out = symbol * pybamm.Vector(sub_vector)
            else:
                # Repeat for secondary points
                matrix = csr_matrix(kron(eye(symbol.shape_for_testing[0]), sub_vector))
                out = pybamm.Matrix(matrix) @ symbol
            out.domain = domain
        elif broadcast_type.startswith("secondary"):
            # Make copies of the child stacked on top of each other
            identity = eye(symbol.shape[0])
            matrix = vstack([identity for _ in range(secondary_domain_size)])
            out = pybamm.Matrix(matrix) @ symbol
        elif broadcast_type.startswith("full"):
            out = symbol * pybamm.Vector(np.ones(full_domain_size), domain=domain)

        out.auxiliary_domains = auxiliary_domains.copy()
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

    def laplacian(self, symbol, discretised_symbol, boundary_conditions):
        """
        Implements the laplacian for a spatial method.

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
            Contains the result of acting the discretised laplacian on
            the child discretised_symbol
        """
        raise NotImplementedError

    def gradient_squared(self, symbol, discretised_symbol, boundary_conditions):
        """
        Implements the inner product of the gradient with itself for a spatial method.

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
            Contains the result of taking the inner product of the result of acting
            the discretised gradient on the child discretised_symbol with itself
        """
        raise NotImplementedError

    def integral(self, child, discretised_child, integration_dimension):
        """
        Implements the integral for a spatial method.

        Parameters
        ----------
        child: :class:`pybamm.Symbol`
            The symbol to which is being integrated
        discretised_child: :class:`pybamm.Symbol`
            The discretised symbol of the correct size
        integration_dimension : str, optional
            The dimension in which to integrate (default is "primary")

        Returns
        -------
        :class: `pybamm.Array`
            Contains the result of acting the discretised integral on
            the child discretised_symbol
        """
        raise NotImplementedError

    def indefinite_integral(self, child, discretised_child, direction):
        """
        Implements the indefinite integral for a spatial method.

        Parameters
        ----------
        child: :class:`pybamm.Symbol`
            The symbol to which is being integrated
        discretised_child: :class:`pybamm.Symbol`
            The discretised symbol of the correct size
        direction : str
            The direction of integration

        Returns
        -------
        :class: `pybamm.Array`
            Contains the result of acting the discretised indefinite integral on
            the child discretised_symbol
        """
        raise NotImplementedError

    def boundary_integral(self, child, discretised_child, region):
        """
        Implements the boundary integral for a spatial method.

        Parameters
        ----------
        child: :class:`pybamm.Symbol`
            The symbol to which is being integrated
        discretised_child: :class:`pybamm.Symbol`
            The discretised symbol of the correct size
        region: str
            The region of the boundary over which to integrate. If region is None
            (default) the integration is carried out over the entire boundary. If
            region is `negative tab` or `positive tab` then the integration is only
            carried out over the appropriate part of the boundary corresponding to
            the tab.

        Returns
        -------
        :class: `pybamm.Array`
            Contains the result of acting the discretised boundary integral on
            the child discretised_symbol
        """
        raise NotImplementedError

    def delta_function(self, symbol, discretised_symbol):
        """
        Implements the delta function on the approriate side for a spatial method.

        Parameters
        ----------
        symbol: :class:`pybamm.Symbol`
            The symbol to which is being integrated
        discretised_symbol: :class:`pybamm.Symbol`
            The discretised symbol of the correct size
        """
        raise NotImplementedError

    def internal_neumann_condition(
        self, left_symbol_disc, right_symbol_disc, left_mesh, right_mesh
    ):
        """
        A method to find the internal neumann conditions between two symbols
        on adjacent subdomains.

        Parameters
        ----------
        left_symbol_disc : :class:`pybamm.Symbol`
            The discretised symbol on the left subdomain
        right_symbol_disc : :class:`pybamm.Symbol`
            The discretised symbol on the right subdomain
        left_mesh : list
            The mesh on the left subdomain
        right_mesh : list
            The mesh on the right subdomain
        """

        raise NotImplementedError

    def preprocess_external_variables(self, var):
        return {}

    def boundary_value_or_flux(self, symbol, discretised_child, bcs=None):
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
        bcs : dict (optional)
            The boundary conditions. If these are supplied and "use bcs" is True in
            the options, then these will be used to improve the accuracy of the
            extrapolation.

        Returns
        -------
        :class:`pybamm.MatrixMultiplication`
            The variable representing the surface value.
        """

        if bcs is None:
            bcs = {}
        if self._get_auxiliary_domain_repeats(discretised_child.auxiliary_domains) > 1:
            raise NotImplementedError("Cannot process 2D symbol in base spatial method")
        if isinstance(symbol, pybamm.BoundaryGradient):
            raise TypeError("Cannot process BoundaryGradient in base spatial method")
        n = sum(self.mesh[dom].npts for dom in discretised_child.domain)
        if symbol.side == "left":
            # coo_matrix takes inputs (data, (row, col)) and puts data[i] at the point
            # (row[i], col[i]) for each index of data. Here we just want a single point
            # with value 1 at (0,0).
            # Convert to a csr_matrix to allow indexing and other functionality
            left_vector = csr_matrix(coo_matrix(([1], ([0], [0])), shape=(1, n)))
            bv_vector = pybamm.Matrix(left_vector)
        elif symbol.side == "right":
            # as above, but now we want a single point with value 1 at (0, n-1)
            right_vector = csr_matrix(coo_matrix(([1], ([0], [n - 1])), shape=(1, n)))
            bv_vector = pybamm.Matrix(right_vector)

        out = bv_vector @ discretised_child
        # boundary value removes domain
        out.clear_domains()
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
        n = submesh.npts

        # Create mass matrix for primary dimension
        prim_mass = eye(n)

        # Get number of points in secondary dimension
        second_dim_repeats = self._get_auxiliary_domain_repeats(symbol.domains)

        # Convert to csr_matrix as required by some solvers
        mass = csr_matrix(kron(eye(second_dim_repeats), prim_mass))
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
        return bin_op._binary_new_copy(disc_left, disc_right)

    def concatenation(self, disc_children):
        """Discrete concatenation object.

        Parameters
        ----------
        disc_children : list
            List of discretised children

        Returns
        -------
        :class:`pybamm.DomainConcatenation`
            Concatenation of the discretised children
        """
        return pybamm.domain_concatenation(disc_children, self.mesh)
