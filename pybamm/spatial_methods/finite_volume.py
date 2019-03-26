#
# Finite Volume discretisation class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np
from scipy.sparse import spdiags
from scipy.sparse import eye
from scipy.sparse import kron


class FiniteVolume(pybamm.SpatialMethod):
    """
    A class which implements the steps specific to the finite volume method during
    discretisation.

    Parameters
    ----------
    mesh : :class:`pybamm.Mesh` (or subclass)
        Contains all the submeshes for discretisation

    **Extends:"": :class:`pybamm.SpatialMethod`
    """

    def __init__(self, mesh):
        # add npts_for_broadcast to mesh domains for this particular discretisation
        for dom in mesh.keys():
            for i in range(len(mesh[dom])):
                mesh[dom][i].npts_for_broadcast = mesh[dom][i].npts
        super().__init__(mesh)

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
        # for finite volume we use the cell centres
        if symbol.name in ["x", "r"]:
            symbol_mesh = self.mesh.combine_submeshes(*symbol.domain)
            return pybamm.Vector(symbol_mesh[0].nodes)
        else:
            raise NotImplementedError("3D meshes not yet implemented")

    def broadcast(self, symbol, domain):
        """
        Broadcast symbol to a specified domain. To do this, calls
        :class:`pybamm.NumpyBroadcast`

        See :meth: `pybamm.SpatialMethod.broadcast`
        """

        broadcasted_symbol = pybamm.NumpyBroadcast(symbol, domain, self.mesh)

        # if the broadcasted symbol evaluates to a constant value, replace the
        # symbol-Vector multiplication with a single array
        if broadcasted_symbol.is_constant():
            broadcasted_symbol = pybamm.Array(
                broadcasted_symbol.evaluate(), domain=broadcasted_symbol.domain
            )

        return broadcasted_symbol

    def gradient(self, symbol, discretised_symbol, boundary_conditions):
        """Matrix-vector multiplication to implement the gradient operator.
        See :meth:`pybamm.SpatialMethod.gradient`
        """
        # Check that boundary condition keys are hashes (ids)
        for key in boundary_conditions.keys():
            assert isinstance(key, int), TypeError(
                "boundary condition keys should be hashes, not {}".format(type(key))
            )
        # Discretise symbol
        domain = symbol.domain
        # Add Dirichlet boundary conditions, if defined
        if symbol.id in boundary_conditions:
            bcs = boundary_conditions[symbol.id]
            # get boundary conditions and edit domain
            if "left" in bcs.keys():
                lbc = bcs["left"]
                domain = [domain[0] + "_left ghost cell"] + domain
            else:
                lbc = None
            if "right" in bcs.keys():
                rbc = bcs["right"]
                domain = domain + [domain[-1] + "_right ghost cell"]
            else:
                rbc = None
            # add ghost nodes
            discretised_symbol = self.add_ghost_nodes(
                symbol, discretised_symbol, lbc, rbc
            )

        # note in 1D spherical grad and normal grad are the same
        gradient_matrix = self.gradient_matrix(domain)

        # set ghost cells
        gradient_matrix.has_left_ghost_cell = discretised_symbol.has_left_ghost_cell
        gradient_matrix.has_right_ghost_cell = discretised_symbol.has_right_ghost_cell

        return gradient_matrix @ discretised_symbol

    def gradient_matrix(self, domain):
        """
        Gradient matrix for finite volumes in the appropriate domain.
        Equivalent to grad(y) = (y[1:] - y[:-1])/dx

        Parameters
        ----------
        domain : list
            The domain(s) in which to compute the gradient matrix

        Returns
        -------
        :class:`pybamm.Matrix`
            The (sparse) finite volume gradient matrix for the domain
        """
        # Create appropriate submesh by combining submeshes in domain
        submesh_list = self.mesh.combine_submeshes(*domain)

        # can just use 1st entry of list to obtain the point etc
        submesh = submesh_list[0]

        # Create 1D matrix using submesh
        n = submesh.npts
        e = 1 / submesh.d_nodes
        data = np.vstack(
            [np.concatenate([-e, np.array([0])]), np.concatenate([np.array([0]), e])]
        )
        diags = np.array([0, 1])
        sub_matrix = spdiags(data, diags, n - 1, n)

        # second dim length
        second_dim_len = len(submesh_list)

        # generate full matrix from the submatrix
        matrix = kron(eye(second_dim_len), sub_matrix)

        return pybamm.Matrix(matrix)

    def divergence(self, symbol, discretised_symbol, boundary_conditions):
        """Matrix-vector multiplication to implement the divergence operator.
        See :meth:`pybamm.SpatialMethod.divergence`
        """
        # Check that boundary condition keys are hashes (ids)
        for key in boundary_conditions.keys():
            assert isinstance(key, int), TypeError(
                "boundary condition keys should be hashes, not {}".format(type(key))
            )

        domain = symbol.domain
        submesh_list = self.mesh.combine_submeshes(*domain)
        # create a bc vector of length equal to the number variables
        # (only has non zero entries for neumann bcs)

        prim_dim = submesh_list[0].npts
        second_dim = len(submesh_list)
        total_pts = prim_dim * second_dim

        # Add Neumann boundary conditions if defined
        if symbol.id in boundary_conditions:
            bcs = boundary_conditions[symbol.id]
            # get boundary conditions and edit domain
            if "left" in bcs.keys():
                lbc = bcs["left"]
                discretised_symbol = pybamm.NumpyConcatenation(lbc, discretised_symbol)
            if "right" in bcs.keys():
                rbc = bcs["right"]
                discretised_symbol = pybamm.NumpyConcatenation(discretised_symbol, rbc)

            # taking divergence removes ghost cells
            discretised_symbol.has_left_ghost_cell = False
            discretised_symbol.has_right_ghost_cell = False

            # doing via loop so that it is easier to implement x varing bcs
            bcs_symbol = pybamm.Vector(np.array([]))  # empty vector
            for i in range(len(submesh_list)):
                # only the interior equations:
                interior = pybamm.Vector(np.zeros(prim_dim - 2))
                left = -lbc / pybamm.Vector(np.array([submesh_list[i].d_edges[0]]))
                right = rbc / pybamm.Vector(np.array([submesh_list[i].d_edges[-1]]))
                bcs_symbol = pybamm.NumpyConcatenation(
                    bcs_symbol, left, interior, right
                )

            # now we must create a matrix of size (npts * (npts -1) )
            # this is a different size to the one created when we have
            # flux boundary conditions so need a flag
            divergence_matrix = self.divergence_matrix(domain, bc_type="neumann")

            # only need interior edges for spherical neumann
            edges = submesh_list[0].edges[1:-1]

        else:
            divergence_matrix = self.divergence_matrix(domain)
            bcs_vec = np.zeros(total_pts)
            bcs_symbol = pybamm.Vector(bcs_vec)
            # need all edges for spherical dirichlet
            edges = submesh_list[0].edges

        # check for particle domain
        if ("negative particle" or "positive particle") in domain:

            # create np.array of repeated submesh[0].nodes
            r_numpy = np.kron(np.ones(second_dim), submesh_list[0].nodes)
            r_edges_numpy = np.kron(np.ones(second_dim), edges)

            r = pybamm.Vector(r_numpy)
            r_edges = pybamm.Vector(r_edges_numpy)

            # for clarity, we are implicitly multiplying the the lbc by r^2=0
            # and the rbc by r^2=1. But lbc is 0 so we don't need to do
            # any r_edges^2 operations on bcs_symbol
            out = (1 / (r ** 2)) * (
                divergence_matrix @ ((r_edges ** 2) * discretised_symbol) + bcs_symbol
            )
        else:
            out = divergence_matrix @ discretised_symbol + bcs_symbol

        return out

    def divergence_matrix(self, domain, bc_type="dirichlet"):
        """
        Divergence matrix for finite volumes in the appropriate domain.
        Equivalent to div(N) = (N[1:] - N[:-1])/dx

        Parameters
        ----------
        domain : list
            The domain(s) in which to compute the divergence matrix

        Returns
        -------
        :class:`pybamm.Matrix`
            The (sparse) finite volume divergence matrix for the domain
        """
        # Create appropriate submesh by combining submeshes in domain
        submesh_list = self.mesh.combine_submeshes(*domain)

        # can just use 1st entry of list to obtain the point etc
        submesh = submesh_list[0]
        e = 1 / submesh.d_edges

        # Create matrix using submesh
        n = submesh.npts + 1
        if bc_type == "dirichlet":
            data = np.vstack(
                [
                    np.concatenate([-e, np.array([0])]),
                    np.concatenate([np.array([0]), e]),
                ]
            )
            diags = np.array([0, 1])
            sub_matrix = spdiags(data, diags, n - 1, n)
        elif bc_type == "neumann":
            # we don't have to act on bc fluxes which are now in
            # the bc vector
            data = np.vstack([-e[1:], e[:-1]])
            diags = np.array([-1, 0])
            sub_matrix = spdiags(data, diags, n - 1, n - 2)
        else:
            raise NotImplementedError(
                "Can only process Neumann or Dirichlet boundary conditions"
            )

        # repeat matrix for each node in secondary dimensions
        second_dim_len = len(submesh_list)
        # generate full matrix from the submatrix
        matrix = kron(eye(second_dim_len), sub_matrix)
        return pybamm.Matrix(matrix)

    def integral(self, domain, symbol, discretised_symbol):
        """Vector-vector dot product to implement the integral operator.
        See :meth:`pybamm.BaseDiscretisation.integral`
        """
        # Calculate integration vector
        integration_vector = self.definite_integral_vector(domain)
        # Check for particle domain
        if ("negative particle" or "positive particle") in symbol.domain:
            submesh_list = self.mesh.combine_submeshes(*symbol.domain)
            second_dim = len(submesh_list)
            r_numpy = np.kron(np.ones(second_dim), submesh_list[0].nodes)
            r = pybamm.Vector(r_numpy)
            out = 2 * np.pi * integration_vector @ (discretised_symbol * r)
        else:
            out = integration_vector @ discretised_symbol
        out.domain = []
        return out

    def definite_integral_vector(self, domain):
        """
        Vector for finite-volume implementation of the definite integral

        .. math::
            I = \\int_{a}^{b}\\!f(s)\\,ds

        for where :math:`a` and :math:`b` are the left-hand and right-hand boundaries of
        the domain respectively

        Parameters
        ----------
        domain : list
            The domain(s) of integration

        Returns
        -------
        :class:`pybamm.Vector`
            The finite volume integral vector for the domain
        """
        # Create appropriate submesh by combining submeshes in domain
        submesh_list = self.mesh.combine_submeshes(*domain)

        # Create vector of ones using submesh
        vector = np.array([])
        for submesh in submesh_list:
            vector = np.append(vector, submesh.d_edges * np.ones_like(submesh.nodes))

        return pybamm.Vector(vector)

    def add_ghost_nodes(self, symbol, discretised_symbol, lbc=None, rbc=None):
        """
        Add Dirichlet boundary conditions via ghost nodes.

        For a boundary condition "y = a at the left-hand boundary",
        we concatenate a ghost node to the start of the vector y with value "2*a - y1"
        where y1 is the value of the first node.
        Similarly for the right-hand boundary condition.

        Currently, Dirichlet boundary conditions can only be applied on state
        variables (e.g. concentration, temperature), and not on expressions.
        To access the value of the first node (y1), we create a "first_node" object
        which is a StateVector whose y_slice is the start of the y_slice of
        discretised_symbol.
        Similarly, the last node is a StateVector whose y_slice is the end of the
        y_slice of discretised_symbol

        Parameters
        ----------
        discretised_symbol : :class:`pybamm.StateVector` (size n)
            The discretised variable (a state vector) to which to add ghost nodes
        lbc : :class:`pybamm.Scalar`
            Dirichlet boundary condition on the left-hand side. Default is None.
        rbc : :class:`pybamm.Scalar`
            Dirichlet boundary condition on the right-hand side. Default is None.

        Returns
        -------
        :class:`pybamm.Concatenation` (size n+1 or n+2)
            Concatenation of the variable (a state vector) and ghost nodes

        """
        if isinstance(discretised_symbol, pybamm.StateVector):
            y_slice_start = discretised_symbol.y_slice.start
            y_slice_stop = discretised_symbol.y_slice.stop
        elif isinstance(discretised_symbol, pybamm.Concatenation):
            y_slice_start = discretised_symbol.children[0].y_slice.start
            y_slice_stop = discretised_symbol.children[-1].y_slice.stop
        else:
            raise TypeError(
                """
                discretised_symbol must be a StateVector or Concatenation, not '{}'
                """.format(
                    type(discretised_symbol)
                )
            )
        y = np.arange(y_slice_start, y_slice_stop)

        # reshape y_slices into more helpful form
        submesh_list = self.mesh.combine_submeshes(*symbol.domain)
        if isinstance(submesh_list[0].npts, list):
            NotImplementedError("Can only take in 1D primary directions")

        size = [len(submesh_list), submesh_list[0].npts]
        y = np.reshape(y, size)
        y_left = y[:, 0]
        y_right = y[:, -1]

        new_discretised_symbol = pybamm.Vector(np.array([]))  # starts empty

        for i in range(len(submesh_list)):
            y_slice_start = y_left[i]
            y_slice_stop = y_right[i]
            # both ghost cells
            left_ghost_cell = 2 * lbc - first_node
            right_ghost_cell = 2 * rbc - last_node
            # left ghost cell
            first_node = pybamm.StateVector(slice(y_slice_start, y_slice_start + 1))
            left_ghost_cell = 2 * lbc - first_node
            # middle symbol
            sub_disc_symbol = pybamm.StateVector(slice(y_slice_start, y_slice_stop + 1))
            # right ghost cell
            last_node = pybamm.StateVector(slice(y_slice_stop, y_slice_stop + 1))
            right_ghost_cell = 2 * rbc - last_node
            if lbc is not None and rbc is not None:
                # concatenate and flag ghost cells
                concatenated_sub_disc_symbol = pybamm.NumpyConcatenation(
                    left_ghost_cell, sub_disc_symbol, right_ghost_cell
                )
                concat.has_left_ghost_cell = True
                concat.has_right_ghost_cell = True
            elif lbc is not None:
                # concatenate and flag ghost cells
                concatenated_sub_disc_symbol = pybamm.NumpyConcatenation(
                    left_ghost_cell, sub_disc_symbol
                )
                concat.has_left_ghost_cell = True
            elif rbc is not None:
                # right ghost cell only
                # concatenate and flag ghost cells
                concatenated_sub_disc_symbol = pybamm.NumpyConcatenation(
                    sub_disc_symbol, right_ghost_cell
                )
                concat.has_right_ghost_cell = True
            else:
                raise ValueError("at least one boundary condition must be provided")
            new_discretised_symbol = pybamm.NumpyConcatenation(
                new_discretised_symbol, concatenated_sub_disc_symbol
            )

        return new_discretised_symbol

    def surface_value(self, discretised_symbol):
        """
        Uses linear extrapolation to get the surface value of a variable in the
        Finite Volume Method.

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
        # Better to make class similar NodeToEdge and pass function?
        # def surface_value(array):
        #     "Linear extrapolation for surface value"
        #     array[-1] + (array[-1] - array[-2]) / 2
        # ... or make StateVector and add?
        y_slice_stop = discretised_symbol.y_slice.stop
        last_node = pybamm.StateVector(slice(y_slice_stop - 1, y_slice_stop))
        penultimate_node = pybamm.StateVector(slice(y_slice_stop - 2, y_slice_stop - 1))
        surface_value = last_node + (last_node - penultimate_node) / 2
        return surface_value

    #######################################################
    # Can probably be moved outside of the spatial method
    ######################################################

    def compute_diffusivity(
        self, symbol, extrapolate_left=False, extrapolate_right=False
    ):
        """
        Compute the diffusivity at cell edges, based on the diffusivity at cell nodes.
        For now we just take the arithemtic mean, though it may be better to take the
        harmonic mean based on [1].

        [1] Recktenwald, Gerald. "The control-volume finite-difference approximation to
        the diffusion equation." (2012).

        Parameters
        ----------
        symbol : :class:`pybamm.Symbol`
            Symbol to be averaged. When evaluated, this symbol returns either a scalar
            or an array of shape (n,), where n is the number of points in the mesh for
            the symbol's domain (n = self.mesh[symbol.domain].npts)
        extrapolate_left : boolean
            Whether to extrapolate one node to the left when computing the
            diffusivity, to account for ghost cells. Default is False
        extrapolate_right : boolean
            Whether to extrapolate one node to the right when computing the
            diffusivity, to account for ghost cells. Default is False

        Returns
        -------
        :class:`pybamm.NodeToEdge`
            Averaged symbol. When evaluated, this returns either a scalar or an array of
            shape (n-1,) as appropriate.
        """

        def arithmetic_mean(array):
            """Calculate the arithemetic mean of an array"""
            mean_array = (array[1:] + array[:-1]) / 2
            if extrapolate_left:
                left_node = array[0] - (array[1] - array[0]) / 2
                mean_array = np.concatenate([np.array([left_node]), mean_array])
            if extrapolate_right:
                right_node = array[-1] - (array[-2] - array[-1]) / 2
                mean_array = np.concatenate([mean_array, np.array([right_node])])
            return mean_array

        return pybamm.NodeToEdge(symbol, arithmetic_mean)


class NodeToEdge(pybamm.SpatialOperator):
    """A node in the expression tree representing a unary operator that evaluates the
    value of its child at cell edges by averaging the value at cell nodes.

    Parameters
    ----------

    name : str
        name of the node
    child : :class:`Symbol`
        child node
    node_to_edge_function : method
        the function used to average; only acts if the child evaluates to a
        one-dimensional numpy array

    **Extends:** :class:`pybamm.SpatialOperator`
    """

    def __init__(self, child, node_to_edge_function):
        """ See :meth:`pybamm.UnaryOperator.__init__()`. """
        super().__init__(
            "node to edge ({})".format(node_to_edge_function.__name__), child
        )
        self._node_to_edge_function = node_to_edge_function

    def evaluate(self, t=None, y=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        evaluated_child = self.children[0].evaluate(t, y)
        # If the evaluated child is a numpy array of shape (n,), do the averaging
        # NOTE: Doing this check every time might be slow?
        if isinstance(evaluated_child, np.ndarray) and len(evaluated_child.shape) == 1:
            return self._node_to_edge_function(evaluated_child)
        # If not, no need to average
        else:
            return evaluated_child
