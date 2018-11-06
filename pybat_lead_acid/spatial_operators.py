import numpy as np

class Operators:
    """Contains functions that calculate the spatial derivatives.

    Parameters
    ----------
    spatial_discretisation : string
        The spatial discretisation scheme (see pybat_lead_acid.solver).
    mesh : pybat_lead_acid.mesh.Mesh() instance
        The mesh used for the spatial discretisation.

    """
    def __init__(self, spatial_discretisation, mesh):
        self.spatial_discretisation = spatial_discretisation
        self.mesh = mesh

    def grad_x(self, y):
        """Calculates the 1D gradient using Finite Volumes.

        Parameters
        ----------
        y : array_like, shape (n,)
            The variable whose gradient is to be calculated.

        Returns
        -------
        array_like, shape (n-1,)
            The gradient, grad(y).

        """
        if self.spatial_discretisation == "Finite Volumes":
            # Run some basic checks on inputs
            assert y.shape == self.mesh.xc.shape, \
                """xc and y should have the same shape,
                but xc.shape = {} and yc.shape = {}""".format(self.mesh.xc.shape,
                                                              y.shape)

            # Calculate internal flux
            return np.diff(y)/np.diff(self.mesh.xc)

    def div_x(self, N):
        """Calculates the 1D divergence using Finite Volumes.

        Parameters
        ----------
        N : array_like, shape (n,)
            The flux whose divergence is to be calculated.

        Returns
        -------
        array_like, shape (n-1,)
            The divergence, div(N).

        """
        if self.spatial_discretisation == "Finite Volumes":
            # Run basic checks on inputs
            assert N.shape == self.mesh.x.shape, \
                """xc and y should have the same shape,
                but x.shape = {} and N.shape = {}""".format(self.mesh.x.shape,
                                                            N.shape)

            return np.diff(N)/np.diff(self.mesh.x)
