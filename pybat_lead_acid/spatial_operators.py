import numpy as np

def get_spatial_operators(spatial_discretisation, mesh):
    """Returns functions that calculate the spatial derivatives.

    Parameters
    ----------
    spatial_discretisation : string
        The spatial discretisation scheme (see pybat_lead_acid.solver).
    mesh : pybat_lead_acid.mesh.Mesh() instance
        The mesh used for the spatial discretisation.

    Returns
    -------
    grad : function
        A function that calculates the gradient along the mesh.
    div : function
        A function that calculates the divergence along the mesh.

    """
    if spatial_discretisation == "Finite Volumes":
        def grad(y):
            """Calculates the 1D gradient using Finite Volumes.

            Parameters
            ----------
            y : array_like, shape (n,)
                The variable whose gradient is to be calculated.

            Returns
            -------
            grad_y : array_like, shape (n-1,)
                The gradient, grad(y).

            """
            # Run some basic checks on inputs
            assert y.shape == mesh.xc.shape, \
                """xc and y should have the same shape,
                but xc.shape = {} and yc.shape = {}""".format(mesh.xc.shape,
                                                              y.shape)

            # Calculate internal flux
            grad_y = np.diff(y)/np.diff(mesh.xc)

            return grad_y

        def div(N):
            """Calculates the 1D divergence using Finite Volumes.

            Parameters
            ----------
            N : array_like, shape (n,)
                The flux whose divergence is to be calculated.

            Returns
            -------
            div_N : array_like, shape (n-1,)
                The divergence, div(N).

            """
            # Run basic checks on inputs
            assert N.shape == mesh.x.shape, \
                """xc and y should have the same shape,
                but x.shape = {} and N.shape = {}""".format(mesh.x.shape,
                                                            N.shape)

            div_N = np.diff(N)/np.diff(mesh.x)
            return div_N

    return grad, div
