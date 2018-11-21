import numpy as np


# TODO: make general Particle class within which Spherical particle also sits.
class SphericalParticle:
    """The general spherical particle class that contains the elements that are
    common to all of subclasses of spherical particles.
    Parameters
    ----------
    name : string
        Indicates whether in negative or positive electrode
            * 'negative particle': negative electrode particle
            * 'positive particle': positive electrode particle
    u: numpy array
        contains the current state of the particle
    ops:
    """

    def __init__(self):
        self.u = []  # going to call this u in every SubModel class
        # so that it is easy to pull out

    def initial_conditions(self, name, param, mesh):
        if name == "negative particle":
            self.u = param.cn0 * np.ones_like(mesh.r)
        elif name == "positive particle":
            self.u = param.cp0 * np.ones_like(mesh.r)

    def update(self, u):
        self.u = u


# only entering the PDE and BCs here (enter bcs in subclass because will
# be effected by deformation, for example)
class StandardParticle(SphericalParticle):
    """The standard particle class: spherically symmetric, standard diffusion,
    no deformation, no stress assisted diffusion
    """

    def boundary_conditions(self, param):
        flux_bcs = 1
        return flux_bcs

    def rhs(self, operators, flux_bcs):
        # Calculate internal flux
        N_internal = -operators.gradr(self.y)

        # Add boundary conditions (Neumann)
        flux_bc_left, flux_bc_right = flux_bcs
        N = np.concatenate([flux_bc_left, N_internal, flux_bc_right])

        # Calculate time derivative
        dudt = -operators.divr(N)
        return dudt
