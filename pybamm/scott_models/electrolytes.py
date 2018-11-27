import numpy as np


class Electrolyte:
    """This contains all methods common to all Electrolytes"""

    def __init__(self):
        self.u = []  # going to call this u in every SubModel class so that
        # it is easy to pull out. This may be a reason to make a
        # dedicated SubModel class from which all
        # sub models: electrolyte, particles, etc derive

    def initial_conditions(self, name, param, mesh):
        self.c = param.c0 * np.ones_like(mesh.xc)
        self.phi = np.ones_like(mesh.xc)

    def update(self, u):
        # TODO: split u up properly
        c, phi = u
        self.c = c
        self.phi = phi


class StefanMaxwell1D(Electrolyte):
    """A class specific to the Stefan-Maxwell Electrolyte"""

    def rhs(self, param, operators, flux_bcs):
        # Calculate internal flux
        N_internal = -operators.gradx(self.u)

        # Add boundary conditions (Neumann)
        flux_bc_left, flux_bc_right = flux_bcs
        N = np.concatenate([flux_bc_left, N_internal, flux_bc_right])

        # Calculate time derivative
        dudt = -operators.divx(N) + param.s * self.j
        return dudt


class NernstPlanck1D(Electrolyte):
    """A class specific to the Nernst-Planck Electrolyte"""

    def rhs(self, param, operators, flux_bcs):
        # Calculate internal flux
        N_internal = -operators.gradx(self.u)

        # Add boundary conditions (Neumann)
        flux_bc_left, flux_bc_right = flux_bcs
        N = np.concatenate([flux_bc_left, N_internal, flux_bc_right])

        # Calculate time derivative
        dudt = -operators.divx(N) + param.s * self.j
        return dudt
