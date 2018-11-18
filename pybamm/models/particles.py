import numpy as np


class SphericalParticle:
    """The general spherical particle class that contains the elements that are common to all of
    subclasses of spherical particles.

    Parameters
    ----------
    name : string
        Indicates whether in negative or positive electrode
            * 'neg': negative electrode
            * 'pos': positive electrode
    c: numpy array
        contains the current state of the particle
    ops:
    """

    def __init__(self, name, param, mesh, operators):
        self.name = name

        if name == 'neg':
            self.c = param.cn0 * np.ones_like(mesh.r)
        elif name == 'pos':
            self.c = param.cp0 * np.ones_like(mesh.r)

        # TODO: figure out how to fit electrochemical reactions in here

        self.grad = operators.gradr
        self.div = operators.divr

    def update(self, c, G):
        self.c = c
        self.



# only thing that is really changing here is the model equation itself
class StandardParticle(SphericalParticle):
    """The standard particle class: spherically symmetric, standard diffusion,
    no deformation, no stress assisted diffusion

    Parameters
    ----------
    name : string
        Indicates whether in negative or positive electrode
            * 'neg': negative electrode
            * 'pos': positive electrode
    c: numpy array
        contains the current state of the particle
    ops:
    """

    def rhs(self):
        # Calculate internal flux
        N_internal = -self.grad(self.c)