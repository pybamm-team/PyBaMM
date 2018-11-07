from pybat_lead_acid.models import components

import numpy as np

KNOWN_MODELS = ["Simple Diffusion",
                ]
# !Remember to update docstring with any new models!

class Model:
    """The (dimensionless) physical/chemical model to be used in the simulation.

    Parameters
    ----------
    name : string
        The model name:
            * "Simple Diffusion" : One-dimensional diffusion equation:
                dt/dt = d2c/dx2

    """
    def __init__(self, name):
        if name not in KNOWN_MODELS:
            raise NotImplementedError("""Model '{}' is not implemented.
                                      Valid choices: one of '{}'."""
                                      .format(name, KNOWN_MODELS))
        self.name = name

    def get_initial_conditions(self, param, mesh):
        """Calculates the initial conditions for the simulation.

        Parameters
        ----------
        param : pybat_lead_acid.parameters.Parameters() instance
            The model parameters.
        mesh : pybat_lead_acid.mesh.Mesh() instance
            The mesh used for discretisation.

        Returns
        -------
        y0 : array_like
            A concatenated vector of all the initial conditions.
        inits_dict : dict
            A dictionary of the initial conditions, for use by other models.

        """
        inits_dict = {}
        if self.name == "Simple Diffusion":
            c0 = np.ones_like(mesh.xc)

            # Create y0 and inits_dict
            y0 = c0
            inits_dict['c0'] = c0

        return y0, inits_dict

    def get_pdes_rhs(self, vars, param, operators):
        """Calculates the spatial derivates of the spatial terms in the PDEs
           and returns the right-hand side to be used by the ODE solver
           (Method of Lines).

        Parameters
        ----------
        vars : pybat_lead_acid.variables.Variables() instance
            The variables of the model.
        param : pybat_lead_acid.parameters.Parameters() instance
            The model parameters.
        grad : function
            The gradient operator.
        div : function
            The divergence operator.

        Returns
        -------
        dydt : array_like
            A concatenated vector of all the derivatives.
        derivs_dict : dict
            A dictionary of the derivatives, for use by other models.

        """
        derivs_dict = {}
        if self.name == "Simple Diffusion":
            lbc = np.array([0])
            rbc = np.array([0])
            dcdt = components.simple_diffusion(vars.c, operators, lbc, rbc)

            # Create dydt and derivs_dict
            dydt = dcdt
            derivs_dict['dcdt'] = dcdt

        return dydt, derivs_dict
