from pybamm.models import components

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
        param : pybamm.parameters.Parameters() instance
            The model parameters.
        mesh : pybamm.mesh.Mesh() instance
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
            inits_dict['c'] = c0

        return y0, inits_dict

    def get_pdes_rhs(self, t, vars, param, operators):
        """Calculates the spatial derivates of the spatial terms in the PDEs
           and returns the right-hand side to be used by the ODE solver
           (Method of Lines).

        Parameters
        ----------
        t : float
            The simulation time.
        vars : pybamm.variables.Variables() instance
            The variables of the model.
        param : pybamm.parameters.Parameters() instance
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
            j = np.concatenate([0*vars.cn + param.icell(t) / param.ln,
                                0*vars.cs,
                                0*vars.cp - param.icell(t) / param.lp])
            source = param.s*j

            dcdt = components.simple_diffusion(vars.c,
                                               operators,
                                               (lbc, rbc),
                                               source=source)

            # Create dydt and derivs_dict
            dydt = dcdt
            derivs_dict['c'] = dcdt

        return dydt, derivs_dict
