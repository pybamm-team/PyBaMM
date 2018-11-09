from pybamm.models import components

import numpy as np

KNOWN_MODELS = ["Electrolyte diffusion",
                ]
# !Remember to update docstring with any new models!

class Model:
    """The (dimensionless) physical/chemical model to be used in the simulation.

    Parameters
    ----------
    name : string
        The model name:
            * "Electrolyte diffusion" : 1D reaction-diffusion equation for
                the electrolyte:
                dc/dt = d/dx(D*dc/dx) + s*j
            * "Electrolyte current" : 1D MacInnes equation for the elecrolyte
                potentials and current density:
                i = kappa * (d(ln(c))/dx - dPhi/dx)
                de/dt = 1/gamma_dl * (di/dx - j)
    tests : dict
        A dictionary for testing the convergence of the numerical solution:
            * {} (default): We are not running in test mode, use built-ins.
            * {'inits': dict of initial conditions,
               'bcs': dict of boundary conditions,
               'sources': dict of source terms
               }: To be used for testing convergence to an exact solution.
    """
    def __init__(self, name, tests={}):
        if name not in KNOWN_MODELS:
            raise NotImplementedError("""Model '{}' is not implemented.
                                      Valid choices: one of '{}'."""
                                      .format(name, KNOWN_MODELS))
        self.name = name
        if tests:
            assert set(tests.keys()) == {'inits', 'bcs', 'sources'}, \
                "tests.keys() must include, 'inits', 'bcs' and 'sources'"
        self.tests = tests

        # Assign variables
        if self.name == "Electrolyte diffusion":
            self.variables = ['c']
        elif self.name == "Electrolyte current":
            self.variables = ['en', 'ep']

    def initial_conditions(self, param, mesh):
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
        if self.name == "Electrolyte diffusion":
            if not self.tests:
                c0 = np.ones_like(mesh.xc)
            else:
                c0 = self.tests['inits']['c']

            # Create y0 and inits_dict
            y0 = c0
            inits_dict['c'] = c0
        elif self.name == "Electrolyte current":
            if not self.tests:
                en0 = np.ones_like(mesh.xcn)
                ep0 = np.ones_like(mesh.xcp)
            else:
                c0 = self.tests['inits']['c']

            # Create y0 and inits_dict
            y0 = np.concatenate([en0, ep0])
            inits_dict['e'] = np.concatenate([en0, ep0])
        return y0, inits_dict

    def pdes_rhs(self, vars, param, operators):
        """Calculates the spatial derivates of the spatial terms in the PDEs
           and returns the right-hand side to be used by the ODE solver
           (Method of Lines).

        Parameters
        ----------
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
        bcs = self.boundary_conditions(vars, param)
        sources = self.sources(vars, param)
        if self.name == "Electrolyte diffusion":
            dcdt = components.electrolyte_diffusion(vars.c,
                                               operators,
                                               bcs['c'],
                                               source=sources['c'])

            # Create dydt and derivs_dict
            dydt = dcdt
            derivs_dict['c'] = dcdt

        return dydt, derivs_dict

    def boundary_conditions(self, vars, param):
        """Returns the boundary conditions for the model (fluxes only).

        Parameters
        ----------
        vars : pybamm.variables.Variables() instance
            The variables of the model.
        param : pybamm.parameters.Parameters() instance
            The model parameters.

        Returns
        -------
        bcs : dict of 2-tuples
            Dictionary of flux boundary conditions:
                {name: (left-hand flux bc, right-hand flux bc)}.

        """
        bcs = {}
        if self.name == "Electrolyte diffusion":
            if not self.tests:
                bcs['c'] = (np.array([0]), np.array([0]))
            else:
                bcs = self.tests['bcs'](vars.t)

        return bcs

    def sources(self, vars, param):
        """Returns the boundary conditions for the model (fluxes only).

        Parameters
        ----------
        vars : pybamm.variables.Variables() instance
            The variables of the model.
        param : pybamm.parameters.Parameters() instance
            The model parameters.

        Returns
        -------
        sources : dict of 2-tuples
            Dictionary of flux boundary conditions:
                {name: (left-hand flux bc, right-hand flux bc)}.

        """
        sources = {}
        if self.name == "Electrolyte diffusion":
            if not self.tests:
                j = np.concatenate([0*vars.cn + param.icell(vars.t) / param.ln,
                                    0*vars.cs,
                                    0*vars.cp - param.icell(vars.t) / param.lp])
                sources['c'] = param.s*j
            else:
                sources = self.tests['sources'](vars.t)

        return sources
