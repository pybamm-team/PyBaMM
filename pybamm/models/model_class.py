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
            * "Electrolyte diffusion": 1D reaction-diffusion equation for
                the electrolyte:
                dc/dt = d/dx(D(c)*dc/dx) + s*j
            * "Electrolyte current": 1D MacInnes equation for the elecrolyte
                potentials and current density:
                i = kappa(c) * (d(ln(c))/dx - dPhi/dx)
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

    def variables(self):
        """
        The variables of the model.
        List of (name, domain)
        """
        if self.name == "Electrolyte diffusion":
            return [('c', 'xc')]
        elif self.name == "Electrolyte current":
            return [('en', 'xcn'), ('ep', 'xcp')]

    def domains(self):
        """The domains in which the model is defined."""
        if self.name == "Electrolyte diffusion":
            return ["xc"]
        elif self.name == "Electrolyte current":
            return ["xcn", "xcp"]


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

        """
        if not self.tests:
            if self.name == "Electrolyte diffusion":
                c0 = param.c0 * np.ones_like(mesh.xc)
                return c0
            elif self.name == "Porosity":
                eps0 = np.concatenate([param.epsn0 * np.ones_like(mesh.xcn),
                                       param.epss0 * np.ones_like(mesh.xcs),
                                       param.epsp0 * np.ones_like(mesh.xcp)])
                return eps0
            elif self.name == "Electrolyte current":
                en0 = param.U_Pb(param.c0) * np.ones_like(mesh.xcn)
                ep0 = param.U_PbO2(param.c0) * np.ones_like(mesh.xcp)
                return np.concatenate([en0, ep0])

        else:
            return self.tests['inits']

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

        """
        bcs = self.boundary_conditions(vars, param)
        sources = self.sources(vars, param)
        if self.name == "Electrolyte diffusion":
            dcdt = components.electrolyte_diffusion(
                vars.c,
                operators["xc"],
                bcs['concentration'],
                source=sources['concentration'])

            return dcdt
        elif self.name == "Porosity":
            pass
        elif self.name == "Electrolyte current":
            dedt = components.elecrolyte_current(
                (vars.cn, vars.en),
                operators["xcn"],
                bcs["current neg"],
                source=sources["current neg"])

            return dedt

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
        if not self.tests:
            bcs = {}
            if self.name == "Electrolyte diffusion":
                bcs['concentration'] = (np.array([0]), np.array([0]))
            elif self.name == "Electrolyte current":
                bcs['current neg'] = (np.array([param.icell(vars.t)]),
                                      np.array([0]))
                bcs['current pos'] = (np.array([0]),
                                      np.array([param.icell(vars.t)]))
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
        sources : dict
            Dictionary of source terms

        """
        if not self.tests:
            if self.name == "Electrolyte diffusion":
                sources = {}
                j = np.concatenate([0*vars.cn + param.icell(vars.t) / param.ln,
                                    0*vars.cs,
                                    0*vars.cp - param.icell(vars.t) / param.lp])
                sources['concentration'] = param.s*j
        else:
            sources = self.tests['sources'](vars.t)

        return sources
