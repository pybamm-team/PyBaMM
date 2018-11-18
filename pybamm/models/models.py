import numpy as np


class SPM:

    def __init__(self, param, mesh):
        self.param = param
        self.mesh = mesh

        # create initial conditions
        cn = param.cn0 * np.ones_like(mesh.rn)
        cp = param.cp0 * np.ones_like(mesh.rp)
        self.y0 = np.concatenate(cn, cp)

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
        dcndt =

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
        bcs["cn"] = (0, vars.current/param.Ln)
        bcs["cp"] = (0, -vars.current/param.Lp)
        return bcs
