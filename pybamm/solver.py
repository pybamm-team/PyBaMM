#
# Solver for the simulation
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np
import scipy.integrate as it

KNOWN_INTEGRATORS = ["BDF", "analytical"]
KNOWN_SPATIAL_DISCRETISATIONS = ["Finite Volumes"]


class Solver(object):
    """Solver for the simulation created in pybamm.sim.

    Parameters
    ----------
    integrator : string, optional
        The method for integration in time:
            * "BDF" (default): scipy.integrate.solve_ivp with method="BDF".
            * "analytical": use explicit analytical solution (if possible).
                Otherwise revert to "BDF".
    spatial_discretisation : string, optional
        The spatial discretisation scheme:
            * "Finite Volumes" (default): Finite Volumes discretisation.
                Cell edges are on the boundaries between the subdomains
    """

    def __init__(
        self, integrator="BDF", spatial_discretisation="Finite Volumes", tol=1e-8
    ):

        if integrator not in KNOWN_INTEGRATORS:
            raise NotImplementedError(
                """Integrator '{}' is not implemented.
                                      Valid choices: one of '{}'.""".format(
                    integrator, KNOWN_INTEGRATORS
                )
            )
        self.integrator = integrator

        if spatial_discretisation not in KNOWN_SPATIAL_DISCRETISATIONS:
            raise NotImplementedError(
                """Spatial discretisation '{}' is not
                                      implemented. Valid choices: one of '{}'.
                                      """.format(
                    spatial_discretisation, KNOWN_SPATIAL_DISCRETISATIONS
                )
            )
        self.spatial_discretisation = spatial_discretisation

        self.tol = tol

    def __str__(self):
        return "{}_{}".format(self.integrator, self.spatial_discretisation)

    def operators(self, mesh):
        """Define the operators in each domain.

        Parameters
        ----------
        mesh : :class:`pybamm.mesh.Mesh` instance
            The mesh on which the operators are defined.

        Returns
        -------
        :class: `pybamm.operators.Operators`
            A class of all the operators.

        """
        return pybamm.Operators(self.spatial_discretisation, mesh)

    def get_simulation_vars(self, sim):
        """Run a simulation.

        Parameters
        ----------
        sim : :class:`pybamm.simulation.Simulation` instance
            The simulation to be solved.

        Returns
        -------
        vars : pybamm.variables.Variables() instance
            The variables of the solved model.

        """

        # Initialise variables
        vars = pybamm.Variables(sim.model)

        # Initialise y for PDE solver
        yinit = sim.model.initial_conditions()

        # Solve ODEs
        def derivs(t, y):
            vars.update(t, y)
            dydt = sim.model.pdes_rhs(vars)
            return dydt

        if self.integrator == "analytical":
            # TODO: implement analytical simulation
            pass
        elif self.integrator == "BDF":
            target_time = sim.mesh.time
            sol = it.solve_ivp(
                derivs,
                (target_time[0], target_time[-1]),
                yinit,
                t_eval=target_time,
                method="BDF",
                rtol=self.tol,
                atol=self.tol,
            )
            # TODO: implement concentration cut-off event

        # Extract variables from y
        vars.update(sol.t, sol.y)

        # Post-process (get potentials)
        vars.dt = np.concatenate([np.array([0]), np.diff(vars.t)])
        # TODO: write post-processing function

        return vars
