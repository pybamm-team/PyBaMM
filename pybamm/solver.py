from pybamm.variables import Variables
from pybamm.spatial_operators import Operators

import scipy.integrate as it

KNOWN_INTEGRATORS = ["BDF", "analytical"]
KNOWN_SPATIAL_DISCRETISATIONS = ["Finite Volumes"]

class Solver:
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
    def __init__(self,
                 integrator="BDF",
                 spatial_discretisation="Finite Volumes"):

        if integrator not in KNOWN_INTEGRATORS:
            raise NotImplementedError("""Integrator '{}' is not implemented.
                                      Valid choices: one of '{}'."""
                                      .format(driver, KNOWN_INTEGRATORS))
        self.integrator = integrator

        if spatial_discretisation not in KNOWN_SPATIAL_DISCRETISATIONS:
            raise NotImplementedError("""Spatial discretisation '{}' is not
                                      implemented. Valid choices: one of '{}'.
                                      """
                                      .format(driver,
                                              KNOWN_SPATIAL_DISCRETISATIONS))
        self.spatial_discretisation = spatial_discretisation

        self.name = ("{}_{}".format(integrator, spatial_discretisation))

    def __str__(self):
        return self.name

    def get_simulation_vars(self, sim):
        """Run a simulation.

        Parameters
        ----------
        sim : pybamm.simulation.Simulation() instance
            The simulation to be solved.

        Returns
        -------
        vars : pybamm.variables.Variables() instance
            The variables of the solved model.

        """
        param = sim.parameters
        mesh = sim.mesh
        model = sim.model

        # Initialise
        yinit, _ = model.get_initial_conditions(param, mesh)

        # Get grad and div
        operators = Operators(self.spatial_discretisation, mesh)

        # Set mesh dependent parameters
        param.set_mesh_dependent_parameters(mesh)

        # Solve ODEs
        def derivs(t, y):
            # TODO: check if it's more expensive to create vars or update it
            vars = Variables(y, param, mesh)
            dydt, _ = model.get_pdes_rhs(t, vars, param, operators)
            return dydt

        if self.integrator == 'analytical' and ANALYTICAL_AVAILABLE:
            # TODO: implement analytical simulation
            pass
        elif self.integrator == 'BDF':
            target_time = mesh.time
            sol = it.solve_ivp(derivs,
                               (target_time[0], target_time[-1]), yinit,
                               t_eval=target_time, method='BDF')
            # TODO: implement concentration cut-off event

        # Extract variables from y
        vars = Variables(sol.y, param, mesh)

        # Post-process (get potentials)
        # TODO: write post-processing function

        return vars
