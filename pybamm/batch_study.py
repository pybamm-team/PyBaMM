#
# BatchStudy class
#
import pybamm
from itertools import product


class BatchStudy:
    """
    A BatchStudy class for comparison of different PyBaMM simulations.

    Parameters
    ----------
    models : dict
        A dictionary of models to be simulated
    experiments : dict (optional)
        A dictionary of experimental conditions under which to solve the model.
        Default is None
    geometries : dict (optional)
        A dictionary of geometries upon which to solve the model
    parameter_values : dict (optional)
        A dictionary of parameters and their corresponding numerical values.
        Default is None
    submesh_types : dict (optional)
        A dictionary of the types of submesh to use on each subdomain.
        Default is None
    var_pts : dict (optional)
        A dictionary of the number of points used by each spatial variable.
        Default is None
    spatial_methods : dict (optional)
        A dictionary of the types of spatial method to use on each domain.
        Default is None
    solvers : dict (optional)
        A dictionary of solvers to use to solve the model. Default is None
    output_variables : dict (optional)
        A dictionary of variables to plot automatically. Default is None
    C_rates : dict (optional)
        A dictionary of C-rates at which you would like to run a constant current
        (dis)charge. Default is None
    repeats : int (optional)
        The number of times `solve` should be called. Default is 1
    permutations : bool (optional)
        If False runs first model with first solver, first experiment
        and second model with second solver, second experiment etc.
        If True runs a cartesian product of models, solvers and experiments.
        Default is False
    """

    INPUT_LIST = [
        "experiments",
        "geometries",
        "parameter_values",
        "submesh_types",
        "var_pts",
        "spatial_methods",
        "solvers",
        "output_variables",
        "C_rates",
    ]

    def __init__(
        self,
        models,
        experiments=None,
        geometries=None,
        parameter_values=None,
        submesh_types=None,
        var_pts=None,
        spatial_methods=None,
        solvers=None,
        output_variables=None,
        C_rates=None,
        repeats=1,
        permutations=False,
    ):
        self.models = models
        self.experiments = experiments
        self.geometries = geometries
        self.parameter_values = parameter_values
        self.submesh_types = submesh_types
        self.var_pts = var_pts
        self.spatial_methods = spatial_methods
        self.solvers = solvers
        self.output_variables = output_variables
        self.C_rates = C_rates
        self.repeats = repeats
        self.permutations = permutations
        self.quick_plot = None

        if not self.permutations:
            for name in self.INPUT_LIST:
                if getattr(self, name) and (
                    len(self.models) != len(getattr(self, name))
                ):
                    raise ValueError(
                        f"Either provide no {name} or an equal number of {name}"
                        f" as the models ({len(self.models)} models given)"
                        f" if permutations=False"
                    )

    def solve(
        self,
        t_eval=None,
        solver=None,
        save_at_cycles=None,
        calc_esoh=True,
        starting_solution=None,
        initial_soc=None,
        **kwargs,
    ):
        """
        For more information on the parameters used in the solve,
        See :meth:`pybamm.Simulation.solve`
        """
        self.sims = []
        iter_func = product if self.permutations else zip

        # Instantiate items in INPUT_LIST based on the value of self.permutations
        inp_values = []
        for name in self.INPUT_LIST:
            if getattr(self, name):
                inp_value = getattr(self, name).values()
            elif self.permutations:
                inp_value = [None]
            else:
                inp_value = [None] * len(self.models)
            inp_values.append(inp_value)

        for (
            model,
            experiment,
            geometry,
            parameter_value,
            submesh_type,
            var_pt,
            spatial_method,
            solver,
            output_variable,
            C_rate,
        ) in iter_func(self.models.values(), *inp_values):
            sim = pybamm.Simulation(
                model,
                experiment=experiment,
                geometry=geometry,
                parameter_values=parameter_value,
                submesh_types=submesh_type,
                var_pts=var_pt,
                spatial_methods=spatial_method,
                solver=solver,
                output_variables=output_variable,
                C_rate=C_rate,
            )
            # Repeat to get average solve time and integration time
            solve_time = 0
            integration_time = 0
            for _ in range(self.repeats):
                sol = sim.solve(
                    t_eval,
                    solver,
                    save_at_cycles,
                    calc_esoh,
                    starting_solution,
                    initial_soc,
                    **kwargs,
                )
                solve_time += sol.solve_time
                integration_time += sol.integration_time
            sim.solution.solve_time = solve_time / self.repeats
            sim.solution.integration_time = integration_time / self.repeats
            self.sims.append(sim)

    def plot(self, output_variables=None, **kwargs):
        """
        For more information on the parameters used in the plot,
        See :meth:`pybamm.Simulation.plot`
        """
        self.quick_plot = pybamm.dynamic_plot(
            self.sims, output_variables=output_variables, **kwargs
        )
        return self.quick_plot

    def create_gif(self, number_of_images=80, duration=0.1, output_filename="plot.gif"):
        """
        Generates x plots over a time span of t_eval and compiles them to create
        a GIF. For more information see :meth:`pybamm.QuickPlot.create_gif`

        Parameters
        ----------
        number_of_images : int, optional
            Number of images/plots to be compiled for a GIF.
        duration : float, optional
            Duration of visibility of a single image/plot in the created GIF.
        output_filename : str, optional
            Name of the generated GIF file.

        """
        if not hasattr(self, "sims"):
            raise ValueError("The simulations have not been solved yet.")
        if self.quick_plot is None:
            self.quick_plot = pybamm.QuickPlot(self.sims)

        self.quick_plot.create_gif(
            number_of_images=number_of_images,
            duration=duration,
            output_filename=output_filename,
        )
