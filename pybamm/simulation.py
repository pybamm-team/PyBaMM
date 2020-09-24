#
# Simulation class
#
import pickle
import pybamm
import numpy as np
import copy
import warnings
import sys


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":  # pragma: no cover
            # Jupyter notebook or qtconsole
            cfg = get_ipython().config
            nb = len(cfg["InteractiveShell"].keys()) == 0
            return nb
        elif shell == "TerminalInteractiveShell":  # pragma: no cover
            return False  # Terminal running IPython
        elif shell == "Shell":  # pragma: no cover
            return True  # Google Colab notebook
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def constant_current_constant_voltage_constant_power(variables):
    I = variables["Current [A]"]
    V = variables["Terminal voltage [V]"]
    s_I = pybamm.InputParameter("Current switch")
    s_V = pybamm.InputParameter("Voltage switch")
    s_P = pybamm.InputParameter("Power switch")
    n_cells = pybamm.Parameter("Number of cells connected in series to make a battery")
    return (
        s_I * (I - pybamm.InputParameter("Current input [A]"))
        + s_V * (V - pybamm.InputParameter("Voltage input [V]") / n_cells)
        + s_P * (V * I - pybamm.InputParameter("Power input [W]") / n_cells)
    )


class Simulation:
    """A Simulation class for easy building and running of PyBaMM simulations.

    Parameters
    ----------
    model : :class:`pybamm.BaseModel`
        The model to be simulated
    experiment : :class:`pybamm.Experiment` (optional)
        The experimental conditions under which to solve the model
    geometry: :class:`pybamm.Geometry` (optional)
        The geometry upon which to solve the model
    parameter_values: :class:`pybamm.ParameterValues` (optional)
        Parameters and their corresponding numerical values.
    submesh_types: dict (optional)
        A dictionary of the types of submesh to use on each subdomain
    var_pts: dict (optional)
        A dictionary of the number of points used by each spatial
        variable
    spatial_methods: dict (optional)
        A dictionary of the types of spatial method to use on each
        domain (e.g. pybamm.FiniteVolume)
    solver: :class:`pybamm.BaseSolver` (optional)
        The solver to use to solve the model.
    output_variables: list (optional)
        A list of variables to plot automatically
    C_rate: float (optional)
        The C_rate at which you would like to run a constant current
        (dis)charge at.
    """

    def __init__(
        self,
        model,
        experiment=None,
        geometry=None,
        parameter_values=None,
        submesh_types=None,
        var_pts=None,
        spatial_methods=None,
        solver=None,
        output_variables=None,
        C_rate=None,
    ):
        self.parameter_values = parameter_values or model.default_parameter_values

        if isinstance(model, pybamm.lithium_ion.BasicDFNHalfCell):
            raise NotImplementedError(
                "BasicDFNHalfCell is not compatible with Simulations yet."
            )

        if experiment is None:
            # Check to see if the current is provided as data (i.e. drive cycle)
            current = self._parameter_values.get("Current function [A]")
            if isinstance(current, pybamm.Interpolant):
                self.operating_mode = "drive cycle"
            elif isinstance(current, tuple):
                raise NotImplementedError(
                    "Drive cycle from data has been deprecated. "
                    + "Define an Interpolant instead."
                )
            else:
                self.operating_mode = "without experiment"
                if C_rate:
                    self.C_rate = C_rate
                    self._parameter_values.update(
                        {
                            "Current function [A]": self.C_rate
                            * self._parameter_values["Cell capacity [A.h]"]
                        }
                    )

            self._unprocessed_model = model
            self.model = model
        else:
            self.set_up_experiment(model, experiment)

        self.geometry = geometry or self.model.default_geometry
        self.submesh_types = submesh_types or self.model.default_submesh_types
        self.var_pts = var_pts or self.model.default_var_pts
        self.spatial_methods = spatial_methods or self.model.default_spatial_methods
        self.solver = solver or self.model.default_solver
        self.output_variables = output_variables

        # Initialize empty built states
        self._model_with_set_params = None
        self._built_model = None
        self._mesh = None
        self._disc = None
        self._solution = None

        # ignore runtime warnings in notebooks
        if is_notebook():  # pragma: no cover
            import warnings

            warnings.filterwarnings("ignore")

    def set_up_experiment(self, model, experiment):
        """
        Set up a simulation to run with an experiment. This creates a dictionary of
        inputs (current/voltage/power, running time, stopping condition) for each
        operating condition in the experiment. The model will then be solved by
        integrating the model successively with each group of inputs, one group at a
        time.
        """
        self.operating_mode = "with experiment"

        # Update model
        new_model = model.new_copy(build=False)
        new_model.submodels[
            "external circuit"
        ] = pybamm.external_circuit.FunctionControl(
            new_model.param, constant_current_constant_voltage_constant_power
        )
        new_model.submodels[
            "experiment events"
        ] = pybamm.external_circuit.ExperimentEvents(new_model.param)
        new_model.build_model()
        self._unprocessed_model = new_model
        self.model = new_model

        if not isinstance(experiment, pybamm.Experiment):
            raise TypeError("experiment must be a pybamm `Experiment` instance")

        # Save the experiment
        self.experiment = experiment
        # Update parameter values with experiment parameters
        self._parameter_values.update(experiment.parameters)
        # Create a new submodel for each set of operating conditions and update
        # parameters and events accordingly
        self._experiment_inputs = []
        self._experiment_times = []
        for op, events in zip(experiment.operating_conditions, experiment.events):
            if op[1] in ["A", "C"]:
                # Update inputs for constant current
                if op[1] == "A":
                    I = op[0]
                else:
                    # Scale C-rate with capacity to obtain current
                    capacity = self._parameter_values["Cell capacity [A.h]"]
                    I = op[0] * capacity
                operating_inputs = {
                    "Current switch": 1,
                    "Voltage switch": 0,
                    "Power switch": 0,
                    "Current input [A]": I,
                    "Voltage input [V]": 0,  # doesn't matter
                    "Power input [W]": 0,  # doesn't matter
                }
            elif op[1] == "V":
                # Update inputs for constant voltage
                V = op[0]
                operating_inputs = {
                    "Current switch": 0,
                    "Voltage switch": 1,
                    "Power switch": 0,
                    "Current input [A]": 0,  # doesn't matter
                    "Voltage input [V]": V,
                    "Power input [W]": 0,  # doesn't matter
                }
            elif op[1] == "W":
                # Update inputs for constant power
                P = op[0]
                operating_inputs = {
                    "Current switch": 0,
                    "Voltage switch": 0,
                    "Power switch": 1,
                    "Current input [A]": 0,  # doesn't matter
                    "Voltage input [V]": 0,  # doesn't matter
                    "Power input [W]": P,
                }
            # Update period
            operating_inputs["period"] = op[3]
            # Update events
            if events is None:
                # make current and voltage values that won't be hit
                operating_inputs.update(
                    {"Current cut-off [A]": -1e10, "Voltage cut-off [V]": -1e10}
                )
            elif events[1] in ["A", "C"]:
                # update current cut-off, make voltage a value that won't be hit
                if events[1] == "A":
                    I = events[0]
                else:
                    # Scale C-rate with capacity to obtain current
                    capacity = self._parameter_values["Cell capacity [A.h]"]
                    I = events[0] * capacity
                operating_inputs.update(
                    {"Current cut-off [A]": I, "Voltage cut-off [V]": -1e10}
                )
            elif events[1] == "V":
                # update voltage cut-off, make current a value that won't be hit
                V = events[0]
                operating_inputs.update(
                    {"Current cut-off [A]": -1e10, "Voltage cut-off [V]": V}
                )

            self._experiment_inputs.append(operating_inputs)
            # Add time to the experiment times
            dt = op[2]
            if dt is None:
                # max simulation time: 1 week
                dt = 7 * 24 * 3600
            self._experiment_times.append(dt)

    def set_parameters(self):
        """
        A method to set the parameters in the model and the associated geometry.
        """

        if self.model_with_set_params:
            return None

        if self._parameter_values._dict_items == {}:
            # Don't process if parameter values is empty
            self._model_with_set_params = self._unprocessed_model
        else:
            self._model_with_set_params = self._parameter_values.process_model(
                self._unprocessed_model, inplace=False
            )
            self._parameter_values.process_geometry(self._geometry)
        self.model = self._model_with_set_params

    def build(self, check_model=True):
        """
        A method to build the model into a system of matrices and vectors suitable for
        performing numerical computations. If the model has already been built or
        solved then this function will have no effect.
        This method will automatically set the parameters
        if they have not already been set.

        Parameters
        ----------
        check_model : bool, optional
            If True, model checks are performed after discretisation (see
            :meth:`pybamm.Discretisation.process_model`). Default is True.
        """

        if self.built_model:
            return None
        elif self.model.is_discretised:
            self._model_with_set_params = self.model
            self._built_model = self.model
        else:
            self.set_parameters()
            self._mesh = pybamm.Mesh(self._geometry, self._submesh_types, self._var_pts)
            self._disc = pybamm.Discretisation(self._mesh, self._spatial_methods)
            self._built_model = self._disc.process_model(
                self._model_with_set_params, inplace=False, check_model=check_model
            )

    def solve(
        self,
        t_eval=None,
        solver=None,
        external_variables=None,
        inputs=None,
        check_model=True,
    ):
        """
        A method to solve the model. This method will automatically build
        and set the model parameters if not already done so.

        Parameters
        ----------
        t_eval : numeric type, optional
            The times (in seconds) at which to compute the solution. Can be
            provided as an array of times at which to return the solution, or as a
            list `[t0, tf]` where `t0` is the initial time and `tf` is the final time.
            If provided as a list the solution is returned at 100 points within the
            interval `[t0, tf]`.

            If not using an experiment or running a drive cycle simulation (current
            provided as data) `t_eval` *must* be provided.

            If running an experiment the values in `t_eval` are ignored, and the
            solution times are specified by the experiment.

            If None and the parameter "Current function [A]" is read from data
            (i.e. drive cycle simulation) the model will be solved at the times
            provided in the data.
        solver : :class:`pybamm.BaseSolver`
            The solver to use to solve the model.
        external_variables : dict
            A dictionary of external variables and their corresponding
            values at the current time. The variables must correspond to
            the variables that would normally be found by solving the
            submodels that have been made external.
        inputs : dict, optional
            Any input parameters to pass to the model when solving
        check_model : bool, optional
            If True, model checks are performed after discretisation (see
            :meth:`pybamm.Discretisation.process_model`). Default is True.
        """
        # Setup
        self.build(check_model=check_model)
        if solver is None:
            solver = self.solver

        if self.operating_mode in ["without experiment", "drive cycle"]:

            if self.operating_mode == "without experiment":
                if t_eval is None:
                    raise pybamm.SolverError(
                        "'t_eval' must be provided if not using an experiment or "
                        "simulating a drive cycle. 't_eval' can be provided as an "
                        "array of times at which to return the solution, or as a "
                        "list [t0, tf] where t0 is the initial time and tf is the "
                        "final time. "
                        "For a constant current (dis)charge the suggested 't_eval'  "
                        "is [0, 3700/C] where C is the C-rate."
                    )

            elif self.operating_mode == "drive cycle":
                # For drive cycles (current provided as data) we perform additional
                # tests on t_eval (if provided) to ensure the returned solution
                # captures the input.
                time_data = self._parameter_values["Current function [A]"].data[:, 0]
                # If no t_eval is provided, we use the times provided in the data.
                if t_eval is None:
                    pybamm.logger.info("Setting t_eval as specified by the data")
                    t_eval = time_data
                # If t_eval is provided we first check if it contains all of the
                # times in the data to within 10-12. If it doesn't, we then check
                # that the largest gap in t_eval is smaller than the smallest gap in
                # the time data (to ensure the resolution of t_eval is fine enough).
                # We only raise a warning here as users may genuinely only want
                # the solution returned at some specified points.
                elif (
                    set(np.round(time_data, 12)).issubset(set(np.round(t_eval, 12)))
                ) is False:
                    warnings.warn(
                        """
                        t_eval does not contain all of the time points in the data
                        set. Note: passing t_eval = None automatically sets t_eval
                        to be the points in the data.
                        """,
                        pybamm.SolverWarning,
                    )
                    dt_data_min = np.min(np.diff(time_data))
                    dt_eval_max = np.max(np.diff(t_eval))
                    if dt_eval_max > dt_data_min + sys.float_info.epsilon:
                        warnings.warn(
                            """
                            The largest timestep in t_eval ({}) is larger than
                            the smallest timestep in the data ({}). The returned
                            solution may not have the correct resolution to accurately
                            capture the input. Try refining t_eval. Alternatively,
                            passing t_eval = None automatically sets t_eval to be the
                            points in the data.
                            """.format(
                                dt_eval_max, dt_data_min
                            ),
                            pybamm.SolverWarning,
                        )

            self._solution = solver.solve(
                self.built_model,
                t_eval,
                external_variables=external_variables,
                inputs=inputs,
            )
            self.t_eval = self._solution.t * self.model.timescale.evaluate()

        elif self.operating_mode == "with experiment":
            if t_eval is not None:
                pybamm.logger.warning(
                    "Ignoring t_eval as solution times are specified by the experiment"
                )
            # Re-initialize solution, e.g. for solving multiple times with different
            # inputs without having to build the simulation again
            self._solution = None
            # Step through all experimental conditions
            inputs = inputs or {}
            pybamm.logger.info("Start running experiment")
            timer = pybamm.Timer()
            for idx, (exp_inputs, dt) in enumerate(
                zip(self._experiment_inputs, self._experiment_times)
            ):
                pybamm.logger.info(self.experiment.operating_conditions_strings[idx])
                inputs.update(exp_inputs)
                # Make sure we take at least 2 timesteps
                npts = max(int(round(dt / exp_inputs["period"])) + 1, 2)
                self.step(
                    dt,
                    solver=solver,
                    npts=npts,
                    external_variables=external_variables,
                    inputs=inputs,
                )
                # Only allow events specified by experiment
                if not (
                    self._solution.termination == "final time"
                    or "[experiment]" in self._solution.termination
                ):
                    pybamm.logger.warning(
                        "\n\n\tExperiment is infeasible: '{}' ".format(
                            self._solution.termination
                        )
                        + "was triggered during '{}'. ".format(
                            self.experiment.operating_conditions_strings[idx]
                        )
                        + "Try reducing current, shortening the time interval, "
                        "or reducing the period.\n\n"
                    )
                    break
            pybamm.logger.info(
                "Finish experiment simulation, took {}".format(
                    timer.format(timer.time())
                )
            )

        return self.solution

    def step(
        self, dt, solver=None, npts=2, external_variables=None, inputs=None, save=True
    ):
        """
        A method to step the model forward one timestep. This method will
        automatically build and set the model parameters if not already done so.

        Parameters
        ----------
        dt : numeric type
            The timestep over which to step the solution
        solver : :class:`pybamm.BaseSolver`
            The solver to use to solve the model.
        npts : int, optional
            The number of points at which the solution will be returned during
            the step dt. Default is 2 (returns the solution at t0 and t0 + dt).
        external_variables : dict
            A dictionary of external variables and their corresponding
            values at the current time. The variables must correspond to
            the variables that would normally be found by solving the
            submodels that have been made external.
        inputs : dict, optional
            Any input parameters to pass to the model when solving
        save : bool
            Turn on to store the solution of all previous timesteps
        """
        self.build()

        if solver is None:
            solver = self.solver

        self._solution = solver.step(
            self._solution,
            self.built_model,
            dt,
            npts=npts,
            external_variables=external_variables,
            inputs=inputs,
            save=save,
        )

        return self.solution

    def get_variable_array(self, *variables):
        """
        A helper function to easily obtain a dictionary of arrays of values
        for a list of variables at the latest timestep.

        Parameters
        ----------
        variable: str
            The name of the variable/variables you wish to obtain the arrays for.

        Returns
        -------
        variable_arrays: dict
            A dictionary of the variable names and their corresponding
            arrays.
        """

        variable_arrays = [
            self.built_model.variables[var].evaluate(
                self.solution.t[-1], self.solution.y[:, -1]
            )
            for var in variables
        ]

        if len(variable_arrays) == 1:
            return variable_arrays[0]
        else:
            return tuple(variable_arrays)

    def plot(self, output_variables=None, quick_plot_vars=None, **kwargs):
        """
        A method to quickly plot the outputs of the simulation. Creates a
        :class:`pybamm.QuickPlot` object (with keyword arguments 'kwargs') and
        then calls :meth:`pybamm.QuickPlot.dynamic_plot`.

        Parameters
        ----------
        output_variables: list, optional
            A list of the variables to plot.
        quick_plot_vars: list, optional
            A list of the variables to plot. Deprecated, use output_variables instead.
        **kwargs
            Additional keyword arguments passed to
            :meth:`pybamm.QuickPlot.dynamic_plot`.
            For a list of all possible keyword arguments see :class:`pybamm.QuickPlot`.
        """

        if quick_plot_vars is not None:
            raise NotImplementedError(
                "'quick_plot_vars' has been deprecated. Use 'output_variables' instead."
            )

        if self._solution is None:
            raise ValueError(
                "Model has not been solved, please solve the model before plotting."
            )

        if output_variables is None:
            output_variables = self.output_variables

        self.quick_plot = pybamm.dynamic_plot(
            self._solution, output_variables=output_variables, **kwargs
        )

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = copy.copy(model)
        self._model_class = model.__class__

    @property
    def model_with_set_params(self):
        return self._model_with_set_params

    @property
    def built_model(self):
        return self._built_model

    @property
    def geometry(self):
        return self._geometry

    @geometry.setter
    def geometry(self, geometry):
        self._geometry = geometry.copy()

    @property
    def parameter_values(self):
        return self._parameter_values

    @parameter_values.setter
    def parameter_values(self, parameter_values):
        self._parameter_values = parameter_values.copy()

    @property
    def submesh_types(self):
        return self._submesh_types

    @submesh_types.setter
    def submesh_types(self, submesh_types):
        self._submesh_types = submesh_types.copy()

    @property
    def mesh(self):
        return self._mesh

    @property
    def var_pts(self):
        return self._var_pts

    @var_pts.setter
    def var_pts(self, var_pts):
        self._var_pts = var_pts.copy()

    @property
    def spatial_methods(self):
        return self._spatial_methods

    @spatial_methods.setter
    def spatial_methods(self, spatial_methods):
        self._spatial_methods = spatial_methods.copy()

    @property
    def solver(self):
        return self._solver

    @solver.setter
    def solver(self, solver):
        self._solver = solver.copy()

    @property
    def output_variables(self):
        return self._output_variables

    @output_variables.setter
    def output_variables(self, output_variables):
        self._output_variables = copy.copy(output_variables)

    @property
    def solution(self):
        return self._solution

    def specs(
        self,
        geometry=None,
        parameter_values=None,
        submesh_types=None,
        var_pts=None,
        spatial_methods=None,
        solver=None,
        output_variables=None,
        C_rate=None,
    ):
        "Deprecated method for setting specs"
        raise NotImplementedError(
            "The 'specs' method has been deprecated. "
            "Create a new simulation for each different case instead."
        )

    def save(self, filename):
        """Save simulation using pickle"""
        if self.model.convert_to_format == "python":
            # We currently cannot save models in the 'python' format
            raise NotImplementedError(
                """
                Cannot save simulation if model format is python.
                Set model.convert_to_format = 'casadi' instead.
                """
            )
        # Clear solver problem (not pickle-able, will automatically be recomputed)
        if (
            isinstance(self._solver, pybamm.CasadiSolver)
            and self._solver.integrator_specs != {}
        ):
            self._solver.integrator_specs = {}
        with open(filename, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


def load_sim(filename):
    """Load a saved simulation"""
    return pybamm.load(filename)
