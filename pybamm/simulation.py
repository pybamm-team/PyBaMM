#
# Simulation class
#
from __future__ import annotations

import pickle
import pybamm
import numpy as np
import hashlib
import warnings
import sys
from functools import lru_cache
from datetime import timedelta
from pybamm.util import import_optional_dependency

from pybamm.expression_tree.operations.serialise import Serialise


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
        else:  # pragma: no cover
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


class Simulation:
    """A Simulation class for easy building and running of PyBaMM simulations.

    Parameters
    ----------
    model : :class:`pybamm.BaseModel`
        The model to be simulated
    experiment : :class:`pybamm.Experiment` or string or list (optional)
        The experimental conditions under which to solve the model. If a string is
        passed, the experiment is constructed as `pybamm.Experiment([experiment])`. If
        a list is passed, the experiment is constructed as
        `pybamm.Experiment(experiment)`.
    geometry: :class:`pybamm.Geometry` (optional)
        The geometry upon which to solve the model
    parameter_values: :class:`pybamm.ParameterValues` (optional)
        Parameters and their corresponding numerical values.
    submesh_types: dict (optional)
        A dictionary of the types of submesh to use on each subdomain
    var_pts: dict (optional)
        A dictionary of the number of points used by each spatial variable
    spatial_methods: dict (optional)
        A dictionary of the types of spatial method to use on each
        domain (e.g. pybamm.FiniteVolume)
    solver: :class:`pybamm.BaseSolver` (optional)
        The solver to use to solve the model.
    output_variables: list (optional)
        A list of variables to plot automatically
    C_rate: float (optional)
        The C-rate at which you would like to run a constant current (dis)charge.
    discretisation_kwargs: dict (optional)
        Any keyword arguments to pass to the Discretisation class.
        See :class:`pybamm.Discretisation` for details.
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
        discretisation_kwargs=None,
    ):
        self._parameter_values = parameter_values or model.default_parameter_values
        self._unprocessed_parameter_values = self._parameter_values

        if experiment is None:
            # Check to see if the current is provided as data (i.e. drive cycle)
            current = self._parameter_values.get("Current function [A]")
            if isinstance(current, pybamm.Interpolant):
                self.operating_mode = "drive cycle"
            else:
                self.operating_mode = "without experiment"
                if C_rate:
                    self.C_rate = C_rate
                    self._parameter_values.update(
                        {
                            "Current function [A]": self.C_rate
                            * self._parameter_values["Nominal cell capacity [A.h]"]
                        }
                    )
        else:
            if isinstance(experiment, (str, pybamm.step.BaseStep)):
                experiment = pybamm.Experiment([experiment])
            elif isinstance(experiment, list):
                experiment = pybamm.Experiment(experiment)
            elif not isinstance(experiment, pybamm.Experiment):
                raise TypeError(
                    "experiment must be a pybamm `Experiment` instance, a single "
                    "experiment step, or a list of experiment steps"
                )

            self.operating_mode = "with experiment"
            # Save the experiment
            self.experiment = experiment.copy()

        self._unprocessed_model = model
        self._model = model

        self._geometry = geometry or self._model.default_geometry
        self._submesh_types = submesh_types or self._model.default_submesh_types
        self._var_pts = var_pts or self._model.default_var_pts
        self._spatial_methods = spatial_methods or self._model.default_spatial_methods
        self._solver = solver or self._model.default_solver
        self._output_variables = output_variables
        self._discretisation_kwargs = discretisation_kwargs or {}

        # Initialize empty built states
        self._model_with_set_params = None
        self._built_model = None
        self._built_initial_soc = None
        self.steps_to_built_models = None
        self.steps_to_built_solvers = None
        self._mesh = None
        self._disc = None
        self._solution = None
        self.quick_plot = None

        # Initialise instances of Simulation class with the same random seed
        self._set_random_seed()

        # ignore runtime warnings in notebooks
        if is_notebook():  # pragma: no cover
            import warnings

            warnings.filterwarnings("ignore")

        self.get_esoh_solver = lru_cache()(self._get_esoh_solver)

    def __getstate__(self):
        """
        Return dictionary of picklable items
        """
        result = self.__dict__.copy()
        result["get_esoh_solver"] = None  # Exclude LRU cache
        return result

    def __setstate__(self, state):
        """
        Unpickle, restoring unpicklable relationships
        """
        self.__dict__ = state
        self.get_esoh_solver = lru_cache()(self._get_esoh_solver)

    # If the solver being used is CasadiSolver or its variants, set a fixed
    # random seed during class initialization to the SHA-256 hash of the class
    # name for purposes of reproducibility.
    def _set_random_seed(self):
        if isinstance(self._solver, pybamm.CasadiSolver) or isinstance(
            self._solver, pybamm.CasadiAlgebraicSolver
        ):
            np.random.seed(
                int(hashlib.sha256(self.__class__.__name__.encode()).hexdigest(), 16)
                % (2**32)
            )

    def set_up_and_parameterise_experiment(self):
        """
        Create and parameterise the models for each step in the experiment.

        This increases set-up time since several models to be processed, but
        reduces simulation time since the model formulation is efficient.
        """
        parameter_values = self._parameter_values.copy()
        # Set the initial temperature to be the temperature of the first step
        # We can set this globally for all steps since any subsequent steps will either
        # start at the temperature at the end of the previous step (if non-isothermal
        # model), or will use the "Ambient temperature" input (if isothermal model).
        # In either case, the initial temperature is not used for any steps except
        # the first.
        init_temp = self.experiment.steps[0].temperature
        if init_temp is not None:
            parameter_values["Initial temperature [K]"] = init_temp

        # Process each step
        self.experiment_unique_steps_to_model = {}
        for step in self.experiment.unique_steps:
            parameterised_model = step.process_model(self._model, parameter_values)
            self.experiment_unique_steps_to_model[step.basic_repr()] = (
                parameterised_model
            )

        # Set up rest model if experiment has start times
        if self.experiment.initial_start_time:
            # duration doesn't matter, we just need the model
            rest_step = pybamm.step.rest(duration=1)
            # Change ambient temperature to be an input, which will be changed at
            # solve time
            parameter_values["Ambient temperature [K]"] = "[input]"
            parameterised_model = rest_step.process_model(self._model, parameter_values)
            self.experiment_unique_steps_to_model["Rest for padding"] = (
                parameterised_model
            )

    def set_parameters(self):
        """
        A method to set the parameters in the model and the associated geometry.
        """

        if self.model_with_set_params:
            return

        self._model_with_set_params = self._parameter_values.process_model(
            self._unprocessed_model, inplace=False
        )
        self._parameter_values.process_geometry(self._geometry)
        self._model = self._model_with_set_params

    def set_initial_soc(self, initial_soc, inputs=None):
        if self._built_initial_soc != initial_soc:
            # reset
            self._model_with_set_params = None
            self._built_model = None
            self.steps_to_built_models = None
            self.steps_to_built_solvers = None

        options = self.model.options
        param = self._model.param
        if options["open-circuit potential"] == "MSMR":
            self._parameter_values = (
                self._unprocessed_parameter_values.set_initial_ocps(
                    initial_soc, param=param, inplace=False, options=options
                )
            )
        elif options["working electrode"] == "positive":
            self._parameter_values = (
                self._unprocessed_parameter_values.set_initial_stoichiometry_half_cell(
                    initial_soc,
                    param=param,
                    inplace=False,
                    options=options,
                    inputs=inputs,
                )
            )
        else:
            self._parameter_values = (
                self._unprocessed_parameter_values.set_initial_stoichiometries(
                    initial_soc,
                    param=param,
                    inplace=False,
                    options=options,
                    inputs=inputs,
                )
            )

        # Save solved initial SOC in case we need to re-build the model
        self._built_initial_soc = initial_soc

    def build(self, initial_soc=None, inputs=None):
        """
        A method to build the model into a system of matrices and vectors suitable for
        performing numerical computations. If the model has already been built or
        solved then this function will have no effect.
        This method will automatically set the parameters
        if they have not already been set.

        Parameters
        ----------
        initial_soc : float, optional
            Initial State of Charge (SOC) for the simulation. Must be between 0 and 1.
            If given, overwrites the initial concentrations provided in the parameter
            set.
        inputs : dict, optional
            A dictionary of input parameters to pass to the model when solving.
        """
        if initial_soc is not None:
            self.set_initial_soc(initial_soc, inputs=inputs)

        if self.built_model:
            return
        elif self._model.is_discretised:
            self._model_with_set_params = self._model
            self._built_model = self._model
        else:
            self.set_parameters()
            self._mesh = pybamm.Mesh(self._geometry, self._submesh_types, self._var_pts)
            self._disc = pybamm.Discretisation(
                self._mesh, self._spatial_methods, **self._discretisation_kwargs
            )
            self._built_model = self._disc.process_model(
                self._model_with_set_params, inplace=False
            )
            # rebuilt model so clear solver setup
            self._solver._model_set_up = {}

    def build_for_experiment(self, initial_soc=None, inputs=None):
        """
        Similar to :meth:`Simulation.build`, but for the case of simulating an
        experiment, where there may be several models and solvers to build.
        """
        if initial_soc is not None:
            self.set_initial_soc(initial_soc, inputs)

        if self.steps_to_built_models:
            return
        else:
            self.set_up_and_parameterise_experiment()

            # Can process geometry with default parameter values (only electrical
            # parameters change between parameter values)
            self._parameter_values.process_geometry(self._geometry)
            # Only needs to set up mesh and discretisation once
            self._mesh = pybamm.Mesh(self._geometry, self._submesh_types, self._var_pts)
            self._disc = pybamm.Discretisation(
                self._mesh, self._spatial_methods, **self._discretisation_kwargs
            )
            # Process all the different models
            self.steps_to_built_models = {}
            self.steps_to_built_solvers = {}
            for (
                step,
                model_with_set_params,
            ) in self.experiment_unique_steps_to_model.items():
                # It's ok to modify the model with set parameters in place as it's
                # not returned anywhere
                built_model = self._disc.process_model(
                    model_with_set_params, inplace=True
                )
                solver = self._solver.copy()
                self.steps_to_built_solvers[step] = solver
                self.steps_to_built_models[step] = built_model

    def solve(
        self,
        t_eval=None,
        solver=None,
        save_at_cycles=None,
        calc_esoh=True,
        starting_solution=None,
        initial_soc=None,
        callbacks=None,
        showprogress=False,
        inputs=None,
        **kwargs,
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
        solver : :class:`pybamm.BaseSolver`, optional
            The solver to use to solve the model. If None, Simulation.solver is used
        save_at_cycles : int or list of ints, optional
            Which cycles to save the full sub-solutions for. If None, all cycles are
            saved. If int, every multiple of save_at_cycles is saved. If list, every
            cycle in the list is saved. The first cycle (cycle 1) is always saved.
        calc_esoh : bool, optional
            Whether to include eSOH variables in the summary variables. If `False`
            then only summary variables that do not require the eSOH calculation
            are calculated. Default is True.
        starting_solution : :class:`pybamm.Solution`
            The solution to start stepping from. If None (default), then self._solution
            is used. Must be None if not using an experiment.
        initial_soc : float, optional
            Initial State of Charge (SOC) for the simulation. Must be between 0 and 1.
            If given, overwrites the initial concentrations provided in the parameter
            set.
        callbacks : list of callbacks, optional
            A list of callbacks to be called at each time step. Each callback must
            implement all the methods defined in :class:`pybamm.callbacks.BaseCallback`.
        showprogress : bool, optional
            Whether to show a progress bar for cycling. If true, shows a progress bar
            for cycles. Has no effect when not used with an experiment.
            Default is False.
        **kwargs
            Additional key-word arguments passed to `solver.solve`.
            See :meth:`pybamm.BaseSolver.solve`.
        """
        # Setup
        if solver is None:
            solver = self._solver

        callbacks = pybamm.callbacks.setup_callbacks(callbacks)
        logs = {}

        inputs = inputs or {}

        if self.operating_mode in ["without experiment", "drive cycle"]:
            self.build(initial_soc=initial_soc, inputs=inputs)
            if save_at_cycles is not None:
                raise ValueError(
                    "'save_at_cycles' option can only be used if simulating an "
                    "Experiment "
                )
            if starting_solution is not None:
                raise ValueError(
                    "starting_solution can only be provided if simulating an Experiment"
                )
            if (
                self.operating_mode == "without experiment"
                or "ElectrodeSOH" in self._model.name
            ):
                if t_eval is None:
                    raise pybamm.SolverError(
                        "'t_eval' must be provided if not using an experiment or "
                        "simulating a drive cycle. 't_eval' can be provided as an "
                        "array of times at which to return the solution, or as a "
                        "list [t0, tf] where t0 is the initial time and tf is the "
                        "final time. "
                        "For a constant current (dis)charge the suggested 't_eval'  "
                        "is [0, 3700/C] where C is the C-rate. "
                        "For example, run\n\n"
                        "\tsim.solve([0, 3700])\n\n"
                        "for a 1C discharge."
                    )

            elif self.operating_mode == "drive cycle":
                # For drive cycles (current provided as data) we perform additional
                # tests on t_eval (if provided) to ensure the returned solution
                # captures the input.
                time_data = self._parameter_values["Current function [A]"].x[0]
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
                        stacklevel=2,
                    )
                    dt_data_min = np.min(np.diff(time_data))
                    dt_eval_max = np.max(np.diff(t_eval))
                    if dt_eval_max > dt_data_min + sys.float_info.epsilon:
                        warnings.warn(
                            f"""
                            The largest timestep in t_eval ({dt_eval_max}) is larger than
                            the smallest timestep in the data ({dt_data_min}). The returned
                            solution may not have the correct resolution to accurately
                            capture the input. Try refining t_eval. Alternatively,
                            passing t_eval = None automatically sets t_eval to be the
                            points in the data.
                            """,
                            pybamm.SolverWarning,
                            stacklevel=2,
                        )

            self._solution = solver.solve(
                self.built_model, t_eval, inputs=inputs, **kwargs
            )

        elif self.operating_mode == "with experiment":
            callbacks.on_experiment_start(logs)
            self.build_for_experiment(initial_soc=initial_soc, inputs=inputs)
            if t_eval is not None:
                pybamm.logger.warning(
                    "Ignoring t_eval as solution times are specified by the experiment"
                )
            # Re-initialize solution, e.g. for solving multiple times with different
            # inputs without having to build the simulation again
            self._solution = starting_solution
            # Step through all experimental conditions
            user_inputs = inputs
            timer = pybamm.Timer()

            # Set up eSOH solver (for summary variables)
            esoh_solver = self.get_esoh_solver(calc_esoh)

            if starting_solution is None:
                starting_solution_cycles = []
                starting_solution_summary_variables = []
                starting_solution_first_states = []
            elif not hasattr(starting_solution, "all_summary_variables"):
                (
                    cycle_solution,
                    cycle_sum_vars,
                    cycle_first_state,
                ) = pybamm.make_cycle_solution(
                    [starting_solution],
                    esoh_solver=esoh_solver,
                    save_this_cycle=True,
                    inputs=user_inputs,
                )
                starting_solution_cycles = [cycle_solution]
                starting_solution_summary_variables = [cycle_sum_vars]
                starting_solution_first_states = [cycle_first_state]
            else:
                starting_solution_cycles = starting_solution.cycles.copy()
                starting_solution_summary_variables = (
                    starting_solution.all_summary_variables.copy()
                )
                starting_solution_first_states = (
                    starting_solution.all_first_states.copy()
                )

            # set simulation initial_start_time
            if starting_solution is None:
                initial_start_time = self.experiment.initial_start_time
            else:
                initial_start_time = starting_solution.initial_start_time

            if (
                initial_start_time is None
                and self.experiment.initial_start_time is not None
            ):
                raise ValueError(
                    "When using experiments with `start_time`, the starting_solution "
                    "must have a `start_time` too."
                )

            cycle_offset = len(starting_solution_cycles)
            all_cycle_solutions = starting_solution_cycles
            all_summary_variables = starting_solution_summary_variables
            all_first_states = starting_solution_first_states
            current_solution = starting_solution or pybamm.EmptySolution()

            voltage_stop = self.experiment.termination.get("voltage")
            time_stop = self.experiment.termination.get("time")
            logs["stopping conditions"] = {"voltage": voltage_stop, "time": time_stop}

            idx = 0
            num_cycles = len(self.experiment.cycle_lengths)
            feasible = True  # simulation will stop if experiment is infeasible

            # Add initial padding rest if current time is earlier than first start time
            # This could be the case when using a starting solution
            if starting_solution is not None:
                step = self.experiment.steps[0]
                if step.start_time is not None:
                    rest_time = (
                        step.start_time
                        - (
                            initial_start_time
                            + timedelta(seconds=float(current_solution.t[-1]))
                        )
                    ).total_seconds()
                    if rest_time > pybamm.settings.step_start_offset:
                        # logs["step operating conditions"] = "Initial rest for padding"
                        # callbacks.on_step_start(logs)

                        inputs = {
                            **user_inputs,
                            "Ambient temperature [K]": (
                                step.temperature
                                or self._parameter_values["Ambient temperature [K]"]
                            ),
                            "start time": current_solution.t[-1],
                        }
                        steps = current_solution.cycles[-1].steps
                        step_solution = current_solution.cycles[-1].steps[-1]

                        step_solution_with_rest = self.run_padding_rest(
                            kwargs, rest_time, step_solution, inputs
                        )
                        steps[-1] = step_solution + step_solution_with_rest

                        cycle_solution, _, _ = pybamm.make_cycle_solution(
                            steps, esoh_solver=esoh_solver, save_this_cycle=True
                        )
                        old_cycles = current_solution.cycles.copy()
                        old_cycles[-1] = cycle_solution
                        current_solution += step_solution_with_rest
                        current_solution.cycles = old_cycles

                        # Update _solution
                        self._solution = current_solution

            # check if a user has tqdm installed
            if showprogress:
                tqdm = import_optional_dependency("tqdm")
                cycle_lengths = tqdm.tqdm(
                    self.experiment.cycle_lengths,
                    desc="Cycling",
                )
            else:
                cycle_lengths = self.experiment.cycle_lengths

            for cycle_num, cycle_length in enumerate(
                cycle_lengths,
                start=1,
            ):
                logs["cycle number"] = (
                    cycle_num + cycle_offset,
                    num_cycles + cycle_offset,
                )
                logs["elapsed time"] = timer.time()
                callbacks.on_cycle_start(logs)

                steps = []
                cycle_solution = None

                # Decide whether we should save this cycle
                save_this_cycle = (
                    # always save cycle 1
                    cycle_num == 1
                    # always save last cycle
                    or cycle_num == num_cycles
                    # None: save all cycles
                    or save_at_cycles is None
                    # list: save all cycles in the list
                    or (
                        isinstance(save_at_cycles, list)
                        and cycle_num + cycle_offset in save_at_cycles
                    )
                    # int: save all multiples
                    or (
                        isinstance(save_at_cycles, int)
                        and (cycle_num + cycle_offset) % save_at_cycles == 0
                    )
                )
                for step_num in range(1, cycle_length + 1):
                    # Use 1-indexing for printing cycle number as it is more
                    # human-intuitive
                    step = self.experiment.steps[idx]
                    start_time = current_solution.t[-1]

                    # If step has an end time, dt must take that into account
                    if step.end_time is not None:
                        dt = min(
                            step.duration,
                            (
                                step.end_time
                                - (
                                    initial_start_time
                                    + timedelta(seconds=float(start_time))
                                )
                            ).total_seconds(),
                        )
                    else:
                        dt = step.duration

                    # if dt + starttime is larger than time_stop, set dt to time_stop - starttime
                    if time_stop is not None:
                        dt = min(dt, time_stop - start_time)

                    step_str = str(step)
                    model = self.steps_to_built_models[step.basic_repr()]
                    solver = self.steps_to_built_solvers[step.basic_repr()]

                    logs["step number"] = (step_num, cycle_length)
                    logs["step operating conditions"] = step_str
                    logs["step duration"] = step.duration
                    callbacks.on_step_start(logs)

                    inputs = {
                        **user_inputs,
                        "start time": start_time,
                    }
                    # Make sure we take at least 2 timesteps
                    npts = max(int(round(dt / step.period)) + 1, 2)
                    try:
                        step_solution = solver.step(
                            current_solution,
                            model,
                            dt,
                            t_eval=np.linspace(0, dt, npts),
                            save=False,
                            inputs=inputs,
                            **kwargs,
                        )
                    except pybamm.SolverError as error:
                        if (
                            "non-positive at initial conditions" in error.message
                            and "[experiment]" in error.message
                        ):
                            step_solution = pybamm.EmptySolution(
                                "Event exceeded in initial conditions", t=start_time
                            )
                        else:
                            logs["error"] = error
                            callbacks.on_experiment_error(logs)
                            feasible = False
                            # If none of the cycles worked, raise an error
                            if cycle_num == 1 and step_num == 1:
                                raise error
                            # Otherwise, just stop this cycle
                            break

                    step_termination = step_solution.termination

                    # Add a padding rest step if necessary
                    if step.next_start_time is not None:
                        rest_time = (
                            step.next_start_time
                            - (
                                initial_start_time
                                + timedelta(seconds=float(step_solution.t[-1]))
                            )
                        ).total_seconds()
                        if rest_time > pybamm.settings.step_start_offset:
                            logs["step number"] = (step_num, cycle_length)
                            logs["step operating conditions"] = "Rest for padding"
                            callbacks.on_step_start(logs)

                            inputs = {
                                **user_inputs,
                                "Ambient temperature [K]": (
                                    step.temperature
                                    or self._parameter_values["Ambient temperature [K]"]
                                ),
                                "start time": step_solution.t[-1],
                            }

                            step_solution_with_rest = self.run_padding_rest(
                                kwargs, rest_time, step_solution, inputs=inputs
                            )
                            step_solution += step_solution_with_rest

                    steps.append(step_solution)

                    # If there haven't been any successful steps yet in this cycle, then
                    # carry the solution over from the previous cycle (but
                    # `step_solution` should still be an EmptySolution so that in the
                    # list of returned step solutions we can see which steps were
                    # skipped)
                    if (
                        cycle_solution is None
                        and isinstance(step_solution, pybamm.EmptySolution)
                        and not isinstance(current_solution, pybamm.EmptySolution)
                    ):
                        cycle_solution = current_solution.last_state
                    else:
                        cycle_solution = cycle_solution + step_solution

                    current_solution = cycle_solution

                    logs["experiment time"] = cycle_solution.t[-1]
                    callbacks.on_step_end(logs)

                    logs["termination"] = step_solution.termination

                    # Check for some cases that would make the experiment end early
                    if step_termination == "final time" and step.uses_default_duration:
                        # reached the default duration of a step (typically we should
                        # reach an event before the default duration)
                        callbacks.on_experiment_infeasible_time(logs)
                        feasible = False
                        break

                    elif not (
                        isinstance(step_solution, pybamm.EmptySolution)
                        or step_termination == "final time"
                        or "[experiment]" in step_termination
                    ):
                        # Step has reached an event that is not specified in the
                        # experiment
                        callbacks.on_experiment_infeasible_event(logs)
                        feasible = False
                        break

                    elif time_stop is not None and logs["experiment time"] >= time_stop:
                        # reached the time limit of the experiment
                        break

                    else:
                        # Increment index for next iteration, then continue
                        idx += 1

                if save_this_cycle or feasible is False:
                    self._solution = self._solution + cycle_solution

                # At the final step of the inner loop we save the cycle
                if len(steps) > 0:
                    # Check for EmptySolution
                    if all(isinstance(step, pybamm.EmptySolution) for step in steps):
                        if len(steps) == 1:
                            raise pybamm.SolverError(
                                f"Step '{step_str}' is infeasible "
                                "due to exceeded bounds at initial conditions. "
                                "If this step is part of a longer cycle, "
                                "round brackets should be used to indicate this, "
                                "e.g.:\n pybamm.Experiment([(\n"
                                "\tDischarge at C/5 for 10 hours or until 3.3 V,\n"
                                "\tCharge at 1 A until 4.1 V,\n"
                                "\tHold at 4.1 V until 10 mA\n"
                                "])"
                            )
                        else:
                            this_cycle = self.experiment.cycles[cycle_num - 1]
                            raise pybamm.SolverError(
                                f"All steps in the cycle {this_cycle} are infeasible "
                                "due to exceeded bounds at initial conditions."
                            )
                    cycle_sol = pybamm.make_cycle_solution(
                        steps,
                        esoh_solver=esoh_solver,
                        save_this_cycle=save_this_cycle,
                        inputs=user_inputs,
                    )
                    cycle_solution, cycle_sum_vars, cycle_first_state = cycle_sol
                    all_cycle_solutions.append(cycle_solution)
                    all_summary_variables.append(cycle_sum_vars)
                    all_first_states.append(cycle_first_state)

                    logs["summary variables"] = cycle_sum_vars

                # Calculate capacity_start using the first cycle
                if cycle_num == 1:
                    # Note capacity_start could be defined as
                    # self._parameter_values["Nominal cell capacity [A.h]"] instead
                    if "capacity" in self.experiment.termination:
                        capacity_start = all_summary_variables[0]["Capacity [A.h]"]
                        logs["start capacity"] = capacity_start
                        value, typ = self.experiment.termination["capacity"]
                        if typ == "Ah":
                            capacity_stop = value
                        elif typ == "%":
                            capacity_stop = value / 100 * capacity_start
                    else:
                        capacity_stop = None
                    logs["stopping conditions"]["capacity"] = capacity_stop

                logs["elapsed time"] = timer.time()

                # Add minimum voltage to summary variable logs if there is a voltage stop
                # See PR #3995
                if voltage_stop is not None:
                    min_voltage = np.min(cycle_solution["Battery voltage [V]"].data)
                    logs["summary variables"]["Minimum voltage [V]"] = min_voltage

                callbacks.on_cycle_end(logs)

                # Break if stopping conditions are met
                # Logging is done in the callbacks
                if capacity_stop is not None:
                    capacity_now = cycle_sum_vars["Capacity [A.h]"]
                    if not np.isnan(capacity_now) and capacity_now <= capacity_stop:
                        break

                if voltage_stop is not None:
                    if min_voltage <= voltage_stop[0]:
                        break

                # Break if the experiment is infeasible (or errored)
                if feasible is False:
                    break

            if self.solution is not None and len(all_cycle_solutions) > 0:
                self.solution.cycles = all_cycle_solutions
                self.solution.set_summary_variables(all_summary_variables)
                self.solution.all_first_states = all_first_states

            callbacks.on_experiment_end(logs)

            # record initial_start_time of the solution
            self.solution.initial_start_time = initial_start_time

        return self.solution

    def run_padding_rest(self, kwargs, rest_time, step_solution, inputs):
        model = self.steps_to_built_models["Rest for padding"]
        solver = self.steps_to_built_solvers["Rest for padding"]

        # Make sure we take at least 2 timesteps. The period is hardcoded to 10
        # minutes,the user can always override it by adding a rest step
        npts = max(int(round(rest_time / 600)) + 1, 2)

        step_solution_with_rest = solver.step(
            step_solution,
            model,
            rest_time,
            t_eval=np.linspace(0, rest_time, npts),
            save=False,
            inputs=inputs,
            **kwargs,
        )

        return step_solution_with_rest

    def step(
        self,
        dt,
        solver=None,
        t_eval=None,
        save=True,
        starting_solution=None,
        inputs=None,
        **kwargs,
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
        t_eval : list or numpy.ndarray, optional
            An array of times at which to return the solution during the step
            (Note: t_eval is the time measured from the start of the step, so should start at 0 and end at dt).
            By default, the solution is returned at t0 and t0 + dt.
        save : bool
            Turn on to store the solution of all previous timesteps
        starting_solution : :class:`pybamm.Solution`
            The solution to start stepping from. If None (default), then self._solution
            is used
        **kwargs
            Additional key-word arguments passed to `solver.solve`.
            See :meth:`pybamm.BaseSolver.step`.
        """
        if self.operating_mode in ["without experiment", "drive cycle"]:
            self.build()

        if solver is None:
            solver = self._solver

        if starting_solution is None:
            starting_solution = self._solution

        self._solution = solver.step(
            starting_solution,
            self.built_model,
            dt,
            t_eval=t_eval,
            save=save,
            inputs=inputs,
            **kwargs,
        )

        return self.solution

    def _get_esoh_solver(self, calc_esoh):
        if (
            calc_esoh is False
            or isinstance(self._model, pybamm.lead_acid.BaseModel)
            or isinstance(self._model, pybamm.equivalent_circuit.Thevenin)
            or self._model.options["working electrode"] != "both"
        ):
            return None

        return pybamm.lithium_ion.ElectrodeSOHSolver(
            self._parameter_values, self._model.param, options=self._model.options
        )

    def plot(self, output_variables=None, **kwargs):
        """
        A method to quickly plot the outputs of the simulation. Creates a
        :class:`pybamm.QuickPlot` object (with keyword arguments 'kwargs') and
        then calls :meth:`pybamm.QuickPlot.dynamic_plot`.

        Parameters
        ----------
        output_variables: list, optional
            A list of the variables to plot.
        **kwargs
            Additional keyword arguments passed to
            :meth:`pybamm.QuickPlot.dynamic_plot`.
            For a list of all possible keyword arguments see :class:`pybamm.QuickPlot`.
        """

        if self._solution is None:
            raise ValueError(
                "Model has not been solved, please solve the model before plotting."
            )

        if output_variables is None:
            output_variables = self._output_variables

        self.quick_plot = pybamm.dynamic_plot(
            self._solution, output_variables=output_variables, **kwargs
        )

        return self.quick_plot

    def create_gif(self, number_of_images=80, duration=0.1, output_filename="plot.gif"):
        """
        Generates x plots over a time span of t_eval and compiles them to create
        a GIF. For more information see :meth:`pybamm.QuickPlot.create_gif`

        Parameters
        ----------
        number_of_images : int (optional)
            Number of images/plots to be compiled for a GIF.
        duration : float (optional)
            Duration of visibility of a single image/plot in the created GIF.
        output_filename : str (optional)
            Name of the generated GIF file.

        """
        if self.solution is None:
            raise ValueError("The simulation has not been solved yet.")
        if self.quick_plot is None:
            self.quick_plot = pybamm.QuickPlot(self._solution)

        self.quick_plot.create_gif(
            number_of_images=number_of_images,
            duration=duration,
            output_filename=output_filename,
        )

    @property
    def model(self):
        return self._model

    @property
    def model_with_set_params(self):
        return self._model_with_set_params

    @property
    def built_model(self):
        return self._built_model

    @property
    def geometry(self):
        return self._geometry

    @property
    def parameter_values(self):
        return self._parameter_values

    @property
    def submesh_types(self):
        return self._submesh_types

    @property
    def mesh(self):
        return self._mesh

    @property
    def var_pts(self):
        return self._var_pts

    @property
    def spatial_methods(self):
        return self._spatial_methods

    @property
    def solver(self):
        return self._solver

    @property
    def output_variables(self):
        return self._output_variables

    @property
    def solution(self):
        return self._solution

    def save(self, filename):
        """Save simulation using pickle module.

        Parameters
        ----------
        filename : str
            The file extension can be arbitrary, but it is common to use ".pkl" or ".pickle"
        """
        if self._model.convert_to_format == "python":
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

        if self.steps_to_built_solvers is not None:
            for solver in self.steps_to_built_solvers.values():
                if (
                    isinstance(solver, pybamm.CasadiSolver)
                    and solver.integrator_specs != {}
                ):
                    solver.integrator_specs = {}

        with open(filename, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def save_model(
        self,
        filename: str | None = None,
        mesh: bool = False,
        variables: bool = False,
    ):
        """
        Write out a discretised model to a JSON file

        Parameters
        ----------
        mesh: bool
            The mesh used to discretise the model. If false, plotting tools will not
            be available when the model is read back in and solved.
        variables: bool
            The discretised variables. Not required to solve a model, but if false
            tools will not be availble. Will automatically save meshes as well, required
            for plotting tools.
        filename: str, optional
            The desired name of the JSON file. If no name is provided, one will be
            created based on the model name, and the current datetime.
        """
        mesh = self.mesh if (mesh or variables) else None
        variables = self.built_model.variables if variables else None

        if self.operating_mode == "with experiment":
            raise NotImplementedError(
                """
                Serialising models coupled to experiments is not yet supported.
                """
            )

        if self.built_model:
            Serialise().save_model(
                self.built_model, filename=filename, mesh=mesh, variables=variables
            )
        else:
            raise NotImplementedError(
                """
                PyBaMM can only serialise a discretised model.
                Ensure the model has been built (e.g. run `build()`) before saving.
                """
            )

    def plot_voltage_components(
        self,
        ax=None,
        show_legend=True,
        split_by_electrode=False,
        show_plot=True,
        **kwargs_fill,
    ):
        """
        Generate a plot showing the component overpotentials that make up the voltage

        Parameters
        ----------
        ax : matplotlib Axis, optional
            The axis on which to put the plot. If None, a new figure and axis is created.
        show_legend : bool, optional
            Whether to display the legend. Default is True.
        split_by_electrode : bool, optional
            Whether to show the overpotentials for the negative and positive electrodes
            separately. Default is False.
        show_plot : bool, optional
            Whether to show the plots. Default is True. Set to False if you want to
            only display the plot after plt.show() has been called.
        kwargs_fill
            Keyword arguments, passed to ax.fill_between.

        """
        if self.solution is None:
            raise ValueError("The simulation has not been solved yet.")

        return pybamm.plot_voltage_components(
            self.solution,
            ax=ax,
            show_legend=show_legend,
            split_by_electrode=split_by_electrode,
            show_plot=show_plot,
            **kwargs_fill,
        )


def load_sim(filename):
    """Load a saved simulation"""
    return pybamm.load(filename)
