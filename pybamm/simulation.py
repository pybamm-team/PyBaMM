#
# Simulation class
#
import pickle
import pybamm
import numpy as np
import copy
import warnings
import sys
from functools import lru_cache


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
    experiment : :class:`pybamm.Experiment` (optional)
        The experimental conditions under which to solve the model
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
        self._unprocessed_parameter_values = self.parameter_values

        if isinstance(model, pybamm.lithium_ion.BasicDFNHalfCell):
            if experiment is not None:
                raise NotImplementedError(
                    "BasicDFNHalfCell is not compatible with experiment simulations."
                )

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
            if not isinstance(experiment, pybamm.Experiment):
                raise TypeError("experiment must be a pybamm `Experiment` instance")

            self.operating_mode = "with experiment"
            # Save the experiment
            self.experiment = experiment.copy()

        self._unprocessed_model = model
        self.model = model

        self.geometry = geometry or self.model.default_geometry
        self.submesh_types = submesh_types or self.model.default_submesh_types
        self.var_pts = var_pts or self.model.default_var_pts
        self.spatial_methods = spatial_methods or self.model.default_spatial_methods
        self.solver = solver or self.model.default_solver
        self.output_variables = output_variables

        # Initialize empty built states
        self._model_with_set_params = None
        self._built_model = None
        self._built_initial_soc = None
        self.op_conds_to_built_models = None
        self.op_conds_to_built_solvers = None
        self._mesh = None
        self._disc = None
        self._solution = None
        self.quick_plot = None

        # ignore runtime warnings in notebooks
        if is_notebook():  # pragma: no cover
            import warnings

            warnings.filterwarnings("ignore")

    def set_up_and_parameterise_experiment(self):
        """
        Set up a simulation to run with an experiment. This creates a dictionary of
        inputs (current/voltage/power, running time, stopping condition) for each
        operating condition in the experiment. The model will then be solved by
        integrating the model successively with each group of inputs, one group at a
        time.
        This needs to be done here and not in the Experiment class because the nominal
        cell capacity (from the parameters) is used to convert C-rate to current.
        """
        # Update experiment using capacity
        capacity = self._parameter_values["Nominal cell capacity [A.h]"]
        for op_conds in self.experiment.operating_conditions:
            op_type = op_conds["type"]
            if op_conds["dc_data"] is not None:
                # If operating condition includes a drive cycle, define the interpolant
                drive_cycle_interpolant = pybamm.Interpolant(
                    op_conds["dc_data"][:, 0],
                    op_conds["dc_data"][:, 1],
                    pybamm.t - pybamm.InputParameter("start time"),
                )
                if op_type == "current":
                    op_conds["Current input [A]"] = drive_cycle_interpolant
                if op_type == "voltage":
                    op_conds["Voltage input [V]"] = drive_cycle_interpolant
                if op_type == "power":
                    op_conds["Power input [W]"] = drive_cycle_interpolant
            else:
                if op_type == "C-rate":
                    Crate = op_conds.pop("C-rate input [-]")
                    op_conds["type"] = "current"
                    op_conds["Current input [A]"] = Crate * capacity
                elif op_type == "current":
                    Crate = op_conds["Current input [A]"] / capacity

            # Update events
            events = op_conds.pop("events")
            if events is not None:
                event_type = events.pop("type")
                if event_type == "C-rate":
                    # Scale C-rate with capacity to obtain current
                    events["Current input [A]"] = (
                        events.pop("C-rate input [-]") * capacity
                    )
                # Update the dictionary of operating conditions, replacing
                # "xxx input [unit]" with "xxx cut-off [unit]"
                op_conds.update(
                    {
                        key.replace("input", "cut-off"): value
                        for key, value in events.items()
                    }
                )

            # Add time to the experiment times
            dt = op_conds["time"]
            if dt is None:
                if op_conds["type"] in ["current", "CCCV"]:
                    # Current control: max simulation time: 3 * max simulation time
                    # based on C-rate
                    dt = 3 / abs(Crate) * 3600  # seconds
                    if op_conds["type"] == "CCCV":
                        dt *= 5  # 5x longer for CCCV
                else:
                    # max simulation time: 1 day
                    dt = 24 * 3600  # seconds
            op_conds["time"] = dt

        # Set up model for experiment
        self.set_up_and_parameterise_model_for_experiment()

    def set_up_and_parameterise_model_for_experiment(self):
        """
        Set up self.model to be able to run the experiment (new version).
        In this version, a new model is created for each step.

        This increases set-up time since several models to be processed, but
        reduces simulation time since the model formulation is efficient.
        """
        self.op_type_to_model = {}
        self.op_string_to_model = {}
        for op_number, op in enumerate(self.experiment.operating_conditions):
            # Create model for this operating condition type (current/voltage/power)
            # if it has not already been seen before
            if op["type"] not in self.op_type_to_model:
                if op["type"] == "current":
                    new_model, submodel = self.model, None
                else:
                    # Voltage or power control
                    # Create a new model where the current density is now a variable
                    # To do so, we replace all instances of the current density in the
                    # model with a current density variable, which is obtained from the
                    # FunctionControl submodel
                    # check which kind of external circuit model we need (differential
                    # or algebraic)
                    if op["type"] == "voltage":
                        submodel_class = pybamm.external_circuit.VoltageFunctionControl
                    elif op["type"] == "power":
                        submodel_class = pybamm.external_circuit.PowerFunctionControl
                    elif op["type"] == "CCCV":
                        submodel_class = pybamm.external_circuit.CCCVFunctionControl

                    new_model = self.model.new_copy()
                    # Build the new submodel and update the model with it
                    submodel = submodel_class(new_model.param, new_model.options)
                    variables = new_model.variables
                    submodel.variables = submodel.get_fundamental_variables()
                    variables.update(submodel.variables)
                    submodel.variables.update(submodel.get_coupled_variables(variables))
                    variables.update(submodel.variables)
                    submodel.set_rhs(variables)
                    submodel.set_algebraic(variables)
                    submodel.set_initial_conditions(variables)
                    new_model.rhs.update(submodel.rhs)
                    new_model.algebraic.update(submodel.algebraic)
                    new_model.initial_conditions.update(submodel.initial_conditions)

                self.op_type_to_model[op["type"]] = (new_model, submodel)

            if op["string"] not in self.op_string_to_model:
                model, submodel = self.op_type_to_model[op["type"]]
                # Create a new model for this operating condition, since we will update
                # the events differently (based on parameter values and inputs) for
                # different models of the same type (current/voltage/power)
                new_model = model.new_copy()
                self.update_new_model_events(new_model, op)
                # Update parameter values
                new_parameter_values = self.parameter_values.copy()
                self._original_temperature = new_parameter_values[
                    "Ambient temperature [K]"
                ]
                experiment_parameter_values = self.get_experiment_parameter_values(
                    op, op_number
                )
                new_parameter_values.update(
                    experiment_parameter_values, check_already_exists=False
                )
                # Set the "current function" to be the variable defined in the submodel
                if submodel is not None:
                    new_parameter_values["Current function [A]"] = submodel.variables[
                        "Current [A]"
                    ]
                parameterised_model = new_parameter_values.process_model(
                    new_model, inplace=False
                )
                self.op_string_to_model[op["string"]] = parameterised_model

    def update_new_model_events(self, new_model, op):
        if "Current cut-off [A]" in op:
            if op["type"] == "CCCV":
                # for the CCCV model we need to make sure that the current
                # cut-off is only reached at the end of the CV phase
                # Current is negative for a charge so this event will be
                # negative until it is zero
                # So we take away a large number times a heaviside switch
                # for the CV phase to make sure that the event can only be
                # hit during CV
                new_model.events.append(
                    pybamm.Event(
                        "Current cut-off (CCCV) [A] [experiment]",
                        -new_model.variables["Current [A]"]
                        - abs(pybamm.InputParameter("Current cut-off [A]"))
                        + 1e4
                        * (
                            new_model.variables["Battery voltage [V]"]
                            < (pybamm.InputParameter("Voltage input [V]") - 1e-4)
                        ),
                    )
                )
            else:
                new_model.events.append(
                    pybamm.Event(
                        "Current cut-off [A] [experiment]",
                        abs(new_model.variables["Current [A]"])
                        - pybamm.InputParameter("Current cut-off [A]"),
                    )
                )

        # add voltage events to the model
        if "Voltage cut-off [V]" in op:
            # The voltage event should be positive at the start of charge/
            # discharge. We use the sign of the current or power input to
            # figure out whether the voltage event is greater than the starting
            # voltage (charge) or less (discharge) and set the sign of the
            # event accordingly
            if op["type"] == "power":
                inp = op["Power input [W]"]
            else:
                inp = op["Current input [A]"]
            sign = np.sign(inp)
            if sign > 0:
                name = "Discharge"
            else:
                name = "Charge"
            if sign != 0:
                # Event should be positive at initial conditions for both
                # charge and discharge
                new_model.events.append(
                    pybamm.Event(
                        f"{name} voltage cut-off [V] [experiment]",
                        sign
                        * (
                            new_model.variables["Battery voltage [V]"]
                            - pybamm.InputParameter("Voltage cut-off [V]")
                        ),
                    )
                )

        # Keep the min and max voltages as safeguards but add some tolerances
        # so that they are not triggered before the voltage limits in the
        # experiment
        for i, event in enumerate(new_model.events):
            if event.name in ["Minimum voltage [V]", "Maximum voltage [V]"]:
                new_model.events[i] = pybamm.Event(
                    event.name, event.expression + 1, event.event_type
                )

    def get_experiment_parameter_values(self, op, op_number):
        experiment_parameter_values = {}
        if op["type"] == "current":
            experiment_parameter_values.update(
                {"Current function [A]": op["Current input [A]"]}
            )
        if op["type"] == "CCCV":
            experiment_parameter_values.update(
                {"CCCV current function [A]": op["Current input [A]"]}
            )
        if op["type"] in ["voltage", "CCCV"]:
            experiment_parameter_values.update(
                {"Voltage function [V]": op["Voltage input [V]"]}
            )
        if op["type"] == "power":
            experiment_parameter_values.update(
                {"Power function [W]": op["Power input [W]"]}
            )

        if op["temperature"] is not None:
            ambient_temperature = op["temperature"] + 273.15
            experiment_parameter_values.update(
                {"Ambient temperature [K]": ambient_temperature}
            )

            # If at the first operation, then the intial temperature
            # should be the ambient temperature.
            if op_number == 0:
                experiment_parameter_values.update(
                    {
                        "Initial temperature [K]": ambient_temperature,
                    }
                )
        else:
            experiment_parameter_values.update(
                {"Ambient temperature [K]": self._original_temperature}
            )

        return experiment_parameter_values

    def set_parameters(self):
        """
        A method to set the parameters in the model and the associated geometry.
        """

        if self.model_with_set_params:
            return

        if self._parameter_values._dict_items == {}:
            # Don't process if parameter values is empty
            self._model_with_set_params = self._unprocessed_model
        else:
            self._model_with_set_params = self._parameter_values.process_model(
                self._unprocessed_model, inplace=False
            )
            self._parameter_values.process_geometry(self.geometry)
        self.model = self._model_with_set_params

    def set_initial_soc(self, initial_soc):
        if self._built_initial_soc != initial_soc:
            # reset
            self._model_with_set_params = None
            self._built_model = None
            self.op_conds_to_built_models = None
            self.op_conds_to_built_solvers = None

        param = self.model.param
        self.parameter_values = (
            self._unprocessed_parameter_values.set_initial_stoichiometries(
                initial_soc, param=param, inplace=False
            )
        )
        # Save solved initial SOC in case we need to re-build the model
        self._built_initial_soc = initial_soc

    def build(self, check_model=True, initial_soc=None):
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
        initial_soc : float, optional
            Initial State of Charge (SOC) for the simulation. Must be between 0 and 1.
            If given, overwrites the initial concentrations provided in the parameter
            set.
        """
        if initial_soc is not None:
            self.set_initial_soc(initial_soc)

        if self.built_model:
            return
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
            # rebuilt model so clear solver setup
            self._solver._model_set_up = {}

    def build_for_experiment(self, check_model=True, initial_soc=None):
        """
        Similar to :meth:`Simulation.build`, but for the case of simulating an
        experiment, where there may be several models and solvers to build.
        """
        if initial_soc is not None:
            self.set_initial_soc(initial_soc)

        if self.op_conds_to_built_models:
            return
        else:
            self.set_up_and_parameterise_experiment()

            # Can process geometry with default parameter values (only electrical
            # parameters change between parameter values)
            self._parameter_values.process_geometry(self._geometry)
            # Only needs to set up mesh and discretisation once
            self._mesh = pybamm.Mesh(self._geometry, self._submesh_types, self._var_pts)
            self._disc = pybamm.Discretisation(self._mesh, self._spatial_methods)
            # Process all the different models
            self.op_conds_to_built_models = {}
            self.op_conds_to_built_solvers = {}
            for op_cond, model_with_set_params in self.op_string_to_model.items():
                # It's ok to modify the model with set parameters in place as it's
                # not returned anywhere
                built_model = self._disc.process_model(
                    model_with_set_params, inplace=True, check_model=check_model
                )
                solver = self.solver.copy()
                self.op_conds_to_built_solvers[op_cond] = solver
                self.op_conds_to_built_models[op_cond] = built_model

    def solve(
        self,
        t_eval=None,
        solver=None,
        check_model=True,
        save_at_cycles=None,
        calc_esoh=True,
        starting_solution=None,
        initial_soc=None,
        callbacks=None,
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
        check_model : bool, optional
            If True, model checks are performed after discretisation (see
            :meth:`pybamm.Discretisation.process_model`). Default is True.
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
        **kwargs
            Additional key-word arguments passed to `solver.solve`.
            See :meth:`pybamm.BaseSolver.solve`.
        """
        # Setup
        if solver is None:
            solver = self.solver

        callbacks = pybamm.callbacks.setup_callbacks(callbacks)
        logs = {}

        if self.operating_mode in ["without experiment", "drive cycle"]:
            self.build(check_model=check_model, initial_soc=initial_soc)
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
                or self.model.name == "ElectrodeSOH model"
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

            self._solution = solver.solve(self.built_model, t_eval, **kwargs)

        elif self.operating_mode == "with experiment":
            callbacks.on_experiment_start(logs)
            self.build_for_experiment(check_model=check_model, initial_soc=initial_soc)
            if t_eval is not None:
                pybamm.logger.warning(
                    "Ignoring t_eval as solution times are specified by the experiment"
                )
            # Re-initialize solution, e.g. for solving multiple times with different
            # inputs without having to build the simulation again
            self._solution = starting_solution
            # Step through all experimental conditions
            user_inputs = kwargs.get("inputs", {})
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
                    starting_solution.steps,
                    esoh_solver=esoh_solver,
                    save_this_cycle=True,
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

            cycle_offset = len(starting_solution_cycles)
            all_cycle_solutions = starting_solution_cycles
            all_summary_variables = starting_solution_summary_variables
            all_first_states = starting_solution_first_states
            current_solution = starting_solution or pybamm.EmptySolution()

            voltage_stop = self.experiment.termination.get("voltage")
            logs["stopping conditions"] = {"voltage": voltage_stop}

            idx = 0
            num_cycles = len(self.experiment.cycle_lengths)
            feasible = True  # simulation will stop if experiment is infeasible
            for cycle_num, cycle_length in enumerate(
                self.experiment.cycle_lengths, start=1
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
                    op_conds = self.experiment.operating_conditions[idx]
                    dt = op_conds["time"]
                    op_conds_str = op_conds["string"]
                    model = self.op_conds_to_built_models[op_conds_str]
                    solver = self.op_conds_to_built_solvers[op_conds_str]

                    logs["step number"] = (step_num, cycle_length)
                    logs["step operating conditions"] = op_conds_str
                    callbacks.on_step_start(logs)

                    start_time = current_solution.t[-1]
                    kwargs["inputs"] = {
                        **user_inputs,
                        **op_conds,
                        "start time": start_time,
                    }
                    # Make sure we take at least 2 timesteps
                    npts = max(int(round(dt / op_conds["period"])) + 1, 2)
                    try:
                        step_solution = solver.step(
                            current_solution,
                            model,
                            dt,
                            npts=npts,
                            save=False,
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

                    steps.append(step_solution)

                    cycle_solution = cycle_solution + step_solution
                    current_solution = cycle_solution

                    callbacks.on_step_end(logs)

                    logs["termination"] = step_solution.termination
                    # Only allow events specified by experiment
                    if not (
                        isinstance(step_solution, pybamm.EmptySolution)
                        or step_solution.termination == "final time"
                        or "[experiment]" in step_solution.termination
                    ):
                        callbacks.on_experiment_infeasible(logs)
                        feasible = False
                        break

                    # Increment index for next iteration
                    idx += 1

                if save_this_cycle or feasible is False:
                    self._solution = self._solution + cycle_solution

                # At the final step of the inner loop we save the cycle
                if len(steps) > 0:
                    # Check for EmptySolution
                    if all(isinstance(step, pybamm.EmptySolution) for step in steps):
                        if len(steps) == 1:
                            raise pybamm.SolverError(
                                f"Step '{op_conds_str}' is infeasible "
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
                            this_cycle = self.experiment.operating_conditions_cycles[
                                cycle_num - 1
                            ]
                            raise pybamm.SolverError(
                                f"All steps in the cycle {this_cycle} are infeasible "
                                "due to exceeded bounds at initial conditions."
                            )
                    cycle_sol = pybamm.make_cycle_solution(
                        steps, esoh_solver=esoh_solver, save_this_cycle=save_this_cycle
                    )
                    cycle_solution, cycle_sum_vars, cycle_first_state = cycle_sol
                    all_cycle_solutions.append(cycle_solution)
                    all_summary_variables.append(cycle_sum_vars)
                    all_first_states.append(cycle_first_state)

                    logs["summary variables"] = cycle_sum_vars

                # Calculate capacity_start using the first cycle
                if cycle_num == 1:
                    # Note capacity_start could be defined as
                    # self.parameter_values["Nominal cell capacity [A.h]"] instead
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
                callbacks.on_cycle_end(logs)

                # Break if stopping conditions are met
                # Logging is done in the callbacks
                if capacity_stop is not None:
                    capacity_now = cycle_sum_vars["Capacity [A.h]"]
                    if not np.isnan(capacity_now) and capacity_now <= capacity_stop:
                        break

                if voltage_stop is not None:
                    min_voltage = cycle_sum_vars["Minimum voltage [V]"]
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

        return self.solution

    def step(
        self, dt, solver=None, npts=2, save=True, starting_solution=None, **kwargs
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
            solver = self.solver

        if starting_solution is None:
            starting_solution = self._solution

        self._solution = solver.step(
            starting_solution, self.built_model, dt, npts=npts, save=save, **kwargs
        )

        return self.solution

    @lru_cache
    def get_esoh_solver(self, calc_esoh):
        if (
            calc_esoh is False
            or isinstance(self.model, pybamm.lead_acid.BaseModel)
            or isinstance(self.model, pybamm.equivalent_circuit.Thevenin)
            or self.model.options["working electrode"] != "both"
        ):
            return None

        return pybamm.lithium_ion.ElectrodeSOHSolver(
            self.parameter_values, self.model.param
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
            output_variables = self.output_variables

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

    @model.setter
    def model(self, model):
        self._model = copy.copy(model)

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
