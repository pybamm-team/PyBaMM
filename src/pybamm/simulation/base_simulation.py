from __future__ import annotations

import pickle
import warnings
from copy import copy
from functools import lru_cache

import numpy as np

import pybamm
import pybamm.telemetry
from pybamm.models.base_model import ModelSolutionObservability


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


class BaseSimulation:
    """A Simulation class for easy building and running of PyBaMM simulations.

    Parameters
    ----------
    model : :class:`pybamm.BaseModel`
        The model to be simulated
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

    MODE_WITHOUT_EXPERIMENT = "without experiment"
    MODE_DRIVE_CYCLE = "drive cycle"
    MODE_WITH_EXPERIMENT = "with experiment"

    def __init__(
        self,
        model,
        geometry=None,
        parameter_values=None,
        submesh_types=None,
        var_pts=None,
        spatial_methods=None,
        solver=None,
        output_variables=None,
        C_rate=None,
        discretisation_kwargs=None,
        cache_esoh=True,
    ):
        self._parameter_values = parameter_values or model.default_parameter_values
        self._unprocessed_parameter_values = self._parameter_values

        current = self._parameter_values.get("Current function [A]")
        if isinstance(current, pybamm.Interpolant):
            self.operating_mode = self.MODE_DRIVE_CYCLE
        else:
            self.operating_mode = self.MODE_WITHOUT_EXPERIMENT
            if C_rate:
                self.C_rate = C_rate
                self._parameter_values.update(
                    {
                        "Current function [A]": self.C_rate
                        * self._parameter_values["Nominal cell capacity [A.h]"]
                    }
                )

        model = model.new_copy()
        self._unprocessed_model = model
        self._model = model

        self._geometry = geometry or self._model.default_geometry
        self._submesh_types = submesh_types or self._model.default_submesh_types
        self._var_pts = var_pts or self._model.default_var_pts
        self._spatial_methods = spatial_methods or self._model.default_spatial_methods
        self._solver = solver or self._model.default_solver
        self._output_variables = output_variables
        self._discretisation_kwargs = discretisation_kwargs or {}

        if bool(getattr(self._solver, "output_variables", [])):
            model.disable_solution_observability(
                ModelSolutionObservability.SOLVER_OUTPUT_VARIABLES
            )

        # Initialize empty built states
        self._model_with_set_params = None
        self._built_model = None
        self._built_initial_soc = None
        self._built_nominal_capacity = None
        self._cache_esoh = cache_esoh
        self._esoh_fingerprint = None
        self._initial_soc_solver = None
        self._mesh = None
        self._disc = None
        self._solution = None
        self.quick_plot = None
        self._needs_ic_rebuild = False

        if is_notebook():  # pragma: no cover
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

    def set_parameters(self):
        msg = (
            "pybamm.set_parameters is deprecated and not meant to be accessed by users."
        )
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        self._set_parameters()

    def _set_parameters(self):
        """
        A method to set the parameters in the model and the associated geometry.
        """
        if self._model_with_set_params:
            return

        self._model_with_set_params = self._parameter_values.process_model(
            self._unprocessed_model,
            inplace=False,
            delayed_variable_processing=True,
        )
        self._parameter_values.process_geometry(self._geometry)
        self._model = self._model_with_set_params

    @staticmethod
    def _pv_fingerprint(pv):
        """Hash all parameter values to detect any in-place modifications."""
        parts = []
        for k in sorted(pv.keys()):
            v = pv[k]
            if isinstance(v, int | float):
                parts.append((k, v))
            else:
                parts.append((k, id(v)))
        return tuple(parts)

    @staticmethod
    def _normalize_inputs(inputs):
        """Convert input values to hashable, comparison-safe types."""
        items = []
        for k in sorted(inputs.keys()):
            v = inputs[k]
            if isinstance(v, np.ndarray):
                items.append((k, v.tobytes()))
            elif isinstance(v, int | float):
                items.append((k, float(v)))
            else:
                items.append((k, id(v)))
        return tuple(items)

    def _compute_esoh_fingerprint(self, initial_soc, direction, inputs):
        """Compute a fingerprint of all eSOH-relevant state to detect changes.

        Delegates to the model-specific fingerprint function in
        ``pybamm.lithium_ion.compute_esoh_fingerprint``, which evaluates the
        exact quantities that determine the eSOH result for this model type.
        Falls back to raw inputs if the model-specific evaluation fails.
        """
        pv = self._unprocessed_parameter_values
        # Hash the full parameter store as a safety net: the model-specific
        # fingerprint only evaluates a handful of scalar quantities, so it
        # cannot detect changes to non-numeric parameters such as OCP
        # functions that also affect the eSOH result.
        pv_fp = self._pv_fingerprint(pv)

        try:
            evals = pybamm.lithium_ion.compute_esoh_fingerprint(
                pv, self._model.param, self._model.options, inputs
            )
        except Exception:
            evals = self._normalize_inputs(inputs) if inputs else ()

        return (initial_soc, direction, pv_fp, evals)

    def _create_esoh_solver(self, direction, initial_soc):
        """Create the appropriate eSOH solver/sim for this model type."""
        options = self._model.options
        pv = self._unprocessed_parameter_values
        param = self._model.param

        if options.get("open-circuit potential") == "MSMR" or (
            options.get("working electrode") != "positive"
            and not pybamm.lithium_ion.check_if_composite(options, "positive")
            and not pybamm.lithium_ion.check_if_composite(options, "negative")
        ):
            return pybamm.lithium_ion.ElectrodeSOHSolver(
                pv,
                direction=direction,
                param=param,
                options=options,
            )
        elif options.get("working electrode") == "positive":
            model = pybamm.lithium_ion.ElectrodeSOHHalfCell(
                "ElectrodeSOH",
                direction=direction,
                options=options,
            )
            return pybamm.Simulation(model, parameter_values=pv)
        else:
            if isinstance(initial_soc, str) and initial_soc.strip().endswith("V"):
                initialization_method = "voltage"
            else:
                initialization_method = "SOC"
            model = pybamm.lithium_ion.ElectrodeSOHComposite(
                options,
                direction,
                initialization_method=initialization_method,
            )
            from pybamm.models.full_battery_models.lithium_ion.electrode_soh import (
                get_esoh_default_solver,
            )

            return pybamm.Simulation(
                model,
                parameter_values=pv,
                solver=get_esoh_default_solver(),
            )

    def set_initial_state(self, initial_soc, direction=None, inputs=None):
        if self._cache_esoh:
            fingerprint = self._compute_esoh_fingerprint(initial_soc, direction, inputs)
            if fingerprint == self._esoh_fingerprint:
                return
        else:
            normalized = self._normalize_inputs(inputs) if inputs else ()
            fingerprint = (initial_soc, direction, normalized)
            if fingerprint == self._esoh_fingerprint:
                return

        self._needs_ic_rebuild = True

        param = self._model.param
        options = self._model.options

        if self._cache_esoh:
            if self._initial_soc_solver is None:
                self._initial_soc_solver = self._create_esoh_solver(
                    direction, initial_soc
                )
            self._parameter_values = pybamm.lithium_ion.set_initial_state(
                initial_soc,
                self._unprocessed_parameter_values,
                direction=direction,
                param=param,
                inplace=False,
                options=options,
                inputs=inputs,
                esoh_solver=self._initial_soc_solver,
            )
        else:
            self._parameter_values = (
                self._unprocessed_parameter_values.set_initial_state(
                    initial_soc,
                    direction=direction,
                    param=param,
                    inplace=False,
                    options=options,
                    inputs=inputs,
                )
            )

        # Save solved initial SOC in case we need to re-build the model
        self._built_initial_soc = initial_soc
        self._esoh_fingerprint = fingerprint

    def set_initial_soc(self, initial_soc, direction, inputs=None):
        msg = "pybamm.simulation.set_initial_soc is deprecated, please use set_initial_state."
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return self.set_initial_state(
            initial_soc=initial_soc, direction=direction, inputs=inputs
        )

    def _prepare_solve(
        self, t_eval, solver, calc_esoh, callbacks, inputs, warn_stacklevel=3
    ):
        """Common setup for solve methods. Returns resolved values."""
        pybamm.telemetry.capture("simulation-solved")
        t_eval = copy(t_eval)
        if solver is None:
            solver = self._solver
        if calc_esoh is None:
            calc_esoh = self._model.calc_esoh
        else:
            if calc_esoh and not self._model.calc_esoh:
                calc_esoh = False
                warnings.warn(
                    "Model is not suitable for calculating eSOH, "
                    "setting `calc_esoh` to False",
                    UserWarning,
                    stacklevel=warn_stacklevel,
                )
        callbacks = pybamm.callbacks.setup_callbacks(callbacks)
        inputs = inputs or {}
        return t_eval, solver, calc_esoh, callbacks, inputs

    def _get_built_models(self):
        """Return list of built models that need IC recomputation."""
        models = []
        if self._built_model is not None:
            models.append(self._built_model)
        return models

    def _recompute_initial_conditions(self):
        """Recompute initial conditions on built model(s) without full rebuild."""
        unprocessed_by_name = {
            var.name: expr
            for var, expr in self._unprocessed_model.initial_conditions.items()
        }

        models = self._get_built_models()

        for built_model in models:
            new_param_ics = {}
            for var, existing in built_model.initial_conditions.items():
                if var.name in unprocessed_by_name:
                    new_param_ics[var] = self._parameter_values.process_symbol(
                        unprocessed_by_name[var.name]
                    )
                else:
                    new_param_ics[var] = existing

            processed_ics = self._disc.process_dict(new_param_ics, ics=True)
            slices = [built_model.y_slices[var][0] for var in processed_ics]
            sorted_eqs = [
                eq for _, eq in sorted(zip(slices, processed_ics.values(), strict=True))
            ]
            concat_ics = pybamm.numpy_concatenation(*sorted_eqs)

            built_model.initial_conditions = processed_ics
            built_model.concatenated_initial_conditions = concat_ics

        self._needs_ic_rebuild = False

    def build(self, initial_soc=None, direction=None, inputs=None):
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
            self.set_initial_state(initial_soc, direction=direction, inputs=inputs)

        if self._built_model:
            if self._needs_ic_rebuild:
                self._recompute_initial_conditions()
            return
        if self._model.is_discretised:
            self._model_with_set_params = self._model
            self._built_model = self._model
        else:
            self._set_parameters()
            self._mesh = pybamm.Mesh(self._geometry, self._submesh_types, self._var_pts)
            self._disc = pybamm.Discretisation(
                self._mesh, self._spatial_methods, **self._discretisation_kwargs
            )
            self._built_model = self._disc.process_model(
                self._model_with_set_params,
                inplace=False,
                delayed_variable_processing=True,
            )
            self._solver._model_set_up = {}
        self._needs_ic_rebuild = False

    def solve(
        self,
        t_eval=None,
        solver=None,
        calc_esoh=None,
        initial_soc=None,
        direction=None,
        callbacks=None,
        inputs=None,
        t_interp=None,
        **kwargs,
    ):
        """
        A method to solve the model. This method will automatically build
        and set the model parameters if not already done so.

        Parameters
        ----------
        t_eval : numeric type, optional
            The times at which to stop the integration due to a discontinuity in time.
            Can be provided as an array of times at which to return the solution, or as
            a list `[t0, tf]` where `t0` is the initial time and `tf` is the final
            time. If the solver does not support intra-solve interpolation, providing
            `t_eval` as a list returns the solution at 100 points within the interval
            `[t0, tf]`. Otherwise, the solution is returned at the times specified in
            `t_interp` or as a result of the adaptive time-stepping solution. See the
            `t_interp` argument for more details.

            If not using an experiment or running a drive cycle simulation (current
            provided as data) `t_eval` *must* be provided.

            If None and the parameter "Current function [A]" is read from data
            (i.e. drive cycle simulation) the model will be solved at the times
            provided in the data.
        solver : :class:`pybamm.BaseSolver`, optional
            The solver to use to solve the model. If None, Simulation.solver is used
        calc_esoh : bool, optional
            Whether to include eSOH variables in the summary variables. If `False`
            then only summary variables that do not require the eSOH calculation
            are calculated.
            If given, overwrites the default provided by the model.
        initial_soc : float, optional
            Initial State of Charge (SOC) for the simulation. Must be between 0 and 1.
            If given, overwrites the initial concentrations provided in the parameter
            set.
        callbacks : list of callbacks, optional
            A list of callbacks to be called at each time step. Each callback must
            implement all the methods defined in :class:`pybamm.callbacks.BaseCallback`.
        t_interp : None, list or ndarray, optional
            The times (in seconds) at which to interpolate the solution. Defaults to None.
            Only valid for solvers that support intra-solve interpolation (`IDAKLUSolver`).
        **kwargs
            Additional key-word arguments passed to `solver.solve`.
            See :meth:`pybamm.BaseSolver.solve`.
        """
        t_eval, solver, calc_esoh, callbacks, inputs = self._prepare_solve(
            t_eval, solver, calc_esoh, callbacks, inputs
        )

        self.build(initial_soc=initial_soc, direction=direction, inputs=inputs)

        if (
            self.operating_mode == self.MODE_WITHOUT_EXPERIMENT
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

        elif self.operating_mode == self.MODE_DRIVE_CYCLE:
            time_data = self._parameter_values["Current function [A]"].x[0]
            if t_eval is None:
                pybamm.logger.info("Setting t_eval as specified by the data")
                t_eval = time_data
            elif not solver.supports_t_eval_discontinuities and not set(
                np.round(time_data, 12)
            ).issubset(set(np.round(t_eval, 12))):
                # Warn if t_eval misses data points or has coarser resolution
                warnings.warn(
                    "t_eval does not contain all of the time points in the data "
                    "set. Note: passing t_eval = None automatically sets t_eval "
                    "to be the points in the data.",
                    pybamm.SolverWarning,
                    stacklevel=2,
                )
                dt_data_min = np.min(np.diff(time_data))
                dt_eval_max = np.max(np.diff(t_eval))
                if dt_eval_max > np.nextafter(dt_data_min, np.inf):
                    warnings.warn(
                        f"The largest timestep in t_eval ({dt_eval_max}) is larger "
                        f"than the smallest timestep in the data ({dt_data_min}). "
                        "The returned solution may not have the correct resolution "
                        "to accurately capture the input. Try refining t_eval or "
                        "passing t_eval = None.",
                        pybamm.SolverWarning,
                        stacklevel=2,
                    )
        self._solution = solver.solve(
            self._built_model,
            t_eval,
            inputs=inputs,
            t_interp=t_interp,
            **kwargs,
        )

        return self._solution

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
        # Copy t_eval to avoid modifying the original
        t_eval = copy(t_eval)

        if self.operating_mode in (self.MODE_WITHOUT_EXPERIMENT, self.MODE_DRIVE_CYCLE):
            self.build()

        if solver is None:
            solver = self._solver

        if starting_solution is None:
            starting_solution = self._solution

        self._solution = solver.step(
            starting_solution,
            self._built_model,
            dt,
            t_eval=t_eval,
            save=save,
            inputs=inputs,
            **kwargs,
        )

        return self._solution

    def _get_esoh_solver(self, calc_esoh, direction):
        if calc_esoh is False:
            return None

        return pybamm.lithium_ion.ElectrodeSOHSolver(
            self._parameter_values,
            param=self._model.param,
            direction=direction,
            options=self._model.options,
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
        if self._solution is None:
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
            tools will not be available. Will automatically save meshes as well, required
            for plotting tools.
        filename: str, optional
            The desired name of the JSON file. If no name is provided, one will be
            created based on the model name, and the current datetime.
        """
        mesh = self._mesh if (mesh or variables) else None
        variables = self._built_model.variables if variables else None

        if self._built_model:
            from pybamm.expression_tree.operations.serialise import Serialise

            Serialise().save_model(
                self._built_model, filename=filename, mesh=mesh, variables=variables
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
        electrode_phases=("primary", "primary"),
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
        electrode_phases : (str, str), optional
            The phases for which to plot the anode and cathode overpotentials, respectively.
            Default is `("primary", "primary")`.
        show_plot : bool, optional
            Whether to show the plots. Default is True. Set to False if you want to
            only display the plot after plt.show() has been called.
        kwargs_fill
            Keyword arguments, passed to ax.fill_between.

        """
        if self._solution is None:
            raise ValueError("The simulation has not been solved yet.")

        return pybamm.plot_voltage_components(
            self._solution,
            ax=ax,
            show_legend=show_legend,
            split_by_electrode=split_by_electrode,
            electrode_phases=electrode_phases,
            show_plot=show_plot,
            **kwargs_fill,
        )


def load_sim(filename):
    """Load a saved simulation"""
    return pybamm.load(filename)
