from __future__ import annotations

import pickle
import warnings
from copy import copy
from datetime import timedelta
from functools import lru_cache

import numpy as np

import pybamm
import pybamm.telemetry
from pybamm.models.base_model import ModelSolutionObservability
from pybamm.solvers.base_solver import process
from pybamm.util import import_optional_dependency


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
            from .models.full_battery_models.lithium_ion.electrode_soh import (
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

        self._solver._model_set_up = {}
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


class Simulation(BaseSimulation):
    """Extends :class:`BaseSimulation` with experiment support.

    Accepts all parameters from :class:`BaseSimulation`, plus:

    Parameters
    ----------
    experiment : :class:`pybamm.Experiment` or string or list (optional)
        The experimental conditions under which to solve the model. If a string is
        passed, the experiment is constructed as `pybamm.Experiment([experiment])`. If
        a list is passed, the experiment is constructed as
        `pybamm.Experiment(experiment)`.
    experiment_model_mode : str, optional
        How to construct experiment models. Options are:
        ``"legacy"`` (default), which always uses one model per step; and
        ``"unified"``, which requires the shared experiment model path.
    """

    _PADDING_REST_KEY = "Rest for padding"
    _STEP_INDEX_INPUT = "Experiment step index"
    _TERMINATION_TIME = "experiment time limit reached"
    _TERMINATION_VOLTAGE = "experiment voltage limit reached"
    _TERMINATION_CAPACITY = "experiment capacity limit reached"
    _TERMINATION_FINAL_TIME = "final time"
    _TERMINATION_EXPERIMENT_TAG = "[experiment]"
    _COMBINED_TERMINATION_EVENT = "Combined termination [experiment]"

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
        cache_esoh=True,
        experiment_model_mode="legacy",
    ):
        super().__init__(
            model,
            geometry=geometry,
            parameter_values=parameter_values,
            submesh_types=submesh_types,
            var_pts=var_pts,
            spatial_methods=spatial_methods,
            solver=solver,
            output_variables=output_variables,
            C_rate=C_rate,
            discretisation_kwargs=discretisation_kwargs,
            cache_esoh=cache_esoh,
        )

        if experiment is not None:
            if isinstance(experiment, str | pybamm.step.BaseStep):
                experiment = pybamm.Experiment([experiment])
            elif isinstance(experiment, list):
                experiment = pybamm.Experiment(experiment)
            elif not isinstance(experiment, pybamm.Experiment):
                raise TypeError(
                    "experiment must be a pybamm `Experiment` instance, a single "
                    "experiment step, or a list of experiment steps"
                )

            self.operating_mode = self.MODE_WITH_EXPERIMENT
            self.experiment = experiment.copy()

        # Initialize experiment-specific built states
        self._built_experiment_model = None
        self._built_experiment_solver = None
        self.steps_to_built_models = None
        self.steps_to_built_solvers = None
        self.model_state_mappers = {}
        self._compiled_model_state_mappers = {}
        self._experiment_model_mode = self._normalise_experiment_model_mode(
            experiment_model_mode
        )
        self._experiment_uses_unified_model = False
        self._experiment_unified_model_key = "Unified experiment"
        self._experiment_step_indices = []
        self._experiment_padding_rest_index = None
        self._experiment_includes_padding_rest = False

    def __getstate__(self):
        """
        Return dictionary of picklable items
        """
        result = super().__getstate__()
        result["model_state_mappers"] = {}
        result["_compiled_model_state_mappers"] = {}
        result["_built_experiment_model"] = None
        result["_built_experiment_solver"] = None
        result["steps_to_built_models"] = None
        result["steps_to_built_solvers"] = None
        result["experiment_unique_steps_to_model"] = None
        return result

    # Defaults for backward-compatible unpickling of older Simulation pickles
    _SETSTATE_DEFAULTS = {
        "model_state_mappers": dict,
        "_compiled_model_state_mappers": dict,
        "_built_experiment_model": None,
        "_built_experiment_solver": None,
        "experiment_unique_steps_to_model": None,
        "_experiment_uses_unified_model": False,
        "_experiment_unified_model_key": "Unified experiment",
        "_experiment_step_indices": list,
        "_experiment_padding_rest_index": None,
        "_experiment_includes_padding_rest": False,
    }

    def __setstate__(self, state):
        """
        Unpickle, restoring unpicklable relationships
        """
        super().__setstate__(state)
        for attr, default in self._SETSTATE_DEFAULTS.items():
            if attr not in self.__dict__:
                value = default() if callable(default) else default
                setattr(self, attr, value)
        self._experiment_model_mode = self._normalise_experiment_model_mode(
            self.__dict__.get("_experiment_model_mode", "legacy")
        )

    def set_up_and_parameterise_experiment(self, solve_kwargs=None):
        msg = "pybamm.simulation.set_up_and_parameterise_experiment is deprecated and not meant to be accessed by users."
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        self._set_up_and_parameterise_experiment(solve_kwargs=solve_kwargs)

    def _update_experiment_models_for_capacity(self, inputs, solve_kwargs=None):
        """
        Check if the nominal capacity has changed and update the experiment models
        if needed. This re-processes the models without rebuilding the mesh and
        discretisation.
        """
        current_capacity = self._parameter_values.get(
            "Nominal cell capacity [A.h]", None
        )

        if self._built_nominal_capacity == current_capacity:
            return

        pybamm.logger.info(
            f"Nominal capacity changed from {self._built_nominal_capacity} to "
            f"{current_capacity}. Re-processing experiment models."
        )

        self._set_up_and_parameterise_experiment(solve_kwargs)
        self._discretise_experiment_models()
        self._build_experiment_state_mappers(inputs)
        self._built_nominal_capacity = current_capacity

    def _discretise_experiment_models(self):
        """Discretise all experiment models and populate solver/model lookup dicts."""
        self._built_experiment_model = None
        self._built_experiment_solver = None
        self.steps_to_built_models = {}
        self.steps_to_built_solvers = {}
        if self._experiment_uses_unified_model:
            model_with_set_params = self.experiment_unique_steps_to_model[
                self._experiment_unified_model_key
            ]
            built_model = self._disc.process_model(
                model_with_set_params,
                inplace=True,
                delayed_variable_processing=True,
            )
            solver = self._solver.copy()
            self._built_experiment_model = built_model
            self._built_experiment_solver = solver
            for step in self.experiment.unique_steps:
                self.steps_to_built_models[step.basic_repr()] = built_model
                self.steps_to_built_solvers[step.basic_repr()] = solver
        else:
            for (
                step,
                model_with_set_params,
            ) in self.experiment_unique_steps_to_model.items():
                built_model = self._disc.process_model(
                    model_with_set_params,
                    inplace=True,
                    delayed_variable_processing=True,
                )
                solver = self._solver.copy()
                self.steps_to_built_solvers[step] = solver
                self.steps_to_built_models[step] = built_model

    @staticmethod
    def _experiment_step_index_input_name():
        warnings.warn(
            "Simulation._experiment_step_index_input_name() is deprecated, "
            "use Simulation._STEP_INDEX_INPUT instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return Simulation._STEP_INDEX_INPUT

    @staticmethod
    def _normalise_experiment_model_mode(mode):
        if mode not in ("unified", "legacy"):
            raise ValueError(
                "experiment_model_mode must be one of 'unified' or 'legacy'"
            )
        return mode

    def _get_unified_experiment_model_blockers(self):
        if self.experiment is None:
            return ["no experiment is attached to the simulation"]

        for step in self.experiment.steps:
            if step.is_drive_cycle:
                return ["drive-cycle experiment steps are not yet supported"]
            if isinstance(step, pybamm.step.BaseStep) and not isinstance(
                step,
                (
                    pybamm.step.BaseStepExplicit,
                    pybamm.step.BaseStepImplicit,
                ),
            ):
                return [f"unsupported experiment step type '{type(step).__name__}'"]
            if (
                isinstance(step, pybamm.step.CustomStepImplicit)
                and step.control != "algebraic"
            ):
                return ["CustomStepImplicit with differential control is not supported"]

        if self._solver.ode_solver:
            return ["unified experiment model requires a DAE-capable solver"]

        return []

    def _experiment_can_use_unified_model(self):
        return not self._get_unified_experiment_model_blockers()

    def _set_up_unified_experiment_model(self, parameter_values):
        self._experiment_uses_unified_model = True
        self._experiment_step_indices = list(range(1, len(self.experiment.steps) + 1))
        self._experiment_includes_padding_rest = bool(
            self.experiment.initial_start_time
        )
        self._experiment_padding_rest_index = (
            len(self.experiment.steps) + 1
            if self._experiment_includes_padding_rest
            else None
        )

        new_model = self._model.new_copy()
        new_parameter_values = parameter_values.copy()
        # Temperatures may change between steps, so the unified model must read the
        # ambient temperature from step-level inputs instead of baking in one value.
        new_parameter_values["Ambient temperature [K]"] = "[input]"

        # Build one conditional control residual that selects the active step's
        # control law via the experiment step index input.
        step_control_builders = [
            step.get_control_residual for step in self.experiment.steps
        ]
        if self._experiment_includes_padding_rest:
            padding_rest_step = pybamm.step.Rest(duration=1)
            step_control_builders.append(padding_rest_step.get_control_residual)
        submodel = pybamm.external_circuit.ExperimentFunctionControl(
            new_model.param,
            new_model.options,
            self._STEP_INDEX_INPUT,
            step_control_builders,
        )
        # Reuse the implicit-step wiring so the experiment-wide controller owns the
        # current variable in the same way as voltage/power/resistance steps do today.
        new_model, variables = pybamm.step.BaseStepImplicit.add_control_submodel(
            new_model,
            submodel,
            new_parameter_values,
        )

        # Combine each step's local termination expression into one experiment event,
        # selecting the active branch with the step index input.
        termination_branches = [
            step.get_combined_termination_expression(variables)
            for step in self.experiment.steps
        ]
        if self._experiment_includes_padding_rest:
            termination_branches.append(pybamm.Scalar(1))
        combined_termination_expression = pybamm.Conditional(
            pybamm.InputParameter(self._STEP_INDEX_INPUT),
            *termination_branches,
        )
        new_model.events.append(
            pybamm.Event(
                self._COMBINED_TERMINATION_EVENT,
                combined_termination_expression,
            )
        )
        pybamm.step.BaseStep.update_voltage_safety_events(new_model)

        processed_model = new_parameter_values.process_model(
            new_model,
            inplace=True,
            delayed_variable_processing=True,
        )
        self.experiment_unique_steps_to_model = {
            self._experiment_unified_model_key: processed_model,
        }
        # Keep the legacy lookup keys alive even though compatible steps now share one
        # processed model instance.
        for step in self.experiment.unique_steps:
            self.experiment_unique_steps_to_model[step.basic_repr()] = processed_model

    def _build_experiment_step_inputs(
        self,
        user_inputs,
        step,
        start_time,
        active_step_index=None,
        include_temperature=True,
    ):
        temperature = (
            step.temperature or self._parameter_values["Ambient temperature [K]"]
        )
        if self._experiment_uses_unified_model:
            return self._build_unified_experiment_inputs(
                user_inputs,
                active_step_index,
                start_time,
                temperature,
            )
        inputs = {
            **user_inputs,
            "start time": start_time,
        }
        if include_temperature:
            inputs["Ambient temperature [K]"] = temperature
        return inputs

    def _build_unified_experiment_inputs(
        self, user_inputs, active_step_index, start_time, temperature
    ):
        inputs = {
            **user_inputs,
            "Ambient temperature [K]": temperature,
            "start time": start_time,
            self._STEP_INDEX_INPUT: active_step_index,
        }
        return inputs

    def _get_built_experiment_model(self, step_or_key):
        if self._experiment_uses_unified_model:
            return self._built_experiment_model
        if isinstance(step_or_key, str):
            return self.steps_to_built_models[step_or_key]
        return self.steps_to_built_models[step_or_key.basic_repr()]

    def _get_built_experiment_solver(self, step_or_key):
        if self._experiment_uses_unified_model:
            return self._built_experiment_solver
        if isinstance(step_or_key, str):
            return self.steps_to_built_solvers[step_or_key]
        return self.steps_to_built_solvers[step_or_key.basic_repr()]

    def _evaluate_step_termination_expression_from_solution(
        self, term, step_solution, step
    ):
        # Some custom terminations reference symbols that are not directly evaluable
        # from the raw event expression, but are available as processed solution
        # variables. Rebuild a minimal variables dict from the solved step state and
        # re-run the termination expression against that view of the solution.
        if step_solution.t_event is not None:
            t = float(np.asarray(step_solution.t_event).reshape(-1)[0])
        else:
            t = float(step_solution.t[-1])

        class SolutionVariables(dict):
            def __missing__(self, key):
                value = step_solution[key](t)
                self[key] = value
                return value

        try:
            value = term.get_event_expression(SolutionVariables(), step)
        except (KeyError, NotImplementedError):  # pragma: no cover
            return None

        return value

    def _decode_combined_step_termination(self, step_solution, step, model, inputs):
        if (
            step_solution.termination
            != f"event: {self._COMBINED_TERMINATION_EVENT}"
        ):
            return step_solution.termination

        final_event_values = {}
        for term in step.termination:
            event = term.get_event(model.variables, step)
            if event is None:
                continue

            value = None
            # First try the exact symbolic event expression that the unified model used.
            # This is the closest match to what the solver just triggered.
            t = (
                step_solution.t_event
                if step_solution.t_event is not None
                else step_solution.t[-1]
            )
            y = (
                step_solution.y_event
                if step_solution.y_event is not None
                else step_solution.y[:, -1]
            )
            try:
                value = event.expression.evaluate(t=t, y=y, inputs=inputs)
            except NotImplementedError:  # pragma: no cover
                value = None

            # If the raw expression still contains unevaluated symbols, fall back to
            # the processed variables on the solved step. This is slower, but it works
            # for custom terminations built from model outputs.
            if value is None:
                value = self._evaluate_step_termination_expression_from_solution(
                    term, step_solution, step
                )

            if value is not None:
                final_event_values[event.name] = value

        if not final_event_values:
            return step_solution.termination

        termination_event = min(final_event_values, key=final_event_values.get)
        step_solution.termination = f"event: {termination_event}"
        return step_solution.termination

    def _set_up_and_parameterise_experiment(self, solve_kwargs=None):
        """
        Create and parameterise the models for each step in the experiment.

        This increases set-up time since several models to be processed, but
        reduces simulation time since the model formulation is efficient.
        """
        parameter_values = self._parameter_values.copy()

        # some parameters are used to control the experiment, and should not be
        # input parameters
        restrict_list = {"Initial temperature [K]", "Ambient temperature [K]"}
        for step in self.experiment.steps:
            if step.is_implicit():
                restrict_list.update(step.get_parameter_values([]).keys())
            else:
                restrict_list.update(["Current function [A]"])
        for key in restrict_list:
            if key in parameter_values.keys() and isinstance(
                parameter_values[key], pybamm.InputParameter
            ):
                raise pybamm.ModelError(
                    f"Cannot use '{key}' as an input parameter in this experiment. "
                    f"This experiment is controlled via the following parameters: {restrict_list}. "
                    f"None of these parameters are able to be input parameters."
                )

        if (
            solve_kwargs is not None
            and "calculate_sensitivities" in solve_kwargs
            and solve_kwargs["calculate_sensitivities"]
        ):
            for step in self.experiment.steps:
                if any(
                    isinstance(
                        term,
                        pybamm.experiment.step.step_termination.BaseTermination,
                    )
                    for term in step.termination
                ):
                    pybamm.logger.warning(
                        f"Step '{step}' has a termination condition based on an event. Sensitivity calculation will be inaccurate "
                        "if the time of each step event changes rapidly with respect to the parameters. "
                    )
                    break

        # Set the initial temperature to be the temperature of the first step
        # We can set this globally for all steps since any subsequent steps will either
        # start at the temperature at the end of the previous step (if non-isothermal
        # model), or will use the "Ambient temperature" input (if isothermal model).
        # In either case, the initial temperature is not used for any steps except
        # the first.
        init_temp = self.experiment.steps[0].temperature
        if init_temp is not None:
            parameter_values["Initial temperature [K]"] = init_temp

        blockers = self._get_unified_experiment_model_blockers()
        if self._experiment_model_mode == "unified":
            if blockers:
                raise pybamm.ModelError(
                    "Cannot build a unified experiment model: "
                    + "; ".join(blockers)
                    + ". Use 'legacy' mode or a compatible solver/experiment."
                )
            self._set_up_unified_experiment_model(parameter_values)
            return

        self._experiment_uses_unified_model = False
        self._built_experiment_model = None
        self._built_experiment_solver = None
        self._experiment_step_indices = []
        self._experiment_padding_rest_index = None
        self._experiment_includes_padding_rest = False

        self.experiment_unique_steps_to_model = {}
        for step in self.experiment.unique_steps:
            new_model = step.process_model(
                self._model,
                parameter_values,
                delayed_variable_processing=True,
            )
            self.experiment_unique_steps_to_model[step.basic_repr()] = new_model

        # Set up rest model if experiment has start times
        if self.experiment.initial_start_time:
            # duration doesn't matter, we just need the model
            rest_step = pybamm.step.Rest(duration=1)
            # Change ambient temperature to be an input, which will be changed at
            # solve time
            parameter_values["Ambient temperature [K]"] = "[input]"
            new_model = rest_step.process_model(
                self._model,
                parameter_values,
                delayed_variable_processing=True,
            )
            self.experiment_unique_steps_to_model[self._PADDING_REST_KEY] = new_model

    def _get_built_models(self):
        """Return list of built models that need IC recomputation."""
        models = super()._get_built_models()
        if self._built_experiment_model is not None:
            models.append(self._built_experiment_model)
        elif self.steps_to_built_models is not None:
            models.extend(self.steps_to_built_models.values())
        return models

    def _recompute_initial_conditions(self):
        super()._recompute_initial_conditions()
        # Also clear per-step solver caches so they re-process the updated models
        if self.steps_to_built_solvers is not None:
            for solver in self.steps_to_built_solvers.values():
                solver._model_set_up = {}
        if self._built_experiment_solver is not None:
            self._built_experiment_solver._model_set_up = {}

    def _build_experiment_state_mappers(self, inputs: dict):
        self.model_state_mappers = {}
        self._compiled_model_state_mappers = {}

        if (
            not self.experiment
            or not self.steps_to_built_models
            or self._experiment_uses_unified_model
        ):
            return

        ordered_steps = self.experiment.steps
        previous_model = None
        for step in ordered_steps:
            model = self.steps_to_built_models[step.basic_repr()]
            if previous_model is not None and previous_model is not model:
                key = (previous_model, model)
                if key not in self.model_state_mappers:
                    self.model_state_mappers[key] = (
                        model.build_initial_state_mapper(previous_model)
                    )
            previous_model = model

        rest_model = self.steps_to_built_models.get(self._PADDING_REST_KEY)
        if rest_model is not None:
            unique_models = set(self.steps_to_built_models.values())
            for model in unique_models:
                if model is rest_model:
                    continue
                to_rest_key = (model, rest_model)
                if to_rest_key not in self.model_state_mappers:
                    self.model_state_mappers[to_rest_key] = (
                        rest_model.build_initial_state_mapper(model)
                    )
                from_rest_key = (rest_model, model)
                if from_rest_key not in self.model_state_mappers:
                    self.model_state_mappers[from_rest_key] = (
                        model.build_initial_state_mapper(rest_model)
                    )

        for (previous_model, next_model), mapper in self.model_state_mappers.items():
            ordered_compile_inputs = {
                name: inputs.get(name, 0)
                for name in sorted(ip.name for ip in previous_model.input_parameters)
            }
            vars_for_processing = pybamm.BaseSolver._get_vars_for_processing(
                previous_model, ordered_compile_inputs
            )
            if not hasattr(previous_model, "calculate_sensitivities"):
                previous_model.calculate_sensitivities = []
            f, jac, jacp, _jac_action = process(mapper, "mapper", vars_for_processing)
            self._compiled_model_state_mappers[(previous_model, next_model)] = (
                f,
                jac,
                jacp,
            )

    def _get_state_mapper_for_solution(self, solution, model):
        if not self._compiled_model_state_mappers or isinstance(
            solution, pybamm.EmptySolution
        ):
            return None
        if not solution.all_models:
            return None
        from_model = solution.all_models[-1]
        return self._compiled_model_state_mappers.get((from_model, model))

    @staticmethod
    def _should_save_cycle(
        cycle_num, num_cycles, cycle_offset, save_at_cycles
    ):
        """Determine whether to persist full solution data for this cycle."""
        return (
            cycle_num == 1
            or cycle_num == num_cycles
            or save_at_cycles is None
            or (
                isinstance(save_at_cycles, list)
                and cycle_num + cycle_offset in save_at_cycles
            )
            or (
                isinstance(save_at_cycles, int)
                and (cycle_num + cycle_offset) % save_at_cycles == 0
            )
        )

    def _check_infeasible_steps(self, steps, step, step_str, cycle_num):
        """Raise or skip when all steps in a cycle produced EmptySolution."""
        if len(steps) == 1:
            if step.skip_ok:
                pybamm.logger.warning(
                    f"Step '{step_str}' is infeasible at initial conditions, "
                    "but skip_ok is True. Skipping step."
                )
                self._solution.termination = steps[0].termination
                return True  # signal: continue to next cycle
            raise pybamm.SolverError(
                f"Step '{step_str}' is infeasible "
                "due to exceeded bounds at initial conditions. "
                "If this step is part of a longer cycle, "
                "round brackets should be used to indicate this, "
                "e.g.:\n pybamm.Experiment([(\n"
                "\tDischarge at C/5 for 10 hours or until 3.3 V,\n"
                "\tCharge at 1 A until 4.1 V,\n"
                "\tHold at 4.1 V until 10 mA\n"
                ")])\n"
                "Otherwise, set skip_ok=True when instantiating the step "
                "to skip this step."
            )

        this_cycle = self.experiment.cycles[cycle_num - 1]
        all_steps_skipped = all(
            this_step.skip_ok
            for this_step in this_cycle
            if isinstance(this_step, pybamm.step.BaseStep)
        )
        msg = f"All steps in the cycle {this_cycle} are infeasible due to exceeded bounds at initial conditions."
        if all_steps_skipped:
            msg += " skip_ok is True for all steps. Please recheck the experiment."
        raise pybamm.SolverError(msg)

    def build_for_experiment(
        self, initial_soc=None, direction=None, inputs=None, solve_kwargs=None
    ):
        """
        Similar to :meth:`Simulation.build`, but for the case of simulating an
        experiment, where there may be several models and solvers to build.
        """
        if initial_soc is not None:
            self.set_initial_state(initial_soc, direction=direction, inputs=inputs)

        if self._built_experiment_model is not None or self.steps_to_built_models:
            if self._needs_ic_rebuild:
                self._recompute_initial_conditions()
            self._update_experiment_models_for_capacity(inputs, solve_kwargs)
            return

        self._set_up_and_parameterise_experiment(solve_kwargs)
        self._parameter_values.process_geometry(self._geometry)
        self._mesh = pybamm.Mesh(self._geometry, self._submesh_types, self._var_pts)
        self._disc = pybamm.Discretisation(
            self._mesh, self._spatial_methods, **self._discretisation_kwargs
        )
        self._discretise_experiment_models()

        if inputs is None:
            inputs = {}
        self._build_experiment_state_mappers(inputs)
        self._built_nominal_capacity = self._parameter_values.get(
            "Nominal cell capacity [A.h]", None
        )
        self._needs_ic_rebuild = False

    def solve(
        self,
        t_eval=None,
        solver=None,
        save_at_cycles=None,
        calc_esoh=None,
        starting_solution=None,
        initial_soc=None,
        direction=None,
        callbacks=None,
        showprogress=False,
        inputs=None,
        t_interp=None,
        **kwargs,
    ):
        """Solve the model, with experiment support.

        Accepts all parameters from :meth:`BaseSimulation.solve`, plus:

        Parameters
        ----------
        save_at_cycles : int or list of ints, optional
            Which cycles to save the full sub-solutions for. If None, all cycles are
            saved. If int, every multiple of save_at_cycles is saved. If list, every
            cycle in the list is saved. The first cycle (cycle 1) is always saved.
        starting_solution : :class:`pybamm.Solution`
            The solution to start stepping from. If None (default), then self._solution
            is used. Must be None if not using an experiment.
        showprogress : bool, optional
            Whether to show a progress bar for cycling. If true, shows a progress bar
            for cycles. Has no effect when not used with an experiment.
            Default is False.
        """
        if self.operating_mode != self.MODE_WITH_EXPERIMENT:
            if save_at_cycles is not None:
                raise ValueError(
                    "'save_at_cycles' option can only be used if simulating an "
                    "Experiment "
                )
            if starting_solution is not None:
                raise ValueError(
                    "starting_solution can only be provided if simulating an Experiment"
                )
            return super().solve(
                t_eval=t_eval,
                solver=solver,
                calc_esoh=calc_esoh,
                initial_soc=initial_soc,
                direction=direction,
                callbacks=callbacks,
                inputs=inputs,
                t_interp=t_interp,
                **kwargs,
            )

        # --- Experiment solve path ---
        t_eval, solver, calc_esoh, callbacks, inputs = self._prepare_solve(
            t_eval, solver, calc_esoh, callbacks, inputs
        )
        logs = {}

        callbacks.on_experiment_start(logs)

        if isinstance(inputs, list):
            raise pybamm.SolverError(
                "Solving with a list of input sets is not supported with experiments."
            )

        self.build_for_experiment(
            initial_soc=initial_soc,
            direction=direction,
            inputs=inputs,
            solve_kwargs=kwargs,
        )
        if t_eval is not None:
            pybamm.logger.warning(
                "Ignoring t_eval as solution times are specified by the experiment"
            )
        # Re-initialize solution, e.g. for solving multiple times with different
        # inputs without having to build the simulation again
        self._solution = starting_solution
        user_inputs = inputs
        timer = pybamm.Timer()

        # Set up eSOH solver (for summary variables)
        esoh_solver = self.get_esoh_solver(calc_esoh, direction)

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

        initial_start_time = (
            self.experiment.initial_start_time
            if starting_solution is None
            else starting_solution.initial_start_time
        )

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
        stop_experiment = False
        experiment_termination = None

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
                if rest_time > 0:
                    inputs = self._build_experiment_step_inputs(
                        user_inputs,
                        step,
                        current_solution.t[-1],
                        self._experiment_padding_rest_index,
                    )

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

                    self._solution = current_solution

        experiment_steps = self.experiment.steps
        uses_unified = self._experiment_uses_unified_model
        step_indices = self._experiment_step_indices

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
            save_this_cycle = self._should_save_cycle(
                cycle_num, num_cycles, cycle_offset, save_at_cycles
            )
            for step_num in range(1, cycle_length + 1):
                step = experiment_steps[idx]
                start_time = current_solution.t[-1]

                dt = step.duration
                if step.end_time is not None:
                    remaining = (
                        step.end_time
                        - (initial_start_time + timedelta(seconds=float(start_time)))
                    ).total_seconds()
                    dt = min(dt, remaining)

                if time_stop is not None:
                    dt = min(dt, time_stop - start_time)
                    if dt <= 0:
                        experiment_termination = self._TERMINATION_TIME
                        stop_experiment = True
                        break

                step_str = str(step)
                model = self._get_built_experiment_model(step)
                solver = self._get_built_experiment_solver(step)

                logs["step number"] = (step_num, cycle_length)
                logs["step operating conditions"] = step_str
                logs["step duration"] = step.duration
                callbacks.on_step_start(logs)

                active_step_index = step_indices[idx] if uses_unified else None
                inputs = self._build_experiment_step_inputs(
                    user_inputs,
                    step,
                    start_time,
                    active_step_index,
                    include_temperature=uses_unified,
                )

                t_eval, t_interp_processed = step.setup_timestepping(
                    solver, dt, t_interp
                )

                state_mapper = self._get_state_mapper_for_solution(
                    current_solution, model
                )

                try:
                    step_solution = solver.step(
                        current_solution,
                        model,
                        dt,
                        t_eval,
                        t_interp=t_interp_processed,
                        save=False,
                        inputs=inputs,
                        state_mapper=state_mapper,
                        **kwargs,
                    )
                except pybamm.SolverError as error:
                    if (
                        "non-positive at initial conditions" in error.message
                        and "[experiment]" in error.message
                    ):
                        step_solution = pybamm.EmptySolution(
                            "Event exceeded in initial conditions",
                            t=start_time,
                        )
                    else:
                        logs["error"] = error
                        callbacks.on_experiment_error(logs)
                        feasible = False
                        # If none of the cycles worked, raise an error
                        if cycle_num == 1 and step_num == 1:
                            raise error from error
                        # Otherwise, just stop this cycle
                        break

                if uses_unified:
                    step_termination = self._decode_combined_step_termination(
                        step_solution, step, model, inputs
                    )
                else:
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
                    if rest_time > 0:
                        logs["step number"] = (step_num, cycle_length)
                        logs["step operating conditions"] = self._PADDING_REST_KEY
                        callbacks.on_step_start(logs)

                        inputs = self._build_experiment_step_inputs(
                            user_inputs,
                            step,
                            step_solution.t[-1],
                            self._experiment_padding_rest_index,
                        )

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
                if step_termination == self._TERMINATION_FINAL_TIME and step.uses_default_duration:
                    # reached the default duration of a step (typically we should
                    # reach an event before the default duration)
                    callbacks.on_experiment_infeasible_time(logs)
                    feasible = False
                    break

                elif not (
                    isinstance(step_solution, pybamm.EmptySolution)
                    or step_termination == self._TERMINATION_FINAL_TIME
                    or self._TERMINATION_EXPERIMENT_TAG in step_termination
                ):
                    # Step has reached an event that is not specified in the
                    # experiment
                    callbacks.on_experiment_infeasible_event(logs)
                    feasible = False
                    break

                elif time_stop is not None and logs["experiment time"] >= time_stop:
                    experiment_termination = self._TERMINATION_TIME
                    stop_experiment = True
                    break

                else:
                    idx += 1

            if cycle_solution is not None and (
                save_this_cycle or feasible is False or stop_experiment
            ):
                self._solution = self._solution + cycle_solution

            if steps:
                if all(
                    isinstance(s, pybamm.EmptySolution) for s in steps
                ):
                    if self._check_infeasible_steps(
                        steps, step, step_str, cycle_num
                    ):
                        continue
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
            else:
                capacity_stop = logs["stopping conditions"].get("capacity")

            logs["elapsed time"] = timer.time()

            # Add minimum voltage to summary variable logs if there is a voltage stop
            min_voltage = None
            if voltage_stop is not None:
                min_voltage = np.min(cycle_solution["Battery voltage [V]"].data)
                logs["Minimum voltage [V]"] = min_voltage

            callbacks.on_cycle_end(logs)

            # Break if stopping conditions are met
            # Logging is done in the callbacks
            if capacity_stop is not None:
                capacity_now = cycle_sum_vars["Capacity [A.h]"]
                if not np.isnan(capacity_now) and capacity_now <= capacity_stop:
                    experiment_termination = self._TERMINATION_CAPACITY
                    stop_experiment = True
                    break

            if voltage_stop is not None and min_voltage <= voltage_stop[0]:
                experiment_termination = self._TERMINATION_VOLTAGE
                stop_experiment = True
                break

            if not feasible:
                break

            if stop_experiment:
                break

        if self._solution is not None and len(all_cycle_solutions) > 0:
            self._solution.cycles = all_cycle_solutions
            self._solution.update_summary_variables(all_summary_variables)
            self._solution.all_first_states = all_first_states

        if self._solution is not None and experiment_termination is not None:
            self._solution.termination = experiment_termination

        callbacks.on_experiment_end(logs)

        # record initial_start_time of the solution
        self._solution.initial_start_time = initial_start_time

        return self._solution

    def run_padding_rest(self, kwargs, rest_time, step_solution, inputs):
        model = self._get_built_experiment_model(self._PADDING_REST_KEY)
        solver = self._get_built_experiment_solver(self._PADDING_REST_KEY)
        state_mapper = self._get_state_mapper_for_solution(step_solution, model)

        # Make sure we take at least 2 timesteps. The period is hardcoded to 10
        # minutes,the user can always override it by adding a rest step
        npts = max(round(rest_time / 600) + 1, 2)

        step_solution_with_rest = solver.step(
            step_solution,
            model,
            rest_time,
            t_eval=np.linspace(0, rest_time, npts),
            save=False,
            inputs=inputs,
            state_mapper=state_mapper,
            **kwargs,
        )

        return step_solution_with_rest

    def save(self, filename):
        if self.steps_to_built_solvers is not None:
            for solver in self.steps_to_built_solvers.values():
                if (
                    isinstance(solver, pybamm.CasadiSolver)
                    and solver.integrator_specs != {}
                ):
                    solver.integrator_specs = {}
        if (
            isinstance(self._built_experiment_solver, pybamm.CasadiSolver)
            and self._built_experiment_solver.integrator_specs != {}
        ):
            self._built_experiment_solver.integrator_specs = {}

        super().save(filename)

    def save_model(
        self,
        filename: str | None = None,
        mesh: bool = False,
        variables: bool = False,
    ):
        if self.operating_mode == self.MODE_WITH_EXPERIMENT:
            raise NotImplementedError(
                """
                Serialising models coupled to experiments is not yet supported.
                """
            )

        super().save_model(filename=filename, mesh=mesh, variables=variables)


def load_sim(filename):
    """Load a saved simulation"""
    return pybamm.load(filename)
