#
# Solution class
#
import casadi
import json
import numbers
import numpy as np
import pickle
import pybamm
import pandas as pd
from scipy.io import savemat


class NumpyEncoder(json.JSONEncoder):
    """
    Numpy serialiser helper class that converts numpy arrays to a list
    https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # won't be called since we only need to convert numpy arrays
        return json.JSONEncoder.default(self, obj)  # pragma: no cover


class Solution(object):
    """
    Class containing the solution of, and various attributes associated with, a PyBaMM
    model.

    Parameters
    ----------
    all_ts : :class:`numpy.array`, size (n,) (or list of these)
        A one-dimensional array containing the times at which the solution is evaluated.
        A list of times can be provided instead to initialize a solution with
        sub-solutions.
    all_ys : :class:`numpy.array`, size (m, n) (or list of these)
        A two-dimensional array containing the values of the solution. y[i, :] is the
        vector of solutions at time t[i].
        A list of ys can be provided instead to initialize a solution with
        sub-solutions.
    all_models : :class:`pybamm.BaseModel`
        The model that was used to calculate the solution.
        A list of models can be provided instead to initialize a solution with
        sub-solutions that have been calculated using those models.
    all_inputs : dict (or list of these)
        The inputs that were used to calculate the solution
        A list of inputs can be provided instead to initialize a solution with
        sub-solutions.
    t_event : :class:`numpy.array`, size (1,)
        A zero-dimensional array containing the time at which the event happens.
    y_event : :class:`numpy.array`, size (m,)
        A one-dimensional array containing the value of the solution at the time when
        the event happens.
    termination : str
        String to indicate why the solution terminated

    sensitivities: bool or dict
        True if sensitivities included as the solution of the explicit forwards
        equations.  False if no sensitivities included/wanted. Dict if sensitivities are
        provided as a dict of {parameter: sensitivities} pairs.

    """

    def __init__(
        self,
        all_ts,
        all_ys,
        all_models,
        all_inputs,
        t_event=None,
        y_event=None,
        termination="final time",
        sensitivities=False,
    ):
        if not isinstance(all_ts, list):
            all_ts = [all_ts]
        if not isinstance(all_ys, list):
            all_ys = [all_ys]
        if not isinstance(all_models, list):
            all_models = [all_models]
        self._all_ts = all_ts
        self._all_ys = all_ys
        self._all_ys_and_sens = all_ys
        self._all_models = all_models

        # Set up inputs
        if not isinstance(all_inputs, list):
            all_inputs_copy = dict(all_inputs)
            for key, value in all_inputs_copy.items():
                if isinstance(value, numbers.Number):
                    all_inputs_copy[key] = np.array([value])
            self.all_inputs = [all_inputs_copy]
        else:
            self.all_inputs = all_inputs

        self.sensitivities = sensitivities

        self._t_event = t_event
        self._y_event = y_event
        self._termination = termination

        self.has_symbolic_inputs = any(
            isinstance(v, casadi.MX) for v in self.all_inputs[0].values()
        )

        # Copy the timescale_eval and lengthscale_evals if they exist
        if hasattr(all_models[0], "timescale_eval"):
            self.timescale_eval = all_models[0].timescale_eval
        else:
            self.timescale_eval = all_models[0].timescale.evaluate()

        if hasattr(all_models[0], "length_scales_eval"):
            self.length_scales_eval = all_models[0].length_scales_eval
        else:
            self.length_scales_eval = {
                domain: scale.evaluate()
                for domain, scale in all_models[0].length_scales.items()
            }

        # Events
        self._t_event = t_event
        self._y_event = y_event
        self._termination = termination
        self.closest_event_idx = None

        # Initialize times
        self.set_up_time = None
        self.solve_time = None
        self.integration_time = None

        # initiaize empty variables and data
        self._variables = pybamm.FuzzyDict()
        self.data = pybamm.FuzzyDict()

        # Add self as sub-solution for compatibility with ProcessedVariable
        self._sub_solutions = [self]

        # initialize empty cycles
        self._cycles = []

        # Initialize empty summary variables
        self._summary_variables = None

        # Solution now uses CasADi
        pybamm.citations.register("Andersson2019")

    def extract_explicit_sensitivities(self):
        # if we got here, we havn't set y yet
        self.set_y()

        # extract sensitivities from full y solution
        self._y, self._sensitivities = self._extract_explicit_sensitivities(
            self.all_models[0], self.y, self.t, self.all_inputs[0]
        )

        # make sure we remove all sensitivities from all_ys
        for index, (model, ys, ts, inputs) in enumerate(
            zip(self.all_models, self.all_ys, self.all_ts, self.all_inputs)
        ):
            self._all_ys[index], _ = self._extract_explicit_sensitivities(
                model, ys, ts, inputs
            )

    def _extract_explicit_sensitivities(self, model, y, t_eval, inputs):
        """
        given a model and a solution y, extracts the sensitivities

        Parameters
        --------
        model : :class:`pybamm.BaseModel`
            A model that has been already setup by this base solver
        y: ndarray
            The solution of the full explicit sensitivity equations
        t_eval: ndarray
            The evaluation times
        inputs: dict
            parameter inputs

        Returns
        -------
        y: ndarray
            The solution of the ode/dae in model
        sensitivities: dict of (string: ndarray)
            A dictionary of parameter names, and the corresponding solution of
            the sensitivity equations
        """

        n_states = model.len_rhs_and_alg
        n_rhs = model.len_rhs
        n_alg = model.len_alg
        # Get the point where the algebraic equations start
        if model.len_rhs != 0:
            n_p = model.len_rhs_sens // model.len_rhs
        else:
            n_p = model.len_alg_sens // model.len_alg
        len_rhs_and_sens = model.len_rhs + model.len_rhs_sens

        n_t = len(t_eval)
        # y gets the part of the solution vector that correspond to the
        # actual ODE/DAE solution

        # save sensitivities as a dictionary
        # first save the whole sensitivity matrix
        # reshape using Fortran order to get the right array:
        #   t0_x0_p0, t0_x0_p1, ..., t0_x0_pn
        #   t0_x1_p0, t0_x1_p1, ..., t0_x1_pn
        #   ...
        #   t0_xn_p0, t0_xn_p1, ..., t0_xn_pn
        #   t1_x0_p0, t1_x0_p1, ..., t1_x0_pn
        #   t1_x1_p0, t1_x1_p1, ..., t1_x1_pn
        #   ...
        #   t1_xn_p0, t1_xn_p1, ..., t1_xn_pn
        #   ...
        #   tn_x0_p0, tn_x0_p1, ..., tn_x0_pn
        #   tn_x1_p0, tn_x1_p1, ..., tn_x1_pn
        #   ...
        #   tn_xn_p0, tn_xn_p1, ..., tn_xn_pn
        # 1, Extract rhs and alg sensitivities and reshape into 3D matrices
        # with shape (n_p, n_states, n_t)
        if isinstance(y, casadi.DM):
            y_full = y.full()
        else:
            y_full = y
        ode_sens = y_full[n_rhs:len_rhs_and_sens, :].reshape(n_p, n_rhs, n_t)
        alg_sens = y_full[len_rhs_and_sens + n_alg :, :].reshape(n_p, n_alg, n_t)
        # 2. Concatenate into a single 3D matrix with shape (n_p, n_states, n_t)
        # i.e. along first axis
        full_sens_matrix = np.concatenate([ode_sens, alg_sens], axis=1)
        # Transpose and reshape into a (n_states * n_t, n_p) matrix
        full_sens_matrix = full_sens_matrix.transpose(2, 1, 0).reshape(
            n_t * n_states, n_p
        )

        # Save the full sensitivity matrix
        sensitivity = {"all": full_sens_matrix}

        # also save the sensitivity wrt each parameter (read the columns of the
        # sensitivity matrix)
        start = 0
        for name in model.calculate_sensitivities:
            inp = inputs[name]
            input_size = inp.shape[0]
            end = start + input_size
            sensitivity[name] = full_sens_matrix[:, start:end]
            start = end

        y_dae = np.vstack(
            [
                y[: model.len_rhs, :],
                y[len_rhs_and_sens : len_rhs_and_sens + model.len_alg, :],
            ]
        )
        return y_dae, sensitivity

    @property
    def t(self):
        """Times at which the solution is evaluated"""
        try:
            return self._t
        except AttributeError:
            self.set_t()
            return self._t

    def set_t(self):
        self._t = np.concatenate(self.all_ts)
        if any(np.diff(self._t) <= 0):
            raise ValueError("Solution time vector must be strictly increasing")

    @property
    def y(self):
        """Values of the solution"""
        try:
            return self._y
        except AttributeError:
            self.set_y()

            # if y is evaluated before sensitivities then need to extract them
            if isinstance(self._sensitivities, bool) and self._sensitivities:
                self.extract_explicit_sensitivities()

            return self._y

    @property
    def sensitivities(self):
        """Values of the sensitivities. Returns a dict of param_name: np_array"""
        if isinstance(self._sensitivities, bool):
            if self._sensitivities:
                self.extract_explicit_sensitivities()
            else:
                self._sensitivities = {}
        return self._sensitivities

    @sensitivities.setter
    def sensitivities(self, value):
        """Updates the sensitivity"""
        # sensitivities must be a dict or bool
        if not isinstance(value, (bool, dict)):
            raise TypeError("sensitivities arg needs to be a bool or dict")
        self._sensitivities = value

    def set_y(self):
        try:
            if isinstance(self.all_ys[0], (casadi.DM, casadi.MX)):
                self._y = casadi.horzcat(*self.all_ys)
            else:
                self._y = np.hstack(self.all_ys)
        except ValueError:
            raise pybamm.SolverError(
                "The solution is made up from different models, so `y` cannot be "
                "computed explicitly."
            )

    @property
    def all_ts(self):
        return self._all_ts

    @property
    def all_ys(self):
        return self._all_ys

    @property
    def all_models(self):
        """Model(s) used for solution"""
        return self._all_models

    @property
    def all_inputs_casadi(self):
        try:
            return self._all_inputs_casadi
        except AttributeError:
            self._all_inputs_casadi = [
                casadi.vertcat(*inp.values()) for inp in self.all_inputs
            ]
            return self._all_inputs_casadi

    @property
    def t_event(self):
        """Time at which the event happens"""
        return self._t_event

    @property
    def y_event(self):
        """Value of the solution at the time of the event"""
        return self._y_event

    @property
    def termination(self):
        """Reason for termination"""
        return self._termination

    @termination.setter
    def termination(self, value):
        """Updates the reason for termination"""
        self._termination = value

    @property
    def first_state(self):
        """
        A Solution object that only contains the first state. This is faster to evaluate
        than the full solution when only the first state is needed (e.g. to initialize
        a model with the solution)
        """
        try:
            return self._first_state
        except AttributeError:
            new_sol = Solution(
                self.all_ts[0][:1],
                self.all_ys[0][:, :1],
                self.all_models[:1],
                self.all_inputs[:1],
                None,
                None,
                "success",
            )
            new_sol._all_inputs_casadi = self.all_inputs_casadi[:1]
            new_sol._sub_solutions = self.sub_solutions[:1]

            new_sol.solve_time = 0
            new_sol.integration_time = 0
            new_sol.set_up_time = 0

            self._first_state = new_sol
            return self._first_state

    @property
    def last_state(self):
        """
        A Solution object that only contains the final state. This is faster to evaluate
        than the full solution when only the final state is needed (e.g. to initialize
        a model with the solution)
        """
        try:
            return self._last_state
        except AttributeError:
            new_sol = Solution(
                self.all_ts[-1][-1:],
                self.all_ys[-1][:, -1:],
                self.all_models[-1:],
                self.all_inputs[-1:],
                self.t_event,
                self.y_event,
                self.termination,
            )
            new_sol._all_inputs_casadi = self.all_inputs_casadi[-1:]
            new_sol._sub_solutions = self.sub_solutions[-1:]

            new_sol.solve_time = 0
            new_sol.integration_time = 0
            new_sol.set_up_time = 0

            self._last_state = new_sol
            return self._last_state

    @property
    def total_time(self):
        return self.set_up_time + self.solve_time

    @property
    def cycles(self):
        return self._cycles

    @cycles.setter
    def cycles(self, cycles):
        self._cycles = cycles

    @property
    def summary_variables(self):
        return self._summary_variables

    def set_summary_variables(self, all_summary_variables):
        summary_variables = {var: [] for var in all_summary_variables[0]}
        for sum_vars in all_summary_variables:
            for name, value in sum_vars.items():
                summary_variables[name].append(value)

        summary_variables["Cycle number"] = range(1, len(all_summary_variables) + 1)
        self.all_summary_variables = all_summary_variables
        self._summary_variables = pybamm.FuzzyDict(
            {name: np.array(value) for name, value in summary_variables.items()}
        )

    def update(self, variables):
        """Add ProcessedVariables to the dictionary of variables in the solution"""
        # make sure that sensitivities are extracted if required
        if isinstance(self._sensitivities, bool) and self._sensitivities:
            self.extract_explicit_sensitivities()

        # Convert single entry to list
        if isinstance(variables, str):
            variables = [variables]
        # Process
        for key in variables:
            pybamm.logger.debug("Post-processing {}".format(key))
            # If there are symbolic inputs then we need to make a
            # ProcessedSymbolicVariable
            if self.has_symbolic_inputs is True:
                var = pybamm.ProcessedSymbolicVariable(
                    self.all_models[0].variables[key], self
                )

            # Otherwise a standard ProcessedVariable is ok
            else:
                vars_pybamm = [model.variables[key] for model in self.all_models]

                # Iterate through all models, some may be in the list several times and
                # therefore only get set up once
                vars_casadi = []
                for model, ys, inputs, var_pybamm in zip(
                    self.all_models, self.all_ys, self.all_inputs, vars_pybamm
                ):
                    if key in model._variables_casadi:
                        var_casadi = model._variables_casadi[key]
                    else:
                        t_MX = casadi.MX.sym("t")
                        y_MX = casadi.MX.sym("y", ys.shape[0])
                        symbolic_inputs_dict = {
                            key: casadi.MX.sym("input", value.shape[0])
                            for key, value in inputs.items()
                        }
                        symbolic_inputs = casadi.vertcat(
                            *[p for p in symbolic_inputs_dict.values()]
                        )

                        # Convert variable to casadi
                        # Make all inputs symbolic first for converting to casadi
                        var_sym = var_pybamm.to_casadi(
                            t_MX, y_MX, inputs=symbolic_inputs_dict
                        )

                        var_casadi = casadi.Function(
                            "variable", [t_MX, y_MX, symbolic_inputs], [var_sym]
                        )
                        model._variables_casadi[key] = var_casadi
                    vars_casadi.append(var_casadi)

                var = pybamm.ProcessedVariable(vars_pybamm, vars_casadi, self)

            # Save variable and data
            self._variables[key] = var
            self.data[key] = var.data

    def __getitem__(self, key):
        """Read a variable from the solution. Variables are created 'just in time', i.e.
        only when they are called.

        Parameters
        ----------
        key : str
            The name of the variable

        Returns
        -------
        :class:`pybamm.ProcessedVariable`
            A variable that can be evaluated at any time or spatial point. The
            underlying data for this variable is available in its attribute ".data"
        """

        # return it if it exists
        if key in self._variables:
            return self._variables[key]
        else:
            # otherwise create it, save it and then return it
            self.update(key)
            return self._variables[key]

    def plot(self, output_variables=None, **kwargs):
        """
        A method to quickly plot the outputs of the solution. Creates a
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
        return pybamm.dynamic_plot(self, output_variables=output_variables, **kwargs)

    def save(self, filename):
        """Save the whole solution using pickle"""
        # No warning here if len(self.data)==0 as solution can be loaded
        # and used to process new variables

        with open(filename, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def save_data(
            self, filename=None, variables=None,
            to_format="pickle", short_names=None
    ):
        """
        Save solution data only (raw arrays)

        Parameters
        ----------
        filename : str, optional
            The name of the file to save data to. If None, then a str is returned
        variables : list, optional
            List of variables to save. If None, saves all of the variables that have
            been created so far
        to_format : str, optional
            The format to save to. Options are:

            - 'pickle' (default): creates a pickle file with the data dictionary
            - 'matlab': creates a .mat file, for loading in matlab
            - 'csv': creates a csv file (0D variables only)
            - 'json': creates a json file
        short_names : dict, optional
            Dictionary of shortened names to use when saving. This may be necessary when
            saving to MATLAB, since no spaces or special characters are allowed in
            MATLAB variable names. Note that not all the variables need to be given
            a short name.

        Returns
        -------
        data : str, optional
            str if 'csv' or 'json' is chosen and filename is None, otherwise None


        """
        if variables is None:
            # variables not explicitly provided -> save all variables that have been
            # computed
            data = self.data
        else:
            # otherwise, save only the variables specified
            data = {}
            for name in variables:
                data[name] = self[name].data
        if len(data) == 0:
            raise ValueError(
                """
                Solution does not have any data. Please provide a list of variables
                to save.
                """
            )

        # Use any short names if provided
        data_short_names = {}
        short_names = short_names or {}
        for name, var in data.items():
            # change to short name if it exists
            if name in short_names:
                data_short_names[short_names[name]] = var
            else:
                data_short_names[name] = var

        if to_format == "pickle":
            if filename is None:
                raise ValueError(
                    "pickle format must be written to a file"
                )
            with open(filename, "wb") as f:
                pickle.dump(data_short_names, f, pickle.HIGHEST_PROTOCOL)
        elif to_format == "matlab":
            if filename is None:
                raise ValueError(
                    "matlab format must be written to a file"
                )
            # Check all the variable names only contain a-z, A-Z or _ or numbers
            for name in data_short_names.keys():
                # Check the string only contains the following ASCII:
                # a-z (97-122)
                # A-Z (65-90)
                # _ (95)
                # 0-9 (48-57) but not in the first position
                for i, s in enumerate(name):
                    if not (
                        97 <= ord(s) <= 122
                        or 65 <= ord(s) <= 90
                        or ord(s) == 95
                        or (i > 0 and 48 <= ord(s) <= 57)
                    ):
                        raise ValueError(
                            "Invalid character '{}' found in '{}'. ".format(s, name)
                            + "MATLAB variable names must only contain a-z, A-Z, _, "
                            "or 0-9 (except the first position). "
                            "Use the 'short_names' argument to pass an alternative "
                            "variable name, e.g. \n\n"
                            "\tsolution.save_data(filename, "
                            "['Electrolyte concentration'], to_format='matlab, "
                            "short_names={'Electrolyte concentration': 'c_e'})"
                        )
            savemat(filename, data_short_names)
        elif to_format == "csv":
            for name, var in data_short_names.items():
                if var.ndim >= 2:
                    raise ValueError(
                        "only 0D variables can be saved to csv, but '{}' is {}D".format(
                            name, var.ndim - 1
                        )
                    )
            df = pd.DataFrame(data_short_names)
            return df.to_csv(filename, index=False)
        elif to_format == "json":
            if filename is None:
                return json.dumps(data_short_names, cls=NumpyEncoder)
            else:
                with open(filename, "w") as outfile:
                    json.dump(data_short_names, outfile, cls=NumpyEncoder)
        else:
            raise ValueError("format '{}' not recognised".format(to_format))

    @property
    def sub_solutions(self):
        """List of sub solutions that have been
        concatenated to form the full solution"""

        return self._sub_solutions

    def __add__(self, other):
        """Adds two solutions together, e.g. when stepping"""
        if not isinstance(other, Solution):
            raise pybamm.SolverError(
                "Only a Solution or None can be added to a Solution"
            )
        # Special case: new solution only has one timestep and it is already in the
        # existing solution. In this case, return a copy of the existing solution
        if (
            len(other.all_ts) == 1
            and len(other.all_ts[0]) == 1
            and other.all_ts[0][0] == self.all_ts[-1][-1]
        ):
            new_sol = self.copy()
            # Update termination using the latter solution
            new_sol._termination = other.termination
            new_sol._t_event = other._t_event
            new_sol._y_event = other._y_event
            return new_sol

        # Update list of sub-solutions
        if other.all_ts[0][0] == self.all_ts[-1][-1]:
            # Skip first time step if it is repeated
            all_ts = self.all_ts + [other.all_ts[0][1:]] + other.all_ts[1:]
            all_ys = self.all_ys + [other.all_ys[0][:, 1:]] + other.all_ys[1:]
        else:
            all_ts = self.all_ts + other.all_ts
            all_ys = self.all_ys + other.all_ys

        new_sol = Solution(
            all_ts,
            all_ys,
            self.all_models + other.all_models,
            self.all_inputs + other.all_inputs,
            other.t_event,
            other.y_event,
            other.termination,
            bool(self.sensitivities),
        )

        new_sol.closest_event_idx = other.closest_event_idx
        new_sol._all_inputs_casadi = self.all_inputs_casadi + other.all_inputs_casadi

        # Set solution time
        new_sol.solve_time = self.solve_time + other.solve_time
        new_sol.integration_time = self.integration_time + other.integration_time

        # Set sub_solutions
        new_sol._sub_solutions = self.sub_solutions + other.sub_solutions

        return new_sol

    def __radd__(self, other):
        """
        Right-side adding with special handling for the case None + Solution (returns
        Solution)
        """
        if other is None:
            return self.copy()
        else:
            raise pybamm.SolverError(
                "Only a Solution or None can be added to a Solution"
            )

    def copy(self):
        new_sol = self.__class__(
            self.all_ts,
            self.all_ys,
            self.all_models,
            self.all_inputs,
            self.t_event,
            self.y_event,
            self.termination,
        )
        new_sol._all_inputs_casadi = self.all_inputs_casadi
        new_sol._sub_solutions = self.sub_solutions
        new_sol.closest_event_idx = self.closest_event_idx

        new_sol.solve_time = self.solve_time
        new_sol.integration_time = self.integration_time
        new_sol.set_up_time = self.set_up_time

        return new_sol


def make_cycle_solution(step_solutions, esoh_sim=None, save_this_cycle=True):
    """
    Function to create a Solution for an entire cycle, and associated summary variables

    Parameters
    ----------
    step_solutions : list of :class:`Solution`
        Step solutions that form the entire cycle
    esoh_sim : :class:`pybamm.Simulation`, optional
        A simulation, whose model should be a :class:`pybamm.lithium_ion.ElectrodeSOH`
        model, which is used to calculate some of the summary variables. If `None`
        (default) then only summary variables that do not require the eSOH calculation
        are calculated. See [1] for more details on eSOH variables.
    save_this_cycle : bool, optional
        Whether to save the entire cycle variables or just the summary variables.
        Default True

    Returns
    -------
    cycle_solution : :class:`pybamm.Solution` or None
        The Solution object for this cycle, or None (if save_this_cycle is False)
    cycle_summary_variables : dict
        Dictionary of summary variables for this cycle

    References
    ----------
    .. [1] Mohtat, P., Lee, S., Siegel, J. B., & Stefanopoulou, A. G. (2019). Towards
    better estimability of electrode-specific state of health: Decoding the cell
    expansion. Journal of Power Sources, 427, 101-111.

    """
    sum_sols = step_solutions[0].copy()
    for step_solution in step_solutions[1:]:
        sum_sols = sum_sols + step_solution

    cycle_solution = Solution(
        sum_sols.all_ts,
        sum_sols.all_ys,
        sum_sols.all_models,
        sum_sols.all_inputs,
        sum_sols.t_event,
        sum_sols.y_event,
        sum_sols.termination,
    )
    cycle_solution._all_inputs_casadi = sum_sols.all_inputs_casadi
    cycle_solution._sub_solutions = sum_sols.sub_solutions

    cycle_solution.solve_time = sum_sols.solve_time
    cycle_solution.integration_time = sum_sols.integration_time
    cycle_solution.set_up_time = sum_sols.set_up_time

    cycle_solution.steps = step_solutions

    cycle_summary_variables = get_cycle_summary_variables(cycle_solution, esoh_sim)

    cycle_first_state = cycle_solution.first_state

    if save_this_cycle:
        cycle_solution.cycle_summary_variables = cycle_summary_variables
    else:
        cycle_solution = None

    return cycle_solution, cycle_summary_variables, cycle_first_state


def get_cycle_summary_variables(cycle_solution, esoh_sim):
    model = cycle_solution.all_models[0]
    cycle_summary_variables = pybamm.FuzzyDict({})

    # Measured capacity variables
    if "Discharge capacity [A.h]" in model.variables:
        Q = cycle_solution["Discharge capacity [A.h]"].data
        min_Q = np.min(Q)
        max_Q = np.max(Q)

        cycle_summary_variables.update(
            {
                "Minimum measured discharge capacity [A.h]": min_Q,
                "Maximum measured discharge capacity [A.h]": max_Q,
                "Measured capacity [A.h]": max_Q - min_Q,
            }
        )

    # Degradation variables
    degradation_variables = model.summary_variables
    first_state = cycle_solution.first_state
    last_state = cycle_solution.last_state
    for var in degradation_variables:
        data_first = first_state[var].data
        data_last = last_state[var].data
        cycle_summary_variables[var] = data_last[0]
        var_lowercase = var[0].lower() + var[1:]
        cycle_summary_variables["Change in " + var_lowercase] = (
            data_last[0] - data_first[0]
        )

    # eSOH variables (full-cell lithium-ion model only, for now)
    if (
        esoh_sim is not None
        and isinstance(model, pybamm.lithium_ion.BaseModel)
        and model.half_cell is False
    ):
        V_min = esoh_sim.parameter_values["Lower voltage cut-off [V]"]
        V_max = esoh_sim.parameter_values["Upper voltage cut-off [V]"]
        C_n = last_state["Negative electrode capacity [A.h]"].data[0]
        C_p = last_state["Positive electrode capacity [A.h]"].data[0]
        n_Li = last_state["Total lithium in particles [mol]"].data[0]
        if esoh_sim.solution is not None:
            # initialize with previous solution if it is available
            esoh_sim.built_model.set_initial_conditions_from(esoh_sim.solution)
            solver = None
        else:
            x_100_init = np.max(cycle_solution["Negative electrode SOC"].data)
            # make sure x_0 > 0
            C_init = np.minimum(0.95 * (C_n * x_100_init), max_Q - min_Q)

            # Solve the esoh model and add outputs to the summary variables
            # use CasadiAlgebraicSolver if there are interpolants
            if isinstance(
                esoh_sim.parameter_values["Negative electrode OCP [V]"], tuple
            ) or isinstance(
                esoh_sim.parameter_values["Positive electrode OCP [V]"], tuple
            ):
                solver = pybamm.CasadiAlgebraicSolver()
                # Choose x_100_init so as not to violate the interpolation limits
                if isinstance(
                    esoh_sim.parameter_values["Positive electrode OCP [V]"], tuple
                ):
                    y_100_min = np.min(
                        esoh_sim.parameter_values["Positive electrode OCP [V]"][1][:, 0]
                    )
                    x_100_max = (
                        n_Li * pybamm.constants.F.value / 3600 - y_100_min * C_p
                    ) / C_n
                    x_100_init = np.minimum(x_100_init, 0.99 * x_100_max)
            else:
                solver = None
            # Update initial conditions using the cycle solution
            esoh_sim.build()
            esoh_sim.built_model.set_initial_conditions_from(
                {"x_100": x_100_init, "C": C_init}
            )
        inputs = {
            "V_min": V_min,
            "V_max": V_max,
            "C_n": C_n,
            "C_p": C_p,
            "n_Li": n_Li,
        }

        try:
            esoh_sol = esoh_sim.solve([0], inputs=inputs, solver=solver)
        except pybamm.SolverError:  # pragma: no cover
            raise pybamm.SolverError(
                "Could not solve for summary variables, run "
                "`sim.solve(calc_esoh=False)` to skip this step"
            )
        for var in esoh_sim.built_model.variables:
            cycle_summary_variables[var] = esoh_sol[var].data[0]

        cycle_summary_variables["Capacity [A.h]"] = cycle_summary_variables["C"]

    return cycle_summary_variables
