#
# Summary Variable class
#
from __future__ import annotations
import pybamm
import numpy as np


class SummaryVariables:
    """
    Class for managing and calculating summary variables from a PyBaMM solution.

    Parameters
    ----------
    solution : :class:`pybamm.Solution`
        The solution object to be used for creating the processed variables.
    cycle_summary_variables : list, optional
        A list of cycle summary variables.
    esoh_solver : :class:`pybamm.lithium_ion.ElectrodeSOHSolver`, optional
        Solver for electrode state-of-health (eSOH) calculations.
    user_inputs : dict, optional
        Additional user inputs for calculations.
    """

    def __init__(
        self, solution, cycle_summary_variables=None, esoh_solver=None, user_inputs=None
    ):
        self.user_inputs = user_inputs or {}
        self.model = solution.all_models[0]
        self.esoh_solver = esoh_solver
        self._variables = {}  # Store computed variables
        self._esoh_variables = []
        self._possible_variables = self.model.summary_variables  # minus esoh variables
        self.cycle_number = None

        # Initialize based on cycle information
        if cycle_summary_variables:
            self._initialize_for_cycles(cycle_summary_variables)
        else:
            self.first_state = solution.first_state
            self.last_state = solution.last_state
            self.cycles = None

        # Flag if eSOH calculations are needed
        self.calc_esoh = (
            self.esoh_solver is not None
            and isinstance(self.model, pybamm.lithium_ion.BaseModel)
            and self.model.options.electrode_types["negative"] == "porous"
            and "Negative electrode capacity [A.h]" in self.model.variables
            and "Positive electrode capacity [A.h]" in self.model.variables
        )

    def _initialize_for_cycles(self, cycle_summary_variables):
        """Initialize attributes for cycle-based calculations."""
        self.first_state = None
        self.last_state = None
        self.cycles = cycle_summary_variables
        self.cycle_number = np.arange(1, len(self.cycles) + 1)
        first_cycle = self.cycles[0]
        self.esoh_solver = first_cycle.esoh_solver
        self.user_inputs = first_cycle.user_inputs

    @property
    def all_variables(self):
        """Return all possible summary variables, including eSOH variables."""
        try:
            return self._all_variables
        except AttributeError:
            base_vars = self._possible_variables.copy()
            base_vars.extend(
                f"Change in {var[0].lower() + var[1:]}"
                for var in self._possible_variables
            )

            if self._esoh_variables:
                base_vars.extend(self._esoh_variables)
            elif self.calc_esoh:
                esoh_model = self.esoh_solver._get_electrode_soh_sims_full().model
                esoh_vars = esoh_model.variables.keys()
                base_vars.extend(esoh_vars)
                self._esoh_variables = esoh_vars

            self._all_variables = base_vars
            return self._all_variables

    def __getitem__(self, key):
        """
        Access or compute a summary variable by its name.

        Parameters
        ----------
        key : str
            The name of the variable

        Returns
        -------
        float or np.array
        """

        if key in self._variables:
            # return it if it exists
            return self._variables[key]
        elif key not in self.all_variables:
            # check it's listed as a summary variable
            raise KeyError(f"Variable '{key}' is not a summary variable.")
        else:
            # otherwise create it, save it and then return it
            if self.calc_esoh and key in self._esoh_variables:
                self.update_esoh()
            else:
                base_key = key.removeprefix("Change in ")
                base_key = base_key[0].upper() + base_key[1:]
                # this will create 'X' and 'Change in x' at the same time
                self.update(base_key)
            return self._variables[key]

    def update(self, var):
        """Compute and store a variable and its change."""
        var_lowercase = var[0].lower() + var[1:]
        if self.cycles:
            self._update_multiple_cycles(var, var_lowercase)
        else:
            self._update(var, var_lowercase)

    def _update_multiple_cycles(self, var, var_lowercase):
        """Update variables for where more than one cycle exists."""
        var_cycle = [cycle[var] for cycle in self.cycles]
        change_var_cycle = [
            cycle[f"Change in {var_lowercase}"] for cycle in self.cycles
        ]
        self._variables[var] = var_cycle
        self._variables[f"Change in {var_lowercase}"] = change_var_cycle

    def _update(self, var, var_lowercase):
        """Update variables for state-based data."""
        data_first = self.first_state[var].data
        data_last = self.last_state[var].data
        self._variables[var] = data_last[0]
        self._variables[f"Change in {var_lowercase}"] = data_last[0] - data_first[0]

    def update_esoh(self):
        """Create all aggregated eSOH variables"""
        if self.cycles is not None:
            var_cycle = [cycle._get_esoh_variables() for cycle in self.cycles]
            aggregated_vars = {k: [] for k in var_cycle[0].keys()}
            for cycle in var_cycle:
                for k, v in cycle.items():
                    aggregated_vars[k].append(v)
            self._variables.update(aggregated_vars)
        else:
            self._variables.update(self._get_esoh_variables())

    def _get_esoh_variables(self):
        """Compute eSOH variables for a single solution."""
        Q_n = self.last_state["Negative electrode capacity [A.h]"].data[0]
        Q_p = self.last_state["Positive electrode capacity [A.h]"].data[0]
        Q_Li = self.last_state["Total lithium capacity in particles [A.h]"].data[0]
        all_inputs = {**self.user_inputs, "Q_n": Q_n, "Q_p": Q_p, "Q_Li": Q_Li}
        try:
            esoh_sol = self.esoh_solver.solve(inputs=all_inputs)
        except pybamm.SolverError as error:  # pragma: no cover
            raise pybamm.SolverError(
                "Could not solve for eSOH summary variables"
            ) from error

        return esoh_sol
