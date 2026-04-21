from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

import pybamm

from .base_simulation import BaseSimulation
from .eis_utils import SymbolReplacer


class EISSimulation(BaseSimulation):
    """Frequency-domain EIS simulation built on :class:`BaseSimulation`.

    Parameters
    ----------
    model : :class:`pybamm.BaseModel`
        The model to be simulated.
    parameter_values : :class:`pybamm.ParameterValues`, optional
        Parameters and their corresponding numerical values.
    geometry : :class:`pybamm.Geometry`, optional
        The geometry upon which to solve the model.
    submesh_types : dict, optional
        A dictionary of the types of submesh to use on each subdomain.
    var_pts : dict, optional
        A dictionary of the number of points used by each spatial variable.
    spatial_methods : dict, optional
        A dictionary of the types of spatial method to use on each domain.
    """

    def __init__(
        self,
        model,
        parameter_values=None,
        geometry=None,
        submesh_types=None,
        var_pts=None,
        spatial_methods=None,
    ):
        timer = pybamm.Timer()
        model_name = model.name

        parameter_values = parameter_values or model.default_parameter_values

        # Validate required variables and surface form before any processing
        self._validate_model_for_eis(model)

        # Compute impedance scale factor before model transformation
        V_scale = getattr(model.variables["Voltage [V]"], "scale", 1)
        I_scale = getattr(model.variables["Current [A]"], "scale", 1)
        self._z_scale = parameter_values.evaluate(V_scale / I_scale)

        pybamm.logger.info(f"Setting up {model_name} for EIS")
        eis_model = self._set_up_model_for_eis(model)

        parameter_values["Current function [A]"] = 0

        super().__init__(
            eis_model,
            parameter_values=parameter_values,
            geometry=geometry,
            submesh_types=submesh_types,
            var_pts=var_pts,
            spatial_methods=spatial_methods,
        )

        self.set_up_time = timer.time()
        pybamm.logger.info(
            f"Finished setting up {model_name} for EIS "
            f"(set-up time: {self.set_up_time})"
        )
        pybamm.citations.register("Hallemans2025")

    @staticmethod
    def _validate_model_for_eis(model):
        """Validate that a model is suitable for frequency-domain EIS.

        Raises
        ------
        ValueError
            If the model is missing required variables or options.
        """
        required_vars = ["Voltage [V]", "Current [A]"]
        for var in required_vars:
            if var not in model.variables:
                raise ValueError(
                    f"Model must contain variable '{var}' for EIS simulation"
                )

        surface_form = model.options.get("surface form", "false")
        if surface_form not in ("differential", "algebraic"):
            raise ValueError(
                f"EIS simulation requires 'surface form' model option to be "
                f"'differential' or 'algebraic', got '{surface_form}'. "
                f"Use e.g. pybamm.lithium_ion.SPM("
                f'options={{"surface form": "differential"}})'
            )

    @staticmethod
    def _set_up_model_for_eis(model):
        """Prepare a model for frequency-domain EIS.

        Creates a copy of the model with voltage and current as algebraic
        state variables, suitable for extracting the mass matrix and Jacobian
        needed by the frequency-domain solver.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            Original model.

        Returns
        -------
        new_model : :class:`pybamm.BaseModel`
            Modified model copy.
        """

        new_model = model.new_copy()

        # Add voltage as an algebraic state variable
        V_cell = pybamm.Variable("Voltage variable [V]")
        new_model.variables["Voltage variable [V]"] = V_cell
        V = new_model.variables["Voltage [V]"]
        new_model.algebraic[V_cell] = V_cell - V
        new_model.initial_conditions[V_cell] = new_model.param.ocv_init

        # Replace current with a FunctionControl variable
        external_circuit_variables = pybamm.external_circuit.FunctionControl(
            model.param, None, model.options, control="algebraic"
        ).get_fundamental_variables()

        symbol_replacement_map = {
            new_model.variables[name]: variable
            for name, variable in external_circuit_variables.items()
            if name in new_model.variables
        }
        replacer = SymbolReplacer(
            symbol_replacement_map, process_initial_conditions=False
        )
        replacer.process_model(new_model, inplace=True)

        # Add algebraic equation for current: I = I_applied (= 0 at equilibrium)
        I_var = new_model.variables["Current variable [A]"]
        I_ = new_model.variables["Current [A]"]
        I_applied = pybamm.FunctionParameter(
            "Current function [A]", {"Time [s]": pybamm.t}
        )
        new_model.algebraic[I_var] = I_ - I_applied
        new_model.initial_conditions[I_var] = 0

        return new_model

    def _build_matrix_problem(self, inputs_dict=None):
        """Build the mass matrix, Jacobian, and forcing vector.

        The mass matrix ``M`` and forcing vector ``b`` are cached after the
        first call because they do not depend on the operating point.  Only
        the Jacobian is re-evaluated when initial conditions change (e.g.
        different SOC).

        Parameters
        ----------
        inputs_dict : dict, optional
            Input parameters to pass to the model.

        Returns
        -------
        M : scipy.sparse.csc_matrix
            Mass matrix in CSC format.
        neg_J : scipy.sparse.csc_matrix
            Negated Jacobian in CSC format (pre-negated for the solve loop).
        b : np.ndarray
            Forcing vector with unit perturbation on the current variable.
        """
        model = self._built_model
        inputs_dict = inputs_dict or {}

        # Convert inputs to casadi format for Jacobian evaluation
        if model.convert_to_format == "casadi":
            from casadi import vertcat

            casadi_inputs = vertcat(*inputs_dict.values()) if inputs_dict else []
        else:
            casadi_inputs = inputs_dict

        # Only compile Jacobian/model functions on first call; the compiled
        # functions persist on the model and work with any inputs/y0 values.
        if getattr(model, "jac_rhs_algebraic_eval", None) is None:
            solver = pybamm.BaseSolver()
            solver.set_up(model, inputs=inputs_dict)

        y0 = model.concatenated_initial_conditions.evaluate(0, inputs=inputs_dict)
        J_sparse = model.jac_rhs_algebraic_eval(0, y0, casadi_inputs).sparse()
        neg_J = -J_sparse if isinstance(J_sparse, csc_matrix) else -csc_matrix(J_sparse)

        # M and b are independent of operating point and cached after first call,
        # but we have a defensive guard to invalidate it if the state vector size changes
        n = y0.shape[0]
        if not hasattr(self, "_cached_M") or self._cached_b.shape[0] != n:
            self._cached_M = csc_matrix(model.mass_matrix.entries)
            self._cached_b = np.zeros(n)
            self._cached_b[-1] = -1

        return self._cached_M, neg_J, self._cached_b

    @staticmethod
    def _calculate_impedance(frequency, M, neg_J, b):
        """Calculate impedance at a single frequency.

        Parameters
        ----------
        frequency : float
            Frequency in Hz.
        M : scipy.sparse.csc_matrix
            Mass matrix.
        neg_J : scipy.sparse.csc_matrix
            Negated Jacobian.
        b : np.ndarray
            Forcing vector.

        Returns
        -------
        z : complex
            Complex impedance value (unscaled).
        """
        A = 1.0j * 2 * np.pi * frequency * M + neg_J
        x = spsolve(A, b)
        # Voltage is penultimate, current is last (by construction in
        # _set_up_model_for_eis)
        return -x[-2] / x[-1]

    def solve(self, frequencies, inputs=None, initial_soc=None):
        """Compute impedance at the given frequencies.

        Solves the linear system ``(i*omega*M - J) x = b`` at each frequency
        using a direct sparse solver.

        Parameters
        ----------
        frequencies : array-like
            Frequencies in Hz at which to compute impedance.
        inputs : dict, optional
            Input parameters to pass to the model.
        initial_soc : float or str, optional
            Initial State of Charge. If given, the model is rebuilt with the
            new SOC before solving.

        Returns
        -------
        :class:`pybamm.EISSolution`
            Solution containing frequencies and complex impedance values.
        """
        model_name = self._model.name
        pybamm.logger.info(f"Start calculating impedance for {model_name}")
        timer = pybamm.Timer()

        if initial_soc is not None:
            self.build(initial_soc=initial_soc)
        elif self._built_model is None:
            self.build()

        M, neg_J, b = self._build_matrix_problem(inputs_dict=inputs)

        zs = [self._calculate_impedance(f, M, neg_J, b) for f in frequencies]
        impedance = np.array(zs) * self._z_scale
        self._solution = pybamm.EISSolution(frequencies, impedance)
        self._solution.set_up_time = self.set_up_time

        self.solve_time = timer.time()
        self._solution.solve_time = self.solve_time
        pybamm.logger.info(
            f"Finished calculating impedance for {model_name} "
            f"(solve time: {self.solve_time})"
        )

        return self._solution

    def nyquist_plot(self, **kwargs):
        """Generate a Nyquist plot from the most recent solution.

        Parameters
        ----------
        **kwargs
            Keyword arguments forwarded to :func:`pybamm.nyquist_plot`.

        Returns
        -------
        fig : matplotlib.figure.Figure or None
        ax : matplotlib.axes.Axes
        """
        if self._solution is None:
            raise ValueError(
                "EIS simulation has not been solved yet. Call solve() before plotting."
            )
        return self._solution.nyquist_plot(**kwargs)
