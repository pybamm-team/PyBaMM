from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

import pybamm

from .base_simulation import BaseSimulation


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
    skip_surface_form_check : bool, optional
        If True, skip the 'surface form' model option validation. Defaults to False.
    """

    def __init__(
        self,
        model,
        parameter_values=None,
        geometry=None,
        submesh_types=None,
        var_pts=None,
        spatial_methods=None,
        skip_surface_form_check=False,
    ):
        timer = pybamm.Timer()

        # Validate required variables and surface form before any processing
        self._validate_model_for_eis(model, skip_surface_form_check)

        model_name = model.name

        parameter_values = parameter_values or model.default_parameter_values
        parameter_values = parameter_values.copy()

        pybamm.logger.info(f"Setting up {model_name} for EIS")
        # set current to zero as an algebraic state
        step = pybamm.step.CustomStepImplicit(lambda v: v["Current [A]"] - 0)
        model, parameter_values = step.set_up(model.new_copy(), parameter_values)
        model.initial_conditions[model.variables["Current [A]"]] = pybamm.Scalar(0)

        super().__init__(
            model,
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
    def _validate_model_for_eis(model, skip_surface_form_check=False):
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

        if skip_surface_form_check:
            return
        surface_form = model.options.get("surface form", "false")
        if surface_form not in ("differential", "algebraic"):
            raise ValueError(
                f"EIS simulation requires 'surface form' model option to be "
                f"'differential' or 'algebraic', got '{surface_form}'. "
                f"Use e.g. pybamm.lithium_ion.SPM("
                f'options={{"surface form": "differential"}})'
            )

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
        outputs = pybamm.numpy_concatenation(
            model.get_processed_variable("Voltage [V]"),
            model.get_processed_variable("Current [A]"),
        )
        from casadi import MX, Function, jacobian

        y = MX.sym("y", y0.size)
        output_expression = outputs.to_casadi(t=0, y=y, inputs=inputs_dict)
        self._output_jacobian = Function(
            "eis_outputs", [y], [jacobian(output_expression, y)]
        )(y0).sparse()
        J_sparse = model.jac_rhs_algebraic_eval(0, y0, casadi_inputs).sparse()
        neg_J = -csc_matrix(J_sparse)

        # State ordering and matrices are fixed until the model is rebuilt.
        if getattr(self, "_matrix_model", None) is not model:
            indices = {
                var.name: slices[0].start for var, slices in model.y_slices.items()
            }
            self._current_index = indices["Current variable [A]"]
            self._cached_M = csc_matrix(model.mass_matrix.entries)
            self._cached_b = np.zeros(y0.shape[0])
            self._cached_b[self._current_index] = -1
            self._matrix_model = model

        return self._cached_M, neg_J, self._cached_b

    def _calculate_impedance(self, frequency, M, neg_J, b):
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
            Complex impedance in Ohms.
        """
        A = (1.0j * 2 * np.pi * frequency) * M + neg_J
        x = spsolve(A, b)
        voltage, current = self._output_jacobian @ x
        return -voltage / current

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

        self.build(initial_soc=initial_soc, inputs=inputs)

        M, neg_J, b = self._build_matrix_problem(inputs_dict=inputs)

        zs = [self._calculate_impedance(f, M, neg_J, b) for f in frequencies]
        impedance = np.array(zs)
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
