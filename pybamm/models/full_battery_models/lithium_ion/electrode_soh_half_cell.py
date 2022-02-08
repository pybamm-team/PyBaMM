#
# A model to calculate electrode-specific SOH, adapted to a half-cell
#
import pybamm
import numpy as np


class ElectrodeSOHHalfCell(pybamm.BaseModel):
    """Model to calculate electrode-specific SOH for a half-cell, adapted from [2]_.
    This model is mainly for internal use, to calculate summary variables in a
    simulation.

    .. math::
        V_{max} = U_w(x_{100}),
    .. math::
        V_{min} = U_w(x_{0}),
    .. math::
        x_0 = x_{100} - \\frac{C}{C_w}.

    Subscript w indicates working electrode and subscript c indicates counter electrode.

    References
    ----------
    .. [2] Mohtat, P., Lee, S., Siegel, J. B., & Stefanopoulou, A. G. (2019). Towards
           better estimability of electrode-specific state of health: Decoding the cell
           expansion. Journal of Power Sources, 427, 101-111.


    **Extends:** :class:`pybamm.BaseModel`
    """

    def __init__(self, working_electrode, name="Electrode-specific SOH model"):
        self.working_electrode = working_electrode
        pybamm.citations.register("Mohtat2019")
        super().__init__(name)
        param = pybamm.LithiumIonParameters({"working electrode": working_electrode})

        x_100 = pybamm.Variable("x_100", bounds=(0, 1))
        C = pybamm.Variable("C", bounds=(0, np.inf))
        Cw = pybamm.InputParameter("C_w")
        T_ref = param.T_ref
        if working_electrode == "negative":  # pragma: no cover
            raise NotImplementedError
        elif working_electrode == "positive":
            Uw = param.U_p_dimensional
            x_0 = x_100 + C / Cw

        V_max = pybamm.InputParameter("V_max")
        V_min = pybamm.InputParameter("V_min")

        self.algebraic = {
            x_100: Uw(x_100, T_ref) - V_max,
            C: Uw(x_0, T_ref) - V_min,
        }

        # initial guess must be chosen such that 0 < x_0, x_100 < 1
        # First guess for x_100
        x_100_init = 0.85
        # Make sure x_0 = x_100 - C/C_w > 0
        C_init = param.Q
        C_init = pybamm.minimum(Cw * x_100_init - 0.1, C_init)
        self.initial_conditions = {x_100: x_100_init, C: C_init}

        self.variables = {
            "x_100": x_100,
            "C": C,
            "x_0": x_0,
            "Uw(x_100)": Uw(x_100, T_ref),
            "Uw(x_0)": Uw(x_0, T_ref),
            "C_w": Cw,
            "C_w * (x_100 - x_0)": Cw * (x_100 - x_0),
        }

    @property
    def default_solver(self):
        # Use AlgebraicSolver as CasadiAlgebraicSolver gives unnecessary warnings
        return pybamm.AlgebraicSolver()

    def new_empty_copy(self):
        new_model = ElectrodeSOHHalfCell(self.working_electrode, name=self.name)
        return new_model
