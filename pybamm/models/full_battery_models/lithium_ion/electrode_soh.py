#
# A model to calculate electrode-specific SOH
#
import pybamm
import numpy as np


class ElectrodeSOH(pybamm.BaseModel):
    """Model to calculate electrode-specific SOH, from [1]_.

    .. math::
        n_{Li} &= \\frac{3600}{F}(y_{100}C_p + x_{100}C_n),
        \\
        V_{max} &= U_p(y_{100}) - U_n(x_{100}),
        \\
        V_{min} &= U_p(y_{0}) - U_n(x_{0})
        \\
        x_0 &= x_{100} - \\frac{C}{C_n},
        \\
        y_0 &= y_{100} + \\frac{C}{C_p}.

    References
    ----------
    .. [1] Mohtat, P., Lee, S., Siegel, J. B., & Stefanopoulou, A. G. (2019). Towards
    better estimability of electrode-specific state of health: Decoding the cell
    expansion. Journal of Power Sources, 427, 101-111.

    **Extends:** :class:`pybamm.BaseModel`
    """

    def __init__(self, name="Electrode-specific SOH model"):
        pybamm.citations.register("Mohtat2019")
        super().__init__(name)
        param = pybamm.LithiumIonParameters()

        Un = param.U_n_dimensional
        Up = param.U_p_dimensional
        T_ref = param.T_ref

        x_100 = pybamm.Variable("x_100", bounds=(0, 1))
        C = pybamm.Variable("C", bounds=(0, np.inf))

        V_max = pybamm.InputParameter("V_max")
        V_min = pybamm.InputParameter("V_min")
        C_n = pybamm.InputParameter("C_n")
        C_p = pybamm.InputParameter("C_p")
        n_Li = pybamm.InputParameter("n_Li")

        y_100 = (n_Li * param.F / 3600 - x_100 * C_n) / C_p
        x_0 = x_100 - C / C_n
        y_0 = y_100 + C / C_p

        self.algebraic = {
            x_100: Up(y_100, T_ref) - Un(x_100, T_ref) - V_max,
            C: Up(y_0, T_ref) - Un(x_0, T_ref) - V_min,
        }

        # initial guess must be chosen such that 0 < x_0, y_0, x_100, y_100 < 1
        # First guess for x_100
        x_100_init = 0.85
        # Make sure x_0 = x_100 - C/C_n > 0
        C_init = param.Q
        C_init = pybamm.minimum(C_n * x_100_init - 0.1, C_init)
        # Make sure y_100 > 0
        # x_100_init = pybamm.minimum(n_Li * param.F / 3600 / C_n - 0.01, x_100_init)
        self.initial_conditions = {x_100: x_100_init, C: C_init}

        self.variables = {
            "x_100": x_100,
            "y_100": y_100,
            "C": C,
            "x_0": x_0,
            "y_0": y_0,
            "Un(x_100)": Un(x_100, T_ref),
            "Un(x_0)": Un(x_0, T_ref),
            "Up(y_100)": Up(y_100, T_ref),
            "Up(y_0)": Up(y_0, T_ref),
            "Up(y_100) - Un(x_100)": Up(y_100, T_ref) - Un(x_100, T_ref),
            "Up(y_0) - Un(x_0)": Up(y_0, T_ref) - Un(x_0, T_ref),
            "n_Li_100": 3600 / param.F * (y_100 * C_p + x_100 * C_n),
            "n_Li_0": 3600 / param.F * (y_0 * C_p + x_0 * C_n),
            "n_Li": n_Li,
            "C_n": C_n,
            "C_p": C_p,
            "C_n * (x_100 - x_0)": C_n * (x_100 - x_0),
            "C_p * (y_100 - y_0)": C_p * (y_0 - y_100),
        }

    @property
    def default_solver(self):
        # Use AlgebraicSolver as CasadiAlgebraicSolver gives unnecessary warnings
        return pybamm.AlgebraicSolver()
