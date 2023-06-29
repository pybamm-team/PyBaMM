#
# A model to calculate electrode-specific SOH, adapted to a half-cell
#
import pybamm


class ElectrodeSOHHalfCell(pybamm.BaseModel):
    """Model to calculate electrode-specific SOH for a half-cell, adapted from
    :footcite:t:`Mohtat2019`.
    This model is mainly for internal use, to calculate summary variables in a
    simulation.

    .. math::
        V_{max} = U_w(x_{100}),
    .. math::
        V_{min} = U_w(x_{0}),
    .. math::
        x_0 = x_{100} - \\frac{C}{C_w}.

    Subscript w indicates working electrode and subscript c indicates counter electrode.

    """

    def __init__(self, working_electrode, name="Electrode-specific SOH model"):
        self.working_electrode = working_electrode
        pybamm.citations.register("Mohtat2019")
        super().__init__(name)
        param = pybamm.LithiumIonParameters({"working electrode": working_electrode})

        x_100 = pybamm.Variable("x_100", bounds=(0, 1))
        x_0 = pybamm.Variable("x_0", bounds=(0, 1))
        Q_w = pybamm.InputParameter("Q_w")
        T_ref = param.T_ref
        if working_electrode == "negative":  # pragma: no cover
            raise NotImplementedError
        elif working_electrode == "positive":
            U_w = param.p.prim.U
            Q = Q_w * (x_100 - x_0)

        V_max = param.opc_soc_100_dimensional
        V_min = param.opc_soc_0_dimensional

        self.algebraic = {
            x_100: U_w(x_100, T_ref) - V_max,
            x_0: U_w(x_0, T_ref) - V_min,
        }
        self.initial_conditions = {x_100: 0.8, x_0: 0.2}

        self.variables = {
            "x_100": x_100,
            "x_0": x_0,
            "Q": Q,
            "Uw(x_100)": U_w(x_100, T_ref),
            "Uw(x_0)": U_w(x_0, T_ref),
            "Q_w": Q_w,
            "Q_w * (x_100 - x_0)": Q_w * (x_100 - x_0),
        }

    @property
    def default_solver(self):
        # Use AlgebraicSolver as CasadiAlgebraicSolver gives unnecessary warnings
        return pybamm.AlgebraicSolver()
