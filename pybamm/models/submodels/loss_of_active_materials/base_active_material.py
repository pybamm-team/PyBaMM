#
# Base class for activate material

#
import pybamm


class BaseModel(pybamm.BaseSubModel):
    """Base class for activate material

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def _get_standard_activate_material_variables(
        self, eps_am_n, eps_am_s, eps_am_p, set_leading_order=False
    ):

        eps_am_n_av = pybamm.x_average(eps_am_n)
        eps_am_s_av = pybamm.x_average(eps_am_s)
        eps_am_p_av = pybamm.x_average(eps_am_p)
        eps_am = pybamm.Concatenation(eps_am_n, eps_am_s, eps_am_p)
        am = "active material volume fraction"
        variables = {
            "Active material volume fraction": eps_am,
            "Negative electrode " + am: eps_am_n,
            "Separator " + am: eps_am_s,
            "Positive electrode " + am: eps_am_p,
            "X-averaged negative electrode " + am: eps_am_n_av,
            "X-averaged separator " + am: eps_am_s_av,
            "X-averaged positive electrode " + am: eps_am_p_av,
            "Negative electrode LAM ratio": eps_am_n / self.param.epsilon_s_n,
            "X-averaged negative electrode LAM ratio": eps_am_n_av
            / self.param.epsilon_s_n,
            "Positive electrode LAM ratio": eps_am_p / self.param.epsilon_s_p,
            "X-averaged positive electrode LAM ratio": eps_am_p_av
            / self.param.epsilon_s_p,
        }

        if set_leading_order is True:
            leading_order_variables = {
                "Leading-order " + name.lower(): var for name, var in variables.items()
            }
            variables.update(leading_order_variables)

        return variables

    def _get_standard_active_material_change_variables(
        self, deps_am_n_dt, deps_am_s_dt, deps_am_p_dt, set_leading_order=False
    ):

        deps_am_n_dt_av = pybamm.x_average(deps_am_n_dt)
        deps_am_s_dt_av = pybamm.x_average(deps_am_s_dt)
        deps_am_p_dt_av = pybamm.x_average(deps_am_p_dt)
        deps_am_dt = pybamm.Concatenation(deps_am_n_dt, deps_am_s_dt, deps_am_p_dt)
        am = "active material volume fraction"
        variables = {
            "Active material volume fraction change": deps_am_dt,
            f"Negative electrode {am} change": deps_am_n_dt,
            f"Separator {am} change": deps_am_s_dt,
            f"Positive electrode {am} change": deps_am_p_dt,
            f"X-averaged negative electrode {am} change": deps_am_n_dt_av,
            f"X-averaged separator {am} change": deps_am_s_dt_av,
            f"X-averaged positive electrode {am} change": deps_am_p_dt_av,
        }

        if set_leading_order is True:
            variables.update(
                {
                    "Leading-order x-averaged "
                    + f"negative electrode {am} change": deps_am_n_dt_av,
                    f"Leading-order x-averaged separator {am} change": deps_am_s_dt_av,
                    "Leading-order x-averaged "
                    + f"positive electrode {am} change": deps_am_p_dt_av,
                }
            )

        return variables

    def set_events(self, variables):
        am = "active material volume fraction"
        eps_am_n = variables[f"Negative electrode {am}"]
        eps_am_p = variables[f"Positive electrode {am}"]
        self.events.append(
            pybamm.Event(
                f"Zero negative electrode {am} cut-off",
                pybamm.min(eps_am_n),
                pybamm.EventType.TERMINATION,
            )
        )
        self.events.append(
            pybamm.Event(
                f"Max negative electrode {am} cut-off",
                pybamm.max(eps_am_n) - 1,
                pybamm.EventType.TERMINATION,
            )
        )

        self.events.append(
            pybamm.Event(
                f"Zero positive electrode {am} cut-off",
                pybamm.min(eps_am_p),
                pybamm.EventType.TERMINATION,
            )
        )

        self.events.append(
            pybamm.Event(
                f"Max positive electrode {am} cut-off",
                pybamm.max(eps_am_p) - 1,
                pybamm.EventType.TERMINATION,
            )
        )
