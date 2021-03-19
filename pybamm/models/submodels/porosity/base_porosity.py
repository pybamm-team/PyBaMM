#
# Base class for porosity
#
import pybamm


class BaseModel(pybamm.BaseSubModel):
    """Base class for porosity

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, options):
        super().__init__(param, options=options)

    def _get_standard_porosity_variables(
        self, eps_n, eps_s, eps_p, set_leading_order=False
    ):

        eps_n_av = pybamm.x_average(eps_n)
        eps_s_av = pybamm.x_average(eps_s)
        eps_p_av = pybamm.x_average(eps_p)
        eps = pybamm.Concatenation(eps_n, eps_s, eps_p)

        variables = {
            "Porosity": eps,
            "Negative electrode porosity": eps_n,
            "Separator porosity": eps_s,
            "Positive electrode porosity": eps_p,
            "X-averaged negative electrode porosity": eps_n_av,
            "X-averaged separator porosity": eps_s_av,
            "X-averaged positive electrode porosity": eps_p_av,
        }

        if set_leading_order is True:
            leading_order_variables = {
                "Leading-order " + name.lower(): var for name, var in variables.items()
            }
            variables.update(leading_order_variables)

        return variables

    def _get_standard_porosity_change_variables(
        self, deps_n_dt, deps_s_dt, deps_p_dt, set_leading_order=False
    ):

        deps_n_dt_av = pybamm.x_average(deps_n_dt)
        deps_s_dt_av = pybamm.x_average(deps_s_dt)
        deps_p_dt_av = pybamm.x_average(deps_p_dt)
        deps_dt = pybamm.Concatenation(deps_n_dt, deps_s_dt, deps_p_dt)

        variables = {
            "Porosity change": deps_dt,
            "Negative electrode porosity change": deps_n_dt,
            "Separator porosity change": deps_s_dt,
            "Positive electrode porosity change": deps_p_dt,
            "X-averaged negative electrode porosity change": deps_n_dt_av,
            "X-averaged separator porosity change": deps_s_dt_av,
            "X-averaged positive electrode porosity change": deps_p_dt_av,
        }

        if set_leading_order is True:
            variables.update(
                {
                    "Leading-order x-averaged "
                    + "negative electrode porosity change": deps_n_dt_av,
                    "Leading-order x-averaged separator porosity change": deps_s_dt_av,
                    "Leading-order x-averaged "
                    + "positive electrode porosity change": deps_p_dt_av,
                }
            )

        return variables

    def set_events(self, variables):
        eps_n = variables["Negative electrode porosity"]
        eps_p = variables["Positive electrode porosity"]
        self.events.append(
            pybamm.Event(
                "Zero negative electrode porosity cut-off",
                pybamm.min(eps_n),
                pybamm.EventType.TERMINATION,
            )
        )
        self.events.append(
            pybamm.Event(
                "Max negative electrode porosity cut-off",
                pybamm.max(eps_n) - 1,
                pybamm.EventType.TERMINATION,
            )
        )

        self.events.append(
            pybamm.Event(
                "Zero positive electrode porosity cut-off",
                pybamm.min(eps_p),
                pybamm.EventType.TERMINATION,
            )
        )

        self.events.append(
            pybamm.Event(
                "Max positive electrode porosity cut-off",
                pybamm.max(eps_p) - 1,
                pybamm.EventType.TERMINATION,
            )
        )
