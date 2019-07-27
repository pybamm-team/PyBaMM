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

    def __init__(self, param):
        super().__init__(param)

    def _get_standard_porosity_variables(self, eps, set_leading_order=False):

        eps_n, eps_s, eps_p = eps.orphans

        variables = {
            "Porosity": eps,
            "Negative electrode porosity": eps_n,
            "Separator porosity": eps_s,
            "Positive electrode porosity": eps_p,
            "X-averaged negative electrode porosity": pybamm.x_average(eps_n),
            "X-averaged separator porosity": pybamm.x_average(eps_s),
            "X-averaged positive electrode porosity": pybamm.x_average(eps_p),
        }

        if set_leading_order is True:
            variables.update(
                {
                    "Leading-order negative electrode porosity": eps_n,
                    "Leading-order separator porosity": eps_s,
                    "Leading-order positive electrode porosity": eps_p,
                    "Leading-order x-averaged "
                    + "negative electrode porosity": pybamm.x_average(eps_n),
                    "Leading-order x-averaged separator porosity": pybamm.x_average(
                        eps_s
                    ),
                    "Leading-order x-averaged "
                    + "positive electrode porosity": pybamm.x_average(eps_p),
                }
            )

        return variables

    def _get_standard_porosity_change_variables(self, deps_dt, set_leading_order=False):

        deps_n_dt, deps_s_dt, deps_p_dt = deps_dt.orphans

        variables = {
            "Porosity change": deps_dt,
            "Negative electrode porosity change": deps_n_dt,
            "Separator porosity change": deps_s_dt,
            "Positive electrode porosity change": deps_p_dt,
            "X-averaged porosity change": pybamm.x_average(deps_dt),
            "X-averaged negative electrode porosity change": pybamm.x_average(
                deps_n_dt
            ),
            "X-averaged separator porosity change": pybamm.x_average(deps_s_dt),
            "X-averaged positive electrode porosity change": pybamm.x_average(
                deps_p_dt
            ),
        }

        if set_leading_order is True:
            variables.update(
                {
                    "Leading-order x-averaged "
                    + "negative electrode porosity change": pybamm.x_average(deps_n_dt),
                    "Leading-order x-averaged "
                    + "separator porosity change": pybamm.x_average(deps_s_dt),
                    "Leading-order x-averaged "
                    + "positive electrode porosity change": pybamm.x_average(deps_p_dt),
                }
            )

        return variables
