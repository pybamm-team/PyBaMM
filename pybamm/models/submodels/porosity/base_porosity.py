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
        eps_n_av = pybamm.x_average(eps_n)
        eps_s_av = pybamm.x_average(eps_s)
        eps_p_av = pybamm.x_average(eps_p)

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
            variables.update(
                {
                    "Leading-order negative electrode porosity": eps_n,
                    "Leading-order separator porosity": eps_s,
                    "Leading-order positive electrode porosity": eps_p,
                    "Leading-order x-averaged negative electrode porosity": eps_n_av,
                    "Leading-order x-averaged separator porosity": eps_s_av,
                    "Leading-order x-averaged positive electrode porosity": eps_p_av,
                }
            )

        return variables

    def _get_standard_porosity_change_variables(self, deps_dt, set_leading_order=False):

        deps_n_dt, deps_s_dt, deps_p_dt = deps_dt.orphans
        deps_n_dt_av = pybamm.x_average(deps_n_dt)
        deps_s_dt_av = pybamm.x_average(deps_s_dt)
        deps_p_dt_av = pybamm.x_average(deps_p_dt)
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
