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

    def _get_standard_porosity_variables(self, eps_dict):
        eps = pybamm.concatenation(*eps_dict.values())

        variables = {"Porosity": eps}

        for domain, eps_k in eps_dict.items():
            Domain = domain.capitalize()
            eps_k_av = pybamm.x_average(eps_k)
            variables.update(
                {
                    f"{Domain} porosity": eps_k,
                    f"X-averaged {domain} porosity": eps_k_av,
                }
            )

        return variables

    def _get_standard_porosity_change_variables(self, depsdt_dict):
        deps_dt = pybamm.concatenation(*depsdt_dict.values())

        variables = {"Porosity change": deps_dt}
        for domain, depsdt_k in depsdt_dict.items():
            Domain = domain.capitalize()
            depsdt_k_av = pybamm.x_average(depsdt_k)
            variables.update(
                {
                    f"{Domain} porosity change [s-1]": depsdt_k,
                    f"X-averaged {domain} porosity change [s-1]": depsdt_k_av,
                }
            )

        return variables
