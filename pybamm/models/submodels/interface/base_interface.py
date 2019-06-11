#
# Base interface class
#

import pybamm


class BaseInterface(pybamm.BaseSubModel):
    """
    Base class for interfacial currents

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, domain):
        super().__init__(param)
        self._domain = domain

    def get_standard_derived_variables(self, derived_variables):
        derived_variables.update(self.get_average_variables(derived_variables))
        derived_variables.update(self.get_dimensional_variables(derived_variables))

        return derived_variables

    def get_dimensional_variables(self, variables):
        dimensional_variables = {}

        return dimensional_variables

    def get_derived_interfacial_currents(self, j_n, j_p, j0_n, j0_p):
        """
        Calculate dimensionless and dimensional variables for the interfacial current
        submodel

        Parameters
        ----------
        j_n : :class:`pybamm.Symbol`
            Interfacial current density in the negative electrode
        j_p : :class:`pybamm.Symbol`
            Interfacial current density in the positive electrode
        j0_n : :class:`pybamm.Symbol`
            Exchange-current density in the negative electrode
        j0_p : :class:`pybamm.Symbol`
            Exchange-current density in the positive electrode

        Returns
        -------
        dict
            Dictionary {string: :class:`pybamm.Symbol`} of relevant variables
        """
        i_typ = self.set_of_parameters.i_typ

        # Broadcast if necessary
        if j_n.domain in [[], ["current collector"]]:
            j_n = pybamm.Broadcast(j_n, ["negative electrode"])
        if j_p.domain in [[], ["current collector"]]:
            j_p = pybamm.Broadcast(j_p, ["positive electrode"])
        if j0_n.domain in [[], ["current collector"]]:
            j0_n = pybamm.Broadcast(j0_n, ["negative electrode"])
        if j0_p.domain in [[], ["current collector"]]:
            j0_p = pybamm.Broadcast(j0_p, ["positive electrode"])

        # Concatenations
        j = pybamm.Concatenation(*[j_n, pybamm.Broadcast(0, ["separator"]), j_p])
        j0 = pybamm.Concatenation(*[j0_n, pybamm.Broadcast(0, ["separator"]), j0_p])

        # Averages
        j_n_av = pybamm.average(j_n)
        j_p_av = pybamm.average(j_p)

        return {
            "Negative electrode interfacial current density": j_n,
            "Positive electrode interfacial current density": j_p,
            "Average negative electrode interfacial current density": j_n_av,
            "Average positive electrode interfacial current density": j_p_av,
            "Interfacial current density": j,
            "Negative electrode exchange-current density": j0_n,
            "Positive electrode exchange-current density": j0_p,
            "Exchange-current density": j0,
            "Negative electrode interfacial current density [A.m-2]": i_typ * j_n,
            "Positive electrode interfacial current density [A.m-2]": i_typ * j_p,
            "Average negative electrode interfacial current density [A.m-2]": i_typ
            * j_n_av,
            "Average positive electrode interfacial current density [A.m-2]": i_typ
            * j_p_av,
            "Interfacial current density [A.m-2]": i_typ * j,
            "Negative electrode exchange-current density [A.m-2]": i_typ * j0_n,
            "Positive electrode exchange-current density [A.m-2]": i_typ * j0_p,
            "Exchange-current density [A.m-2]": i_typ * j0,
        }

