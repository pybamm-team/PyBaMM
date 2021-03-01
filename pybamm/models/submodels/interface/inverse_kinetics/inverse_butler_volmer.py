#
# Inverse Bulter-Volmer class
#
import pybamm
from ..base_interface import BaseInterface


class InverseButlerVolmer(BaseInterface):
    """
    A submodel that implements the inverted form of the Butler-Volmer relation to
    solve for the reaction overpotential.

    Parameters
    ----------
    param
        Model parameters
    domain : iter of str, optional
        The domain(s) in which to compute the interfacial current. Default is None,
        in which case j.domain is used.
    reaction : str
        The name of the reaction being implemented
    options: dict
        A dictionary of options to be passed to the model. In this case "SEI film
        resistance" is the important option. See :class:`pybamm.BaseBatteryModel`

    **Extends:** :class:`pybamm.interface.BaseInterface`

    """

    def __init__(self, param, domain, reaction, options=None):
        super().__init__(param, domain, reaction)
        if options is None:
            options = {"SEI film resistance": "none"}
        self.options = options

    def get_coupled_variables(self, variables):
        ocp, dUdT = self._get_open_circuit_potential(variables)

        j0 = self._get_exchange_current_density(variables)
        j_tot_av = self._get_average_total_interfacial_current_density(variables)
        # Broadcast to match j0's domain
        if j0.domain in [[], ["current collector"]]:
            j_tot = j_tot_av
        else:
            j_tot = pybamm.PrimaryBroadcast(
                j_tot_av, [self.domain.lower() + " electrode"]
            )

        ne = self._get_number_of_electrons_in_reaction()
        # Note: T must have the same domain as j0 and eta_r
        if j0.domain in ["current collector", ["current collector"]]:
            T = variables["X-averaged cell temperature"]
        else:
            T = variables[self.domain + " electrode temperature"]

        # eta_r is the overpotential from inverting Butler-Volmer, regardless of any
        # additional SEI resistance. What changes is how delta_phi is defined in terms
        # of eta_r
        # We use the total resistance to calculate eta_r, but this only introduces
        # negligible errors. For the exact answer, the surface form submodels should
        # be used instead
        eta_r = self._get_overpotential(j_tot, j0, ne, T)

        # With SEI resistance (distributed and averaged have the same effect here)
        if self.options["SEI film resistance"] != "none":
            if self.domain == "Negative":
                R_sei = self.param.R_sei_n
            elif self.domain == "Positive":
                R_sei = self.param.R_sei_p
            L_sei = variables[
                "Total " + self.domain.lower() + " electrode SEI thickness"
            ]
            eta_sei = -j_tot * L_sei * R_sei
        # Without SEI resistance
        else:
            eta_sei = pybamm.Scalar(0)

        delta_phi = eta_r + ocp - eta_sei

        variables.update(
            self._get_standard_total_interfacial_current_variables(j_tot_av)
        )
        variables.update(self._get_standard_exchange_current_variables(j0))
        variables.update(self._get_standard_overpotential_variables(eta_r))
        variables.update(
            self._get_standard_surface_potential_difference_variables(delta_phi)
        )
        variables.update(self._get_standard_sei_film_overpotential_variables(eta_sei))
        variables.update(self._get_standard_ocp_variables(ocp, dUdT))

        return variables

    def _get_overpotential(self, j, j0, ne, T):
        return (2 * (1 + self.param.Theta * T) / ne) * pybamm.arcsinh(j / (2 * j0))


class CurrentForInverseButlerVolmer(BaseInterface):
    """
    Submodel for the current associated with the inverse Butler-Volmer formulation. This
    has to be created as a separate submodel because of how the interfacial currents
    are calculated:

    1. Calculate eta_r from the total average current j_tot_av = I_app / L
    2. Calculate j_sei from eta_r
    3. Calculate j = j_tot_av - j_sei

    To be able to do step 1, then step 2, then step 3 requires two different submodels
    for step 1 and step 2

    This introduces a little bit of error because eta_r is calculated using j_tot_av
    instead of j. But since j_sei is very small, this error is very small. The "surface
    form" model solves a differential or algebraic equation for delta_phi, which gives
    the exact right answer. Comparing the two approaches shows almost no difference.

    Parameters
    ----------
    param
        Model parameters
    domain : iter of str, optional
        The domain(s) in which to compute the interfacial current. Default is None,
        in which case j.domain is used.
    reaction : str
        The name of the reaction being implemented

    **Extends:** :class:`pybamm.interface.BaseInterface`

    """

    def __init__(self, param, domain, reaction):
        super().__init__(param, domain, reaction)

    def get_coupled_variables(self, variables):
        j_tot = variables[
            "X-averaged "
            + self.domain.lower()
            + " electrode total interfacial current density"
        ]
        j_sei = variables[self.domain + " electrode SEI interfacial current density"]
        j_stripping = variables[
            self.domain + " electrode lithium plating interfacial current density"
        ]
        j = j_tot - j_sei - j_stripping

        variables.update(self._get_standard_interfacial_current_variables(j))

        if (
            "Negative electrode" + self.reaction_name + " interfacial current density"
            in variables
            and "Positive electrode"
            + self.reaction_name
            + " interfacial current density"
            in variables
            and self.Reaction_icd not in variables
        ):
            variables.update(
                self._get_standard_whole_cell_interfacial_current_variables(variables)
            )
            variables.update(
                self._get_standard_whole_cell_exchange_current_variables(variables)
            )

        return variables
