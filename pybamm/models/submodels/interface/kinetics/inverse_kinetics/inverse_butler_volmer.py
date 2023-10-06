#
# Inverse Bulter-Volmer class
#
import pybamm
from ...base_interface import BaseInterface


class InverseButlerVolmer(BaseInterface):
    """
    A submodel that implements the inverted form of the Butler-Volmer relation to
    solve for the reaction overpotential.

    Parameters
    ----------
    param
        Model parameters
    domain : iter of str, optional
        The domain(s) in which to compute the interfacial current.
    reaction : str
        The name of the reaction being implemented
    options: dict
        A dictionary of options to be passed to the model. In this case "SEI film
        resistance" is the important option. See :class:`pybamm.BaseBatteryModel`

    """

    def __init__(self, param, domain, reaction, options=None):
        super().__init__(param, domain, reaction, options=options)

    def get_coupled_variables(self, variables):
        domain, Domain = self.domain_Domain
        reaction_name = self.reaction_name

        ocp = variables[f"{Domain} electrode {reaction_name}open-circuit potential [V]"]

        j0 = self._get_exchange_current_density(variables)
        # Broadcast to match j0's domain

        j_tot_av, a_j_tot_av = self._get_average_total_interfacial_current_density(
            variables
        )
        if j0.domain in [[], ["current collector"]]:
            j_tot = j_tot_av
        else:
            j_tot = pybamm.PrimaryBroadcast(j_tot_av, [f"{domain} electrode"])
        variables.update(
            self._get_standard_total_interfacial_current_variables(j_tot, a_j_tot_av)
        )

        ne = self._get_number_of_electrons_in_reaction()
        # Note: T must have the same domain as j0 and eta_r
        if self.options.electrode_types[domain] == "planar":
            T = variables["X-averaged cell temperature [K]"]
            u = variables["Lithium metal interface utilisation"]
        elif j0.domain in ["current collector", ["current collector"]]:
            T = variables["X-averaged cell temperature [K]"]
            u = variables[f"X-averaged {domain} electrode interface utilisation"]
        else:
            T = variables[f"{Domain} electrode temperature [K]"]
            u = variables[f"{Domain} electrode interface utilisation"]

        # eta_r is the overpotential from inverting Butler-Volmer, regardless of any
        # additional SEI resistance. What changes is how delta_phi is defined in terms
        # of eta_r
        # We use the total resistance to calculate eta_r, but this only introduces
        # negligible errors. For the exact answer, the surface form submodels should
        # be used instead
        eta_r = self._get_overpotential(j_tot, j0, ne, T, u)

        # With SEI resistance (distributed and averaged have the same effect here)
        if self.options["SEI film resistance"] != "none":
            R_sei = self.phase_param.R_sei
            if self.options.electrode_types[domain] == "planar":
                L_sei = variables[f"{Domain} total SEI thickness [m]"]
            else:
                L_sei = variables[f"X-averaged {domain} total SEI thickness [m]"]
            eta_sei = -j_tot * L_sei * R_sei
        # Without SEI resistance
        else:
            eta_sei = pybamm.Scalar(0)
        variables.update(self._get_standard_sei_film_overpotential_variables(eta_sei))

        delta_phi = eta_r + ocp - eta_sei  # = phi_s - phi_e

        variables.update(self._get_standard_exchange_current_variables(j0))
        variables.update(self._get_standard_overpotential_variables(eta_r))
        variables.update(
            self._get_standard_average_surface_potential_difference_variables(
                pybamm.x_average(delta_phi)
            )
        )

        return variables

    def _get_overpotential(self, j, j0, ne, T, u):
        param = self.param
        return (2 * (param.R * T) / param.F / ne) * pybamm.arcsinh(j / (2 * j0 * u))


class CurrentForInverseButlerVolmer(BaseInterface):
    """
    Submodel for the current associated with the inverse Butler-Volmer formulation. This
    has to be created as a separate submodel because of how the interfacial currents
    are calculated:

    1. Calculate eta_r from the total average current j_tot_av = I_app / (a*L)
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
        The domain(s) in which to compute the interfacial current.
    reaction : str
        The name of the reaction being implemented
    options: dict, optional
        A dictionary of options to be passed to the model.
    """

    def __init__(self, param, domain, reaction, options=None):
        super().__init__(param, domain, reaction, options=options)

    def get_coupled_variables(self, variables):
        domain, Domain = self.domain_Domain

        j_tot = variables[
            f"X-averaged {domain} electrode total interfacial current density [A.m-2]"
        ]
        j_sei = variables[f"{Domain} electrode SEI interfacial current density [A.m-2]"]
        j_stripping = variables[
            f"{Domain} electrode lithium plating interfacial current density [A.m-2]"
        ]
        j = j_tot - j_sei - j_stripping

        variables.update(self._get_standard_interfacial_current_variables(j))
        variables.update(
            self._get_standard_volumetric_current_density_variables(variables)
        )

        return variables


class CurrentForInverseButlerVolmerLithiumMetal(BaseInterface):
    """
    Submodel for the current associated with the inverse Butler-Volmer formulation in
    a lithium metal cell. This is simply equal to the current collector current density.

    Parameters
    ----------
    param
        Model parameters
    domain : iter of str, optional
        The domain(s) in which to compute the interfacial current.
    reaction : str
        The name of the reaction being implemented
    options : dict, optional
        A dictionary of options to be passed to the model.
    """

    def __init__(self, param, domain, reaction, options=None):
        super().__init__(param, domain, reaction, options=options)

    def get_coupled_variables(self, variables):
        i_boundary_cc = variables["Current collector current density [A.m-2]"]
        j = i_boundary_cc

        variables.update(self._get_standard_interfacial_current_variables(j))

        return variables
