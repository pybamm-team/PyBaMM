#
# Base inverse kinetics class
#
import pybamm
from pybamm.models.submodels.interface.base_interface import BaseInterface


class BaseInverseKinetics(BaseInterface):
    """
    Base submodel for inverse kinetics.

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

    def build(self):
        domain = self.domain
        reaction_name = self.reaction_name
        planar = self.options.electrode_types[domain] == "planar"

        ocp = self._coupled_electrode_variable(
            f"electrode {reaction_name}open-circuit potential [V]"
        )
        j0 = self._get_exchange_current_density()

        # Broadcast to match j0's domain
        j_tot_av, a_j_tot_av = self._get_average_total_interfacial_current_density()
        if j0.domain in [[], ["current collector"]]:
            j_tot = j_tot_av
        else:
            j_tot = pybamm.PrimaryBroadcast(j_tot_av, [f"{domain} electrode"])
        variables = self._get_standard_total_interfacial_current_variables(
            j_tot, a_j_tot_av
        )

        ne = self._get_number_of_electrons_in_reaction()
        # Note: T must have the same domain as j0 and eta_r
        if planar:
            T = self.coupled_variable(
                "X-averaged cell temperature [K]", domain="current collector"
            )
            u = self.coupled_variable(
                "Lithium metal interface utilisation", domain="current collector"
            )
        elif j0.domain in ["current collector", ["current collector"]]:
            T = self.coupled_variable(
                "X-averaged cell temperature [K]", domain="current collector"
            )
            u = self._coupled_electrode_variable(
                "electrode interface utilisation", x_average=True
            )
        else:
            T = self._coupled_electrode_variable("electrode temperature [K]")
            u = self._coupled_electrode_variable("electrode interface utilisation")

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
            L_sei = self._get_coupled_sei_thickness(x_average=not planar)
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
        self.variables.update(variables)

    def _get_overpotential(self, j, j0, ne, T, u):
        # Use a specialized arcsinh2(a,b) = arcsinh(a/b) to avoid division by zero
        # errors when j0 is close to zero
        return (2 * (self.param.R * T) / self.param.F / ne) * pybamm.arcsinh2(
            j, 2 * j0 * u
        )


class CurrentForInverseKinetics(BaseInterface):
    """
    Submodel for the current associated with the inverse kinetics formulation. This
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

    def build(self):
        Domain = self.domain.capitalize()

        # The inverse kinetics submodel uses the x-averaged total current density
        j_tot, _ = self._get_average_total_interfacial_current_density()
        j_sei = self._coupled_electrode_variable(
            "electrode SEI interfacial current density [A.m-2]"
        )
        j_stripping = self._coupled_electrode_variable(
            "electrode lithium plating interfacial current density [A.m-2]"
        )
        j = j_tot - j_sei - j_stripping

        variables = self._get_standard_interfacial_current_variables(j)
        j = variables[f"{Domain} electrode interfacial current density [A.m-2]"]
        variables.update(self._get_standard_volumetric_current_density_variables(j))
        self.variables.update(variables)


class CurrentForInverseKineticsLithiumMetal(BaseInterface):
    """
    Submodel for the current associated with the inverse kinetics formulation in
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

    def build(self):
        i_boundary_cc = self.coupled_variable(
            "Current collector current density [A.m-2]", domain="current collector"
        )
        self.variables.update(
            self._get_standard_interfacial_current_variables(i_boundary_cc)
        )
