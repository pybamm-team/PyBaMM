#
# Base kinetics class
#
import pybamm
from pybamm.models.submodels.interface.base_interface import BaseInterface


class BaseKinetics(BaseInterface):
    """
    Base submodel for kinetics

    Parameters
    ----------
    param :
        model parameters
    domain : str
        The domain to implement the model, either: 'Negative' or 'Positive'.
    reaction : str
        The name of the reaction being implemented
    options: dict
        A dictionary of options to be passed to the model.
        See :class:`pybamm.BaseBatteryModel`
    phase : str, optional
        Phase of the particle (default is "primary")
    x_average : bool, optional
        Whether the electrolyte and particle concentrations are uniform in x.
        Default is False.
    x_average_surface_potential : bool, optional
        Whether the surface potential difference is uniform in x. Default is False.
    """

    def __init__(
        self,
        param,
        domain,
        reaction,
        options,
        phase="primary",
        x_average=False,
        x_average_surface_potential=False,
    ):
        super().__init__(param, domain, reaction, options, phase, x_average)
        self.x_average_surface_potential = x_average_surface_potential

    def build(self):
        domain, Domain = self.domain_Domain
        reaction_name = self.reaction_name
        phase_name = self.phase_name
        domain_options = getattr(self.options, domain)
        planar = self.options.electrode_types[domain] == "planar"
        size_distribution = self.size_distribution and not planar
        variables = {}

        j_tot_var = None
        if (
            self.options["total interfacial current density as a state"] == "true"
            and "main" in self.reaction
        ):
            j_tot_var = pybamm.Variable(
                f"Total {domain} electrode {phase_name}"
                "interfacial current density variable [A.m-2]",
                domain=f"{domain} electrode",
                auxiliary_domains={"secondary": "current collector"},
            )
            j_tot_var.print_name = "j_tot"
            variables.update(
                {
                    f"Total {domain} electrode {phase_name}"
                    "interfacial current density variable [A.m-2]": j_tot_var,
                    f"X-averaged total {domain} electrode {phase_name}"
                    "interfacial current density variable [A.m-2]": pybamm.x_average(
                        j_tot_var
                    ),
                }
            )

        # Get surface potential difference and open-circuit potential. The
        # x-averaged ones are used when the reaction is evaluated x-averaged.
        if planar:  # li metal electrode (half-cell)
            delta_phi = self.coupled_variable(
                "Lithium metal interface surface potential difference [V]",
                domain="current collector",
            )
            ocp = self._coupled_electrode_variable(
                f"electrode {reaction_name}open-circuit potential [V]",
                x_average=self.options["thermal"] != "x-full",
            )
        elif size_distribution:
            delta_phi = self._coupled_electrode_variable(
                "electrode surface potential difference [V]",
                x_average=self.x_average_surface_potential,
            )
            delta_phi = pybamm.PrimaryBroadcast(
                delta_phi, [f"{domain} {phase_name}particle size"]
            )
            ocp = self._coupled_electrode_variable(
                f"electrode {reaction_name}open-circuit potential distribution [V]",
                x_average=self.x_average_surface_potential,
                size=True,
            )
        else:
            delta_phi = self._coupled_electrode_variable(
                "electrode surface potential difference [V]",
                x_average=self.x_average_surface_potential,
            )
            ocp = self._coupled_electrode_variable(
                f"electrode {reaction_name}open-circuit potential [V]"
            )

        # Get exchange-current density. For MSMR models we calculate the exchange
        # current density for each reaction, then sum these to give a total exchange
        # current density. Note: this is only used for the "exchange current density"
        # variables. For the interfacial current density variables, we sum the
        # interfacial currents from each reaction.
        msmr = domain_options["intercalation kinetics"] == "MSMR"
        if msmr:
            N = int(domain_options["number of MSMR reactions"])
            j0_by_reaction = [
                self._get_exchange_current_density_by_reaction(i) for i in range(N)
            ]
            for i, j0_i in enumerate(j0_by_reaction):
                variables.update(
                    self._get_standard_exchange_current_by_reaction_variables(j0_i, i)
                )
            j0 = sum(j0_by_reaction)
        else:
            j0 = self._get_exchange_current_density()

        # Get reaction overpotential
        eta_r = delta_phi - ocp

        # Get average interfacial current density
        j_tot_av, a_j_tot_av = self._get_average_total_interfacial_current_density()
        # Add SEI resistance
        if planar:
            R_sei = self.phase_param.R_sei
            L_sei = self._get_coupled_sei_thickness()  # on interface
            eta_sei = -j_tot_av * L_sei * R_sei
        elif self.options["SEI film resistance"] == "average":
            R_sei = self.phase_param.R_sei
            L_sei_av = self._get_coupled_sei_thickness(x_average=True)
            eta_sei = -j_tot_av * L_sei_av * R_sei
        elif self.options["SEI film resistance"] == "distributed":
            R_sei = self.phase_param.R_sei
            L_sei = self._get_coupled_sei_thickness()
            # "distributed" makes the total interfacial current density a state
            eta_sei = -j_tot_var * L_sei * R_sei
            if self.size_distribution and eta_r.domains["secondary"] != [
                f"{domain} electrode"
            ]:
                eta_sei = pybamm.x_average(eta_sei)
        else:
            eta_sei = pybamm.Scalar(0)
        eta_r += eta_sei

        j0 = self._broadcast_to_eta_r_domain(j0, eta_r)

        # Get number of electrons in reaction
        ne = self._get_number_of_electrons_in_reaction()

        # Get kinetics. Note: T and u must have the same domain as j0 and eta_r
        T, u = self._get_temperature_and_utilisation(j0)

        # For MSMR, the total current density is the sum over all reactions
        if msmr:
            j = 0
            for i, j0_i in enumerate(j0_by_reaction):
                j_i = self._get_kinetics_by_reaction(j0_i, ne, eta_r, T, u, i)
                variables.update(self._get_standard_icd_by_reaction_variables(j_i, i))
                j += j_i
        else:
            j = self._get_kinetics(j0, ne, eta_r, T, u)

        if j.domain == [f"{domain} particle size"] or j.domain == [
            f"{domain} {phase_name}particle size"
        ]:
            # If j depends on particle size, get size-dependent "distribution"
            # variables first
            variables.update(
                self._get_standard_size_distribution_interfacial_current_variables(j)
            )
            variables.update(
                self._get_standard_size_distribution_exchange_current_variables(j0)
            )
            variables.update(
                self._get_standard_size_distribution_overpotential_variables(eta_r)
            )

        current_variables = self._get_standard_interfacial_current_variables(j)
        variables.update(current_variables)
        variables.update(
            self._get_standard_total_interfacial_current_variables(j_tot_av, a_j_tot_av)
        )
        variables.update(self._get_standard_exchange_current_variables(j0))
        variables.update(self._get_standard_overpotential_variables(eta_r))
        if not planar:
            j = current_variables[
                f"{Domain} electrode {reaction_name}interfacial current density [A.m-2]"
            ]
            variables.update(self._get_standard_volumetric_current_density_variables(j))

        if self.reaction in [
            "lithium-ion main",
            "lithium metal plating",
            "lead-acid main",
        ]:
            variables.update(
                self._get_standard_sei_film_overpotential_variables(eta_sei)
            )

        self.variables.update(variables)

        if j_tot_var is not None:
            # Algebraic equation to set the variable j_tot_var equal to the sum of
            # currents j_tot = a_j_tot / a
            a_j_tot = self.coupled_variable(
                f"Sum of {domain} electrode {phase_name}"
                "volumetric interfacial current densities [A.m-3]",
                domains=j_tot_var.domains,
            )
            a = self._coupled_electrode_variable(
                f"electrode {phase_name}surface area to volume ratio [m-1]"
            )
            self.algebraic[j_tot_var] = j_tot_var - a_j_tot / a
            # Set initial guess to zero
            self.initial_conditions[j_tot_var] = pybamm.Scalar(0)

    def _get_temperature_and_utilisation(self, j0):
        """Temperature and interface utilisation on the domain of ``j0``."""
        domain = self.domain
        phase_name = self.phase_name
        x_averaged_T = self.coupled_variable(
            "X-averaged cell temperature [K]", domain="current collector"
        )
        if self.options.electrode_types[domain] == "planar":
            u = self.coupled_variable(
                "Lithium metal interface utilisation", domain="current collector"
            )
            return x_averaged_T, u
        if j0.domain in ["current collector", ["current collector"]]:
            u = self._coupled_electrode_variable(
                "electrode interface utilisation", x_average=True
            )
            return x_averaged_T, u
        if j0.domain == [f"{domain} {phase_name}particle size"]:
            if j0.domains["secondary"] != [f"{domain} electrode"]:
                T = x_averaged_T
                u = self._coupled_electrode_variable(
                    "electrode interface utilisation", x_average=True
                )
            else:
                T = self._coupled_electrode_variable("electrode temperature [K]")
                u = self._coupled_electrode_variable("electrode interface utilisation")
            # Broadcast T onto "particle size" domain
            T = pybamm.PrimaryBroadcast(T, [f"{domain} {phase_name}particle size"])
            return T, u
        T = self._coupled_electrode_variable("electrode temperature [K]")
        u = self._coupled_electrode_variable("electrode interface utilisation")
        return T, u

    def _broadcast_to_eta_r_domain(self, symbol, eta_r):
        """Broadcast an x-averaged symbol onto the electrode if eta_r is not."""
        if symbol.secondary_domain == [
            "current collector"
        ] and eta_r.secondary_domain == [f"{self.domain} electrode"]:
            symbol = pybamm.SecondaryBroadcast(symbol, [f"{self.domain} electrode"])
        return symbol
