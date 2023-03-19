#
# Base kinetics class
#
import pybamm
from ..base_interface import BaseInterface


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
    """

    def __init__(self, param, domain, reaction, options, phase="primary"):
        super().__init__(param, domain, reaction, options=options, phase=phase)

    def get_fundamental_variables(self):
        domain = self.domain
        phase_name = self.phase_name

        if (
            self.options["total interfacial current density as a state"] == "true"
            and "main" in self.reaction
        ):
            j = pybamm.Variable(
                f"Total {domain} electrode {phase_name}"
                "interfacial current density variable [A.m-2]",
                domain=f"{domain} electrode",
                auxiliary_domains={"secondary": "current collector"},
            )

            variables = {
                f"Total {domain} electrode {phase_name}"
                "interfacial current density variable [A.m-2]": j,
                f"X-averaged total {domain} electrode {phase_name}"
                "interfacial current density variable [A.m-2]": pybamm.x_average(j),
            }
            return variables
        else:
            return {}

    def get_coupled_variables(self, variables):
        domain, Domain = self.domain_Domain
        reaction_name = self.reaction_name
        phase_name = self.phase_name

        if self.reaction == "lithium metal plating":  # li metal electrode (half-cell)
            delta_phi = variables[
                "Lithium metal interface surface potential difference [V]"
            ]
        else:
            delta_phi = variables[
                f"{Domain} electrode surface potential difference [V]"
            ]
            # If delta_phi was broadcast, take only the orphan.
            if isinstance(delta_phi, pybamm.Broadcast):
                delta_phi = delta_phi.orphans[0]
        # For "particle-size distribution" models, delta_phi must then be
        # broadcast to "particle size" domain
        domain_options = getattr(self.options, domain)
        if (
            self.reaction == "lithium-ion main"
            and domain_options["particle size"] == "distribution"
        ):
            delta_phi = pybamm.PrimaryBroadcast(delta_phi, [f"{domain} particle size"])

        # Get exchange-current density
        j0 = self._get_exchange_current_density(variables)
        # Get open-circuit potential variables and reaction overpotential
        if (
            domain_options["particle size"] == "distribution"
            and self.options.electrode_types[domain] == "porous"
        ):
            ocp = variables[
                f"{Domain} electrode {reaction_name}"
                "open-circuit potential distribution [V]"
            ]
        else:
            ocp = variables[
                f"{Domain} electrode {reaction_name}open-circuit potential [V]"
            ]
        # If ocp was broadcast, take only the orphan.
        if isinstance(ocp, pybamm.Broadcast):
            ocp = ocp.orphans[0]
        eta_r = delta_phi - ocp

        # Get average interfacial current density
        j_tot_av, a_j_tot_av = self._get_average_total_interfacial_current_density(
            variables
        )
        # Add SEI resistance in the negative electrode
        if self.domain == "negative":
            if self.options.electrode_types["negative"] == "planar":
                R_sei = self.phase_param.R_sei
                L_sei = variables[
                    f"Total {phase_name}SEI thickness [m]"
                ]  # on interface
                eta_sei = -j_tot_av * L_sei * R_sei
            elif self.options["SEI film resistance"] == "average":
                R_sei = self.phase_param.R_sei
                L_sei_av = variables[f"X-averaged total {phase_name}SEI thickness [m]"]
                eta_sei = -j_tot_av * L_sei_av * R_sei
            elif self.options["SEI film resistance"] == "distributed":
                R_sei = self.phase_param.R_sei
                L_sei = variables[f"Total {phase_name}SEI thickness [m]"]
                j_tot = variables[
                    f"Total negative electrode {phase_name}"
                    "interfacial current density variable [A.m-2]"
                ]

                # Override print_name
                j_tot.print_name = "j_tot"

                eta_sei = -j_tot * L_sei * R_sei
            else:
                eta_sei = pybamm.Scalar(0)
            eta_r += eta_sei

        # Get number of electrons in reaction
        ne = self._get_number_of_electrons_in_reaction()
        # Get kinetics. Note: T and u must have the same domain as j0 and eta_r
        if self.options.electrode_types[domain] == "planar":
            T = variables["X-averaged cell temperature [K]"]
            u = variables["Lithium metal interface utilisation"]
        elif j0.domain in ["current collector", ["current collector"]]:
            T = variables["X-averaged cell temperature [K]"]
            u = variables[f"X-averaged {domain} electrode interface utilisation"]
        elif j0.domain == [f"{domain} particle size"]:
            if j0.domains["secondary"] != [f"{domain} electrode"]:
                T = variables["X-averaged cell temperature [K]"]
                u = variables[f"X-averaged {domain} electrode interface utilisation"]
            else:
                T = variables[f"{Domain} electrode temperature [K]"]
                u = variables[f"{Domain} electrode interface utilisation"]

            # Broadcast T onto "particle size" domain
            T = pybamm.PrimaryBroadcast(T, [f"{domain} particle size"])
        else:
            T = variables[f"{Domain} electrode temperature [K]"]
            u = variables[f"{Domain} electrode interface utilisation"]

        # Update j, except in the "distributed SEI resistance" model, where j will be
        # found by solving an algebraic equation.
        # (In the "distributed SEI resistance" model, we have already defined j)
        j = self._get_kinetics(j0, ne, eta_r, T, u)

        if j.domain == [f"{domain} particle size"]:
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

        variables.update(self._get_standard_interfacial_current_variables(j))

        variables.update(
            self._get_standard_total_interfacial_current_variables(j_tot_av, a_j_tot_av)
        )
        variables.update(self._get_standard_exchange_current_variables(j0))
        variables.update(self._get_standard_overpotential_variables(eta_r))

        variables.update(
            self._get_standard_volumetric_current_density_variables(variables)
        )

        if self.domain == "negative" and self.reaction in [
            "lithium-ion main",
            "lithium metal plating",
            "lead-acid main",
        ]:
            variables.update(
                self._get_standard_sei_film_overpotential_variables(eta_sei)
            )

        return variables

    def set_algebraic(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        if (
            self.options["total interfacial current density as a state"] == "true"
            and "main" in self.reaction
        ):
            j_tot_var = variables[
                f"Total {domain} electrode {phase_name}"
                "interfacial current density variable [A.m-2]"
            ]

            # Override print_name
            j_tot_var.print_name = "j_tot"

            a_j_tot = variables[
                f"Sum of {domain} electrode {phase_name}"
                "volumetric interfacial current densities [A.m-3]"
            ]
            a = variables[
                f"{Domain} electrode {phase_name}surface area to volume ratio [m-1]"
            ]

            # Algebraic equation to set the variable j_tot_var
            # equal to the sum of currents j_tot = a_j_tot / a
            self.algebraic[j_tot_var] = j_tot_var - a_j_tot / a

    def set_initial_conditions(self, variables):
        domain = self.domain
        phase_name = self.phase_name

        if (
            self.options["total interfacial current density as a state"] == "true"
            and "main" in self.reaction
        ):
            j_tot_var = variables[
                f"Total {domain} electrode {phase_name}"
                "interfacial current density variable [A.m-2]"
            ]
            # Set initial guess to zero
            self.initial_conditions[j_tot_var] = pybamm.Scalar(0)
