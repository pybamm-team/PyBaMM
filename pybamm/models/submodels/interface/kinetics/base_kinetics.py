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

    **Extends:** :class:`pybamm.interface.BaseInterface`
    """

    def __init__(self, param, domain, reaction, options):
        super().__init__(param, domain, reaction, options=options)

    def get_fundamental_variables(self):
        if (
            self.options["total interfacial current density as a state"] == "true"
            and "main" in self.reaction
        ):
            j = pybamm.Variable(
                "Total "
                + self.domain.lower()
                + " electrode interfacial current density variable",
                domain=self.domain.lower() + " electrode",
                auxiliary_domains={"secondary": "current collector"},
            )

            variables = {
                "Total "
                + self.domain.lower()
                + " electrode interfacial current density variable": j
            }
            return variables
        else:
            return {}

    def get_coupled_variables(self, variables):
        if self.reaction == "lithium metal plating":  # li metal electrode (half-cell)
            delta_phi = variables[
                "Lithium metal interface surface potential difference"
            ]
        else:
            delta_phi = variables[
                self.domain + " electrode surface potential difference"
            ]
            # If delta_phi was broadcast, take only the orphan.
            if isinstance(delta_phi, pybamm.Broadcast):
                delta_phi = delta_phi.orphans[0]
        # For "particle-size distribution" models, delta_phi must then be
        # broadcast to "particle size" domain
        if (
            self.reaction == "lithium-ion main"
            and self.options["particle size"] == "distribution"
        ):
            delta_phi = pybamm.PrimaryBroadcast(
                delta_phi, [self.domain.lower() + " particle size"]
            )

        # Get exchange-current density
        j0 = self._get_exchange_current_density(variables)
        # Get open-circuit potential variables and reaction overpotential
        ocp, dUdT = self._get_open_circuit_potential(variables)
        eta_r = delta_phi - ocp

        # Get average interfacial current density
        j_tot_av = self._get_average_total_interfacial_current_density(variables)
        # j = j_tot_av + (j - pybamm.x_average(j))  # enforce true average

        # Add SEI resistance in the negative electrode
        if self.domain == "Negative":
            if self.half_cell or self.options["SEI film resistance"] == "average":
                R_sei = self.param.R_sei
                L_sei = variables["Total SEI thickness"]
                eta_sei = -j_tot_av * L_sei * R_sei
            elif self.options["SEI film resistance"] == "distributed":
                R_sei = self.param.R_sei
                L_sei = variables["Total SEI thickness"]
                j_tot = variables[
                    "Total negative electrode interfacial current density variable"
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
        if self.half_cell and self.domain == "Negative":
            T = variables["X-averaged cell temperature"]
            u = variables["Lithium metal interface utilisation"]
        elif j0.domain in ["current collector", ["current collector"]]:
            T = variables["X-averaged cell temperature"]
            u = variables[
                "X-averaged " + self.domain.lower() + " electrode interface utilisation"
            ]
        elif j0.domain == [self.domain.lower() + " particle size"]:
            if j0.domains["secondary"] != [self.domain.lower() + " electrode"]:
                T = variables["X-averaged cell temperature"]
                u = variables[
                    "X-averaged "
                    + self.domain.lower()
                    + " electrode interface utilisation"
                ]
            else:
                T = variables[self.domain + " electrode temperature"]
                u = variables[self.domain + " electrode interface utilisation"]

            # Broadcast T onto "particle size" domain
            T = pybamm.PrimaryBroadcast(T, [self.domain.lower() + " particle size"])
        else:
            T = variables[self.domain + " electrode temperature"]
            u = variables[self.domain + " electrode interface utilisation"]

        # Update j, except in the "distributed SEI resistance" model, where j will be
        # found by solving an algebraic equation.
        # (In the "distributed SEI resistance" model, we have already defined j)
        j = self._get_kinetics(j0, ne, eta_r, T, u)

        if j.domain == [self.domain.lower() + " particle size"]:
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
            variables.update(
                self._get_standard_size_distribution_ocp_variables(ocp, dUdT)
            )

        variables.update(self._get_standard_interfacial_current_variables(j))

        variables.update(
            self._get_standard_total_interfacial_current_variables(j_tot_av)
        )
        variables.update(self._get_standard_exchange_current_variables(j0))
        variables.update(self._get_standard_overpotential_variables(eta_r))
        variables.update(self._get_standard_ocp_variables(ocp, dUdT))

        if self.domain == "Negative" and self.reaction in [
            "lithium-ion main",
            "lithium metal plating",
            "lead-acid main",
        ]:
            variables.update(
                self._get_standard_sei_film_overpotential_variables(eta_sei)
            )

        if (
            (
                self.half_cell
                or (
                    "Negative electrode"
                    + self.reaction_name
                    + " interfacial current density"
                )
                in variables
            )
            and (
                "Positive electrode"
                + self.reaction_name
                + " interfacial current density"
            )
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

    def set_algebraic(self, variables):
        if (
            self.options["total interfacial current density as a state"] == "true"
            and "main" in self.reaction
        ):
            j_tot_var = variables[
                "Total "
                + self.domain.lower()
                + " electrode interfacial current density variable"
            ]

            # Override print_name
            j_tot_var.print_name = "j_tot"

            j_tot = variables[
                "Sum of "
                + self.domain.lower()
                + " electrode interfacial current densities"
            ]
            # Algebraic equation to set the variable j_tot_var
            # equal to the sum of currents j_tot
            self.algebraic[j_tot_var] = j_tot_var - j_tot

    def set_initial_conditions(self, variables):
        if (
            self.options["total interfacial current density as a state"] == "true"
            and "main" in self.reaction
        ):
            param = self.param
            j_tot_var = variables[
                "Total "
                + self.domain.lower()
                + " electrode interfacial current density variable"
            ]
            current_at_0 = (
                pybamm.FunctionParameter("Current function [A]", {"Time [s]": 0})
                / param.I_typ
                * pybamm.sign(param.I_typ)
            )
            if self.domain == "Negative":
                j_tot_av_init = current_at_0 / param.l_n
            elif self.domain == "Positive":
                j_tot_av_init = -current_at_0 / param.l_p

            self.initial_conditions[j_tot_var] = j_tot_av_init

    def _get_dj_dc(self, variables):
        """
        Default to calculate derivative of interfacial current density with respect to
        concentration. Can be overwritten by specific kinetic functions.
        """
        (
            c_e,
            delta_phi,
            j0,
            ne,
            ocp,
            T,
            u,
        ) = self._get_interface_variables_for_first_order(variables)
        j = self._get_kinetics(j0, ne, delta_phi - ocp, T, u)
        return j.diff(c_e)

    def _get_dj_ddeltaphi(self, variables):
        """
        Default to calculate derivative of interfacial current density with respect to
        surface potential difference. Can be overwritten by specific kinetic functions.
        """
        _, delta_phi, j0, ne, ocp, T, u = self._get_interface_variables_for_first_order(
            variables
        )
        j = self._get_kinetics(j0, ne, delta_phi - ocp, T, u)
        return j.diff(delta_phi)

    def _get_interface_variables_for_first_order(self, variables):
        # This is a bit of a hack, but we need to wrap electrolyte concentration with
        # the NotConstant class
        # to differentiate it from the electrolyte concentration inside the
        # surface potential difference when taking j.diff(c_e) later on
        c_e_0 = pybamm.NotConstant(
            variables["Leading-order x-averaged electrolyte concentration"]
        )
        hacked_variables = {
            **variables,
            self.domain
            + " electrolyte concentration": pybamm.PrimaryBroadcast(
                c_e_0, self.domain_for_broadcast
            ),
        }
        delta_phi = variables[
            "Leading-order x-averaged "
            + self.domain.lower()
            + " electrode surface potential difference"
        ]
        j0 = self._get_exchange_current_density(hacked_variables)
        ne = self._get_number_of_electrons_in_reaction()
        ocp = self._get_open_circuit_potential(hacked_variables)[0]
        if j0.domain in ["current collector", ["current collector"]]:
            T = variables["X-averaged cell temperature"]
            u = variables[
                "X-averaged " + self.domain.lower() + " electrode interface utilisation"
            ]
        else:
            T = variables[self.domain + " electrode temperature"]
            u = variables[self.domain + " electrode interface utilisation"]
        return c_e_0, delta_phi, j0, ne, ocp, T, u

    def _get_j_diffusion_limited_first_order(self, variables):
        """
        First-order correction to the interfacial current density due to
        diffusion-limited effects. For a general model the correction term is zero,
        since the reaction is not diffusion-limited
        """
        return pybamm.Scalar(0)
