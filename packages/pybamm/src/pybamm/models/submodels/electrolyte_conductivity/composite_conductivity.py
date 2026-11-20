#
# Composite electrolyte potential employing stefan-maxwell
#
import pybamm

from .base_electrolyte_conductivity import BaseElectrolyteConductivity


class Composite(BaseElectrolyteConductivity):
    """Base class for conservation of charge in the electrolyte employing the
    Stefan-Maxwell constitutive equations.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str, optional
        The domain in which the model holds
    options : dict, optional
        A dictionary of options to be passed to the model.
    higher_order_terms : str
        What kind of higher-order terms to use ('composite' or 'first-order')
    """

    def __init__(self, param, domain=None, options=None):
        super().__init__(param, domain, options=options)

    @property
    def _use_surface_form_sparsity_fixes(self) -> bool:
        return (
            self.options["surface form"] != "false"
            and self.options.electrode_types["negative"] == "porous"
            and self.domain in (None, "negative")
        )

    def _higher_order_macinnes_function(self, x):
        "Function to differentiate between composite and first-order models"
        tol = pybamm.settings.tolerances["macinnes__c_e"]
        x = pybamm.maximum(x, tol)
        return pybamm.log(x)

    def build(self):
        if self.options.electrode_types["negative"] == "planar":
            delta_phi_n_av = None
        else:
            delta_phi_n_av = self.coupled_variable(
                "X-averaged negative electrode surface potential difference [V]",
                domain="current collector",
            )
        self._build_composite(delta_phi_n_av)

    def _build_composite(self, delta_phi_n_av):
        """
        Set the composite electrolyte potential and current variables, the
        boundary conditions on the electrolyte potential and, with a surface form,
        the algebraic states that keep the Jacobian sparse.

        Parameters
        ----------
        delta_phi_n_av : :class:`pybamm.Symbol` or None
            The x-averaged negative electrode surface potential difference, or None
            for a planar negative electrode
        """
        param = self.param
        macinnes = self._higher_order_macinnes_function
        planar_negative = self.options.electrode_types["negative"] == "planar"

        def coupled_average(name):
            return self.coupled_variable(name, domain="current collector")

        c_e_av = coupled_average("X-averaged electrolyte concentration [mol.m-3]")
        i_boundary_cc = coupled_average("Current collector current density [A.m-2]")
        T_av = coupled_average("X-averaged cell temperature [K]")
        tor_s_av = coupled_average(
            "X-averaged separator electrolyte transport efficiency"
        )
        tor_p_av = coupled_average(
            "X-averaged positive electrolyte transport efficiency"
        )
        c_e_s = self._coupled_domain_variable(
            "Separator electrolyte concentration [mol.m-3]", "separator"
        )
        c_e_p = self._coupled_domain_variable(
            "Positive electrolyte concentration [mol.m-3]", "positive electrode"
        )
        if not planar_negative:
            c_e_n = self._coupled_domain_variable(
                "Negative electrolyte concentration [mol.m-3]", "negative electrode"
            )
            phi_s_n_av = coupled_average("X-averaged negative electrode potential [V]")
            tor_n_av = coupled_average(
                "X-averaged negative electrolyte transport efficiency"
            )

        T_av_s = pybamm.PrimaryBroadcast(T_av, "separator")
        T_av_p = pybamm.PrimaryBroadcast(T_av, "positive electrode")

        RT_F_av = param.R * T_av / param.F
        RT_F_av_s = param.R * T_av_s / param.F
        RT_F_av_p = param.R * T_av_p / param.F

        L_n = param.n.L
        L_s = param.s.L
        L_p = param.p.L
        L_x = param.L_x
        x_s = pybamm.standard_spatial_vars.x_s
        x_p = pybamm.standard_spatial_vars.x_p

        # bulk conductivities
        kappa_s_av = param.kappa_e(c_e_av, T_av) * tor_s_av
        kappa_p_av = param.kappa_e(c_e_av, T_av) * tor_p_av

        chi_av = param.chi(c_e_av, T_av)
        chi_av_s = pybamm.PrimaryBroadcast(chi_av, "separator")
        chi_av_p = pybamm.PrimaryBroadcast(chi_av, "positive electrode")

        # electrolyte current
        if planar_negative:
            i_e_n = None
        else:
            x_n = pybamm.standard_spatial_vars.x_n
            chi_av_n = pybamm.PrimaryBroadcast(chi_av, "negative electrode")
            T_av_n = pybamm.PrimaryBroadcast(T_av, "negative electrode")
            RT_F_av_n = param.R * T_av_n / param.F
            kappa_n_av = param.kappa_e(c_e_av, T_av) * tor_n_av
            i_e_n = i_boundary_cc * x_n / L_n
        i_e_s = pybamm.PrimaryBroadcast(i_boundary_cc, "separator")
        i_e_p = i_boundary_cc * (L_x - x_p) / L_p
        i_e = pybamm.concatenation(i_e_n, i_e_s, i_e_p)

        phi_e_dict = {}

        # electrolyte potential
        if planar_negative:
            phi_e_li = self._coupled_lithium_metal_electrolyte_potential()
            c_e_n = pybamm.boundary_value(c_e_s, "left")
            phi_e_const = (
                phi_e_li
                - chi_av * RT_F_av * macinnes(c_e_n / c_e_av)
                + (i_boundary_cc / kappa_s_av) * L_n
            )
        else:
            if self._use_surface_form_sparsity_fixes:
                macinnes_c_e_n = pybamm.Variable(
                    "X-averaged negative MacInnes function",
                    domain="current collector",
                )
            else:
                macinnes_c_e_n = pybamm.x_average(macinnes(c_e_n / c_e_av))

            phi_e_const = (
                -delta_phi_n_av
                + phi_s_n_av
                - (chi_av * RT_F_av * macinnes_c_e_n)
                - ((i_boundary_cc * L_n) * (1 / (3 * kappa_n_av) - 1 / kappa_s_av))
            )

            phi_e_n = (
                phi_e_const
                + (chi_av_n * RT_F_av_n * macinnes(c_e_n / c_e_av))
                - (i_boundary_cc / kappa_n_av) * (x_n**2 - L_n**2) / (2 * L_n)
                - i_boundary_cc * L_n / kappa_s_av
            )
            phi_e_dict["negative electrode"] = phi_e_n

        phi_e_s = (
            phi_e_const
            + (chi_av_s * RT_F_av_s * macinnes(c_e_s / c_e_av))
            - (i_boundary_cc / kappa_s_av) * x_s
        )

        phi_e_p = (
            phi_e_const
            + (chi_av_p * RT_F_av_p * macinnes(c_e_p / c_e_av))
            - (i_boundary_cc / kappa_p_av)
            * (x_p * (2 * L_x - x_p) + L_p**2 - L_x**2)
            / (2 * L_p)
            - i_boundary_cc * (L_x - L_p) / kappa_s_av
        )

        phi_e_dict["separator"] = phi_e_s
        phi_e_dict["positive electrode"] = phi_e_p

        # concentration overpotential
        macinnes_c_e_p = pybamm.x_average(macinnes(c_e_p / c_e_av))
        if planar_negative:
            macinnes_c_e_n = pybamm.x_average(macinnes(c_e_n / c_e_av))
            ohmic_n = 0
        else:
            ohmic_n = L_n / (3 * kappa_n_av)

        eta_c_av = chi_av * RT_F_av * (macinnes_c_e_p - macinnes_c_e_n)

        # average electrolyte ohmic losses
        delta_phi_e_av = -(i_boundary_cc) * (
            ohmic_n + L_s / kappa_s_av + L_p / (3 * kappa_p_av)
        )

        variables = self._get_standard_potential_variables(phi_e_dict)
        variables.update(self._get_standard_current_variables(i_e))
        variables.update(self._get_split_overpotential(eta_c_av, delta_phi_e_av))
        i_e.print_name = "i_e"

        if self._use_surface_form_sparsity_fixes:
            # Algebraic states for x-averages read by every equation keep the
            # Jacobian sparse
            phi_e_p_av = pybamm.Variable(
                "X-averaged positive electrolyte potential [V]",
                domain="current collector",
            )
            variables.update(
                {
                    "X-averaged negative MacInnes function": macinnes_c_e_n,
                    "X-averaged positive electrolyte potential [V]": phi_e_p_av,
                }
            )
            self.algebraic[macinnes_c_e_n] = macinnes_c_e_n - pybamm.x_average(
                macinnes(c_e_n / c_e_av)
            )
            self.algebraic[phi_e_p_av] = phi_e_p_av - pybamm.x_average(phi_e_p)
            self.initial_conditions[macinnes_c_e_n] = pybamm.Scalar(0)
            self.initial_conditions[phi_e_p_av] = pybamm.Scalar(0)

        self.variables.update(variables)
        self.boundary_conditions.update(
            self._get_electrolyte_potential_boundary_conditions(
                variables["Electrolyte potential [V]"]
            )
        )
