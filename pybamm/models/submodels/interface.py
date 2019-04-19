#
# Equations for the electrode-electrolyte interface
#
import pybamm
import autograd.numpy as np


class InterfacialCurrent(pybamm.SubModel):
    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

    def set_homogeneous_interfacial_current(self):
        """ Homogeneous reaction at the electrode-electrolyte interface """
        icell = pybamm.electrical_parameters.current_with_time

        j_n = pybamm.Broadcast(
            icell / pybamm.geometric_parameters.l_n, ["negative electrode"]
        )
        j_p = pybamm.Broadcast(
            -icell / pybamm.geometric_parameters.l_p, ["positive electrode"]
        )

        self.set_derived_interfacial_currents(j_n, j_p)

    def set_exchange_current_densities(self, variables, intercalation=True):
        """The exchange current-density as a function of concentration

        Parameters
        ----------
        variables : dict
            Dictionary of {string: :class:`pybamm.Symbol`}, which can be read to find
            already-calculated variables
        intercalation : bool
            Whether intercalation occurs in the model.
        """
        param = self.set_of_parameters
        c_e = variables["Electrolyte concentration"]
        c_e_n, c_e_s, c_e_p = c_e.orphans

        if intercalation:
            c_s_n_surf = variables["Negative particle surface concentration"]
            c_s_p_surf = variables["Positive particle surface concentration"]
            j0_n = (1 / param.C_r_n) * (
                c_e_n ** (1 / 2) * c_s_n_surf ** (1 / 2) * (1 - c_s_n_surf) ** (1 / 2)
            )
            j0_p = (param.gamma_p / param.C_r_p) * (
                c_e_p ** (1 / 2) * c_s_p_surf ** (1 / 2) * (1 - c_s_p_surf) ** (1 / 2)
            )
        else:
            j0_n = param.m_n * c_e_n
            c_w_p = (1 - c_e_p * param.V_e) / param.V_w
            j0_p = param.m_p * (c_e_p ** 2 * c_w_p)

        j0 = pybamm.Concatenation(*[j0_n, pybamm.Broadcast(0, ["separator"]), j0_p])

        # Update Variables and compute dimensional variables
        i_typ = param.i_typ
        self.variables.update(
            {
                "Negative electrode exchange-current density": j0_n,
                "Positive electrode exchange-current density": j0_p,
                "Exchange-current density": j0,
                "Negative electrode exchange-current density [A m-2]": i_typ * j0_n,
                "Positive electrode exchange-current density [A m-2]": i_typ * j0_p,
                "Exchange-current density [A m-2]": i_typ * j0,
            }
        )

    def set_interfacial_current_butler_volmer(self, variables):
        """
        Butler-Volmer reactions

        .. math::
            j = j_0(c) * \\sinh(\\phi - U(c)),

            \\text{where} \\phi = \\Phi_\\text{s} - \\Phi

        Parameters
        ----------
        variables : dict
            Dictionary of {string: :class:`pybamm.Symbol`}, which can be read to find
            already-calculated variables
        """
        param = self.set_of_parameters

        # Unpack variables
        eta_r_n = variables["Negative reaction overpotential"]
        eta_r_p = variables["Positive reaction overpotential"]
        j0_n = variables["Negative electrode exchange-current density"]
        j0_p = variables["Positive electrode exchange-current density"]

        # Compute Butler-Volmer
        j_n = j0_n * pybamm.Function(np.sinh, (param.ne_n / 2) * eta_r_n)
        j_p = j0_p * pybamm.Function(np.sinh, (param.ne_p / 2) * eta_r_p)

        self.set_derived_interfacial_currents(j_n, j_p)

    def get_inverse_butler_volmer(self, variables):
        """
        Inverts the Butler-Volmer relation to solve for the reaction overpotential.

        Parameters
        ----------
        variables : dict
            Dictionary of {string: :class:`pybamm.Symbol`}, which can be read to find
            already-calculated variables

        """
        param = self.set_of_parameters

        # Unpack variables
        j_n = variables["Negative electrode interfacial current density"]
        j_p = variables["Positive electrode interfacial current density"]
        j0_n = variables["Negative electrode exchange-current density"]
        j0_p = variables["Positive electrode exchange-current density"]

        # Invert Butler-Volmer
        eta_r_n = (2 / param.ne_n) * pybamm.Function(np.arcsinh, j_n / j0_n)
        eta_r_p = (2 / param.ne_p) * pybamm.Function(np.arcsinh, j_p / j0_p)

        return eta_r_n, eta_r_p

    def set_derived_interfacial_currents(self, j_n, j_p):
        i_typ = self.set_of_parameters.i_typ

        j = pybamm.Concatenation(*[j_n, pybamm.Broadcast(0, ["separator"]), j_p])

        self.variables.update(
            {
                "Negative electrode interfacial current density": j_n,
                "Positive electrode interfacial current density": j_p,
                "Interfacial current density": j,
                "Negative electrode interfacial current density [A m-2]": i_typ * j_n,
                "Positive electrode interfacial current density [A m-2]": i_typ * j_p,
                "Interfacial current density [A m-2]": i_typ * j,
            }
        )
