#
# Equations for the electrode-electrolyte interface
#
import pybamm
import autograd.numpy as np


class InterfacialCurrent(pybamm.SubModel):
    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

    def get_homogeneous_interfacial_current(self, broadcast=True):
        """
        Homogeneous reaction at the electrode-electrolyte interface

        Parameters
        ----------
        broadcast : bool
            Whether to broadcast the result

        Returns
        -------
        dict
            Dictionary {string: :class:`pybamm.Symbol`} of relevant variables
        """
        icell = pybamm.electrical_parameters.current_with_time

        j_n = icell / pybamm.geometric_parameters.l_n
        j_p = -icell / pybamm.geometric_parameters.l_p

        if broadcast:
            return self.get_derived_interfacial_currents(
                pybamm.Broadcast(jn, ["negative electrode"]),
                pybamm.Broadcast(j_p, ["positive electrode"]),
            )
        else:
            return {
                "Negative electrode interfacial current density": j_n,
                "Positive electrode interfacial current density": j_p,
            }

    def get_exchange_current_densities(self, variables, intercalation=True):
        """The exchange current-density as a function of concentration

        Parameters
        ----------
        variables : dict
            Dictionary of {string: :class:`pybamm.Symbol`}, which can be read to find
            already-calculated variables
        intercalation : bool
            Whether intercalation occurs in the model.

        Returns
        -------
        dict
            Dictionary {string: :class:`pybamm.Symbol`} of relevant variables
        """
        param = self.set_of_parameters
        c_e = variables["Electrolyte concentration"]
        # Allow for leading-order case
        if isinstance(c_e, pybamm.Variable):
            c_e_n = c_e
            c_e_p = c_e
        else:
            c_e_n, c_e_s, c_e_p = c_e.orphans

        if intercalation:
            c_s_n_surf = pybamm.surf(variables["Negative particle concentration"])
            c_s_p_surf = pybamm.surf(variables["Positive particle concentration"])
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

        # Compute dimensional variables
        i_typ = param.i_typ
        variables = {
            "Negative electrode exchange-current density": j0_n,
            "Positive electrode exchange-current density": j0_p,
            "Negative electrode exchange-current density [A m-2]": i_typ * j0_n,
            "Positive electrode exchange-current density [A m-2]": i_typ * j0_p,
        }
        if j0_n.domain == []:
            return variables
        else:
            j0 = pybamm.Concatenation(*[j0_n, pybamm.Broadcast(0, ["separator"]), j0_p])
            variables.update(
                {
                    "Exchange-current density": j0,
                    "Exchange-current density [A m-2]": i_typ * j0,
                }
            )
            return variables

    def get_interfacial_current_butler_volmer(self, variables):
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

        Returns
        -------
        dict
            Dictionary {string: :class:`pybamm.Symbol`} of relevant variables
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

        return self.get_derived_interfacial_currents(j_n, j_p)

    def get_inverse_butler_volmer(self, variables):
        """
        Inverts the Butler-Volmer relation to solve for the reaction overpotential.

        Parameters
        ----------
        variables : dict
            Dictionary of {string: :class:`pybamm.Symbol`}, which can be read to find
            already-calculated variables

        Returns
        -------
        dict
            Dictionary {string: :class:`pybamm.Symbol`} of relevant variables

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

    def get_derived_interfacial_currents(self, j_n, j_p):
        """
        Calculate dimensionless and dimensional variables for the interfacial current
        submodel

        Returns
        -------
        dict
            Dictionary {string: :class:`pybamm.Symbol`} of relevant variables
        """
        i_typ = self.set_of_parameters.i_typ

        j = pybamm.Concatenation(*[j_n, pybamm.Broadcast(0, ["separator"]), j_p])

        return {
            "Negative electrode interfacial current density": j_n,
            "Positive electrode interfacial current density": j_p,
            "Interfacial current density": j,
            "Negative electrode interfacial current density [A m-2]": i_typ * j_n,
            "Positive electrode interfacial current density [A m-2]": i_typ * j_p,
            "Interfacial current density [A m-2]": i_typ * j,
        }
