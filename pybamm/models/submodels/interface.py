#
# Equations for the electrode-electrolyte interface
#
import pybamm
import autograd.numpy as np


class InterfacialCurrent(pybamm.SubModel):
    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

    def get_homogeneous_interfacial_current(self, domain=None, broadcast=True):
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
        if domain == ["negative electrode"]:
            if broadcast:
                return pybamm.Broadcast(j_n, ["negative electrode"])
            else:
                return j_n
        elif domain == ["positive electrode"]:
            if broadcast:
                return pybamm.Broadcast(j_p, ["positive electrode"])
            else:
                return j_p
        elif domain == None:
            j_n = self.get_homogeneous_interfacial_current(
                ["negative electrode"], broadcast
            )
            j_p = self.get_homogeneous_interfacial_current(
                ["positive electrode"], broadcast
            )
            return j_n, j_p

    def get_exchange_current(self, c_e, c_s_k_surf=None, domain=None):
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
        domain = domain or c_e.domain

        if domain == ["negative electrode"]:
            return param.m_n * c_e
        elif domain == ["positive electrode"]:
            c_w = (1 - c_e * param.V_e) / param.V_w
            return param.m_p * (c_e ** 2 * c_w)

    def get_exchange_current_intercalation(self, c_e, c_s_k_surf, domain=None):
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
        domain = domain or c_e.domain

        if domain == ["negative electrode"]:
            return (1 / param.C_r_n) * (
                c_e_n ** (1 / 2) * c_s_n_surf ** (1 / 2) * (1 - c_s_n_surf) ** (1 / 2)
            )
        elif domain == ["positive electrode"]:
            return (param.gamma_p / param.C_r_p) * (
                c_e_p ** (1 / 2) * c_s_p_surf ** (1 / 2) * (1 - c_s_p_surf) ** (1 / 2)
            )

    def get_butler_volmer(self, j0, eta_r, domain=None):
        """
        Butler-Volmer reactions

        .. math::
            j = j_0(c) * \\sinh(\\eta_r(c))

        """
        param = self.set_of_parameters

        domain = domain or j_n.domain
        if domain == ["negative electrode"]:
            return j0 * pybamm.Function(np.sinh, (param.ne_n / 2) * eta_r)
        elif domain == ["positive electrode"]:
            return j0 * pybamm.Function(np.sinh, (param.ne_p / 2) * eta_r)

    def get_inverse_butler_volmer(self, j, j0, domain):
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

        domain = domain or j_n.domain
        if domain == ["negative electrode"]:
            return (2 / param.ne_n) * pybamm.Function(np.arcsinh, j / j0)
        elif domain == ["positive electrode"]:
            return (2 / param.ne_p) * pybamm.Function(np.arcsinh, j / j0)

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
