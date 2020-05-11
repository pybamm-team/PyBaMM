#
# Base class for thermal effects
#
import pybamm


class BaseThermal(pybamm.BaseSubModel):
    """Base class for thermal effects

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    cc_dimension: int, optional
        The dimension of the current collectors. Can be 0 (default), 1 or 2.

    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, cc_dimension=0):
        self.cc_dimension = cc_dimension
        super().__init__(param)

    def _get_standard_fundamental_variables(
        self, T_cn, T_n, T_s, T_p, T_cp, T_x_av, T_vol_av
    ):
        """
        Note: here we explicitly pass in the averages for the temperature as computing
        the average temperature in `BaseThermal` using `self._x_average` requires a
        messy hack to avoid raising a `ModelError` (as the key in the equation
        dict gets modified).

        For more information about this method in general,
        see :meth:`pybamm.base_submodel._get_standard_fundamental_variables`
        """
        param = self.param

        # The variable T is the concatenation of the temperature in the negative
        # electrode, separator and positive electrode, for use the electrochemical
        # models
        T = pybamm.Concatenation(T_n, T_s, T_p)

        # Compute averaged temperatures by domain
        T_n_av = pybamm.x_average(T_n)
        T_s_av = pybamm.x_average(T_s)
        T_p_av = pybamm.x_average(T_p)

        # Get the ambient temperature, which can be specified as a function of time
        T_amb_dim = param.T_amb_dim(pybamm.t * param.timescale)
        T_amb = param.T_amb(pybamm.t * param.timescale)

        variables = {
            "Negative current collector temperature": T_cn,
            "Negative current collector temperature [K]": param.Delta_T * T_cn
            + param.T_ref,
            "X-averaged negative electrode temperature": T_n_av,
            "X-averaged negative electrode temperature [K]": param.Delta_T * T_n_av
            + param.T_ref,
            "Negative electrode temperature": T_n,
            "Negative electrode temperature [K]": param.Delta_T * T_n + param.T_ref,
            "X-averaged separator temperature": T_s_av,
            "X-averaged separator temperature [K]": param.Delta_T * T_s_av
            + param.T_ref,
            "Separator temperature": T_s,
            "Separator temperature [K]": param.Delta_T * T_s + param.T_ref,
            "X-averaged positive electrode temperature": T_p_av,
            "X-averaged positive electrode temperature [K]": param.Delta_T * T_p_av
            + param.T_ref,
            "Positive electrode temperature": T_p,
            "Positive electrode temperature [K]": param.Delta_T * T_p + param.T_ref,
            "Positive current collector temperature": T_cp,
            "Positive current collector temperature [K]": param.Delta_T * T_cp
            + param.T_ref,
            "Cell temperature": T,
            "Cell temperature [K]": param.Delta_T * T + param.T_ref,
            "X-averaged cell temperature": T_x_av,
            "X-averaged cell temperature [K]": param.Delta_T * T_x_av + param.T_ref,
            "Volume-averaged cell temperature": T_vol_av,
            "Volume-averaged cell temperature [K]": param.Delta_T * T_vol_av
            + param.T_ref,
            "Ambient temperature [K]": T_amb_dim,
            "Ambient temperature": T_amb,
        }

        return variables

    def _get_standard_coupled_variables(self, variables):

        param = self.param

        T = variables["Cell temperature"]
        T_n, _, T_p = T.orphans

        j_n = variables["Negative electrode interfacial current density"]
        j_p = variables["Positive electrode interfacial current density"]

        eta_r_n = variables["Negative electrode reaction overpotential"]
        eta_r_p = variables["Positive electrode reaction overpotential"]

        dUdT_n = variables["Negative electrode entropic change"]
        dUdT_p = variables["Positive electrode entropic change"]

        i_e = variables["Electrolyte current density"]
        phi_e = variables["Electrolyte potential"]

        i_s_n = variables["Negative electrode current density"]
        i_s_p = variables["Positive electrode current density"]
        phi_s_n = variables["Negative electrode potential"]
        phi_s_p = variables["Positive electrode potential"]

        # Ohmic heating in solid
        Q_ohm_s_cn, Q_ohm_s_cp = self._current_collector_heating(variables)
        Q_ohm_s_n = -pybamm.inner(i_s_n, pybamm.grad(phi_s_n))
        Q_ohm_s_s = pybamm.FullBroadcast(0, ["separator"], "current collector")
        Q_ohm_s_p = -pybamm.inner(i_s_p, pybamm.grad(phi_s_p))
        Q_ohm_s = pybamm.Concatenation(Q_ohm_s_n, Q_ohm_s_s, Q_ohm_s_p)

        # Ohmic heating in electrolyte
        # TODO: change full stefan-maxwell conductivity so that i_e is always
        # a Concatenation
        if isinstance(i_e, pybamm.Concatenation):
            # compute by domain if possible
            i_e_n, i_e_s, i_e_p = i_e.orphans
            phi_e_n, phi_e_s, phi_e_p = phi_e.orphans
            Q_ohm_e_n = -pybamm.inner(i_e_n, pybamm.grad(phi_e_n))
            Q_ohm_e_s = -pybamm.inner(i_e_s, pybamm.grad(phi_e_s))
            Q_ohm_e_p = -pybamm.inner(i_e_p, pybamm.grad(phi_e_p))
            Q_ohm_e = pybamm.Concatenation(Q_ohm_e_n, Q_ohm_e_s, Q_ohm_e_p)
        else:
            Q_ohm_e = -pybamm.inner(i_e, pybamm.grad(phi_e))

        # Total Ohmic heating
        Q_ohm = Q_ohm_s + Q_ohm_e

        # Irreversible electrochemical heating
        Q_rxn_n = j_n * eta_r_n
        Q_rxn_p = j_p * eta_r_p
        Q_rxn = pybamm.Concatenation(
            *[
                Q_rxn_n,
                pybamm.FullBroadcast(0, ["separator"], "current collector"),
                Q_rxn_p,
            ]
        )

        # Reversible electrochemical heating
        Q_rev_n = j_n * (param.Theta ** (-1) + T_n) * dUdT_n
        Q_rev_p = j_p * (param.Theta ** (-1) + T_p) * dUdT_p
        Q_rev = pybamm.Concatenation(
            *[
                Q_rev_n,
                pybamm.FullBroadcast(0, ["separator"], "current collector"),
                Q_rev_p,
            ]
        )

        # Total heating
        Q = Q_ohm + Q_rxn + Q_rev

        # Compute the X-average over the entire cell, including current collectors
        Q_ohm_av = self._x_average(Q_ohm, Q_ohm_s_cn, Q_ohm_s_cp)
        Q_rxn_av = self._x_average(Q_rxn, 0, 0)
        Q_rev_av = self._x_average(Q_rev, 0, 0)
        Q_av = self._x_average(Q, Q_ohm_s_cn, Q_ohm_s_cp)

        # Compute volume-averaged heat source terms
        Q_ohm_vol_av = self._yz_average(Q_ohm_av)
        Q_rxn_vol_av = self._yz_average(Q_rxn_av)
        Q_rev_vol_av = self._yz_average(Q_rev_av)
        Q_vol_av = self._yz_average(Q_av)

        # Dimensional scaling for heat source terms
        Q_scale = param.i_typ * param.potential_scale / param.L_x

        variables.update(
            {
                "Ohmic heating": Q_ohm,
                "Ohmic heating [W.m-3]": Q_ohm * Q_scale,
                "X-averaged Ohmic heating": Q_ohm_av,
                "X-averaged Ohmic heating [W.m-3]": Q_ohm_av * Q_scale,
                "Volume-averaged Ohmic heating": Q_ohm_vol_av,
                "Volume-averaged Ohmic heating [W.m-3]": Q_ohm_vol_av * Q_scale,
                "Irreversible electrochemical heating": Q_rxn,
                "Irreversible electrochemical heating [W.m-3]": Q_rxn * Q_scale,
                "X-averaged irreversible electrochemical heating": Q_rxn_av,
                "X-averaged irreversible electrochemical heating [W.m-3]": Q_rxn_av
                * Q_scale,
                "Volume-averaged irreversible electrochemical heating": Q_rxn_vol_av,
                "Volume-averaged irreversible electrochemical heating "
                + "[W.m-3]": Q_rxn_vol_av * Q_scale,
                "Reversible heating": Q_rev,
                "Reversible heating [W.m-3]": Q_rev * Q_scale,
                "X-averaged reversible heating": Q_rev_av,
                "X-averaged reversible heating [W.m-3]": Q_rev_av * Q_scale,
                "Volume-averaged reversible heating": Q_rev_vol_av,
                "Volume-averaged reversible heating [W.m-3]": Q_rev_vol_av * Q_scale,
                "Total heating": Q,
                "Total heating [W.m-3]": Q * Q_scale,
                "X-averaged total heating": Q_av,
                "X-averaged total heating [W.m-3]": Q_av * Q_scale,
                "Volume-averaged total heating": Q_vol_av,
                "Volume-averaged total heating [W.m-3]": Q_vol_av * Q_scale,
            }
        )
        return variables

    def _current_collector_heating(self, variables):
        "Compute Ohmic heating in current collectors"
        # TODO: implement grad in 0D to return a scalar zero
        # TODO: implement grad_squared in other spatial methods so that the if
        # statement can be removed
        # In the limit of infinitely large current collector conductivity (i.e.
        # 0D current collectors), the Ohmic heating in the current collectors is
        # zero
        if self.cc_dimension == 0:
            Q_s_cn = pybamm.Scalar(0)
            Q_s_cp = pybamm.Scalar(0)
        # Otherwise we compute the Ohmic heating for 1 or 2D current collectors
        elif self.cc_dimension in [1, 2]:
            phi_s_cn = variables["Negative current collector potential"]
            phi_s_cp = variables["Positive current collector potential"]
            if self.cc_dimension == 1:
                Q_s_cn = self.param.sigma_cn_prime * pybamm.inner(
                    pybamm.grad(phi_s_cn), pybamm.grad(phi_s_cn)
                )
                Q_s_cp = self.param.sigma_cp_prime * pybamm.inner(
                    pybamm.grad(phi_s_cp), pybamm.grad(phi_s_cp)
                )
            elif self.cc_dimension == 2:
                # Inner not implemented in 2D -- have to call grad_squared directly
                Q_s_cn = self.param.sigma_cn_prime * pybamm.grad_squared(phi_s_cn)
                Q_s_cp = self.param.sigma_cp_prime * pybamm.grad_squared(phi_s_cp)
        return Q_s_cn, Q_s_cp

    def _x_average(self, var, var_cn, var_cp):
        """
        Computes the X-average over the whole cell (including current collectors)
        from the variable in the cell (negative electrode, separator,
        positive electrode), negative current collector, and positive current
        collector.
        Note: we do this as we cannot create a single variable which is
        the concatenation [var_cn, var, var_cp] since var_cn and var_cp share the
        same domain. (In the N+1D formulation the current collector variables are
        assumed independent of x, so we do not make the distinction between negative
        and positive current collectors in the geometry).
        """
        out = (
            self.param.l_cn * var_cn
            + self.param.l_x * pybamm.x_average(var)
            + self.param.l_cp * var_cp
        ) / self.param.l
        return out

    def _yz_average(self, var):
        "Computes the y-z average"
        # TODO: change the behaviour of z_average and yz_average so the if statement
        # can be removed
        if self.cc_dimension == 0:
            return var
        elif self.cc_dimension == 1:
            return pybamm.z_average(var)
        elif self.cc_dimension == 2:
            return pybamm.yz_average(var)
