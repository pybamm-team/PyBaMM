#
# Base class for thermal effects
#
import pybamm


class BaseThermal(pybamm.BaseSubModel):
    """
    Base class for thermal effects

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.
    """

    def __init__(self, param, options=None):
        super().__init__(param, options=options)

    def _get_standard_fundamental_variables(self, T_dict):
        """
        Note: here we explicitly pass in the averages for the temperature as computing
        the average temperature in `BaseThermal` using `self._x_average` requires a
        workaround to avoid raising a `ModelError` (as the key in the equation
        dict gets modified).

        For more information about this method in general,
        see :meth:`pybamm.base_submodel._get_standard_fundamental_variables`
        """
        param = self.param

        # The variable T is the concatenation of the temperature in the middle domains
        # (e.g. negative electrode, separator and positive electrode for a full cell),
        # excluding current collectors, for use in the electrochemical models
        T_mid = [T_dict[k] for k in self.options.whole_cell_domains]
        T = pybamm.concatenation(*T_mid)

        # Get the ambient temperature, which can be specified as a function of space
        # (y, z) only and time
        y = pybamm.standard_spatial_vars.y
        z = pybamm.standard_spatial_vars.z
        T_amb = param.T_amb(y, z, pybamm.t)
        T_amb_av = self._yz_average(T_amb)

        variables = {
            "Ambient temperature [K]": T_amb,
            "Volume-averaged ambient temperature [K]": T_amb_av,
            "Cell temperature [K]": T,
        }
        for name, var in T_dict.items():
            Name = name.capitalize()
            variables[f"{Name} temperature [K]"] = var
            if name in ["negative electrode", "separator", "positive electrode"]:
                variables[f"X-averaged {name} temperature [K]"] = pybamm.x_average(var)

        # Calculate temperatures in Celsius
        variables_Kelvin = variables.copy()
        for name_K, var in variables_Kelvin.items():
            name_C = name_K.replace("[K]", "[C]")
            variables.update({name_C: var - 273.15})

        return variables

    def _get_standard_coupled_variables(self, variables):
        param = self.param

        # Ohmic heating in solid
        i_s_p = variables["Positive electrode current density [A.m-2]"]
        phi_s_p = variables["Positive electrode potential [V]"]
        Q_ohm_s_cn, Q_ohm_s_cp = self._current_collector_heating(variables)
        if self.options.electrode_types["negative"] == "planar":
            i_boundary_cc = variables["Current collector current density [A.m-2]"]
            T_n = variables["Negative electrode temperature [K]"]
            Q_ohm_s_n = i_boundary_cc**2 / param.n.sigma(T_n)
        else:
            i_s_n = variables["Negative electrode current density [A.m-2]"]
            phi_s_n = variables["Negative electrode potential [V]"]
            Q_ohm_s_n = -pybamm.inner(i_s_n, pybamm.grad(phi_s_n))
        Q_ohm_s_s = pybamm.FullBroadcast(0, ["separator"], "current collector")
        Q_ohm_s_p = -pybamm.inner(i_s_p, pybamm.grad(phi_s_p))
        Q_ohm_s = pybamm.concatenation(Q_ohm_s_n, Q_ohm_s_s, Q_ohm_s_p)

        # Ohmic heating in electrolyte
        # TODO: change full stefan-maxwell conductivity so that i_e is always
        # a Concatenation
        i_e = variables["Electrolyte current density [A.m-2]"]
        phi_e = variables["Electrolyte potential [V]"]
        if isinstance(i_e, pybamm.Concatenation):
            # compute by domain if possible
            phi_e_s = variables["Separator electrolyte potential [V]"]
            phi_e_p = variables["Positive electrolyte potential [V]"]
            if self.options.electrode_types["negative"] == "planar":
                i_e_s, i_e_p = i_e.orphans
                Q_ohm_e_n = pybamm.FullBroadcast(
                    0, ["negative electrode"], "current collector"
                )
            else:
                i_e_n, i_e_s, i_e_p = i_e.orphans
                phi_e_n = variables["Negative electrolyte potential [V]"]
                Q_ohm_e_n = -pybamm.inner(i_e_n, pybamm.grad(phi_e_n))
            Q_ohm_e_s = -pybamm.inner(i_e_s, pybamm.grad(phi_e_s))
            Q_ohm_e_p = -pybamm.inner(i_e_p, pybamm.grad(phi_e_p))
            Q_ohm_e = pybamm.concatenation(Q_ohm_e_n, Q_ohm_e_s, Q_ohm_e_p)
        else:
            # else compute using i_e across all domains
            if self.options.electrode_types["negative"] == "planar":
                Q_ohm_e_n = pybamm.FullBroadcast(
                    0, ["negative electrode"], "current collector"
                )
                Q_ohm_e_s_p = -pybamm.inner(i_e, pybamm.grad(phi_e))
                Q_ohm_e = pybamm.concatenation(Q_ohm_e_n, Q_ohm_e_s_p)
            else:
                Q_ohm_e = -pybamm.inner(i_e, pybamm.grad(phi_e))

        # Total Ohmic heating
        Q_ohm = Q_ohm_s + Q_ohm_e

        num_phases = int(getattr(self.options, "positive")["particle phases"])
        phase_names = [""]
        if num_phases > 1:
            phase_names = ["primary ", "secondary "]

        Q_rxn_p, Q_rev_p = 0, 0
        T_p = variables["Positive electrode temperature [K]"]
        for phase in phase_names:
            a_j_p = variables[
                f"Positive electrode {phase}volumetric interfacial current density [A.m-3]"
            ]
            eta_r_p = variables[f"Positive electrode {phase}reaction overpotential [V]"]
            # Irreversible electrochemical heating
            Q_rxn_p += a_j_p * eta_r_p
            # Reversible electrochemical heating
            dUdT_p = variables[f"Positive electrode {phase}entropic change [V.K-1]"]
            Q_rev_p += a_j_p * T_p * dUdT_p

        num_phases = int(getattr(self.options, "negative")["particle phases"])
        phase_names = [""]
        if num_phases > 1:
            phase_names = ["primary", "secondary"]

        if self.options.electrode_types["negative"] == "planar":
            Q_rxn_n = pybamm.FullBroadcast(
                0, ["negative electrode"], "current collector"
            )
            Q_rev_n = pybamm.FullBroadcast(
                0, ["negative electrode"], "current collector"
            )
        else:
            T_n = variables["Negative electrode temperature [K]"]
            Q_rxn_n = 0
            Q_rev_n = 0
            for phase in phase_names:
                a_j_n = variables[
                    f"Negative electrode {phase}volumetric interfacial current density [A.m-3]"
                ]
                eta_r_n = variables[
                    f"Negative electrode {phase}reaction overpotential [V]"
                ]
                # Irreversible electrochemical heating
                Q_rxn_n += a_j_n * eta_r_n

                # Reversible electrochemical heating
                dUdT_n = variables[f"Negative electrode {phase}entropic change [V.K-1]"]
                Q_rev_n += a_j_n * T_n * dUdT_n

        # Irreversible electrochemical heating
        Q_rxn = pybamm.concatenation(
            Q_rxn_n, pybamm.FullBroadcast(0, "separator", "current collector"), Q_rxn_p
        )

        # Reversible electrochemical heating
        Q_rev = pybamm.concatenation(
            Q_rev_n, pybamm.FullBroadcast(0, "separator", "current collector"), Q_rev_p
        )

        # Total heating
        Q = Q_ohm + Q_rxn + Q_rev

        # Compute the X-average over the entire cell, including current collectors
        # Note: this can still be a function of y and z for higher-dimensional pouch
        # cell models
        Q_ohm_av = self._x_average(Q_ohm, Q_ohm_s_cn, Q_ohm_s_cp)
        Q_rxn_av = self._x_average(Q_rxn, 0, 0)
        Q_rev_av = self._x_average(Q_rev, 0, 0)
        Q_av = self._x_average(Q, Q_ohm_s_cn, Q_ohm_s_cp)

        # Compute the integrated heat source per unit simulated electrode-pair area
        # in W.m-2. Note: this can still be a function of y and z for
        # higher-dimensional pouch cell models
        Q_ohm_Wm2 = Q_ohm_av * param.L
        Q_rxn_Wm2 = Q_rxn_av * param.L
        Q_rev_Wm2 = Q_rev_av * param.L
        Q_Wm2 = Q_av * param.L
        # Now average over the electrode height and width
        Q_ohm_Wm2_av = self._yz_average(Q_ohm_Wm2)
        Q_rxn_Wm2_av = self._yz_average(Q_rxn_Wm2)
        Q_rev_Wm2_av = self._yz_average(Q_rev_Wm2)
        Q_Wm2_av = self._yz_average(Q_Wm2)

        # Compute total heat source terms (in W) over the *entire cell volume*, not
        # the product of electrode height * electrode width * electrode stack thickness
        # Note: we multiply by the number of electrode pairs, since the Q_xx_Wm2_av
        # variables are per electrode pair
        n_elec = param.n_electrodes_parallel
        A = param.L_y * param.L_z  # *modelled* electrode area
        Q_ohm_W = Q_ohm_Wm2_av * n_elec * A
        Q_rxn_W = Q_rxn_Wm2_av * n_elec * A
        Q_rev_W = Q_rev_Wm2_av * n_elec * A
        Q_W = Q_Wm2_av * n_elec * A

        # Compute volume-averaged heat source terms over the *entire cell volume*, not
        # the product of electrode height * electrode width * electrode stack thickness
        V = param.V_cell  # *actual* cell volume
        Q_ohm_vol_av = Q_ohm_W / V
        Q_rxn_vol_av = Q_rxn_W / V
        Q_rev_vol_av = Q_rev_W / V
        Q_vol_av = Q_W / V

        variables.update(
            {
                # Ohmic
                "Ohmic heating [W.m-3]": Q_ohm,
                "X-averaged Ohmic heating [W.m-3]": Q_ohm_av,
                "Volume-averaged Ohmic heating [W.m-3]": Q_ohm_vol_av,
                "Ohmic heating per unit electrode-pair area [W.m-2]": Q_ohm_Wm2,
                "Ohmic heating [W]": Q_ohm_W,
                # Irreversible
                "Irreversible electrochemical heating [W.m-3]": Q_rxn,
                "X-averaged irreversible electrochemical heating [W.m-3]": Q_rxn_av,
                "Volume-averaged irreversible electrochemical heating "
                + "[W.m-3]": Q_rxn_vol_av,
                "Irreversible electrochemical heating per unit "
                + "electrode-pair area [W.m-2]": Q_rxn_Wm2,
                "Irreversible electrochemical heating [W]": Q_rxn_W,
                # Reversible
                "Reversible heating [W.m-3]": Q_rev,
                "X-averaged reversible heating [W.m-3]": Q_rev_av,
                "Volume-averaged reversible heating [W.m-3]": Q_rev_vol_av,
                "Reversible heating per unit electrode-pair area " "[W.m-2]": Q_rev_Wm2,
                "Reversible heating [W]": Q_rev_W,
                # Total
                "Total heating [W.m-3]": Q,
                "X-averaged total heating [W.m-3]": Q_av,
                "Volume-averaged total heating [W.m-3]": Q_vol_av,
                "Total heating per unit electrode-pair area [W.m-2]": Q_Wm2,
                "Total heating [W]": Q_W,
                # Current collector
                "Negative current collector Ohmic heating [W.m-3]": Q_ohm_s_cn,
                "Positive current collector Ohmic heating [W.m-3]": Q_ohm_s_cp,
            }
        )
        return variables

    def _current_collector_heating(self, variables):
        """Compute Ohmic heating in current collectors."""
        cc_dimension = self.options["dimensionality"]

        # Compute the Ohmic heating for 0D current collectors
        if cc_dimension == 0:
            i_boundary_cc = variables["Current collector current density [A.m-2]"]
            Q_s_cn = i_boundary_cc**2 / self.param.n.sigma_cc
            Q_s_cp = i_boundary_cc**2 / self.param.p.sigma_cc
        # Otherwise we compute the Ohmic heating for 1 or 2D current collectors
        # In this limit the current flow is all in the y,z direction in the current
        # collectors
        elif cc_dimension in [1, 2]:
            phi_s_cn = variables["Negative current collector potential [V]"]
            phi_s_cp = variables["Positive current collector potential [V]"]
            # TODO: implement grad_squared in other spatial methods so that the
            # if statement can be removed
            if cc_dimension == 1:
                Q_s_cn = self.param.n.sigma_cc * pybamm.inner(
                    pybamm.grad(phi_s_cn), pybamm.grad(phi_s_cn)
                )
                Q_s_cp = self.param.p.sigma_cc * pybamm.inner(
                    pybamm.grad(phi_s_cp), pybamm.grad(phi_s_cp)
                )
            elif cc_dimension == 2:
                # Inner not implemented in 2D -- have to call grad_squared directly
                Q_s_cn = self.param.n.sigma_cc * pybamm.grad_squared(phi_s_cn)
                Q_s_cp = self.param.p.sigma_cc * pybamm.grad_squared(phi_s_cp)
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
            self.param.n.L_cc * var_cn
            + self.param.L_x * pybamm.x_average(var)
            + self.param.p.L_cc * var_cp
        ) / self.param.L
        return out

    def _yz_average(self, var):
        """Computes the y-z average."""
        # TODO: change the behaviour of z_average and yz_average so the if statement
        # can be removed
        if self.options["dimensionality"] in [0, 1]:
            return pybamm.z_average(var)
        elif self.options["dimensionality"] == 2:
            return pybamm.yz_average(var)
