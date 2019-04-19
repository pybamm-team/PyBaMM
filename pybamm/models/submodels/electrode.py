#
# Equation classes for the electrode
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class Ohm(pybamm.SubModel):
    """Ohm's law + conservation of current for the current in the electrodes.

    Parameters
    ----------
    phi : :class:`pybamm.Symbol`
        The electric potential in the electrodes ("electrode potential")
    j : :class:`pybamm.Symbol`
        An expression tree that represents the interfacial current density at the
        electrode-electrolyte interface
    param : parameter class
        The parameters to use for this submodel
    epsilon : :class:`pybamm.Symbol`
        The (electrolyte/liquid phase) porosity (optional)

    *Extends:* :class:`BaseModel`
    """

    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

    def set_algebraic_system(self, phi_s, j, eps=None):
        param = self.set_of_parameters

        icell = param.current_with_time

        # algebraic model only
        self.rhs = {}

        # different bounday conditions in each electrode
        if phi_s.domain == ["negative electrode"]:
            # if the porosity is not a variable, use the input parameter
            if eps is None:
                eps = param.epsilon_n
            # liion sigma_n may already account for porosity
            i_s_n = -param.sigma_n * (1 - eps) ** param.b * pybamm.grad(phi_s)
            self.algebraic = {phi_s: pybamm.div(i_s_n) + j}
            self.boundary_conditions = {phi_s: {"left": 0}, i_s_n: {"right": 0}}
            self.initial_conditions = {phi_s: 0}
            self.variables = {
                "Negative electrode potential": phi_s,
                "Negative electrode current density": i_s_n,
            }
        elif phi_s.domain == ["positive electrode"]:
            # if porosity is not a variable, use the input parameter
            if eps is None:
                eps = param.epsilon_p
            # liion sigma_p may already account for porosity
            i_s_p = -param.sigma_p * (1 - eps) ** param.b * pybamm.grad(phi_s)
            self.algebraic = {phi_s: pybamm.div(i_s_p) + j}
            self.boundary_conditions = {i_s_p: {"left": 0, "right": icell}}
            self.initial_conditions = {
                phi_s: param.U_p(param.c_p_init) - param.U_n(param.c_n_init)
            }
            self.variables = {
                "Positive electrode potential": phi_s,
                "Positive electrode current density": i_s_p,
            }
        # for whole cell domain call both electrode models and ignore separator
        elif phi_s.domain == ["negative electrode", "separator", "positive electrode"]:
            # if porosity is not a variable, use the input parameter
            if eps is None:
                eps = param.epsilon
            phi_s_n, phi_s_s, phi_s_p = phi_s.orphans
            eps_n, eps_s, eps_p = eps.orphans
            j_n, j_s, j_p = j.orphans
            neg_model = Ohm(phi_s_n, j_n, param, eps=eps_n)
            pos_model = Ohm(phi_s_p, j_p, param, eps=eps_p)
            self.update(neg_model, pos_model)
            # Voltage variable
            voltage = pybamm.BoundaryValue(phi_s, "right") - pybamm.BoundaryValue(
                phi_s, "left"
            )
            self.variables.update({"Terminal voltage": voltage})
        else:
            raise pybamm.DomainError("domain '{}' not recognised".format(phi_s.domain))

        # Set default solver to DAE
        self.default_solver = pybamm.ScikitsDaeSolver()

    def set_explicit_leading_order(self, variables):
        """
        Provides the leading order explicit solution to solid phase current
        conservation with ohm's law.

        Parameters
        ----------
        variables : dict
            Dictionary of {string: :class:`pybamm.Symbol`}, which can be read to find
            already-calculated variables
        """
        # import parameters and spatial variables
        param = self.set_of_parameters
        l_n = param.l_n
        l_p = param.l_p
        x_n = pybamm.standard_spatial_vars.x_n
        x_p = pybamm.standard_spatial_vars.x_p

        # define current
        i_cell = param.current_with_time

        # unpack variables
        ocp_p = variables["Positive electrode open circuit potential"]
        eta_r_p = variables["Positive reaction overpotential"]
        phi_e = variables["Electrolyte potential"]

        # extract right-most ocp, overpotential, and electrolyte potential
        ocp_p_right = pybamm.BoundaryValue(ocp_p, "right")
        eta_r_p_right = pybamm.BoundaryValue(eta_r_p, "right")
        phi_e_right = pybamm.BoundaryValue(phi_e, "right")

        # electode potential
        phi_s_n = pybamm.Broadcast(0, ["negative electrode"])
        phi_s_s = pybamm.Broadcast(0, ["separator"])
        v = ocp_p_right + eta_r_p_right + phi_e_right
        phi_s_p = v + pybamm.Broadcast(0, ["positive electrode"])
        phi_s = pybamm.Concatenation(phi_s_n, phi_s_s, phi_s_p)

        # electrode current
        i_s_n = i_cell - i_cell * x_n / l_n
        i_s_s = pybamm.Broadcast(0, ["separator"])
        i_s_p = i_cell - i_cell * (1 - x_p) / l_p
        i_s = pybamm.Concatenation(i_s_n, i_s_s, i_s_p)

        delta_phi_s_av = pybamm.Scalar(0)

        self.set_variables(phi_s, i_s, delta_phi_s_av, v)

    def set_explicit_combined(self, variables):
        """
        Provides an explicit combined leading and first order solution to solid phase
        current conservation with ohm's law. Note that the returned current density is
        only the leading order approximation.

        Parameters
        ----------
        variables : dict
            Dictionary of {string: :class:`pybamm.Symbol`}, which can be read to find
            already-calculated variables
        """
        # import parameters and spatial vairables
        param = self.set_of_parameters
        l_n = param.l_n
        l_p = param.l_p
        i_cell = param.current_with_time
        x_n = pybamm.standard_spatial_vars.x_n
        x_p = pybamm.standard_spatial_vars.x_p

        # Unpack variables
        phi_e = variables["Electrolyte potential"]
        ocp_p = variables["Positive electrode open circuit potential"]
        eta_r_p = variables["Positive reaction overpotential"]

        # if porosity is not provided, use the input parameter
        try:
            epsilon = variables["Porosity"]
        except KeyError:
            epsilon = param.epsilon
        eps_n, eps_s, eps_p = [e.orphans[0] for e in epsilon.orphans]

        # extract right-most ocp, overpotential, and electrolyte potential
        ocp_p_right = pybamm.BoundaryValue(ocp_p, "right")
        eta_r_p_right = pybamm.BoundaryValue(eta_r_p, "right")
        phi_e_right = pybamm.BoundaryValue(phi_e, "right")

        # electrode potential
        sigma_n_eff = param.sigma_n * (1 - eps_n)
        sigma_p_eff = param.sigma_p * (1 - eps_p)
        phi_s_n = i_cell * x_n * (2 * l_n - x_n) / (2 * sigma_n_eff * l_n)
        phi_s_s = pybamm.Broadcast(0, ["separator"])  # can we put NaN?
        phi_s_p = (ocp_p_right + eta_r_p_right + phi_e_right) + i_cell * (
            (1 - x_p) * (1 - 2 * l_p - x_p) / (2 * sigma_p_eff * l_p)
        )
        phi_s = pybamm.Concatenation(phi_s_n, phi_s_s, phi_s_p)

        # electrode current
        i_s_n = i_cell - i_cell * x_n / l_n
        i_s_s = pybamm.Broadcast(0, ["separator"])
        i_s_p = i_cell - i_cell * (1 - x_p) / l_p
        i_s = pybamm.Concatenation(i_s_n, i_s_s, i_s_p)

        # average solid phase ohmic losses
        delta_phi_s_av = -i_cell / 3 * (l_p / sigma_p_eff + l_n / sigma_n_eff)

        # terminal voltage
        ocv_av = variables["Average open circuit voltage"]
        eta_r_av = variables["Average reaction overpotential"]
        eta_c_av = variables["Average concentration overpotential"]
        delta_phi_e_av = variables["Average electrolyte ohmic losses"]

        v = ocv_av + eta_r_av + eta_c_av + delta_phi_e_av + delta_phi_s_av

        self.set_variables(phi_s, i_s, delta_phi_s_av, v)

    def set_variables(self, phi_s, i_s, delta_phi_s_av, v):
        param = self.set_of_parameters

        # Unpack
        phi_s_n, phi_s_s, phi_s_p = phi_s.orphans
        i_s_n, i_s_s, i_s_p = i_s.orphans

        # Dimensional
        phi_s_n_dim = param.potential_scale * phi_s_n
        phi_s_s_dim = pybamm.Broadcast(0, ["separator"])
        phi_s_p_dim = param.U_p_ref - param.U_n_ref + param.potential_scale * phi_s_p
        phi_s_dim = pybamm.Concatenation(phi_s_n_dim, phi_s_s_dim, phi_s_p_dim)
        i_s_n_dim = param.i_typ * i_s_n
        i_s_p_dim = param.i_typ * i_s_p
        i_s_dim = param.i_typ * i_s
        delta_phi_s_av_dim = param.potential_scale * delta_phi_s_av
        v_dim = param.potential_scale * v

        # Update variables

        self.variables.update(
            {
                "Negative electrode potential": phi_s_n,
                "Positive electrode potential": phi_s_p,
                "Electrode potential": phi_s,
                "Negative electrode current density": i_s_n,
                "Positive electrode current density": i_s_p,
                "Electrode current density": i_s,
                "Average solid phase ohmic losses": delta_phi_s_av,
                "Terminal voltage": v,
                "Negative electrode potential [V]": phi_s_n_dim,
                "Positive electrode potential [V]": phi_s_p_dim,
                "Electrode potential [V]": phi_s_dim,
                "Negative electrode current density [A m-2]": i_s_n_dim,
                "Positive electrode current density [A m-2]": i_s_p_dim,
                "Electrode current density [A m-2]": i_s_dim,
                "Average solid phase ohmic losses [V]": delta_phi_s_av_dim,
                "Terminal voltage [V]": v_dim,
            }
        )
