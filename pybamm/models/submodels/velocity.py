#
# Equation classes for the electrolyte velocity
#
import pybamm


class Velocity(pybamm.BaseSubModel):
    """Electrolyte velocity

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.SubModel`
    """

    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

    def set_algebraic_system(self, variables):
        """
        Algebraic system for pressure and volume-averaged velocity in the electrolyte.

        Parameters
        ----------
        variables : dict
            Dictionary of symbols to use in the model
        """
        # Unpack variables
        param = self.set_of_parameters
        p = variables["Electrolyte pressure"]
        # c_e = variables["Electrolyte concentration"]

        # Set up reactions
        j = variables["Interfacial current density"]
        v_mass = -pybamm.grad(p)
        v_box = v_mass
        _, dVbox_dz = self.get_separator_velocities(variables)

        # Build model
        self.algebraic = {p: pybamm.div(v_box) + dVbox_dz - param.beta * j}
        self.initial_conditions = {p: 0}
        self.boundary_conditions = {
            p: {"left": (0, "Dirichlet"), "right": (0, "Neumann")}
        }
        self.variables = self.get_variables(v_box, dVbox_dz)

    def get_explicit_leading_order(self, variables):
        """
        Provides explicit velocity for the leading-order models, as a post-processing
        step.

        Parameters
        ----------
        variables : dict
            Dictionary of symbols to use in the model
        """
        # Set up
        param = self.set_of_parameters
        x_n = pybamm.standard_spatial_vars.x_n
        x_p = pybamm.standard_spatial_vars.x_p

        j_n = variables["Negative electrode interfacial current density"]
        j_p = variables["Positive electrode interfacial current density"]

        # Volume-averaged velocity
        v_box_n = param.beta_n * pybamm.outer(j_n, x_n)
        v_box_p = param.beta_p * pybamm.outer(j_p, x_p - 1)

        v_box_s, dVbox_dz = self.get_separator_velocities(variables)
        v_box = pybamm.Concatenation(v_box_n, v_box_s, v_box_p)

        return self.get_variables(v_box, dVbox_dz)

    def get_explicit_composite(self, variables):
        """
        Provides explicit velocity for the composite models, as a post-processing step.

        Parameters
        ----------
        variables : dict
            Dictionary of symbols to use in the model
        """
        # Set up
        param = self.set_of_parameters
        x_n = pybamm.standard_spatial_vars.x_n
        x_p = pybamm.standard_spatial_vars.x_p
        j_n = variables["Negative electrode interfacial current density"]
        j_p = variables["Positive electrode interfacial current density"]

        # Volume-averaged velocity
        v_box_n = param.beta_n * pybamm.IndefiniteIntegral(j_n, x_n)
        # Shift v_box_p to be equal to 0 at x_p = 1
        v_box_p = param.beta_p * (
            pybamm.IndefiniteIntegral(j_p, x_p) - pybamm.Integral(j_p, x_p)
        )

        v_box_s, dVbox_dz = self.get_separator_velocities(variables)
        v_box = pybamm.Concatenation(v_box_n, v_box_s, v_box_p)

        return self.get_variables(v_box, dVbox_dz)

    def get_separator_velocities(self, variables):
        """
        Calculate x- and z-components of velocity in the separator

        Parameters
        ----------
        variables : dict
            Dictionary of symbols to use in the model

        Returns
        -------
        v_box_s : :class:`pybamm.Symbol`
            The x-component of velocity in the separator
        dVbox_dz : :class:`pybamm.Symbol`
            The z-component of velocity in the separator
        """
        # Set up
        param = self.set_of_parameters
        l_n = pybamm.geometric_parameters.l_n
        l_s = pybamm.geometric_parameters.l_s
        x_s = pybamm.standard_spatial_vars.x_s

        # Difference in negative and positive electrode velocities determines the
        # velocity in the separator
        i_boundary_cc = variables["Current collector current density"]
        v_box_n_right = param.beta_n * i_boundary_cc
        v_box_p_left = param.beta_p * i_boundary_cc
        d_vbox_s__dx = (v_box_p_left - v_box_n_right) / l_s

        # Simple formula for velocity in the separator
        dVbox_dz = pybamm.Concatenation(
            pybamm.Broadcast(0, "negative electrode"),
            pybamm.Broadcast(-d_vbox_s__dx, "separator"),
            pybamm.Broadcast(0, "positive electrode"),
        )
        v_box_s = d_vbox_s__dx * (x_s - l_n) + v_box_n_right

        return v_box_s, dVbox_dz

    def get_variables(self, v_box, dVbox_dz):
        """
        Calculate dimensionless and dimensional variables for the electrolyte current
        submodel

        Parameters
        ----------
        v_box : :class:`pybamm.Symbol`
            Volume-averaged velocity in the x-direction
        dVbox_dz : :class:`pybamm.Symbol`
            Volume-averaged acceleration in the z-direction (z-derivative of velocity)

        Returns
        -------
        dict
            Dictionary {string: :class:`pybamm.Symbol`} of relevant variables
        """
        vel_scale = self.set_of_parameters.velocity_scale
        L_z = self.set_of_parameters.L_z

        return {
            "Volume-averaged velocity": v_box,
            "Volume-averaged velocity [m.s-1]": vel_scale * v_box,
            "Vertical volume-averaged acceleration": dVbox_dz,
            "Vertical volume-averaged acceleration [m.s-2]": vel_scale / L_z * dVbox_dz,
        }
