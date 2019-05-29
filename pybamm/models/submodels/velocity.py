#
# Equation classes for the electrolyte velocity
#
import pybamm


class Velocity(pybamm.SubModel):
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
        dVbox_dz = 0

        # Build model
        self.algebraic = {p: pybamm.div(v_box) + dVbox_dz - param.beta * j}
        self.initial_conditions = {p: 0}
        self.boundary_conditions = {
            p: {"left": (0, "Dirichlet"), "right": (0, "Neumann")}
        }
        self.variables = self.get_variables(v_box)

    def get_explicit_leading_order(self, reactions):
        """
        Provides explicit velocity for the leading-order models, as a post-processing
        step.

        Parameters
        ----------
        reactions : dict
            Dictionary of reaction variables
        """
        # Set up
        param = self.set_of_parameters
        x_n = pybamm.standard_spatial_vars.x_n
        x_p = pybamm.standard_spatial_vars.x_p

        j_n = reactions["main"]["neg"]["aj"]
        j_p = reactions["main"]["pos"]["aj"]

        # Volume-averaged velocity
        v_box_n = param.beta_n * pybamm.outer(j_n, x_n)
        v_box_p = param.beta_p * pybamm.outer(j_p, x_p - 1)

        v_box, V_box_z = self.get_combined_velocities(v_box_n, v_box_p)

        return self.get_variables(v_box)

    def get_explicit_composite(self, reactions):
        """
        Provides explicit velocity for the composite models, as a post-processing step.

        Parameters
        ----------
        reactions : dict
            Dictionary of reaction variables
        """
        # Set up
        param = self.set_of_parameters
        x_n = pybamm.standard_spatial_vars.x_n
        x_p = pybamm.standard_spatial_vars.x_p
        j_n = reactions["main"]["neg"]["aj"]
        j_p = reactions["main"]["pos"]["aj"]

        # Volume-averaged velocity
        v_box_n = param.beta_n * pybamm.IndefiniteIntegral(j_n, x_n)
        v_box_p = param.beta_p * pybamm.IndefiniteIntegral(j_p, x_p)

        v_box, V_box_z = self.get_combined_velocities(v_box_n, v_box_p)

        return self.get_variables(v_box)

    def get_combined_velocities(self, v_box_n, v_box_p):
        """
        Calculate x- and z-components of velocity in the separator

        Parameters
        ----------
        v_box_n : :class:`pybamm.Symbol`
            The x-component of velocity in the negative electrode
        v_box_p : :class:`pybamm.Symbol`
            The x-component of velocity in the positive electrode

        Returns
        -------
        v_box : :class:`pybamm.Symbol`
            The x-component of velocity in the whole cell
        V_box_z : :class:`pybamm.Symbol`
            The z-component of velocity in the separator
        """
        l_n = pybamm.geometric_parameters.l_n
        l_s = pybamm.geometric_parameters.l_s
        x_s = pybamm.standard_spatial_vars.x_s
        z = pybamm.standard_spatial_vars.z

        # Difference in negative and positive electrode velocities determines the
        # velocity in the separator
        v_box_n_right = pybamm.boundary_value(v_box_n, "right")
        v_box_p_left = pybamm.boundary_value(v_box_p, "left")
        v_box_difference = (v_box_p_left - v_box_n_right) / l_s

        # Simple formula for velocity in the separator
        V_box_z = v_box_difference * z
        v_box_s = v_box_difference * (x_s - l_n) + v_box_n_right

        v_box = pybamm.Concatenation(v_box_n, v_box_s, v_box_p)
        return v_box, V_box_z

    def get_variables(self, v_box):
        """
        Calculate dimensionless and dimensional variables for the electrolyte current
        submodel

        Parameters
        ----------
        v_box : tuple
            Tuple of mass-averaged velocities

        Returns
        -------
        dict
            Dictionary {string: :class:`pybamm.Symbol`} of relevant variables
        """
        vel_scale = self.set_of_parameters.velocity_scale

        return {
            "Volume-averaged velocity": v_box,
            "Volume-averaged velocity [m.s-1]": vel_scale * v_box,
        }
