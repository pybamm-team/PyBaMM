#
# Geometric Parameters
#
import pybamm
from .base_parameters import BaseParameters


class GeometricParameters(BaseParameters):
    """
    Standard geometric parameters

    Layout:
        1. Dimensional Parameters
        2. Dimensional Functions
        3. Scalings
        4. Dimensionless Parameters
        5. Dimensionless Functions
    """

    def __init__(self):

        # Set parameters and scales
        self._set_dimensional_parameters()
        self._set_scales()
        self._set_dimensionless_parameters()

    def _set_dimensional_parameters(self):
        """Defines the dimensional parameters."""

        # Macroscale geometry
        self.L_cn = pybamm.Parameter("Negative current collector thickness [m]")
        self.L_n = pybamm.Parameter("Negative electrode thickness [m]")
        self.L_s = pybamm.Parameter("Separator thickness [m]")
        self.L_p = pybamm.Parameter("Positive electrode thickness [m]")
        self.L_cp = pybamm.Parameter("Positive current collector thickness [m]")
        self.L_x = (
            self.L_n + self.L_s + self.L_p
        )  # Total distance between current collectors
        self.L = self.L_cn + self.L_x + self.L_cp  # Total cell thickness
        self.L_Li = pybamm.Parameter("Lithium counter electrode thickness [m]")
        self.L_y = pybamm.Parameter(
            "Electrode width [m]"
        )  # For a cylindrical cell L_y is the "unwound" length of the electrode
        self.L_z = pybamm.Parameter("Electrode height [m]")
        self.r_inner_dimensional = pybamm.Parameter("Inner cell radius [m]")
        self.r_outer_dimensional = pybamm.Parameter("Outer cell radius [m]")
        self.A_cc = self.L_y * self.L_z  # Current collector cross sectional area
        self.A_cooling = pybamm.Parameter("Cell cooling surface area [m2]")
        self.V_cell = pybamm.Parameter("Cell volume [m3]")

        # Tab geometry (for pouch cells)
        self.L_tab_n = pybamm.Parameter("Negative tab width [m]")
        self.Centre_y_tab_n = pybamm.Parameter("Negative tab centre y-coordinate [m]")
        self.Centre_z_tab_n = pybamm.Parameter("Negative tab centre z-coordinate [m]")
        self.L_tab_p = pybamm.Parameter("Positive tab width [m]")
        self.Centre_y_tab_p = pybamm.Parameter("Positive tab centre y-coordinate [m]")
        self.Centre_z_tab_p = pybamm.Parameter("Positive tab centre z-coordinate [m]")
        self.A_tab_n = self.L_tab_n * self.L_cn  # Area of negative tab
        self.A_tab_p = self.L_tab_p * self.L_cp  # Area of negative tab

        # Microscale geometry
        # Note: for li-ion cells, the definition of the surface area to
        # volume ratio is overwritten in lithium_ion_parameters.py to be computed
        # based on the assumed particle shape
        self.a_n_dim = pybamm.Parameter(
            "Negative electrode surface area to volume ratio [m-1]"
        )
        self.a_p_dim = pybamm.Parameter(
            "Positive electrode surface area to volume ratio [m-1]"
        )
        self.b_e_n = pybamm.Parameter(
            "Negative electrode Bruggeman coefficient (electrolyte)"
        )
        self.b_e_s = pybamm.Parameter("Separator Bruggeman coefficient (electrolyte)")
        self.b_e_p = pybamm.Parameter(
            "Positive electrode Bruggeman coefficient (electrolyte)"
        )
        self.b_s_n = pybamm.Parameter(
            "Negative electrode Bruggeman coefficient (electrode)"
        )
        self.b_s_p = pybamm.Parameter(
            "Positive electrode Bruggeman coefficient (electrode)"
        )

        # Particle-size distribution geometry
        self.R_min_n_dim = pybamm.Parameter("Negative minimum particle radius [m]")
        self.R_min_p_dim = pybamm.Parameter("Positive minimum particle radius [m]")
        self.R_max_n_dim = pybamm.Parameter("Negative maximum particle radius [m]")
        self.R_max_p_dim = pybamm.Parameter("Positive maximum particle radius [m]")
        self.sd_a_n_dim = pybamm.Parameter(
            "Negative area-weighted particle-size standard deviation [m]"
        )
        self.sd_a_p_dim = pybamm.Parameter(
            "Positive area-weighted particle-size standard deviation [m]"
        )

        x_n = pybamm.standard_spatial_vars.x_n
        x_p = pybamm.standard_spatial_vars.x_p
        self.R_n_dimensional = pybamm.FunctionParameter(
            "Negative particle radius [m]",
            {"Through-cell distance (x_n) [m]": x_n * self.L_x},
        )
        self.R_p_dimensional = pybamm.FunctionParameter(
            "Positive particle radius [m]",
            {"Through-cell distance (x_p) [m]": x_p * self.L_x},
        )

    def f_a_dist_n_dimensional(self, R):
        """
        Dimensional negative electrode area-weighted particle-size distribution
        """
        inputs = {
            "Negative particle-size variable [m]": R,
        }
        return pybamm.FunctionParameter(
            "Negative area-weighted particle-size distribution [m-1]",
            inputs,
        )

    def f_a_dist_p_dimensional(self, R):
        """
        Dimensional positive electrode area-weighted particle-size distribution
        """
        inputs = {
            "Positive particle-size variable [m]": R,
        }
        return pybamm.FunctionParameter(
            "Positive area-weighted particle-size distribution [m-1]",
            inputs,
        )

    def _set_scales(self):
        """Define the scales used in the non-dimensionalisation scheme"""

        # Microscale geometry
        # Note: these scales are necessary here to non-dimensionalise the
        # particle size distributions.
        self.R_n_typ = pybamm.xyz_average(self.R_n_dimensional)
        self.R_p_typ = pybamm.xyz_average(self.R_p_dimensional)

    def _set_dimensionless_parameters(self):
        """Defines the dimensionless parameters."""

        # Macroscale Geometry
        self.l_cn = self.L_cn / self.L_x
        self.l_n = self.L_n / self.L_x
        self.l_s = self.L_s / self.L_x
        self.l_p = self.L_p / self.L_x
        self.l_cp = self.L_cp / self.L_x
        self.l_x = self.L_x / self.L_x
        self.l_Li = self.L_Li / self.L_x
        self.l_y = self.L_y / self.L_z
        self.l_z = self.L_z / self.L_z
        self.r_inner = self.r_inner_dimensional / self.r_outer_dimensional
        self.r_outer = self.r_outer_dimensional / self.r_outer_dimensional
        self.a_cc = self.l_y * self.l_z
        self.a_cooling = self.A_cooling / (self.L_z ** 2)
        self.v_cell = self.V_cell / (self.L_x * self.L_z ** 2)

        self.l = self.L / self.L_x
        self.delta = self.L_x / self.L_z  # Pouch cell aspect ratio

        # Tab geometry (for pouch cells)
        self.l_tab_n = self.L_tab_n / self.L_z
        self.centre_y_tab_n = self.Centre_y_tab_n / self.L_z
        self.centre_z_tab_n = self.Centre_z_tab_n / self.L_z
        self.l_tab_p = self.L_tab_p / self.L_z
        self.centre_y_tab_p = self.Centre_y_tab_p / self.L_z
        self.centre_z_tab_p = self.Centre_z_tab_p / self.L_z

        # Particle-size distribution geometry
        self.R_min_n = self.R_min_n_dim / self.R_n_typ
        self.R_min_p = self.R_min_p_dim / self.R_p_typ
        self.R_max_n = self.R_max_n_dim / self.R_n_typ
        self.R_max_p = self.R_max_p_dim / self.R_p_typ
        self.sd_a_n = self.sd_a_n_dim / self.R_n_typ
        self.sd_a_p = self.sd_a_p_dim / self.R_p_typ

        # Particle radius
        self.R_n = self.R_n_dimensional / self.R_n_typ
        self.R_p = self.R_p_dimensional / self.R_p_typ

    def f_a_dist_n(self, R):
        """
        Dimensionless negative electrode area-weighted particle-size distribution
        """
        R_dim = R * self.R_n_typ
        return self.f_a_dist_n_dimensional(R_dim) * self.R_n_typ

    def f_a_dist_p(self, R):
        """
        Dimensionless positive electrode area-weighted particle-size distribution
        """
        R_dim = R * self.R_p_typ
        return self.f_a_dist_p_dimensional(R_dim) * self.R_p_typ


geometric_parameters = GeometricParameters()
