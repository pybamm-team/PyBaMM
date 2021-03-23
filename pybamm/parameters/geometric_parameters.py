#
# Geometric Parameters
#
import pybamm


class GeometricParameters:
    """
    Standard geometric parameters

    Layout:
        1. Dimensional Parameters
        2. Dimensionless Parameters
    """

    def __init__(self):

        # Set parameters
        self._set_dimensional_parameters()
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
        self.L_y = pybamm.Parameter("Electrode width [m]")
        self.L_z = pybamm.Parameter("Electrode height [m]")
        self.L_Li = pybamm.Parameter("Lithium counter electrode thickness [m]")
        self.A_cc = self.L_y * self.L_z  # Area of current collector
        self.A_cooling = pybamm.Parameter("Cell cooling surface area [m2]")
        self.V_cell = pybamm.Parameter("Cell volume [m3]")

        # Tab geometry
        self.L_tab_n = pybamm.Parameter("Negative tab width [m]")
        self.Centre_y_tab_n = pybamm.Parameter("Negative tab centre y-coordinate [m]")
        self.Centre_z_tab_n = pybamm.Parameter("Negative tab centre z-coordinate [m]")
        self.L_tab_p = pybamm.Parameter("Positive tab width [m]")
        self.Centre_y_tab_p = pybamm.Parameter("Positive tab centre y-coordinate [m]")
        self.Centre_z_tab_p = pybamm.Parameter("Positive tab centre z-coordinate [m]")
        self.A_tab_n = self.L_tab_n * self.L_cn  # Area of negative tab
        self.A_tab_p = self.L_tab_p * self.L_cp  # Area of negative tab

        # Microscale geometry
        # Note: parameters related to the particles in li-ion cells are defined
        # in lithium_ion_parameters.py. The definition of the surface area to
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

    def _set_dimensionless_parameters(self):
        """Defines the dimensionless parameters."""

        # Macroscale Geometry
        self.l_cn = self.L_cn / self.L_x
        self.l_n = self.L_n / self.L_x
        self.l_s = self.L_s / self.L_x
        self.l_p = self.L_p / self.L_x
        self.l_cp = self.L_cp / self.L_x
        self.l_x = self.L_x / self.L_x
        self.l_y = self.L_y / self.L_z
        self.l_z = self.L_z / self.L_z
        self.l_Li = self.L_Li / self.L_x
        self.a_cc = self.l_y * self.l_z
        self.a_cooling = self.A_cooling / (self.L_z ** 2)
        self.v_cell = self.V_cell / (self.L_x * self.L_z ** 2)

        self.l = self.L / self.L_x
        self.delta = self.L_x / self.L_z  # Aspect ratio

        # Tab geometry
        self.l_tab_n = self.L_tab_n / self.L_z
        self.centre_y_tab_n = self.Centre_y_tab_n / self.L_z
        self.centre_z_tab_n = self.Centre_z_tab_n / self.L_z
        self.l_tab_p = self.L_tab_p / self.L_z
        self.centre_y_tab_p = self.Centre_y_tab_p / self.L_z
        self.centre_z_tab_p = self.Centre_z_tab_p / self.L_z


geometric_parameters = GeometricParameters()
