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

    def __init__(self, options=None):
        self.options = options
        self.n = DomainGeometricParameters("negative", self)
        self.s = DomainGeometricParameters("separator", self)
        self.p = DomainGeometricParameters("positive", self)
        self.domain_params = {
            "negative": self.n,
            "separator": self.s,
            "positive": self.p,
        }

        # Set parameters and scales
        self._set_dimensional_parameters()
        self._set_scales()
        self._set_dimensionless_parameters()

    def _set_dimensional_parameters(self):
        """Defines the dimensional parameters."""
        for domain in self.domain_params.values():
            domain._set_dimensional_parameters()

        # Macroscale geometry
        self.L_x = (
            self.n.L + self.s.L + self.p.L
        )  # Total distance between current collectors
        self.L = self.n.L_cc + self.L_x + self.p.L_cc  # Total cell thickness
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

    def _set_scales(self):
        """Define the scales used in the non-dimensionalisation scheme"""
        for domain in self.domain_params.values():
            domain._set_scales()

    def _set_dimensionless_parameters(self):
        """Defines the dimensionless parameters."""
        for domain in self.domain_params.values():
            domain._set_dimensionless_parameters()

        # Macroscale Geometry
        self.l_x = self.L_x / self.L_x
        self.l_Li = self.L_Li / self.L_x
        self.l_y = self.L_y / self.L_z
        self.l_z = self.L_z / self.L_z
        self.r_inner = self.r_inner_dimensional / self.r_outer_dimensional
        self.r_outer = self.r_outer_dimensional / self.r_outer_dimensional
        self.a_cc = self.l_y * self.l_z
        self.a_cooling = self.A_cooling / (self.L_z**2)
        self.v_cell = self.V_cell / (self.L_x * self.L_z**2)

        self.l = self.L / self.L_x
        self.delta = self.L_x / self.L_z  # Pouch cell aspect ratio


class DomainGeometricParameters(BaseParameters):
    def __init__(self, domain, main_param):
        self.domain = domain
        self.main_param = main_param

        if self.domain != "separator":
            self.prim = ParticleGeometricParameters(domain, "primary", main_param)
            self.sec = ParticleGeometricParameters(domain, "secondary", main_param)
            self.phase_params = {"primary": self.prim, "secondary": self.sec}
        else:
            self.phase_params = {}

    def _set_dimensional_parameters(self):
        """Defines the dimensional parameters."""
        for phase in self.phase_params.values():
            phase._set_dimensional_parameters()

        if self.domain == "separator":
            self.L = pybamm.Parameter("Separator thickness [m]")
            self.b_e = pybamm.Parameter("Separator Bruggeman coefficient (electrolyte)")
            return

        Domain = self.domain.capitalize()

        # Macroscale geometry
        self.L_cc = pybamm.Parameter(f"{Domain} current collector thickness [m]")
        self.L = pybamm.Parameter(f"{Domain} electrode thickness [m]")

        # Tab geometry (for pouch cells)
        self.L_tab = pybamm.Parameter(f"{Domain} tab width [m]")
        self.Centre_y_tab = pybamm.Parameter(f"{Domain} tab centre y-coordinate [m]")
        self.Centre_z_tab = pybamm.Parameter(f"{Domain} tab centre z-coordinate [m]")
        self.A_tab = self.L_tab * self.L_cc  # Area of tab

        # Microscale geometry
        self.b_e = pybamm.Parameter(
            f"{Domain} electrode Bruggeman coefficient (electrolyte)"
        )
        self.b_s = pybamm.Parameter(
            f"{Domain} electrode Bruggeman coefficient (electrode)"
        )

    def _set_scales(self):
        """Define the scales used in the non-dimensionalisation scheme"""
        for phase in self.phase_params.values():
            phase._set_scales()

    def _set_dimensionless_parameters(self):
        """Defines the dimensionless parameters."""
        for phase in self.phase_params.values():
            phase._set_dimensionless_parameters()
        main = self.main_param

        # Macroscale Geometry
        self.l = self.L / main.L_x
        if self.domain == "separator":
            return

        self.l_cc = self.L_cc / main.L_x

        # Tab geometry (for pouch cells)
        self.l_tab = self.L_tab / main.L_z
        self.centre_y_tab = self.Centre_y_tab / main.L_z
        self.centre_z_tab = self.Centre_z_tab / main.L_z


class ParticleGeometricParameters(BaseParameters):
    def __init__(self, domain, phase, main_param):
        self.domain = domain
        self.phase = phase
        self.main_param = main_param
        self.set_phase_name()

    def _set_dimensional_parameters(self):
        """Defines the dimensional parameters."""
        Domain = self.domain.capitalize()
        pref = self.phase_prefactor

        # Microscale geometry
        # Note: for li-ion cells, the definition of the surface area to
        # volume ratio is overwritten in lithium_ion_parameters.py to be computed
        # based on the assumed particle shape
        self.a_dim = pybamm.Parameter(
            f"{pref}{Domain} electrode surface area to volume ratio [m-1]"
        )

        # Particle-size distribution geometry
        self.R_min_dim = pybamm.Parameter(f"{pref}{Domain} minimum particle radius [m]")
        self.R_max_dim = pybamm.Parameter(f"{pref}{Domain} maximum particle radius [m]")
        self.sd_a_dim = pybamm.Parameter(
            f"{pref}{Domain} area-weighted particle-size standard deviation [m]"
        )

    @property
    def R_dimensional(self):
        if self.domain == "negative":
            x = pybamm.standard_spatial_vars.x_n
        elif self.domain == "positive":
            x = pybamm.standard_spatial_vars.x_p

        inputs = {"Through-cell distance (x) [m]": x * self.main_param.L_x}
        Domain = self.domain.capitalize()
        return pybamm.FunctionParameter(
            f"{self.phase_prefactor}{Domain} particle radius [m]", inputs
        )

    def f_a_dist_dimensional(self, R):
        """
        Dimensional electrode area-weighted particle-size distribution
        """
        Domain = self.domain.capitalize()
        inputs = {f"{self.phase_prefactor}{Domain} particle-size variable [m]": R}
        return pybamm.FunctionParameter(
            f"{self.phase_prefactor}{Domain} "
            "area-weighted particle-size distribution [m-1]",
            inputs,
        )

    def _set_scales(self):
        """Define the scales used in the non-dimensionalisation scheme"""
        # Microscale geometry
        # Note: these scales are necessary here to non-dimensionalise the
        # particle size distributions.
        self.R_typ = pybamm.xyz_average(self.R_dimensional)

    def _set_dimensionless_parameters(self):
        """Defines the dimensionless parameters."""
        # Particle-size distribution geometry
        self.R_min = self.R_min_dim / self.R_typ
        self.R_max = self.R_max_dim / self.R_typ
        self.sd_a = self.sd_a_dim / self.R_typ

        # Particle radius
        self.R = self.R_dimensional / self.R_typ

    def f_a_dist(self, R):
        """
        Dimensionless electrode area-weighted particle-size distribution
        """
        R_dim = R * self.R_typ
        return self.f_a_dist_dimensional(R_dim) * self.R_typ


geometric_parameters = GeometricParameters()
