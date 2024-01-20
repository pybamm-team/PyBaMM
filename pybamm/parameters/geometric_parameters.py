#
# Geometric Parameters
#
import pybamm
from .base_parameters import BaseParameters


class GeometricParameters(BaseParameters):
    """
    Standard geometric parameters
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
        self._set_parameters()

    def _set_parameters(self):
        """Defines the dimensional parameters."""
        for domain in self.domain_params.values():
            domain._set_parameters()

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
        self.r_inner = pybamm.Parameter("Inner cell radius [m]")
        self.r_outer = pybamm.Parameter("Outer cell radius [m]")
        self.A_cc = self.L_y * self.L_z  # Current collector cross sectional area

        # Cell surface area and volume (for thermal models only)
        cell_geometry = self.options.get("cell geometry", None)
        if cell_geometry == "pouch":
            # assuming a single-layer pouch cell for now, see
            # https://github.com/pybamm-team/PyBaMM/issues/1777
            self.A_cooling = 2 * (
                self.L_y * self.L_z + self.L_z * self.L + self.L_y * self.L
            )
            self.V_cell = self.L_y * self.L_z * self.L
        else:
            self.A_cooling = pybamm.Parameter("Cell cooling surface area [m2]")
            self.V_cell = pybamm.Parameter("Cell volume [m3]")


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

    def _set_parameters(self):
        """Defines the dimensional parameters."""
        for phase in self.phase_params.values():
            phase._set_parameters()

        if self.domain == "separator":
            self.L = pybamm.Parameter("Separator thickness [m]")
            self.b_e = pybamm.Parameter("Separator Bruggeman coefficient (electrolyte)")
            self.tau_e = pybamm.Parameter("Separator tortuosity factor (electrolyte)")
            return

        Domain = self.domain.capitalize()

        # Macroscale geometry
        self.L_cc = pybamm.Parameter(f"{Domain} current collector thickness [m]")
        self.L = pybamm.Parameter(f"{Domain} electrode thickness [m]")

        # Tab geometry (for pouch cells)
        self.L_tab = pybamm.Parameter(f"{Domain} tab width [m]")
        self.centre_y_tab = pybamm.Parameter(f"{Domain} tab centre y-coordinate [m]")
        self.centre_z_tab = pybamm.Parameter(f"{Domain} tab centre z-coordinate [m]")
        self.A_tab = self.L_tab * self.L_cc  # Area of tab

        # Microscale geometry
        self.b_e = pybamm.Parameter(
            f"{Domain} electrode Bruggeman coefficient (electrolyte)"
        )
        self.b_s = pybamm.Parameter(
            f"{Domain} electrode Bruggeman coefficient (electrode)"
        )
        self.tau_e = pybamm.Parameter(
            f"{Domain} electrode tortuosity factor (electrolyte)"
        )
        self.tau_s = pybamm.Parameter(
            f"{Domain} electrode tortuosity factor (electrode)"
        )


class ParticleGeometricParameters(BaseParameters):
    def __init__(self, domain, phase, main_param):
        self.domain = domain
        self.phase = phase
        self.main_param = main_param
        self.set_phase_name()

    def _set_parameters(self):
        """Defines the dimensional parameters."""
        Domain = self.domain.capitalize()
        pref = self.phase_prefactor

        # Microscale geometry
        # Note: for li-ion cells, the definition of the surface area to
        # volume ratio is overwritten in lithium_ion_parameters.py to be computed
        # based on the assumed particle shape
        self.a = pybamm.Parameter(
            f"{pref}{Domain} electrode surface area to volume ratio [m-1]"
        )

        # Particle-size distribution geometry
        self.R_min = pybamm.Parameter(f"{pref}{Domain} minimum particle radius [m]")
        self.R_max = pybamm.Parameter(f"{pref}{Domain} maximum particle radius [m]")

    @property
    def R_typ(self):
        # evaluate the typical particle radius
        # in the middle of the electrode
        main = self.main_param
        if self.domain == "negative":
            x = main.n.L / 2
        elif self.domain == "positive":
            x = main.n.L + main.s.L + main.p.L / 2

        inputs = {"Through-cell distance (x) [m]": x}
        Domain = self.domain.capitalize()
        return pybamm.FunctionParameter(
            f"{self.phase_prefactor}{Domain} particle radius [m]", inputs
        )

    @property
    def R(self):
        if self.domain == "negative":
            x = pybamm.standard_spatial_vars.x_n
        elif self.domain == "positive":
            x = pybamm.standard_spatial_vars.x_p

        inputs = {"Through-cell distance (x) [m]": x}
        Domain = self.domain.capitalize()
        return pybamm.FunctionParameter(
            f"{self.phase_prefactor}{Domain} particle radius [m]", inputs
        )

    def f_a_dist(self, R):
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


geometric_parameters = GeometricParameters()
