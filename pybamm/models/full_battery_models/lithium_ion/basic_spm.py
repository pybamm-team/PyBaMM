#
# Basic Single Particle Model (SPM)
#
import pybamm
from .base_lithium_ion_model import BaseModel


class BasicSPM(BaseModel):
    """ingle Particle Model (SPM) model of a lithium-ion battery, from [1]_.

    This class differs from the :class:`pybamm.lithium_ion.SPM` model class in that it
    shows the whole model in a single class. This comes at the cost of flexibility, and
    in general the main SPM class should be used instead.

    Parameters
    ----------
    name : str, optional
        The name of the model.

    References
    ----------
    .. [1] SG Marquis, V Sulzer, R Timms, CP Please and SJ Chapman. “An asymptotic
           derivation of a single particle model with electrolyte”. In: arXiv preprint
           arXiv:1905.12553 (2019).


    **Extends:** :class:`pybamm.lithium_ion.BaseModel`
    """

    def __init__(self, name="Doyle-Fuller-Newman model"):
        super().__init__({}, name)
        param = self.param

        ######################
        # Variables
        ######################
        c_s_n = pybamm.Variable(
            "X-averaged negative particle concentration", domain="negative particle",
        )
        c_s_p = pybamm.Variable(
            "X-averaged positive particle concentration", domain="positive particle",
        )

        # Constant temperature
        T = param.T_init

        ######################
        # Other set-up
        ######################

        # Current density
        i_cell = self.param.current_with_time
        j_n = i_cell / param.l_n
        j_p = -i_cell / param.l_p

        ######################
        # State of Charge
        ######################
        Q = pybamm.Variable("Discharge capacity [A.h]")
        I = param.dimensional_current_with_time
        self.rhs[Q] = I * param.timescale / 3600
        self.initial_conditions[Q] = pybamm.Scalar(0)

        ######################
        # Particles
        ######################

        N_s_n = -param.D_n(c_s_n, T) * pybamm.grad(c_s_n)
        N_s_p = -param.D_p(c_s_p, T) * pybamm.grad(c_s_p)
        self.rhs[c_s_n] = -(1 / param.C_n) * pybamm.div(N_s_n)
        self.rhs[c_s_p] = -(1 / param.C_p) * pybamm.div(N_s_p)
        self.boundary_conditions[c_s_n] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (-param.C_n * j_n / param.a_n, "Neumann"),
        }
        self.boundary_conditions[c_s_p] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (-param.C_p * j_p / param.a_p / self.param.gamma_p, "Neumann"),
        }
        self.initial_conditions[c_s_n] = param.c_n_init
        self.initial_conditions[c_s_p] = param.c_p_init
        c_s_surf_n = pybamm.surf(c_s_n)
        c_s_surf_p = pybamm.surf(c_s_p)
        self.events.update(
            {
                "Minimum negative particle surface concentration": (
                    pybamm.min(c_s_surf_n) - 0.01
                ),
                "Maximum negative particle surface concentration": (1 - 0.01)
                - pybamm.max(c_s_surf_n),
                "Minimum positive particle surface concentration": (
                    pybamm.min(c_s_surf_p) - 0.01
                ),
                "Maximum positive particle surface concentration": (1 - 0.01)
                - pybamm.max(c_s_surf_p),
            }
        )

        ######################
        # (Some) variables
        ######################
        # Interfacial reactions
        j0_n = (
            param.m_n(T)
            / param.C_r_n
            * 1 ** (1 / 2)
            * c_s_surf_n ** (1 / 2)
            * (1 - c_s_surf_n) ** (1 / 2)
        )
        j0_p = (
            param.gamma_p
            * param.m_p(T)
            / param.C_r_p
            * 1 ** (1 / 2)
            * c_s_surf_p ** (1 / 2)
            * (1 - c_s_surf_p) ** (1 / 2)
        )
        eta_n = (2 / param.ne_n) * pybamm.arcsinh(j_n / (2 * j0_n))
        eta_p = (2 / param.ne_p) * pybamm.arcsinh(j_p / (2 * j0_p))
        phi_s_n = 0
        phi_e = -eta_n - param.U_n(c_s_surf_n, T)
        phi_s_p = eta_p + phi_e + param.U_p(c_s_surf_p, T)
        V = phi_s_p

        whole_cell = ["negative electrode", "separator", "positive electrode"]
        self.variables = {
            "Negative particle surface concentration": pybamm.PrimaryBroadcast(
                c_s_surf_n, "negative electrode"
            ),
            "Electrolyte concentration": pybamm.PrimaryBroadcast(1, whole_cell),
            "Positive particle surface concentration": pybamm.PrimaryBroadcast(
                c_s_surf_p, "positive electrode"
            ),
            "Current [A]": I,
            "Negative electrode potential": pybamm.PrimaryBroadcast(
                phi_s_n, "negative electrode"
            ),
            "Electrolyte potential": pybamm.PrimaryBroadcast(phi_e, whole_cell),
            "Positive electrode potential": pybamm.PrimaryBroadcast(
                phi_s_p, "positive electrode"
            ),
            "Terminal voltage": V,
        }
        self.events["Minimum voltage"] = V - param.voltage_low_cut
        self.events["Maximum voltage"] = V - param.voltage_high_cut

    @property
    def default_geometry(self):
        return pybamm.Geometry("1D macro", "1D micro")
