#
# Class for one-dimensional (x-direction) thermal submodel
#
import pybamm

from .base_thermal import BaseThermal


class OneDimensionalX(BaseThermal):
    """
    Class for one-dimensional (x-direction) thermal submodel.
    Note: this model assumes infinitely large electrical and thermal conductivity
    in the current collectors, so that the contribution to the Ohmic heating
    from the current collectors is zero and the boundary conditions are applied
    at the edges of the electrodes (at x=0 and x=1, in non-dimensional coordinates).
    For more information see [1]_ and [2]_.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.

    References
    ----------
    .. [1] R Timms, SG Marquis, V Sulzer, CP Please and SJ Chapman. “Asymptotic
           Reduction of a Lithium-ion Pouch Cell Model”. SIAM Journal on Applied
           Mathematics, 81(3), 765--788, 2021
    .. [2] SG Marquis, R Timms, V Sulzer, CP Please and SJ Chapman. “A Suite of
           Reduced-Order Models of a Single-Layer Lithium-ion Pouch Cell”. Journal
           of The Electrochemical Society, 167(14):140513, 2020
    """

    def __init__(self, param, options=None):
        super().__init__(param, options=options)
        pybamm.citations.register("Timms2021")

    def get_fundamental_variables(self):
        T_dict = {}
        for domain in ["negative electrode", "separator", "positive electrode"]:
            Domain = domain.capitalize()
            T_k = pybamm.Variable(
                f"{Domain} temperature [K]",
                domain=domain,
                auxiliary_domains={"secondary": "current collector"},
                scale=self.param.T_ref,
            )
            T_dict[domain] = T_k

        T = pybamm.concatenation(*T_dict.values())
        T_cn = pybamm.boundary_value(T_dict["negative electrode"], "left")
        T_cp = pybamm.boundary_value(T_dict["positive electrode"], "right")
        T_x_av = self._x_average(T, T_cn, T_cp)
        T_vol_av = self._yz_average(T_x_av)
        T_dict.update(
            {
                "negative current collector": T_cn,
                "positive current collector": T_cp,
                "x-averaged cell": T_x_av,
                "volume-averaged cell": T_vol_av,
            }
        )

        variables = self._get_standard_fundamental_variables(T_dict)
        return variables

    def get_coupled_variables(self, variables):
        variables.update(self._get_standard_coupled_variables(variables))
        return variables

    def set_rhs(self, variables):
        T = variables["Cell temperature [K]"]
        T_n = variables["Negative electrode temperature [K]"]
        T_s = variables["Separator temperature [K]"]
        T_p = variables["Positive electrode temperature [K]"]

        Q = variables["Total heating [W.m-3]"]

        # Define volumetric heat capacity
        rho_c_p = pybamm.concatenation(
            self.param.n.rho_c_p(T_n),
            self.param.s.rho_c_p(T_s),
            self.param.p.rho_c_p(T_p),
        )

        # Define thermal conductivity
        lambda_ = pybamm.concatenation(
            self.param.n.lambda_(T_n),
            self.param.s.lambda_(T_s),
            self.param.p.lambda_(T_p),
        )

        # Fourier's law for heat flux
        q = -lambda_ * pybamm.grad(T)

        # N.B only y-z surface cooling is implemented for this model
        self.rhs = {T: (-pybamm.div(q) + Q) / rho_c_p}

    def set_boundary_conditions(self, variables):
        T = variables["Cell temperature [K]"]
        T_n_left = pybamm.boundary_value(T, "left")
        T_p_right = pybamm.boundary_value(T, "right")
        T_amb = variables["Ambient temperature [K]"]

        # N.B only y-z surface cooling is implemented for this thermal model.
        # Tab and edge cooling is not accounted for.
        self.boundary_conditions = {
            T: {
                "left": (
                    self.param.n.h_cc
                    * (T_n_left - T_amb)
                    / self.param.n.lambda_(T_n_left),
                    "Neumann",
                ),
                "right": (
                    -self.param.p.h_cc
                    * (T_p_right - T_amb)
                    / self.param.p.lambda_(T_p_right),
                    "Neumann",
                ),
            }
        }

    def set_initial_conditions(self, variables):
        T = variables["Cell temperature [K]"]
        self.initial_conditions = {T: self.param.T_init}
