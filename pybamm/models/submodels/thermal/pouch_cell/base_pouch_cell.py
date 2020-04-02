#
# Base class for one- and two-dimensional thermal submodels for use in the "N+1D"
# pouch cell models
#
import pybamm

from ..base_thermal import BaseThermal


class BasePouchCell(BaseThermal):
    """Base class for  one- and two-dimensional thermal submodels for use in the
    "N+1D" pouch cell models. The models are averaged in the x-direction and
    are therefore referred to as 'x-lumped'. For more information see [1]_ and [2]_.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    cc_dimension: int
        The dimension of the current collectors. Can be 1 or 2.

    References
    ----------
    .. [1] R Timms, SG Marquis, V Sulzer, CP Please and SJ Chapman. “Asymptotic
           Reduction of a Lithium-ion Pouch Cell Model”. In preparation, 2020.
    .. [2] SG Marquis, R Timms, V Sulzer, CP Please and SJ Chapman. “A Suite of
           Reduced-Order Models of a Single-Layer Lithium-ion Pouch Cell”. In
           preparation, 2020.

    **Extends:** :class:`pybamm.thermal.BaseModel`
    """

    def __init__(self, param, cc_dimension):
        super().__init__(param, cc_dimension=cc_dimension)

    def get_fundamental_variables(self):

        T_x_av = pybamm.standard_variables.T_av
        T_vol_av = self._yz_average(T_x_av)

        T_cn = T_x_av
        T_n = pybamm.PrimaryBroadcast(T_x_av, "negative electrode")
        T_s = pybamm.PrimaryBroadcast(T_x_av, "separator")
        T_p = pybamm.PrimaryBroadcast(T_x_av, "positive electrode")
        T_cp = T_x_av

        variables = self._get_standard_fundamental_variables(
            T_cn, T_n, T_s, T_p, T_cp, T_x_av, T_vol_av
        )

        return variables

    def get_coupled_variables(self, variables):
        variables.update(self._get_standard_coupled_variables(variables))
        return variables
