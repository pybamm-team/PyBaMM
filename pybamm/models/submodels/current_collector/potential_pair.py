#
# Class for one- and two-dimensional potential pair current collector models
#
import pybamm
from .base_current_collector import BaseModel


class BasePotentialPair(BaseModel):
    """A submodel for Ohm's law plus conservation of current in the current collectors.
    For details on the potential pair formulation see [1]_ and [2]_.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    References
    ----------
    .. [1] R Timms, SG Marquis, V Sulzer, CP Please and SJ Chapman. “Asymptotic
           Reduction of a Lithium-ion Pouch Cell Model”. Submitted, 2020.
    .. [2] SG Marquis, R Timms, V Sulzer, CP Please and SJ Chapman. “A Suite of
           Reduced-Order Models of a Single-Layer Lithium-ion Pouch Cell”. In
           preparation, 2020.

    **Extends:** :class:`pybamm.current_collector.BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

        pybamm.citations.register("timms2020")

    def get_fundamental_variables(self):

        phi_s_cn = pybamm.standard_variables.phi_s_cn

        variables = self._get_standard_negative_potential_variables(phi_s_cn)

        # TODO: grad not implemented for 2D yet
        i_cc = pybamm.Scalar(0)
        i_boundary_cc = pybamm.standard_variables.i_boundary_cc

        variables.update(self._get_standard_current_variables(i_cc, i_boundary_cc))
        # Hack to get the leading-order current collector current density
        # Note that this should be different from the actual (composite) current
        # collector current density for 2+1D models, but not sure how to implement this
        # using current structure of lithium-ion models
        variables["Leading-order current collector current density"] = variables[
            "Current collector current density"
        ]

        return variables

    def set_algebraic(self, variables):

        param = self.param

        phi_s_cn = variables["Negative current collector potential"]
        phi_s_cp = variables["Positive current collector potential"]
        i_boundary_cc = variables["Current collector current density"]

        self.algebraic = {
            phi_s_cn: (param.sigma_cn * param.delta ** 2 * param.l_cn)
            * pybamm.laplacian(phi_s_cn)
            - pybamm.source(i_boundary_cc, phi_s_cn),
            i_boundary_cc: (param.sigma_cp * param.delta ** 2 * param.l_cp)
            * pybamm.laplacian(phi_s_cp)
            + pybamm.source(i_boundary_cc, phi_s_cp),
        }

    def set_initial_conditions(self, variables):

        applied_current = self.param.current_with_time
        phi_s_cn = variables["Negative current collector potential"]
        i_boundary_cc = variables["Current collector current density"]

        self.initial_conditions = {
            phi_s_cn: pybamm.Scalar(0),
            i_boundary_cc: applied_current,
        }


class PotentialPair1plus1D(BasePotentialPair):
    "Base class for a 1+1D potential pair model."

    def __init__(self, param):
        super().__init__(param)

    def set_boundary_conditions(self, variables):

        phi_s_cn = variables["Negative current collector potential"]
        phi_s_cp = variables["Positive current collector potential"]

        param = self.param
        applied_current = variables["Total current density"]
        cc_area = self._get_effective_current_collector_area()

        # cc_area appears here due to choice of non-dimensionalisation
        pos_tab_bc = (
            -applied_current
            * cc_area
            / (param.sigma_cp * param.delta ** 2 * param.l_cp)
        )

        # Boundary condition needs to be on the variables that go into the Laplacian,
        # even though phi_s_cp isn't a pybamm.Variable object
        self.boundary_conditions = {
            phi_s_cn: {
                "negative tab": (pybamm.Scalar(0), "Dirichlet"),
                "no tab": (pybamm.Scalar(0), "Neumann"),
            },
            phi_s_cp: {
                "no tab": (pybamm.Scalar(0), "Neumann"),
                "positive tab": (pos_tab_bc, "Neumann"),
            },
        }

    def _get_effective_current_collector_area(self):
        "In the 1+1D models the current collector effectively has surface area l_z"
        return self.param.l_z


class PotentialPair2plus1D(BasePotentialPair):
    "Base class for a 2+1D potential pair model"

    def __init__(self, param):
        super().__init__(param)

    def set_boundary_conditions(self, variables):

        phi_s_cn = variables["Negative current collector potential"]
        phi_s_cp = variables["Positive current collector potential"]

        param = self.param
        applied_current = variables["Total current density"]
        cc_area = self._get_effective_current_collector_area()

        # Note: we divide by the *numerical* tab area so that the correct total
        # current is applied. That is, numerically integrating the current density
        # around the boundary gives the applied current exactly.

        positive_tab_area = pybamm.BoundaryIntegral(
            pybamm.PrimaryBroadcast(param.l_cp, "current collector"),
            region="positive tab",
        )

        # cc_area appears here due to choice of non-dimensionalisation
        pos_tab_bc = (
            -applied_current
            * cc_area
            / (param.sigma_cp * param.delta ** 2 * positive_tab_area)
        )

        # Boundary condition needs to be on the variables that go into the Laplacian,
        # even though phi_s_cp isn't a pybamm.Variable object
        # In the 2+1D model, the equations for the current collector potentials
        # are solved on a 2D domain and the regions "negative tab" and "positive tab"
        # are the projections of the tabs onto this 2D domain.
        # In the 2D formulation it is assumed that no flux boundary conditions
        # are applied everywhere apart from the tabs.
        # The reference potential is taken to be zero on the negative tab,
        # giving the zero Dirichlet condition on phi_s_cn. Elsewhere, the boundary
        # is insulated, giving no flux conditions on phi_s_cn. This is automatically
        # applied everywhere, apart from the region corresponding to the projection
        # of the positive tab, so we need to explitly apply a zero-flux boundary
        # condition on the region "positive tab" for phi_s_cn.
        # A current is drawn from the positive tab, giving the non-zero Neumann
        # boundary condition on phi_s_cp at "positive tab". Elsewhere, the boundary is
        # insulated, so, as with phi_s_cn, we need to explicitly give the zero-flux
        # condition on the region "negative tab" for phi_s_cp.
        self.boundary_conditions = {
            phi_s_cn: {
                "negative tab": (pybamm.Scalar(0), "Dirichlet"),
                "positive tab": (pybamm.Scalar(0), "Neumann"),
            },
            phi_s_cp: {
                "negative tab": (pybamm.Scalar(0), "Neumann"),
                "positive tab": (pos_tab_bc, "Neumann"),
            },
        }

    def _get_effective_current_collector_area(self):
        "Return the area of the current collector"
        return self.param.l_y * self.param.l_z
