#
# Class for one- and two-dimensional potential pair current collector models
#
import pybamm
from .base_current_collector import BaseModel


class BasePotentialPair(BaseModel):
    """A submodel for Ohm's law plus conservation of current in the current collectors.
    For details on the potential pair formulation see :footcite:t:`Timms2021` and
    :footcite:t:`Marquis2020`.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    """

    def __init__(self, param):
        super().__init__(param)

        pybamm.citations.register("Timms2021")

    def get_fundamental_variables(self):
        phi_s_cn = pybamm.Variable(
            "Negative current collector potential [V]", domain="current collector"
        )

        variables = self._get_standard_negative_potential_variables(phi_s_cn)

        # TODO: grad not implemented for 2D yet
        i_cc = pybamm.Scalar(0)
        i_boundary_cc = pybamm.Variable(
            "Current collector current density [A.m-2]",
            domain="current collector",
            scale=self.param.Q / (self.param.A_cc * self.param.n_electrodes_parallel),
        )

        variables.update(self._get_standard_current_variables(i_cc, i_boundary_cc))

        return variables

    def set_algebraic(self, variables):
        phi_s_cn = variables["Negative current collector potential [V]"]
        phi_s_cp = variables["Positive current collector potential [V]"]
        i_boundary_cc = variables["Current collector current density [A.m-2]"]

        self.algebraic = {
            phi_s_cn: (self.param.n.sigma_cc * self.param.n.L_cc)
            * pybamm.laplacian(phi_s_cn)
            - pybamm.source(i_boundary_cc, phi_s_cn),
            i_boundary_cc: (self.param.p.sigma_cc * self.param.p.L_cc)
            * pybamm.laplacian(phi_s_cp)
            + pybamm.source(i_boundary_cc, phi_s_cp),
        }

    def set_initial_conditions(self, variables):
        phi_s_cn = variables["Negative current collector potential [V]"]
        i_boundary_cc = variables["Current collector current density [A.m-2]"]

        self.initial_conditions = {
            phi_s_cn: pybamm.Scalar(0),
            i_boundary_cc: pybamm.Scalar(0),
        }


class PotentialPair1plus1D(BasePotentialPair):
    """Base class for a 1+1D potential pair model."""

    def __init__(self, param):
        super().__init__(param)

    def set_boundary_conditions(self, variables):
        phi_s_cn = variables["Negative current collector potential [V]"]
        phi_s_cp = variables["Positive current collector potential [V]"]

        applied_current_density = variables["Total current density [A.m-2]"]
        total_current = applied_current_density * self.param.A_cc

        # In the 1+1D model, the behaviour is averaged over the y-direction, so the
        # effective tab area is the cell width multiplied by the current collector
        # thickness
        positive_tab_area = self.param.L_y * self.param.p.L_cc
        pos_tab_bc = -total_current / (self.param.p.sigma_cc * positive_tab_area)

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


class PotentialPair2plus1D(BasePotentialPair):
    """Base class for a 2+1D potential pair model"""

    def __init__(self, param):
        super().__init__(param)

    def set_boundary_conditions(self, variables):
        phi_s_cn = variables["Negative current collector potential [V]"]
        phi_s_cp = variables["Positive current collector potential [V]"]

        applied_current_density = variables["Total current density [A.m-2]"]
        total_current = applied_current_density * self.param.A_cc

        # Note: we divide by the *numerical* tab area so that the correct total
        # current is applied. That is, numerically integrating the current density
        # around the boundary gives the applied current exactly.
        positive_tab_area = pybamm.BoundaryIntegral(
            pybamm.PrimaryBroadcast(self.param.p.L_cc, "current collector"),
            region="positive tab",
        )

        # cc_area appears here due to choice of non-dimensionalisation
        pos_tab_bc = -total_current / (self.param.p.sigma_cc * positive_tab_area)

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
