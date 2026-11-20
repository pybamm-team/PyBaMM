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

    def build(self):
        phi_s_cn = pybamm.Variable(
            "Negative current collector potential [V]", domain="current collector"
        )
        # TODO: grad not implemented for 2D yet
        i_cc = pybamm.Scalar(0)
        i_boundary_cc = pybamm.Variable(
            "Current collector current density [A.m-2]",
            domain="current collector",
            scale=self.param.Q / self.param.A_cc,
        )
        self.variables.update(self._get_standard_negative_potential_variables(phi_s_cn))
        self.variables.update(self._get_standard_current_variables(i_cc, i_boundary_cc))

        phi_s_cp = self.coupled_variable(
            "Positive current collector potential [V]", domain="current collector"
        )
        self.algebraic = {
            phi_s_cn: (self.param.n.sigma_cc * self.param.n.L_cc)
            * pybamm.laplacian(phi_s_cn)
            - pybamm.source(i_boundary_cc, phi_s_cn),
            i_boundary_cc: (self.param.p.sigma_cc * self.param.p.L_cc)
            * pybamm.laplacian(phi_s_cp)
            + pybamm.source(i_boundary_cc, phi_s_cp),
        }
        self.initial_conditions = {
            phi_s_cn: pybamm.Scalar(0),
            i_boundary_cc: pybamm.Scalar(0),
        }

        # Only one electrode pair is modelled, so it carries the cell current divided
        # by the number of electrodes connected in parallel
        applied_current_density = self.coupled_variable("Total current density [A.m-2]")
        electrode_pair_current = (
            applied_current_density * self.param.L_y * self.param.L_z
        )
        # Boundary conditions need to be on the variables that go into the Laplacian,
        # even though phi_s_cp isn't a pybamm.Variable object
        self._set_boundary_conditions(phi_s_cn, phi_s_cp, electrode_pair_current)

    def _set_boundary_conditions(self, phi_s_cn, phi_s_cp, electrode_pair_current):
        """
        Set the boundary conditions on the current collector potentials.

        Parameters
        ----------
        phi_s_cn : :class:`pybamm.Symbol`
            The potential in the negative current collector.
        phi_s_cp : :class:`pybamm.Symbol`
            The potential in the positive current collector.
        electrode_pair_current : :class:`pybamm.Symbol`
            The current carried by the modelled electrode pair.
        """
        raise NotImplementedError


class PotentialPair1plus1D(BasePotentialPair):
    """Base class for a 1+1D potential pair model."""

    def __init__(self, param):
        super().__init__(param)

    def _set_boundary_conditions(self, phi_s_cn, phi_s_cp, electrode_pair_current):
        # The 1+1D model is averaged over y, so the effective tab area is the cell
        # width multiplied by the current collector thickness
        positive_tab_area = self.param.L_y * self.param.p.L_cc
        pos_tab_bc = -electrode_pair_current / (
            self.param.p.sigma_cc * positive_tab_area
        )

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

    def _set_boundary_conditions(self, phi_s_cn, phi_s_cp, electrode_pair_current):
        # Divide by the *numerical* tab area, so that integrating the current density
        # around the boundary gives the applied current exactly
        positive_tab_area = pybamm.BoundaryIntegral(
            pybamm.PrimaryBroadcast(self.param.p.L_cc, "current collector"),
            region="positive tab",
        )

        pos_tab_bc = -electrode_pair_current / (
            self.param.p.sigma_cc * positive_tab_area
        )

        # Tabs are projected onto the 2D domain, whose boundary is insulated except at
        # the tabs, so zero flux is set explicitly on the other electrode's tab.
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
