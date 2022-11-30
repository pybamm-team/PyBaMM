#
# Class for one- and two-dimensional potential pair "quite conductive"
# current collector models
#
import pybamm
from .potential_pair import (
    BasePotentialPair,
    PotentialPair1plus1D,
    PotentialPair2plus1D,
)


class BaseQuiteConductivePotentialPair(BasePotentialPair):
    """A submodel for Ohm's law plus conservation of current in the current collectors,
    in the limit of quite conductive electrodes.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.current_collector.BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

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
            scale=param.Q / (param.A_cc * param.n_electrodes_parallel),
        )

        variables.update(self._get_standard_current_variables(i_cc, i_boundary_cc))

        # Lagrange multiplier for the composite current (enforce average)
        c = pybamm.Variable("Lagrange multiplier")
        variables.update({"Lagrange multiplier": c})

        return variables

    def set_algebraic(self, variables):

        param = self.param
        applied_current = variables["Total current density [A.m-2]"]
        cc_area = self._get_effective_current_collector_area()
        z = pybamm.standard_spatial_vars.z

        phi_s_cn = variables["Negative current collector potential [V]"]
        phi_s_cp = variables["Positive current collector potential [V]"]
        i_boundary_cc = variables["Current collector current density [A.m-2]"]
        i_boundary_cc_0 = variables[
            "Leading-order current collector current density [A.m-2]"
        ]
        c = variables["Lagrange multiplier"]

        # Note that the second argument of 'source' must be the same as the argument
        # in the laplacian (the variable to which the boundary conditions are applied)
        self.algebraic = {
            phi_s_cn: (param.n.sigma_cc * param.n.L_cc) * pybamm.laplacian(phi_s_cn)
            - pybamm.source(i_boundary_cc_0, phi_s_cn),
            i_boundary_cc: (param.p.sigma_cc * param.p.L_cc)
            * pybamm.laplacian(phi_s_cp)
            + pybamm.source(i_boundary_cc_0, phi_s_cp)
            + c * pybamm.PrimaryBroadcast(cc_area, "current collector"),
            c: pybamm.Integral(i_boundary_cc, z) - applied_current / cc_area,
        }

    def set_initial_conditions(self, variables):

        param = self.param
        applied_current = param.current_with_time
        cc_area = self._get_effective_current_collector_area()
        phi_s_cn = variables["Negative current collector potential [V]"]
        i_boundary_cc = variables["Current collector current density [A.m-2]"]
        c = variables["Lagrange multiplier"]

        self.initial_conditions = {
            phi_s_cn: pybamm.Scalar(0),
            i_boundary_cc: applied_current / cc_area,
            c: pybamm.Scalar(0),
        }


class QuiteConductivePotentialPair1plus1D(
    BaseQuiteConductivePotentialPair, PotentialPair1plus1D
):
    def __init__(self, param):
        super().__init__(param)


class QuiteConductivePotentialPair2plus1D(
    BaseQuiteConductivePotentialPair, PotentialPair2plus1D
):
    def __init__(self, param):
        super().__init__(param)
