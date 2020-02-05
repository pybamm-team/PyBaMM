#
# Class for one-dimensional current collectors in which the potential is held
# fixed and the current is determined from the I-V relationship used in the SPM(e)
#
import pybamm
from .base_current_collector import BaseModel


class BaseSetPotentialSingleParticle(BaseModel):
    """A submodel for current collectors which *doesn't* update the potentials
    during solve. This class uses the current-voltage relationship from the
    SPM(e) (see [1]_) to calculate the current.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    References
    ----------
    .. [1] SG Marquis, V Sulzer, R Timms, CP Please and SJ Chapman. “An asymptotic
           derivation of a single particle model with electrolyte”. In: arXiv preprint
           arXiv:1905.12553 (2019).


    **Extends:** :class:`pybamm.current_collector.BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):

        phi_s_cn = pybamm.standard_variables.phi_s_cn

        variables = self._get_standard_negative_potential_variables(phi_s_cn)

        # TO DO: grad not implemented for 2D yet
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

    def set_rhs(self, variables):
        phi_s_cn = variables["Negative current collector potential"]

        # Dummy equations so that PyBaMM doesn't change the potentials during solve
        # i.e. d_phi/d_t = 0. Potentials are set externally between steps.
        self.rhs = {phi_s_cn: pybamm.Scalar(0)}

    def set_algebraic(self, variables):
        ocp_p_av = variables["X-averaged positive electrode open circuit potential"]
        ocp_n_av = variables["X-averaged negative electrode open circuit potential"]
        eta_r_n_av = variables["X-averaged negative electrode reaction overpotential"]
        eta_r_p_av = variables["X-averaged positive electrode reaction overpotential"]
        eta_e_av = variables["X-averaged electrolyte overpotential"]
        delta_phi_s_n_av = variables["X-averaged negative electrode ohmic losses"]
        delta_phi_s_p_av = variables["X-averaged positive electrode ohmic losses"]

        i_boundary_cc = variables["Current collector current density"]
        v_boundary_cc = variables["Local voltage"]
        # The voltage-current expression from the SPM(e)
        local_voltage_expression = (
            ocp_p_av
            - ocp_n_av
            + eta_r_p_av
            - eta_r_n_av
            + eta_e_av
            + delta_phi_s_p_av
            - delta_phi_s_n_av
        )
        self.algebraic = {i_boundary_cc: v_boundary_cc - local_voltage_expression}

    def set_initial_conditions(self, variables):

        applied_current = variables["Total current density"]
        cc_area = self._get_effective_current_collector_area()
        phi_s_cn = variables["Negative current collector potential"]
        i_boundary_cc = variables["Current collector current density"]

        self.initial_conditions = {
            phi_s_cn: pybamm.Scalar(0),
            i_boundary_cc: applied_current / cc_area,
        }


class SetPotentialSingleParticle1plus1D(BaseSetPotentialSingleParticle):
    "Class for 1+1D set potential model"

    def __init__(self, param):
        super().__init__(param)

    def _get_effective_current_collector_area(self):
        "In the 1+1D models the current collector effectively has surface area l_z"
        return self.param.l_z


class SetPotentialSingleParticle2plus1D(BaseSetPotentialSingleParticle):
    "Class for 1+1D set potential model"

    def __init__(self, param):
        super().__init__(param)

    def _get_effective_current_collector_area(self):
        "Return the area of the current collector"
        return self.param.l_y * self.param.l_z
