#
# Sulfuric acid viscosity
#
from pybamm import Scalar


def viscosity_Chapman1968(c_e):
    """
    Dimensional viscosity of sulfuric acid [kg.m-1.s-1], from data in [1]_, as a
    function of the electrolyte concentration c_e [mol.m-3].

    References
    ----------
    .. [1] TW Chapman and J Newman. Compilation of selected thermodynamic and transport
           properties of binary electrolytes in aqueous solution. Technical report,
           California Univ., Berkeley. Lawrence Radiation Lab., 1968.

    """
    c_e /= Scalar(1, "[mol.m-3]")
    return (0.89e-3 + 1.11e-7 * c_e + 3.29e-11 * c_e ** 2) * Scalar(1, "[kg.m-1.s-1]")
