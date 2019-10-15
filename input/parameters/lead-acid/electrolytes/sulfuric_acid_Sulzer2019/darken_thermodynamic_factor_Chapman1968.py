#
# Darken thermodynamic factor of sulfuric acid
#


def darken_thermodynamic_factor_Chapman1968(c_e):
    """
    Dimensional Darken thermodynamic factor of sulfuric acid, from data in
    [1, 2]_, as a function of the electrolyte concentration c_e [mol.m-3].

    References
    ----------
    .. [1] TW Chapman and J Newman. Compilation of selected thermodynamic and transport
           properties of binary electrolytes in aqueous solution. Technical report,
           California Univ., Berkeley. Lawrence Radiation Lab., 1968.
    .. [2] KS Pitzer, RN Roy, and LF Silvester. Thermodynamics of electrolytes. 7.
           sulfuric acid. Journal of the American Chemical Society, 99(15):4930–4936,
           1977.

    """
    return 0.49 + 4.1e-4 * c_e
