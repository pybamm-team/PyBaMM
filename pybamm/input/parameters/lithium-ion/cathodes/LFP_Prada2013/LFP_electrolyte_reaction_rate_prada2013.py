from pybamm import exp, constants


def LFP_electrolyte_reaction_rate_prada2013(c_e, c_s_surf,T): # , 1 
    """
    Reaction rate for Butler-Volmer reactions
    References
    ----------
Representative Volume Element Model of Lithium-ion Battery
Electrodes Based on X-ray Nano-tomography
Ali Ghorbani Kashkooli
SAFE TEMPERATURE CONTROL OF LITHIUM ION BATTERY
SYSTEMS FOR HIGH PERFORMANCE AND LONG LIFE
Mayank Garg
    ----------
    T: :class:`pybamm.Symbol`
        Dimensional temperature
    Returns
    -------
    :class:`pybamm.Symbol`
        Reaction rate
    """

    m_ref = 6 * 10 ** (-7)  # (A/m2)(mol/m3)**1.5 - includes ref concentrations
    E_r = 39570
    arrhenius = exp(E_r / constants.R * (1 / 298.15 - 1 / T))
    c_p_max = Parameter("Maximum concentration in positive electrode [mol.m-3]")


    return (
        m_ref * arrhenius * c_e ** 0.5 * c_s_surf ** 0.5 * (c_p_max - c_s_surf) ** 0.5
    )

