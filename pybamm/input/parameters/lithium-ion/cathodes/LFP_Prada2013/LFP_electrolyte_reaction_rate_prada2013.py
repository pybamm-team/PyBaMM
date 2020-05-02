from pybamm import exp, constants


def LFP_electrolyte_reaction_rate_prada2013(T):
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

    k_ref = 2.5e-13

    # multiply by Faraday's constant to get correct units
    m_ref = constants.F * k_ref

    E_r = 25000
    arrhenius = exp(-E_r / (constants.R * T)) * exp(E_r / (constants.R * 296.15))

    return m_ref * arrhenius
