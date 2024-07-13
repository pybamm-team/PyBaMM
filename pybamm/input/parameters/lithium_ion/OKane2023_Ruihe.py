import pybamm
import numpy as np


def plating_exchange_current_density_OKane2020(c_e, c_Li, T):
    """
    Exchange-current density for Li plating reaction [A.m-2].
    References
    ----------
    .. [1] O’Kane, Simon EJ, Ian D. Campbell, Mohamed WJ Marzook, Gregory J. Offer, and
    Monica Marinescu. "Physical origin of the differential voltage minimum associated
    with lithium plating in Li-ion batteries." Journal of The Electrochemical Society
    167, no. 9 (2020): 090540.
    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_Li : :class:`pybamm.Symbol`
        Plated lithium concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]
    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """

    k_plating = pybamm.Parameter("Lithium plating kinetic rate constant [m.s-1]")

    return pybamm.constants.F * k_plating * c_e


def stripping_exchange_current_density_OKane2020(c_e, c_Li, T):
    """
    Exchange-current density for Li stripping reaction [A.m-2].

    References
    ----------

    .. [1] O’Kane, Simon EJ, Ian D. Campbell, Mohamed WJ Marzook, Gregory J. Offer, and
    Monica Marinescu. "Physical origin of the differential voltage minimum associated
    with lithium plating in Li-ion batteries." Journal of The Electrochemical Society
    167, no. 9 (2020): 090540.

    Parameters
    ----------

    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_Li : :class:`pybamm.Symbol`
        Plated lithium concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------

    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """

    k_plating = pybamm.Parameter("Lithium plating kinetic rate constant [m.s-1]")

    return pybamm.constants.F * k_plating * c_Li


def SEI_limited_dead_lithium_OKane2022(L_sei):
    """
    Decay rate for dead lithium formation [s-1].
    References
    ----------
    .. [1] Simon E. J. O'Kane, Weilong Ai, Ganesh Madabattula, Diega Alonso-Alvarez,
    Robert Timms, Valentin Sulzer, Jaqueline Sophie Edge, Billy Wu, Gregory J. Offer
    and Monica Marinescu. "Lithium-ion battery degradation: how to model it."
    Physical Chemistry: Chemical Physics 24, no. 13 (2022): 7909-7922.
    Parameters
    ----------
    L_sei : :class:`pybamm.Symbol`
        Total SEI thickness [m]
    Returns
    -------
    :class:`pybamm.Symbol`
        Dead lithium decay rate [s-1]
    """

    gamma_0 = pybamm.Parameter("Dead lithium decay constant [s-1]")
    L_inner_0 = pybamm.Parameter("Negative initial inner SEI thickness [m]")
    L_outer_0 = pybamm.Parameter("Negative initial outer SEI thickness [m]")
    L_sei_0 = L_inner_0 + L_outer_0

    gamma = gamma_0 * L_sei_0 / L_sei

    return gamma


def electrolyte_conductivity_base_Landesfeind2019(c_e, T, coeffs):
    """
    Conductivity of LiPF6 in solvent_X as a function of ion concentration and
    temperature. The data comes from [1].

    References
    ----------
    .. [1] Landesfeind, J. and Gasteiger, H.A., 2019. Temperature and Concentration
    Dependence of the Ionic Transport Properties of Lithium-Ion Battery Electrolytes.
    Journal of The Electrochemical Society, 166(14), pp.A3079-A3097.

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature
    coeffs: :class:`pybamm.Symbol`
        Fitting parameter coefficients

    Returns
    -------
    :class:`pybamm.Symbol`
        Electrolyte conductivity
    """
    c = c_e / 1000  # mol.m-3 -> mol.l
    p1, p2, p3, p4, p5, p6 = coeffs
    A = p1 * (1 + (T - p2))
    B = 1 + p3 * pybamm.sqrt(c) + p4 * (1 + p5 * pybamm.exp(1000 / T)) * c
    C = 1 + c**4 * (p6 * pybamm.exp(1000 / T))
    sigma_e = A * c * B / C  # mS.cm-1

    return sigma_e / 10


def electrolyte_diffusivity_base_Landesfeind2019(c_e, T, coeffs):
    """
    Diffusivity of LiPF6 in solvent_X as a function of ion concentration and
    temperature. The data comes from [1].

    References
    ----------
    .. [1] Landesfeind, J. and Gasteiger, H.A., 2019. Temperature and Concentration
    Dependence of the Ionic Transport Properties of Lithium-Ion Battery Electrolytes.
    Journal of The Electrochemical Society, 166(14), pp.A3079-A3097.

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature
    coeffs: :class:`pybamm.Symbol`
        Fitting parameter coefficients

    Returns
    -------
    :class:`pybamm.Symbol`
        Electrolyte diffusivity
    """
    c = c_e / 1000  # mol.m-3 -> mol.l
    p1, p2, p3, p4 = coeffs
    A = p1 * pybamm.exp(p2 * c)
    B = pybamm.exp(p3 / T)
    C = pybamm.exp(p4 * c / T)
    D_e = A * B * C * 1e-10  # m2/s

    return D_e


def electrolyte_TDF_base_Landesfeind2019(c_e, T, coeffs):
    """
    Thermodynamic factor (TDF) of LiPF6 in solvent_X as a function of ion concentration
    and temperature. The data comes from [1].

    References
    ----------
    .. [1] Landesfeind, J. and Gasteiger, H.A., 2019. Temperature and Concentration
    Dependence of the Ionic Transport Properties of Lithium-Ion Battery Electrolytes.
    Journal of The Electrochemical Society, 166(14), pp.A3079-A3097.

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature
    coeffs: :class:`pybamm.Symbol`
        Fitting parameter coefficients

    Returns
    -------
    :class:`pybamm.Symbol`
        Electrolyte thermodynamic factor
    """
    c = c_e / 1000  # mol.m-3 -> mol.l
    p1, p2, p3, p4, p5, p6, p7, p8, p9 = coeffs
    tdf = (
        p1
        + p2 * c
        + p3 * T
        + p4 * c**2
        + p5 * c * T
        + p6 * T**2
        + p7 * c**3
        + p8 * c**2 * T
        + p9 * c * T**2
    )

    return tdf


def electrolyte_t_plus_base_Landesfeind2019(c_e, T, coeffs):
    """
    Transference number of LiPF6 in solvent_X as a function of ion concentration and
    temperature. The data comes from [1].

    References
    ----------
    .. [1] Landesfeind, J. and Gasteiger, H.A., 2019. Temperature and Concentration
    Dependence of the Ionic Transport Properties of Lithium-Ion Battery Electrolytes.
    Journal of The Electrochemical Society, 166(14), pp.A3079-A3097.

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature
    coeffs: :class:`pybamm.Symbol`
        Fitting parameter coefficients

    Returns
    -------
    :class:`pybamm.Symbol`
        Electrolyte transference number
    """
    c = c_e / 1000  # mol.m-3 -> mol.l
    p1, p2, p3, p4, p5, p6, p7, p8, p9 = coeffs
    tplus = (
        p1
        + p2 * c
        + p3 * T
        + p4 * c**2
        + p5 * c * T
        + p6 * T**2
        + p7 * c**3
        + p8 * c**2 * T
        + p9 * c * T**2
    )

    return tplus


def copper_heat_capacity_CRC(T):
    """
    Copper specific heat capacity as a function of the temperature from [1].

    References
    ----------
    .. [1] William M. Haynes (Ed.). "CRC handbook of chemistry and physics". CRC Press
    (2014).

    Parameters
    ----------
    T: :class:`pybamm.Symbol`
       Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
       Specific heat capacity
    """

    cp = 1.445e-6 * T**3 - 1.946e-3 * T**2 + 0.9633 * T + 236

    return cp


def aluminium_heat_capacity_CRC(T):
    """
    Aluminium specific heat capacity as a function of the temperature from [1].

    References
    ----------
    .. [1] William M. Haynes (Ed.). "CRC handbook of chemistry and physics". CRC Press
    (2014).

    Parameters
    ----------
    T: :class:`pybamm.Symbol`
       Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
       Specific heat capacity
    """

    cp = 4.503e-6 * T**3 - 6.256e-3 * T**2 + 3.281 * T + 355.7

    return cp


def copper_thermal_conductivity_CRC(T):
    """
    Copper thermal conductivity as a function of the temperature from [1].

    References
    ----------
    .. [1] William M. Haynes (Ed.). "CRC handbook of chemistry and physics". CRC Press
    (2014).

    Parameters
    ----------
    T: :class:`pybamm.Symbol`
       Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
       Thermal conductivity
    """

    lambda_th = -5.409e-7 * T**3 + 7.054e-4 * T**2 - 0.3727 * T + 463.6

    return lambda_th


def graphite_LGM50_diffusivity_ORegan2022(sto, T):
    """
    LG M50 Graphite diffusivity as a function of stochiometry, in this case the
    diffusivity is taken to be a constant. The value is taken from [1].

    References
    ----------
    .. [1] Kieran O’Regan, Ferran Brosa Planella, W. Dhammika Widanage, and Emma
    Kendrick. "Thermal-electrochemical parameters of a high energy lithium-ion
    cylindrical battery." Electrochimica Acta 425 (2022): 140700

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
       Electrode stochiometry
    T: :class:`pybamm.Symbol`
       Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
       Solid diffusivity
    """

    a0 = 11.17
    a1 = -1.553
    a2 = -6.136
    a3 = -9.725
    a4 = 1.85
    b1 = 0.2031
    b2 = 0.5375
    b3 = 0.9144
    b4 = 0.5953
    c0 = -15.11
    c1 = 0.0006091
    c2 = 0.06438
    c3 = 0.0578
    c4 = 0.001356
    d = 2092

    D_ref = (
        10
        ** (
            a0 * sto
            + c0
            + a1 * pybamm.exp(-((sto - b1) ** 2) / c1)
            + a2 * pybamm.exp(-((sto - b2) ** 2) / c2)
            + a3 * pybamm.exp(-((sto - b3) ** 2) / c3)
            + a4 * pybamm.exp(-((sto - b4) ** 2) / c4)
        )
        * 3.0321  # correcting factor (see O'Regan et al 2021)
    )

    #E_D_s = d * pybamm.constants.R
    E_D_s = pybamm.Parameter(
        "Negative electrode diffusivity activation energy [J.mol-1]")
    #print(E_D_s)
    arrhenius = pybamm.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius


def graphite_LGM50_ocp_Chen2020(sto):
    """
    LG M50 Graphite open-circuit potential as a function of stochiometry, fit taken
    from [1].

    References
    ----------
    .. [1] Chang-Hui Chen, Ferran Brosa Planella, Kieran O’Regan, Dominika Gastol, W.
    Dhammika Widanage, and Emma Kendrick. "Development of Experimental Techniques for
    Parameterization of Multi-scale Lithium-ion Battery Models." Journal of the
    Electrochemical Society 167 (2020): 080534.

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
       Electrode stochiometry

    Returns
    -------
    :class:`pybamm.Symbol`
       Open-circuit potential
    """

    U = (
        1.9793 * pybamm.exp(-39.3631 * sto)
        + 0.2482
        - 0.0909 * pybamm.tanh(29.8538 * (sto - 0.1234))
        - 0.04478 * pybamm.tanh(14.9159 * (sto - 0.2769))
        - 0.0205 * pybamm.tanh(30.4444 * (sto - 0.6103))
    )

    return U


def graphite_LGM50_electrolyte_exchange_current_density_ORegan2022(
    c_e, c_s_surf, c_s_max, T
):
    """
    Exchange-current density for Butler-Volmer reactions between graphite and LiPF6 in
    EC:DMC.

    References
    ----------
    .. [1] Kieran O’Regan, Ferran Brosa Planella, W. Dhammika Widanage, and Emma
    Kendrick. "Thermal-electrochemical parameters of a high energy lithium-ion
    cylindrical battery." Electrochimica Acta 425 (2022): 140700

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_s_surf : :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    c_s_max : :class:`pybamm.Symbol`
        Maximum particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """

    i_ref = 2.668  # (A/m2)
    alpha = 0.792
    E_r = 4e4
    arrhenius = pybamm.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    c_e_ref = pybamm.Parameter("Typical electrolyte concentration [mol.m-3]")

    return (
        i_ref
        * arrhenius
        * (c_e / c_e_ref) ** (1 - alpha)
        * (c_s_surf / c_s_max) ** alpha
        * (1 - c_s_surf / c_s_max) ** (1 - alpha)
    )


def graphite_LGM50_heat_capacity_ORegan2022(T):
    """
    Wet negative electrode specific heat capacity as a function of the temperature from
    [1].

    References
    ----------
    .. [1] Kieran O’Regan, Ferran Brosa Planella, W. Dhammika Widanage, and Emma
    Kendrick. "Thermal-electrochemical parameters of a high energy lithium-ion
    cylindrical battery." Electrochimica Acta 425 (2022): 140700

    Parameters
    ----------
    T: :class:`pybamm.Symbol`
       Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
       Specific heat capacity
    """

    # value for the dry porous electrode (i.e. electrode + air, and we neglect the air
    # contribution to density)
    cp_dry = 4.932e-4 * T**3 - 0.491 * T**2 + 169.4 * T - 1.897e4
    rho_dry = 1740
    theta_dry = rho_dry * cp_dry

    # value for the bulk electrolyte
    rho_e = 1280
    cp_e = 229
    eps_e = pybamm.Parameter("Negative electrode porosity")
    theta_e = rho_e * cp_e

    # value for the wet separator
    theta_wet = theta_dry + theta_e * eps_e
    rho_wet = rho_dry + rho_e * eps_e
    cp_wet = theta_wet / rho_wet

    return cp_wet


def graphite_LGM50_thermal_conductivity_ORegan2022(T):
    """
    Wet negative electrode thermal conductivity as a function of the temperature from
    [1].

    References
    ----------
    .. [1] Kieran O’Regan, Ferran Brosa Planella, W. Dhammika Widanage, and Emma
    Kendrick. "Thermal-electrochemical parameters of a high energy lithium-ion
    cylindrical battery." Electrochimica Acta 425 (2022): 140700

    Parameters
    ----------
    T: :class:`pybamm.Symbol`
       Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
       Thermal conductivity
    """

    lambda_wet = -2.61e-4 * T**2 + 0.1726 * T - 24.49

    return lambda_wet


def graphite_LGM50_entropic_change_ORegan2022(sto, c_s_max):
    """
    LG M50 Graphite entropic change in open circuit potential (OCP) at a temperature of
    298.15K as a function of the stochiometry. The fit is taken from [1].

    References
    ----------
    .. [1] Kieran O’Regan, Ferran Brosa Planella, W. Dhammika Widanage, and Emma
    Kendrick. "Thermal-electrochemical parameters of a high energy lithium-ion
    cylindrical battery." Electrochimica Acta 425 (2022): 140700

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
       Electrode stochiometry

    Returns
    -------
    :class:`pybamm.Symbol`
       Entropic change [V.K-1]
    """

    a0 = -0.1112
    a1 = -0.09002 * 0  # fixed fit (see discussion O'Regan et al 2021)
    a2 = 0.3561
    b1 = 0.4955
    b2 = 0.08309
    c0 = 0.02914
    c1 = 0.1122
    c2 = 0.004616
    d1 = 63.9

    dUdT = (
        a0 * sto
        + c0
        + a2 * pybamm.exp(-((sto - b2) ** 2) / c2)
        + a1
        * (pybamm.tanh(d1 * (sto - (b1 - c1))) - pybamm.tanh(d1 * (sto - (b1 + c1))))
    ) / 1000  # fit in mV / K

    return dUdT


def nmc_LGM50_electronic_conductivity_ORegan2022(T):
    """
    Positive electrode electronic conductivity as a function of the temperature from
    [1].

    References
    ----------
    .. [1] Kieran O’Regan, Ferran Brosa Planella, W. Dhammika Widanage, and Emma
    Kendrick. "Thermal-electrochemical parameters of a high energy lithium-ion
    cylindrical battery." Electrochimica Acta 425 (2022): 140700

    Parameters
    ----------
    T: :class:`pybamm.Symbol`
       Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
       Thermal conductivity
    """

    E_r = 3.5e3
    arrhenius = pybamm.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    sigma = 0.8473 * arrhenius

    return sigma


def nmc_LGM50_diffusivity_ORegan2022(sto, T):
    """
    NMC diffusivity as a function of stoichiometry, in this case the
    diffusivity is taken to be a constant. The value is taken from [1].

    References
    ----------
    .. [1] Kieran O’Regan, Ferran Brosa Planella, W. Dhammika Widanage, and Emma
    Kendrick. "Thermal-electrochemical parameters of a high energy lithium-ion
    cylindrical battery." Electrochimica Acta 425 (2022): 140700

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
       Electrode stochiometry
    T: :class:`pybamm.Symbol`
       Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
       Solid diffusivity
    """

    a1 = -0.9231
    a2 = -0.4066
    a3 = -0.993
    b1 = 0.3216
    b2 = 0.4532
    b3 = 0.8098
    c0 = -13.96
    c1 = 0.002534
    c2 = 0.003926
    c3 = 0.09924
    d = 1449

    D_ref = (
        10
        ** (
            c0
            + a1 * pybamm.exp(-((sto - b1) ** 2) / c1)
            + a2 * pybamm.exp(-((sto - b2) ** 2) / c2)
            + a3 * pybamm.exp(-((sto - b3) ** 2) / c3)
        )
        * 2.7  # correcting factor (see O'Regan et al 2021)
    )

    # E_D_s = d * pybamm.constants.R
    E_D_s = pybamm.Parameter(
        "Positive electrode diffusivity activation energy [J.mol-1]")
    arrhenius = pybamm.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius


def nmc_LGM50_ocp_Chen2020(sto):
    """
     LG M50 NMC open-circuit potential as a function of stoichiometry.  The fit is
    taken from [1].

     References
     ----------
    .. [1] Chang-Hui Chen, Ferran Brosa Planella, Kieran O’Regan, Dominika Gastol, W.
    Dhammika Widanage, and Emma Kendrick. "Development of Experimental Techniques for
    Parameterization of Multi-scale Lithium-ion Battery Models." Journal of the
    Electrochemical Society 167 (2020): 080534.

     Parameters
     ----------
     sto: :class:`pybamm.Symbol`
       Electrode stochiometry

     Returns
     -------
     :class:`pybamm.Symbol`
        Open-circuit potential
    """

    U = (
        -0.809 * sto
        + 4.4875
        - 0.0428 * pybamm.tanh(18.5138 * (sto - 0.5542))
        - 17.7326 * pybamm.tanh(15.789 * (sto - 0.3117))
        + 17.5842 * pybamm.tanh(15.9308 * (sto - 0.312))
    )

    return U


def nmc_LGM50_electrolyte_exchange_current_density_ORegan2022(
    c_e, c_s_surf, c_s_max, T
):
    """
    Exchange-current density for Butler-Volmer reactions between NMC and LiPF6 in
    EC:DMC.

    References
    ----------
    .. [1] Kieran O’Regan, Ferran Brosa Planella, W. Dhammika Widanage, and Emma
    Kendrick. "Thermal-electrochemical parameters of a high energy lithium-ion
    cylindrical battery." Electrochimica Acta 425 (2022): 140700

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_s_surf : :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    c_s_max : :class:`pybamm.Symbol`
        Maximum particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """
    i_ref = 5.028  # (A/m2)
    alpha = 0.43
    E_r = 2.401e4
    arrhenius = pybamm.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    c_e_ref = pybamm.Parameter("Typical electrolyte concentration [mol.m-3]")

    return (
        i_ref
        * arrhenius
        * (c_e / c_e_ref) ** (1 - alpha)
        * (c_s_surf / c_s_max) ** alpha
        * (1 - c_s_surf / c_s_max) ** (1 - alpha)
    )


def nmc_LGM50_heat_capacity_ORegan2022(T):
    """
    Wet positive electrode specific heat capacity as a function of the temperature from
    [1].

    References
    ----------
    .. [1] Kieran O’Regan, Ferran Brosa Planella, W. Dhammika Widanage, and Emma
    Kendrick. "Thermal-electrochemical parameters of a high energy lithium-ion
    cylindrical battery." Electrochimica Acta 425 (2022): 140700

    Parameters
    ----------
    T: :class:`pybamm.Symbol`
       Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
       Specific heat capacity
    """

    # value for the dry porous electrode (i.e. electrode + air, and we neglect the air
    # contribution to density)
    cp_dry = -8.414e-4 * T**3 + 0.7892 * T**2 - 241.3 * T + 2.508e4
    rho_dry = 3270
    theta_dry = rho_dry * cp_dry

    # value for the bulk electrolyte
    rho_e = 1280
    cp_e = 229
    eps_e = pybamm.Parameter("Positive electrode porosity")
    theta_e = rho_e * cp_e

    # value for the wet separator
    theta_wet = theta_dry + theta_e * eps_e
    rho_wet = rho_dry + rho_e * eps_e
    cp_wet = theta_wet / rho_wet

    return cp_wet

# define Landesfeind exp(initial) and constant 
def electrolyte_conductivity_base_Landesfeind2019_Constant(c_e, T, coeffs):
    # mol.m-3 -> mol.l
    c = (c_e<4000)*c_e / 1000 +  (c_e>4000)* 4 # Mark Ruihe 
    p1, p2, p3, p4, p5, p6 = coeffs
    A = p1 * (1 + (T - p2))
    B = 1 + p3 * pybamm.sqrt(c) + p4 * (1 + p5 * pybamm.exp(1000 / T)) * c
    C = 1 + c ** 4 * (p6 * pybamm.exp(1000 / T))
    sigma_e = A * c * B / C  # mS.cm-1

    return sigma_e / 10

def electrolyte_diffusivity_base_Landesfeind2019_Constant(c_e, T, coeffs):
    # mol.m-3 -> mol.l
    c = (c_e<4000)*c_e / 1000 +  (c_e>4000)* 4 # Mark Ruihe 
    p1, p2, p3, p4 = coeffs
    A = p1 *pybamm.exp(p2 * c)
    B = pybamm.exp(p3 / T)
    C = pybamm.exp(p4 * c / T)
    D_e = A * B * C * 1e-10  # m2/s

    return D_e

def electrolyte_TDF_base_Landesfeind2019_Constant(c_e, T, coeffs):
    c = (c_e<4000)*c_e / 1000 +  (c_e>4000)* 4 # Mark Ruihe 
    p1, p2, p3, p4, p5, p6, p7, p8, p9 = coeffs
    tdf = (
        p1
        + p2 * c
        + p3 * T
        + p4 * c ** 2
        + p5 * c * T
        + p6 * T ** 2
        + p7 * c ** 3
        + p8 * c ** 2 * T
        + p9 * c * T ** 2
    )
    return tdf

def electrolyte_transference_number_base_Landesfeind2019_Constant(c_e, T, coeffs):
    c = (c_e<4000)*c_e / 1000 +  (c_e>4000)* 4 # Mark Ruihe 
    p1, p2, p3, p4, p5, p6, p7, p8, p9 = coeffs
    tplus = (
        p1
        + p2 * c
        + p3 * T
        + p4 * c ** 2
        + p5 * c * T
        + p6 * T ** 2
        + p7 * c ** 3
        + p8 * c ** 2 * T
        + p9 * c * T ** 2
    )

    return tplus
def electrolyte_transference_number_EC_EMC_3_7_Landesfeind2019_Constant(c_e, T):
    coeffs = np.array(
        [
            -1.28e1,
            -6.12,
            8.21e-2,
            9.04e-1,
            3.18e-2,
            -1.27e-4,
            1.75e-2,
            -3.12e-3,
            -3.96e-5,
        ]
    )

    return electrolyte_transference_number_base_Landesfeind2019_Constant(c_e, T, coeffs)

def electrolyte_TDF_EC_EMC_3_7_Landesfeind2019_Constant(c_e, T):
    coeffs = np.array(
        [2.57e1, -4.51e1, -1.77e-1, 1.94, 2.95e-1, 3.08e-4, 2.59e-1, -9.46e-3, -4.54e-4]
    )

    return electrolyte_TDF_base_Landesfeind2019_Constant(c_e, T, coeffs)

def electrolyte_diffusivity_EC_EMC_3_7_Landesfeind2019_Constant(c_e, T):
    coeffs = np.array([1.01e3, 1.01, -1.56e3, -4.87e2])

    return electrolyte_diffusivity_base_Landesfeind2019_Constant(c_e, T, coeffs)

def electrolyte_conductivity_EC_EMC_3_7_Landesfeind2019_Constant(c_e, T):
    coeffs = np.array([5.21e-1, 2.28e2, -1.06, 3.53e-1, -3.59e-3, 1.48e-3])

    return electrolyte_conductivity_base_Landesfeind2019_Constant(c_e, T, coeffs)


def nmc_LGM50_thermal_conductivity_ORegan2022(T):
    """
    Wet positive electrode thermal conductivity as a function of the temperature from
    [1].

    References
    ----------
    .. [1] Kieran O’Regan, Ferran Brosa Planella, W. Dhammika Widanage, and Emma
    Kendrick. "Thermal-electrochemical parameters of a high energy lithium-ion
    cylindrical battery." Electrochimica Acta 425 (2022): 140700

    Parameters
    ----------
    T: :class:`pybamm.Symbol`
       Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
       Thermal conductivity
    """

    lambda_wet = 2.063e-5 * T**2 - 0.01127 * T + 2.331

    return lambda_wet


def nmc_LGM50_entropic_change_ORegan2022(sto, c_s_max):
    """
    LG M50 NMC 811 entropic change in open circuit potential (OCP) at a temperature of
    298.15K as a function of the stochiometry. The fit is taken from [1].

    References
    ----------
    .. [1] Kieran O’Regan, Ferran Brosa Planella, W. Dhammika Widanage, and Emma
    Kendrick. "Thermal-electrochemical parameters of a high energy lithium-ion
    cylindrical battery." Electrochimica Acta 425 (2022): 140700

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
       Electrode stochiometry

    Returns
    -------
    :class:`pybamm.Symbol`
       Entropic change [V.K-1]
    """
    a1 = 0.04006
    a2 = -0.06656
    b1 = 0.2828
    b2 = 0.8032
    c1 = 0.0009855
    c2 = 0.02179

    dUdT = (
        a1 * pybamm.exp(-((sto - b1) ** 2) / c1)
        + a2 * pybamm.exp(-((sto - b2) ** 2) / c2)
    ) / 1000
    # fit in mV / K

    return dUdT


def separator_LGM50_heat_capacity_ORegan2022(T):
    """
    Wet separator specific heat capacity as a function of the temperature from [1].

    References
    ----------
    .. [1] Kieran O’Regan, Ferran Brosa Planella, W. Dhammika Widanage, and Emma
    Kendrick. "Thermal-electrochemical parameters of a high energy lithium-ion
    cylindrical battery." Electrochimica Acta 425 (2022): 140700

    Parameters
    ----------
    T: :class:`pybamm.Symbol`
       Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
       Specific heat capacity
    """

    # value for the dry porous separator (i.e. separator + air, and we neglect the air
    # contribution to density)
    cp_dry = 1.494e-3 * T**3 - 1.444 * T**2 + 475.5 * T - 5.13e4
    rho_dry = 946
    theta_dry = rho_dry * cp_dry

    # value for the bulk electrolyte
    rho_e = 1280
    cp_e = 229
    eps_e = pybamm.Parameter("Separator porosity")
    theta_e = rho_e * cp_e

    # value for the wet separator
    theta_wet = theta_dry + theta_e * eps_e
    rho_wet = rho_dry + rho_e * eps_e
    cp_wet = theta_wet / rho_wet

    return cp_wet


def graphite_volume_change_Ai2020(sto, c_s_max):
    """
    Graphite particle volume change as a function of stochiometry [1, 2].

    References
    ----------
     .. [1] Ai, W., Kraft, L., Sturm, J., Jossen, A., & Wu, B. (2020).
     Electrochemical Thermal-Mechanical Modelling of Stress Inhomogeneity in
     Lithium-Ion Pouch Cells. Journal of The Electrochemical Society, 167(1), 013512
      DOI: 10.1149/2.0122001JES.
     .. [2] Rieger, B., Erhard, S. V., Rumpf, K., & Jossen, A. (2016).
     A new method to model the thickness change of a commercial pouch cell
     during discharge. Journal of The Electrochemical Society, 163(8), A1566-A1575.

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry, dimensionless
        should be R-averaged particle concentration
    Returns
    -------
    t_change:class:`pybamm.Symbol`
        volume change, dimensionless, normalised by particle volume
    """
    p1 = 145.907
    p2 = -681.229
    p3 = 1334.442
    p4 = -1415.710
    p5 = 873.906
    p6 = -312.528
    p7 = 60.641
    p8 = -5.706
    p9 = 0.386
    p10 = -4.966e-05
    t_change = (
        p1 * sto**9
        + p2 * sto**8
        + p3 * sto**7
        + p4 * sto**6
        + p5 * sto**5
        + p6 * sto**4
        + p7 * sto**3
        + p8 * sto**2
        + p9 * sto
        + p10
    )
    return t_change


def graphite_cracking_rate_Ai2020(T_dim):
    """
    Graphite particle cracking rate as a function of temperature [1, 2].

    References
    ----------
     .. [1] Ai, W., Kraft, L., Sturm, J., Jossen, A., & Wu, B. (2020).
     Electrochemical Thermal-Mechanical Modelling of Stress Inhomogeneity in
     Lithium-Ion Pouch Cells. Journal of The Electrochemical Society, 167(1), 013512
      DOI: 10.1149/2.0122001JES.
     .. [2] Deshpande, R., Verbrugge, M., Cheng, Y. T., Wang, J., & Liu, P. (2012).
     Battery cycle life prediction with coupled chemical degradation and fatigue
     mechanics. Journal of the Electrochemical Society, 159(10), A1730.

    Parameters
    ----------
    T_dim: :class:`pybamm.Symbol`
        temperature, [K]

    Returns
    -------
    k_cr: :class:`pybamm.Symbol`
        cracking rate, [m/(Pa.m0.5)^m_cr]
        where m_cr is another Paris' law constant
    """
    k_cr = 3.9e-20
    # Eac_cr = 0  # to be implemented - Mark Ruihe implemented on 17/06/2023
    Eac_cr = pybamm.Parameter(
        "Negative cracking growth activation energy [J.mol-1]")
    arrhenius = pybamm.exp( - Eac_cr / pybamm.constants.R * (1 / T_dim - 1 / 298.15))
    return k_cr * arrhenius


def volume_change_Ai2020(sto, c_s_max):
    """
    Particle volume change as a function of stochiometry [1, 2].

    References
    ----------
     .. [1] > Ai, W., Kraft, L., Sturm, J., Jossen, A., & Wu, B. (2020).
     Electrochemical Thermal-Mechanical Modelling of Stress Inhomogeneity in
     Lithium-Ion Pouch Cells. Journal of The Electrochemical Society, 167(1), 013512
      DOI: 10.1149/2.0122001JES.
     .. [2] > Rieger, B., Erhard, S. V., Rumpf, K., & Jossen, A. (2016).
     A new method to model the thickness change of a commercial pouch cell
     during discharge. Journal of The Electrochemical Society, 163(8), A1566-A1575.

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry, dimensionless
        should be R-averaged particle concentration
    Returns
    -------
    t_change:class:`pybamm.Symbol`
        volume change, dimensionless, normalised by particle volume
    """
    omega = pybamm.Parameter("Positive electrode partial molar volume [m3.mol-1]")
    t_change = omega * c_s_max * sto
    return t_change


def cracking_rate_Ai2020(T_dim):
    """
    Particle cracking rate as a function of temperature [1, 2].

    References
    ----------
     .. [1] > Ai, W., Kraft, L., Sturm, J., Jossen, A., & Wu, B. (2020).
     Electrochemical Thermal-Mechanical Modelling of Stress Inhomogeneity in
     Lithium-Ion Pouch Cells. Journal of The Electrochemical Society, 167(1), 013512
      DOI: 10.1149/2.0122001JES.
     .. [2] > Deshpande, R., Verbrugge, M., Cheng, Y. T., Wang, J., & Liu, P. (2012).
     Battery cycle life prediction with coupled chemical degradation and fatigue
     mechanics. Journal of the Electrochemical Society, 159(10), A1730.

    Parameters
    ----------
    T: :class:`pybamm.Symbol`
        temperature, [K]

    Returns
    -------
    k_cr: :class:`pybamm.Symbol`
        cracking rate, [m/(Pa.m0.5)^m_cr]
        where m_cr is another Paris' law constant
    """
    k_cr = 3.9e-20
    Eac_cr = 0  # to be implemented
    arrhenius = pybamm.exp( - Eac_cr / pybamm.constants.R * (1 / T_dim - 1 / 298.15))
    return k_cr * arrhenius


def graphite_LGM50_delithiation_ocp_OKane2023(sto):
    """
    LG M50 Graphite delithiation open-circuit potential as a function of stochiometry.
    Fitted to unpublished measurements taken by Kieran O'Regan.

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry

    Returns
    -------
    :class:`pybamm.Symbol`
        Open circuit potential
    """

    u_eq = (
        1.051 * pybamm.exp(-26.76 * sto)
        + 0.1916
        - 0.05598 * pybamm.tanh(35.62 * (sto - 0.1356))
        - 0.04483 * pybamm.tanh(14.64 * (sto - 0.2861))
        - 0.02097 * pybamm.tanh(26.28 * (sto - 0.6183))
        - 0.02398 * pybamm.tanh(38.1 * (sto - 1))
    )

    return u_eq


def graphite_LGM50_lithiation_ocp_OKane2023(sto):
    """
    LG M50 Graphite lithiation open-circuit potential as a function of stochiometry.
    Fitted to unpublished measurements taken by Kieran O'Regan.

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry

    Returns
    -------
    :class:`pybamm.Symbol`
        Open circuit potential
    """

    u_eq = (
        0.5476 * pybamm.exp(-422.4 * sto)
        + 0.5705 * pybamm.exp(-36.89 * sto)
        + 0.1336
        - 0.04758 * pybamm.tanh(13.88 * (sto - 0.2101))
        - 0.01761 * pybamm.tanh(36.2 * (sto - 0.5639))
        - 0.0169 * pybamm.tanh(11.42 * (sto - 1))
    )

    return u_eq


def nmc_LGM50_lithiation_ocp_OKane2023(sto):
    """
    LG M50 NMC lithiation open-circuit potential as a function of stoichiometry.
    Fitted to unpublished measurements by Kieran O'Regan.

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry
    Returns
    -------
    :class:`pybamm.Symbol`
        Open-circuit potential
    """

    U = (
        -0.7983 * sto
        + 4.513
        - 0.03269 * pybamm.tanh(19.83 * (sto - 0.5424))
        - 18.23 * pybamm.tanh(14.33 * (sto - 0.2771))
        + 18.05 * pybamm.tanh(14.46 * (sto - 0.2776))
    )

    return U


def nmc_LGM50_delithiation_ocp_OKane2023(sto):
    """
    LG M50 NMC delithiation open-circuit potential as a function of stoichiometry.
    Fitted to unpublished measurements by Kieran O'Regan.

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry
    Returns
    -------
    :class:`pybamm.Symbol`
        Open-circuit potential
    """

    U = (
        -0.7836 * sto
        + 4.513
        - 0.03432 * pybamm.tanh(19.83 * (sto - 0.5424))
        - 19.35 * pybamm.tanh(14.33 * (sto - 0.2771))
        + 19.17 * pybamm.tanh(14.45 * (sto - 0.2776))
    )

    return U


def electrolyte_transference_number_EC_EMC_3_7_Landesfeind2019(c_e, T):
    """
    Transference number of LiPF6 in EC:EMC (3:7 w:w) as a function of ion
    concentration and temperature. The data comes from [1].

    References
    ----------
    .. [1] Landesfeind, J. and Gasteiger, H.A., 2019. Temperature and Concentration
    Dependence of the Ionic Transport Properties of Lithium-Ion Battery Electrolytes.
    Journal of The Electrochemical Society, 166(14), pp.A3079-A3097.

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Electrolyte transference number
    """
    coeffs = np.array(
        [
            -1.28e1,
            -6.12,
            8.21e-2,
            9.04e-1,
            3.18e-2,
            -1.27e-4,
            1.75e-2,
            -3.12e-3,
            -3.96e-5,
        ]
    )

    c_lim = pybamm.Scalar(4000)  # solubility limit

    t_plus = (
        electrolyte_t_plus_base_Landesfeind2019(c_e, T, coeffs) * (c_e <= c_lim) +
        electrolyte_t_plus_base_Landesfeind2019(c_lim, T, coeffs) * (c_e > c_lim)
    )

    return t_plus


def electrolyte_TDF_EC_EMC_3_7_Landesfeind2019(c_e, T):
    """
    Thermodynamic factor (TDF) of LiPF6 in EC:EMC (3:7 w:w) as a function of ion
    concentration and temperature. The data comes from [1].

    References
    ----------
    .. [1] Landesfeind, J. and Gasteiger, H.A., 2019. Temperature and Concentration
    Dependence of the Ionic Transport Properties of Lithium-Ion Battery Electrolytes.
    Journal of The Electrochemical Society, 166(14), pp.A3079-A3097.

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Electrolyte thermodynamic factor
    """
    coeffs = np.array(
        [2.57e1, -4.51e1, -1.77e-1, 1.94, 2.95e-1, 3.08e-4, 2.59e-1, -9.46e-3, -4.54e-4]
    )

    c_lim = pybamm.Scalar(4000)  # solubility limit

    TDF = (
        electrolyte_TDF_base_Landesfeind2019(c_e, T, coeffs) * (c_e <= c_lim) +
        electrolyte_TDF_base_Landesfeind2019(c_lim, T, coeffs) * (c_e > c_lim)
    )

    return TDF


def electrolyte_diffusivity_EC_EMC_3_7_Landesfeind2019(c_e, T):
    """
    Diffusivity of LiPF6 in EC:EMC (3:7 w:w) as a function of ion concentration and
    temperature. The data comes from [1].

    References
    ----------
    .. [1] Landesfeind, J. and Gasteiger, H.A., 2019. Temperature and Concentration
    Dependence of the Ionic Transport Properties of Lithium-Ion Battery Electrolytes.
    Journal of The Electrochemical Society, 166(14), pp.A3079-A3097.

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Electrolyte diffusivity
    """
    coeffs = np.array([1.01e3, 1.01, -1.56e3, -4.87e2])

    c_lim = pybamm.Scalar(4000)  # solubility limit

    D_e = (
        electrolyte_diffusivity_base_Landesfeind2019(c_e, T, coeffs) * (c_e <= c_lim) +
        electrolyte_diffusivity_base_Landesfeind2019(c_lim, T, coeffs) * (c_e > c_lim)
    )

    return D_e


def electrolyte_conductivity_EC_EMC_3_7_Landesfeind2019(c_e, T):
    """
    Conductivity of LiPF6 in EC:EMC (3:7 w:w) as a function of ion concentration and
    temperature. The data comes from [1].

    References
    ----------
    .. [1] Landesfeind, J. and Gasteiger, H.A., 2019. Temperature and Concentration
    Dependence of the Ionic Transport Properties of Lithium-Ion Battery Electrolytes.
    Journal of The Electrochemical Society, 166(14), pp.A3079-A3097.

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Electrolyte conductivity
    """
    coeffs = np.array([5.21e-1, 2.28e2, -1.06, 3.53e-1, -3.59e-3, 1.48e-3])

    c_lim = pybamm.Scalar(4000)  # solubility limit

    sigma_e = (
        electrolyte_conductivity_base_Landesfeind2019(c_e, T, coeffs) * (c_e <= c_lim) +
        electrolyte_conductivity_base_Landesfeind2019(c_lim, T, coeffs) * (c_e > c_lim)
    )

    return sigma_e


# Call dict via a function to avoid errors when editing in place
def get_parameter_values():
    """
    Parameters for an LG M50 cell, from the paper

    > Simon O'Kane, Weilong Ai, Ganesh Madabattula, Diego Alonso-Alvarez, Robert Timms,
    Valentin Sulzer, Jacqueline Edge, Billy Wu, Gregory Offer, and Monica Marinescu.
    ["Lithium-ion battery degradation: how to model
    it."](https://pubs.rsc.org/en/content/articlelanding/2022/cp/d2cp00417h) Physical
    Chemistry: Chemical Physics 24 (2022): 7909-7922

    based on the paper

    > Chang-Hui Chen, Ferran Brosa Planella, Kieran O’Regan, Dominika Gastol, W.
    Dhammika Widanage, and Emma Kendrick. ["Development of Experimental Techniques for
    Parameterization of Multi-scale Lithium-ion Battery
    Models."](https://iopscience.iop.org/article/10.1149/1945-7111/ab9050) Journal of
    the Electrochemical Society 167 (2020): 080534

    and references therein.

    Note: the SEI and plating parameters do not claim to be representative of the true
    parameter values. These are merely the parameter values that were used in the
    referenced papers.
    """

    return {
        "chemistry": "lithium_ion",
        # lithium plating
        "Lithium metal partial molar volume [m3.mol-1]": 1.3e-05,
        "Lithium plating kinetic rate constant [m.s-1]": 1e-09,
        "Exchange-current density for plating [A.m-2]"
        "": plating_exchange_current_density_OKane2020,
        "Exchange-current density for stripping [A.m-2]"
        "": stripping_exchange_current_density_OKane2020,
        "Initial plated lithium concentration [mol.m-3]": 0.0,
        "Typical plated lithium concentration [mol.m-3]": 1000.0,
        "Lithium plating transfer coefficient": 0.65,
        "Dead lithium decay constant [s-1]": 1e-06,
        "Dead lithium decay rate [s-1]": SEI_limited_dead_lithium_OKane2022,
        # sei
        "Negative ratio of lithium moles to SEI moles": 2.0,
        "Negative inner SEI reaction proportion": 0.5,
        "Negative inner SEI partial molar volume [m3.mol-1]": 9.585e-05,
        "Negative outer SEI partial molar volume [m3.mol-1]": 9.585e-05,
        "Negative SEI reaction exchange current density [A.m-2]": 1.5e-07,
        "Negative SEI resistivity [Ohm.m]": 200000.0,
        "Negative outer SEI solvent diffusivity [m2.s-1]": 2.5000000000000002e-22,
        "Negative bulk solvent concentration [mol.m-3]": 4541.0,
        "Negative inner SEI open-circuit potential [V]": 0.1,
        "Negative outer SEI open-circuit potential [V]": 0.8,
        "Negative inner SEI electron conductivity [S.m-1]": 8.95e-14,
        "Negative inner SEI lithium interstitial diffusivity [m2.s-1]": 1e-20,
        "Negative lithium interstitial reference concentration [mol.m-3]": 15.0,
        "Negative initial inner SEI thickness [m]": 1.23625e-08,
        "Negative initial outer SEI thickness [m]": 1.23625e-08,
        "Negative EC initial concentration in electrolyte [mol.m-3]": 4541.0,
        "Negative EC diffusivity [m2.s-1]": 2e-18,
        "Negative SEI kinetic rate constant [m.s-1]": 1e-12,
        "Negative SEI open-circuit potential [V]": 0.4,
        "Negative SEI growth activation energy [J.mol-1]": 38000.0,
        "Negative cracking growth activation energy [J.mol-1]": 0.0,
        "Negative electrode reaction-driven LAM factor [m3.mol-1]": 0.0,
        "Positive ratio of lithium moles to SEI moles": 1.0,
        "Positive inner SEI reaction proportion": 0.0,
        "Positive inner SEI partial molar volume [m3.mol-1]": 9.585e-05,
        "Positive outer SEI partial molar volume [m3.mol-1]": 9.585e-05,
        "Positive SEI reaction exchange current density [A.m-2]": 1.5e-07,
        "Positive SEI resistivity [Ohm.m]": 200000.0,
        "Positive outer SEI solvent diffusivity [m2.s-1]": 2.5000000000000002e-22,
        "Positive bulk solvent concentration [mol.m-3]": 2636.0,
        "Positive inner SEI open-circuit potential [V]": 0.1,
        "Positive outer SEI open-circuit potential [V]": 0.8,
        "Positive inner SEI electron conductivity [S.m-1]": 8.95e-14,
        "Positive inner SEI lithium interstitial diffusivity [m2.s-1]": 1e-20,
        "Positive lithium interstitial reference concentration [mol.m-3]": 15.0,
        "Positive initial inner SEI thickness [m]": 0.0,
        "Positive initial outer SEI thickness [m]": 5e-09,
        "Positive EC initial concentration in electrolyte [mol.m-3]": 4541.0,
        "Positive EC diffusivity [m2.s-1]": 2e-18,
        "Positive SEI kinetic rate constant [m.s-1]": 1e-12,
        "Positive SEI open-circuit potential [V]": 0.4,
        "Positive SEI growth activation energy [J.mol-1]": 38000.0,
        "Positive electrode reaction-driven LAM factor [m3.mol-1]": 0.0,
        # cell
        "Negative current collector thickness [m]": 1.2e-05,
        "Negative electrode thickness [m]": 8.52e-05,
        "Separator thickness [m]": 1.2e-05,
        "Positive electrode thickness [m]": 7.56e-05,
        "Positive current collector thickness [m]": 1.6e-05,
        "Electrode height [m]": 0.065,
        "Electrode width [m]": 1.58,
        "Cell cooling surface area [m2]": 0.00531,
        "Cell volume [m3]": 2.42e-05,
        "Cell thermal expansion coefficient [m.K-1]": 1.1e-06,
        "Negative current collector conductivity [S.m-1]": 58411000.0,
        "Positive current collector conductivity [S.m-1]": 36914000.0,
        "Negative current collector density [kg.m-3]": 8960.0,
        "Positive current collector density [kg.m-3]": 2700.0,
        "Negative current collector specific heat capacity [J.kg-1.K-1]": 385.0,
        "Positive current collector specific heat capacity [J.kg-1.K-1]": 897.0,
        "Negative current collector thermal conductivity [W.m-1.K-1]": 401.0,
        "Positive current collector thermal conductivity [W.m-1.K-1]": 237.0,
        "Nominal cell capacity [A.h]": 5.0,
        "Typical current [A]": 5.0,
        "Current function [A]": 5.0,
        "Contact resistance [Ohm]": 0.013,
        # negative electrode
        "Negative electrode diffusivity activation energy [J.mol-1]":3.03e4, # OREGAN:1.7E4
        "Negative electrode conductivity [S.m-1]": 215.0,
        "Maximum concentration in negative electrode [mol.m-3]": 32544.0,
        "Minimum concentration in negative electrode [mol.m-3]": 0,
        "Negative electrode diffusivity [m2.s-1]"
        "": graphite_LGM50_diffusivity_ORegan2022,
        "Negative electrode OCP [V]": graphite_LGM50_delithiation_ocp_OKane2023,
        "Negative electrode delithiation OCP [V]"
        "": graphite_LGM50_delithiation_ocp_OKane2023,
        "Negative electrode lithiation OCP [V]"
        "": graphite_LGM50_lithiation_ocp_OKane2023,
        "Negative electrode porosity": 0.2223931640762495, # Mark Ruihe change from 0.22239 to consider initial SEI on crack length,
        "Negative electrode active material volume fraction": 0.75,
        "Negative particle radius [m]": 5.86e-06,
        "Negative electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Negative electrode Bruggeman coefficient (electrode)": 0.0,
        "Negative electrode cation signed stoichiometry": -1.0,
        "Negative electrode electrons in reaction": 1.0,
        "Negative electrode charge transfer coefficient": 0.5,
        "Negative electrode double-layer capacity [F.m-2]": 0.2,
        "Negative electrode exchange-current density [A.m-2]"
        "": graphite_LGM50_electrolyte_exchange_current_density_ORegan2022,
        "Negative electrode density [kg.m-3]": 2060.0,
        "Negative electrode specific heat capacity [J.kg-1.K-1]"
        "": graphite_LGM50_heat_capacity_ORegan2022,
        "Negative electrode thermal conductivity [W.m-1.K-1]"
        "": graphite_LGM50_thermal_conductivity_ORegan2022,
        "Negative electrode OCP entropic change [V.K-1]"
        "": graphite_LGM50_entropic_change_ORegan2022,
        "Negative electrode Poisson's ratio": 0.3,
        "Negative electrode Young's modulus [Pa]": 15000000000.0,
        "Negative electrode reference concentration for free of deformation [mol.m-3]"
        "": 0.0,
        "Negative electrode partial molar volume [m3.mol-1]": 3.1e-06,
        "Negative electrode volume change": graphite_volume_change_Ai2020,
        "Negative electrode initial crack length [m]": 2e-08,
        "Negative electrode initial crack width [m]": 1.5e-08,
        "Negative electrode number of cracks per unit area [m-2]": 3180000000000000.0,
        "Negative electrode Paris' law constant b": 1.12,
        "Negative electrode Paris' law constant m": 2.2,
        "Negative electrode cracking rate": graphite_cracking_rate_Ai2020,
        "Negative electrode LAM constant proportional term [s-1]": 2.7778e-07,
        "Negative electrode LAM constant exponential term": 2.0,
        "Negative electrode critical stress [Pa]": 60000000.0,
        # positive electrode
        "Positive electrode diffusivity activation energy [J.mol-1]":2.5e4,# OREGAN:1.2E4
        "Positive electrode conductivity [S.m-1]"
        "": nmc_LGM50_electronic_conductivity_ORegan2022,
        "Maximum concentration in positive electrode [mol.m-3]": 52787.0,
        "Minimum concentration in positive electrode [mol.m-3]": 0, # 12727.0,
        "Positive electrode diffusivity [m2.s-1]": nmc_LGM50_diffusivity_ORegan2022,
        "Positive electrode OCP [V]": nmc_LGM50_lithiation_ocp_OKane2023,
        "Positive electrode delithiation OCP [V]": nmc_LGM50_delithiation_ocp_OKane2023,
        "Positive electrode lithiation OCP [V]": nmc_LGM50_lithiation_ocp_OKane2023,
        "Positive electrode porosity": 0.335,
        "Positive electrode active material volume fraction": 0.665,
        "Positive particle radius [m]": 5.22e-06,
        "Positive electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Positive electrode Bruggeman coefficient (electrode)": 0.0,
        "Positive electrode cation signed stoichiometry": -1.0,
        "Positive electrode electrons in reaction": 1.0,
        "Positive electrode charge transfer coefficient": 0.5,
        "Positive electrode double-layer capacity [F.m-2]": 0.2,
        "Positive electrode exchange-current density [A.m-2]"
        "": nmc_LGM50_electrolyte_exchange_current_density_ORegan2022,
        "Positive electrode density [kg.m-3]": 3699.0,
        "Positive electrode specific heat capacity [J.kg-1.K-1]"
        "": nmc_LGM50_heat_capacity_ORegan2022,
        "Positive electrode thermal conductivity [W.m-1.K-1]"
        "": nmc_LGM50_thermal_conductivity_ORegan2022,
        "Positive electrode OCP entropic change [V.K-1]"
        "": nmc_LGM50_entropic_change_ORegan2022,
        "Positive electrode Poisson's ratio": 0.2,
        "Positive electrode Young's modulus [Pa]": 375000000000.0,
        "Positive electrode reference concentration for free of deformation [mol.m-3]"
        "": 0.0,
        "Positive electrode partial molar volume [m3.mol-1]": 1.25e-05,
        "Positive electrode volume change": volume_change_Ai2020,
        "Positive electrode initial crack length [m]": 2e-08,
        "Positive electrode initial crack width [m]": 1.5e-08,
        "Positive electrode number of cracks per unit area [m-2]": 3180000000000000.0,
        "Positive electrode Paris' law constant b": 1.12,
        "Positive electrode Paris' law constant m": 2.2,
        "Positive electrode cracking rate": cracking_rate_Ai2020,
        "Positive electrode LAM constant proportional term [s-1]": 2.7778e-07,
        "Positive electrode LAM constant exponential term": 2.0,
        "Positive electrode critical stress [Pa]": 375000000.0,
        # separator
        "Separator porosity": 0.47,
        "Separator Bruggeman coefficient (electrolyte)": 1.5,
        "Separator Bruggeman coefficient (electrode)": 1.5,
        "Separator density [kg.m-3]": 1548.0,
        "Separator specific heat capacity [J.kg-1.K-1]"
        "": separator_LGM50_heat_capacity_ORegan2022,
        "Separator thermal conductivity [W.m-1.K-1]": 0.3344,
        # electrolyte
        "Typical electrolyte concentration [mol.m-3]": 1000.0,
        "Initial concentration in electrolyte [mol.m-3]": 1000.0,
        "Cation transference number"
        "": electrolyte_transference_number_EC_EMC_3_7_Landesfeind2019_Constant,
        "Thermodynamic factor": electrolyte_TDF_EC_EMC_3_7_Landesfeind2019_Constant,
        "Electrolyte diffusivity [m2.s-1]"
        "": electrolyte_diffusivity_EC_EMC_3_7_Landesfeind2019_Constant,
        "Electrolyte conductivity [S.m-1]"
        "": electrolyte_conductivity_EC_EMC_3_7_Landesfeind2019_Constant,
        "EC partial molar volume [m3.mol-1]": 6.667e-05,
        # experiment
        "Reference temperature [K]": 298.15,
        "Total heat transfer coefficient [W.m-2.K-1]": 10.0,
        "Ambient temperature [K]": 298.15,
        "Number of electrodes connected in parallel to make a cell": 1.0,
        "Number of cells connected in series to make a battery": 1.0,
        "Lower voltage cut-off [V]": 2.2,
        "Upper voltage cut-off [V]": 4.4,
        "Open-circuit voltage at 0% SOC [V]": 2.5,
        "Open-circuit voltage at 100% SOC [V]": 4.2,
        "Initial concentration in negative electrode [mol.m-3]": 28543.0,
        "Initial concentration in positive electrode [mol.m-3]": 12729.0,
        "Initial temperature [K]": 298.15,
        # citations
        "citations": ["OKane2022", "OKane2020", "Chen2020", "ORegan2022"],
    }
