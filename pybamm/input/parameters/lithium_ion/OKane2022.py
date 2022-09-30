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

    k_plating = Parameter("Lithium plating kinetic rate constant [m.s-1]")

    return constants.F * k_plating * c_e

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

    k_plating = Parameter("Lithium plating kinetic rate constant [m.s-1]")

    return constants.F * k_plating * c_Li

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

    gamma_0 = Parameter("Dead lithium decay constant [s-1]")
    L_inner_0 = Parameter("Initial inner SEI thickness [m]")
    L_outer_0 = Parameter("Initial outer SEI thickness [m]")
    L_sei_0 = L_inner_0 + L_outer_0

    gamma = gamma_0 * L_sei_0 / L_sei

    return gamma

def graphite_LGM50_diffusivity_Chen2020(sto, T):
    """
    LG M50 Graphite diffusivity as a function of stochiometry, in this case the
    diffusivity is taken to be a constant. The value is taken from [1].

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
    T: :class:`pybamm.Symbol`
       Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
       Solid diffusivity
    """

    D_ref = 3.3e-14
    E_D_s = 3.03e4
    # E_D_s not given by Chen et al (2020), so taken from Ecker et al. (2015) instead
    arrhenius = exp(E_D_s / constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius

def graphite_LGM50_electrolyte_exchange_current_density_Chen2020(
    c_e, c_s_surf, c_s_max, T
):
    """
    Exchange-current density for Butler-Volmer reactions between graphite and LiPF6 in
    EC:DMC.

    References
    ----------
    .. [1] Chang-Hui Chen, Ferran Brosa Planella, Kieran O’Regan, Dominika Gastol, W.
    Dhammika Widanage, and Emma Kendrick. "Development of Experimental Techniques for
    Parameterization of Multi-scale Lithium-ion Battery Models." Journal of the
    Electrochemical Society 167 (2020): 080534.

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

    m_ref = 6.48e-7  # (A/m2)(mol/m3)**1.5 - includes ref concentrations
    E_r = 35000
    arrhenius = exp(E_r / constants.R * (1 / 298.15 - 1 / T))

    return (
        m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5
    )

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
    Eac_cr = 0  # to be implemented
    arrhenius = exp(Eac_cr / constants.R * (1 / T_dim - 1 / 298.15))
    return k_cr * arrhenius

def nmc_LGM50_diffusivity_Chen2020(sto, T):
    """
     NMC diffusivity as a function of stoichiometry, in this case the
     diffusivity is taken to be a constant. The value is taken from [1].

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
     T: :class:`pybamm.Symbol`
        Dimensional temperature

     Returns
     -------
     :class:`pybamm.Symbol`
        Solid diffusivity
    """

    D_ref = 4e-15
    E_D_s = 25000  # O'Kane et al. (2022), after Cabanero et al. (2018)
    arrhenius = exp(E_D_s / constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius

def nmc_LGM50_ocp_Chen2020(sto):
    """
    LG M50 NMC open circuit potential as a function of stochiometry, fit taken
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
        Open circuit potential
    """

    u_eq = (
        -0.8090 * sto
        + 4.4875
        - 0.0428 * tanh(18.5138 * (sto - 0.5542))
        - 17.7326 * tanh(15.7890 * (sto - 0.3117))
        + 17.5842 * tanh(15.9308 * (sto - 0.3120))
    )

    return u_eq

def nmc_LGM50_electrolyte_exchange_current_density_Chen2020(c_e, c_s_surf, c_s_max, T):
    """
    Exchange-current density for Butler-Volmer reactions between NMC and LiPF6 in
    EC:DMC.

    References
    ----------
    .. [1] Chang-Hui Chen, Ferran Brosa Planella, Kieran O’Regan, Dominika Gastol, W.
    Dhammika Widanage, and Emma Kendrick. "Development of Experimental Techniques for
    Parameterization of Multi-scale Lithium-ion Battery Models." Journal of the
    Electrochemical Society 167 (2020): 080534.

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_s_surf : :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """
    m_ref = 3.42e-6  # (A/m2)(mol/m3)**1.5 - includes ref concentrations
    E_r = 17800
    arrhenius = exp(E_r / constants.R * (1 / 298.15 - 1 / T))

    return (
        m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5
    )

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
    omega = Parameter("Positive electrode partial molar volume [m3.mol-1]")
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
    arrhenius = exp(Eac_cr / constants.R * (1 / T_dim - 1 / 298.15))
    return k_cr * arrhenius

def electrolyte_diffusivity_Nyman2008_arrhenius(c_e, T):
    """
    Diffusivity of LiPF6 in EC:EMC (3:7) as a function of ion concentration. The data
    comes from [1], with Arrhenius temperature dependence added from [2].

    References
    ----------
    .. [1] A. Nyman, M. Behm, and G. Lindbergh, "Electrochemical characterisation and
    modelling of the mass transport phenomena in LiPF6-EC-EMC electrolyte,"
    Electrochim. Acta, vol. 53, no. 22, pp. 6356–6365, 2008.
    .. [2] Ecker, Madeleine, et al. "Parameterization of a physico-chemical model of
    a lithium-ion battery i. determination of parameters." Journal of the
    Electrochemical Society 162.9 (2015): A1836-A1848.

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    D_c_e = 8.794e-11 * (c_e / 1000) ** 2 - 3.972e-10 * (c_e / 1000) + 4.862e-10

    # Nyman et al. (2008) does not provide temperature dependence
    # So use temperature dependence from Ecker et al. (2015) instead

    E_D_c_e = 17000
    arrhenius = exp(E_D_c_e / constants.R * (1 / 298.15 - 1 / T))

    return D_c_e * arrhenius

def electrolyte_conductivity_Nyman2008_arrhenius(c_e, T):
    """
    Conductivity of LiPF6 in EC:EMC (3:7) as a function of ion concentration. The data
    comes from [1], with Arrhenius temperature dependence added from [2].

    References
    ----------
    .. [1] A. Nyman, M. Behm, and G. Lindbergh, "Electrochemical characterisation and
    modelling of the mass transport phenomena in LiPF6-EC-EMC electrolyte,"
    Electrochim. Acta, vol. 53, no. 22, pp. 6356–6365, 2008.
    .. [2] Ecker, Madeleine, et al. "Parameterization of a physico-chemical model of
    a lithium-ion battery i. determination of parameters." Journal of the
    Electrochemical Society 162.9 (2015): A1836-A1848.

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    sigma_e = (
        0.1297 * (c_e / 1000) ** 3 - 2.51 * (c_e / 1000) ** 1.5 + 3.329 * (c_e / 1000)
    )

    # Nyman et al. (2008) does not provide temperature dependence
    # So use temperature dependence from Ecker et al. (2015) instead

    E_sigma_e = 17000
    arrhenius = exp(E_sigma_e / constants.R * (1 / 298.15 - 1 / T))

    return sigma_e * arrhenius

def get_parameter_values():
    return {'1 + dlnf/dlnc': 1.0,
 'Ambient temperature [K]': 298.15,
 'Bulk solvent concentration [mol.m-3]': 2636.0,
 'Cation transference number': 0.2594,
 'Cell cooling surface area [m2]': 0.00531,
 'Cell thermal expansion coefficient [m.K-1]': 1.1e-06,
 'Cell volume [m3]': 2.42e-05,
 'Current function [A]': 5.0,
 'Dead lithium decay constant [s-1]': 1e-06,
 'Dead lithium decay rate [s-1]': SEI_limited_dead_lithium_OKane2022,
 'EC diffusivity [m2.s-1]': 2e-18,
 'EC initial concentration in electrolyte [mol.m-3]': 4541.0,
 'Electrode height [m]': 0.065,
 'Electrode width [m]': 1.58,
 'Electrolyte conductivity [S.m-1]': electrolyte_conductivity_Nyman2008_arrhenius,
 'Electrolyte diffusivity [m2.s-1]': electrolyte_diffusivity_Nyman2008_arrhenius,
 'Exchange-current density for plating [A.m-2]': plating_exchange_current_density_OKane2020,
 'Exchange-current density for stripping [A.m-2]': stripping_exchange_current_density_OKane2020,
 'Initial concentration in electrolyte [mol.m-3]': 1000.0,
 'Initial concentration in negative electrode [mol.m-3]': 29866.0,
 'Initial concentration in positive electrode [mol.m-3]': 17038.0,
 'Initial inner SEI thickness [m]': 0.0,
 'Initial outer SEI thickness [m]': 5e-09,
 'Initial plated lithium concentration [mol.m-3]': 0.0,
 'Initial temperature [K]': 298.15,
 'Inner SEI electron conductivity [S.m-1]': 8.95e-14,
 'Inner SEI lithium interstitial diffusivity [m2.s-1]': 1e-20,
 'Inner SEI open-circuit potential [V]': 0.1,
 'Inner SEI partial molar volume [m3.mol-1]': 9.585e-05,
 'Inner SEI reaction proportion': 0.0,
 'Lithium interstitial reference concentration [mol.m-3]': 15.0,
 'Lithium metal partial molar volume [m3.mol-1]': 1.3e-05,
 'Lithium plating kinetic rate constant [m.s-1]': 1e-09,
 'Lithium plating transfer coefficient': 0.65,
 'Lower voltage cut-off [V]': 2.5,
 'Maximum concentration in negative electrode [mol.m-3]': 33133.0,
 'Maximum concentration in positive electrode [mol.m-3]': 63104.0,
 'Negative current collector conductivity [S.m-1]': 58411000.0,
 'Negative current collector density [kg.m-3]': 8960.0,
 'Negative current collector specific heat capacity [J.kg-1.K-1]': 385.0,
 'Negative current collector thermal conductivity [W.m-1.K-1]': 401.0,
 'Negative current collector thickness [m]': 1.2e-05,
 'Negative electrode Bruggeman coefficient (electrode)': 1.5,
 'Negative electrode Bruggeman coefficient (electrolyte)': 1.5,
 'Negative electrode LAM constant exponential term': 2.0,
 'Negative electrode LAM constant proportional term [s-1]': 2.7778e-07,
 'Negative electrode OCP [V]': ('graphite_LGM50_ocp_Chen2020,
                                ([array([0.        , 0.03129623, 0.03499902, 0.0387018 , 0.04240458,
       0.04610736, 0.04981015, 0.05351292, 0.05721568, 0.06091845,
       0.06462122, 0.06832399, 0.07202675, 0.07572951, 0.07943227,
       0.08313503, 0.08683779, 0.09054054, 0.09424331, 0.09794607,
       0.10164883, 0.10535158, 0.10905434, 0.1127571 , 0.11645985,
       0.12016261, 0.12386536, 0.12756811, 0.13127086, 0.13497362,
       0.13867638, 0.14237913, 0.14608189, 0.14978465, 0.15348741,
       0.15719018, 0.16089294, 0.1645957 , 0.16829847, 0.17200122,
       0.17570399, 0.17940674, 0.1831095 , 0.18681229, 0.19051504,
       0.1942178 , 0.19792056, 0.20162334, 0.2053261 , 0.20902886,
       0.21273164, 0.2164344 , 0.22013716, 0.22383993, 0.2275427 ,
       0.23124547, 0.23494825, 0.23865101, 0.24235377, 0.24605653,
       0.2497593 , 0.25346208, 0.25716486, 0.26086762, 0.26457039,
       0.26827314, 0.2719759 , 0.27567867, 0.27938144, 0.28308421,
       0.28678698, 0.29048974, 0.29419251, 0.29789529, 0.30159806,
       0.30530083, 0.30900361, 0.31270637, 0.31640913, 0.32011189,
       0.32381466, 0.32751744, 0.33122021, 0.33492297, 0.33862575,
       0.34232853, 0.34603131, 0.34973408, 0.35343685, 0.35713963,
       0.36084241, 0.36454517, 0.36824795, 0.37195071, 0.37565348,
       0.37935626, 0.38305904, 0.38676182, 0.3904646 , 0.39416737,
       0.39787015, 0.40157291, 0.40527567, 0.40897844, 0.41268121,
       0.41638398, 0.42008676, 0.42378953, 0.4274923 , 0.43119506,
       0.43489784, 0.43860061, 0.44230338, 0.44600615, 0.44970893,
       0.45341168, 0.45711444, 0.46081719, 0.46451994, 0.46822269,
       0.47192545, 0.47562821, 0.47933098, 0.48303375, 0.48673651,
       0.49043926, 0.49414203, 0.49784482, 0.50154759, 0.50525036,
       0.50895311, 0.51265586, 0.51635861, 0.52006139, 0.52376415,
       0.52746692, 0.53116969, 0.53487245, 0.53857521, 0.54227797,
       0.54598074, 0.5496835 , 0.55338627, 0.55708902, 0.56079178,
       0.56449454, 0.5681973 , 0.57190006, 0.57560282, 0.57930558,
       0.58300835, 0.58671112, 0.59041389, 0.59411664, 0.59781941,
       0.60152218, 0.60522496, 0.60892772, 0.61263048, 0.61633325,
       0.62003603, 0.6237388 , 0.62744156, 0.63114433, 0.63484711,
       0.63854988, 0.64225265, 0.64595543, 0.64965823, 0.653361  ,
       0.65706377, 0.66076656, 0.66446934, 0.66817212, 0.67187489,
       0.67557767, 0.67928044, 0.68298322, 0.686686  , 0.69038878,
       0.69409156, 0.69779433, 0.70149709, 0.70519988, 0.70890264,
       0.7126054 , 0.71630818, 0.72001095, 0.72371371, 0.72741648,
       0.73111925, 0.73482204, 0.7385248 , 0.74222757, 0.74593034,
       0.74963312, 0.75333589, 0.75703868, 0.76074146, 0.76444422,
       0.76814698, 0.77184976, 0.77555253, 0.77925531, 0.78295807,
       0.78666085, 0.79036364, 0.79406641, 0.79776918, 0.80147197,
       0.80517474, 0.80887751, 0.81258028, 0.81628304, 0.81998581,
       0.82368858, 0.82739136, 0.83109411, 0.83479688, 0.83849965,
       0.84220242, 0.84590519, 0.84960797, 0.85331075, 0.85701353,
       0.86071631, 0.86441907, 0.86812186, 0.87182464, 0.87552742,
       0.87923019, 0.88293296, 0.88663573, 0.89033849, 0.89404126,
       0.89774404, 0.9014468 , 1.        ])],
                                 array([1.81772748, 1.0828807 , 0.99593794, 0.90023398, 0.79649431,
       0.73354429, 0.66664314, 0.64137149, 0.59813869, 0.5670836 ,
       0.54746181, 0.53068399, 0.51304734, 0.49394092, 0.47926274,
       0.46065259, 0.45992726, 0.43801501, 0.42438665, 0.41150269,
       0.40033659, 0.38957134, 0.37756538, 0.36292541, 0.34357086,
       0.3406314 , 0.32299468, 0.31379458, 0.30795386, 0.29207319,
       0.28697687, 0.27405477, 0.2670497 , 0.25857493, 0.25265783,
       0.24826777, 0.2414345 , 0.23362778, 0.22956218, 0.22370236,
       0.22181271, 0.22089651, 0.2194268 , 0.21830064, 0.21845333,
       0.21753715, 0.21719357, 0.21635373, 0.21667822, 0.21738444,
       0.21469313, 0.21541846, 0.21465495, 0.2135479 , 0.21392964,
       0.21074206, 0.20873788, 0.20465319, 0.20205732, 0.19774358,
       0.19444147, 0.19190285, 0.18850531, 0.18581399, 0.18327537,
       0.18157659, 0.17814088, 0.17529686, 0.1719375 , 0.16934161,
       0.16756649, 0.16609676, 0.16414985, 0.16260378, 0.16224113,
       0.160027  , 0.15827096, 0.1588054 , 0.15552238, 0.15580869,
       0.15220118, 0.1511132 , 0.14987253, 0.14874637, 0.14678037,
       0.14620776, 0.14555879, 0.14389819, 0.14359279, 0.14242846,
       0.14038612, 0.13882096, 0.13954628, 0.13946992, 0.13780934,
       0.13973714, 0.13698858, 0.13523254, 0.13441178, 0.1352898 ,
       0.13507985, 0.13647321, 0.13601512, 0.13435452, 0.1334765 ,
       0.1348317 , 0.13275118, 0.13286571, 0.13263667, 0.13456447,
       0.13471718, 0.13395369, 0.13448814, 0.1334765 , 0.13298023,
       0.13259849, 0.13338107, 0.13309476, 0.13275118, 0.13443087,
       0.13315202, 0.132713  , 0.1330184 , 0.13278936, 0.13225491,
       0.13317111, 0.13263667, 0.13187316, 0.13265574, 0.13250305,
       0.13324745, 0.13204496, 0.13242669, 0.13233127, 0.13198769,
       0.13254122, 0.13145325, 0.13298023, 0.13168229, 0.1313578 ,
       0.13235036, 0.13120511, 0.13089971, 0.13109058, 0.13082336,
       0.13011713, 0.129869  , 0.12992626, 0.12942998, 0.12796026,
       0.12862831, 0.12656689, 0.12734947, 0.12509716, 0.12110791,
       0.11839751, 0.11244226, 0.11307214, 0.1092165 , 0.10683058,
       0.10433014, 0.10530359, 0.10056993, 0.09950104, 0.09854668,
       0.09921473, 0.09541635, 0.09980643, 0.0986612 , 0.09560722,
       0.09755413, 0.09612258, 0.09430929, 0.09661885, 0.09366032,
       0.09522548, 0.09535909, 0.09316404, 0.09450016, 0.0930877 ,
       0.09343126, 0.0932404 , 0.09350762, 0.09339309, 0.09291591,
       0.09303043, 0.0926296 , 0.0932404 , 0.09261052, 0.09249599,
       0.09240055, 0.09253416, 0.09209515, 0.09234329, 0.09366032,
       0.09333583, 0.09322131, 0.09264868, 0.09253416, 0.09243873,
       0.09230512, 0.09310678, 0.09165615, 0.09159888, 0.09207606,
       0.09175158, 0.09177067, 0.09236237, 0.09241964, 0.09320222,
       0.09199972, 0.09167523, 0.09322131, 0.09190428, 0.09167523,
       0.09285865, 0.09180884, 0.09150345, 0.09186611, 0.0920188 ,
       0.09320222, 0.09131257, 0.09117896, 0.09133166, 0.09089265,
       0.09058725, 0.09051091, 0.09033912, 0.09041547, 0.0911217 ,
       0.0894611 , 0.08999555, 0.08921297, 0.08881213, 0.08797229,
       0.08709427, 0.08503284, 0.07601531]))),
 'Negative electrode OCP entropic change [V.K-1]': 0.0,
 "Negative electrode Paris' law constant b": 1.12,
 "Negative electrode Paris' law constant m": 2.2,
 "Negative electrode Poisson's ratio": 0.3,
 "Negative electrode Young's modulus [Pa]": 15000000000.0,
 'Negative electrode active material volume fraction': 0.75,
 'Negative electrode cation signed stoichiometry': -1.0,
 'Negative electrode charge transfer coefficient': 0.5,
 'Negative electrode conductivity [S.m-1]': 215.0,
 'Negative electrode cracking rate': graphite_cracking_rate_Ai2020,
 'Negative electrode critical stress [Pa]': 60000000.0,
 'Negative electrode density [kg.m-3]': 1657.0,
 'Negative electrode diffusivity [m2.s-1]': graphite_LGM50_diffusivity_Chen2020,
 'Negative electrode double-layer capacity [F.m-2]': 0.2,
 'Negative electrode electrons in reaction': 1.0,
 'Negative electrode exchange-current density [A.m-2]': graphite_LGM50_electrolyte_exchange_current_density_Chen2020,
 'Negative electrode initial crack length [m]': 2e-08,
 'Negative electrode initial crack width [m]': 1.5e-08,
 'Negative electrode number of cracks per unit area [m-2]': 3180000000000000.0,
 'Negative electrode partial molar volume [m3.mol-1]': 3.1e-06,
 'Negative electrode porosity': 0.25,
 'Negative electrode reaction-driven LAM factor [m3.mol-1]': 0.0,
 'Negative electrode reference concentration for free of deformation [mol.m-3]': 0.0,
 'Negative electrode specific heat capacity [J.kg-1.K-1]': 700.0,
 'Negative electrode thermal conductivity [W.m-1.K-1]': 1.7,
 'Negative electrode thickness [m]': 8.52e-05,
 'Negative electrode volume change': graphite_volume_change_Ai2020,
 'Negative particle radius [m]': 5.86e-06,
 'Nominal cell capacity [A.h]': 5.0,
 'Number of cells connected in series to make a battery': 1.0,
 'Number of electrodes connected in parallel to make a cell': 1.0,
 'Outer SEI open-circuit potential [V]': 0.8,
 'Outer SEI partial molar volume [m3.mol-1]': 9.585e-05,
 'Outer SEI solvent diffusivity [m2.s-1]': 2.5000000000000002e-22,
 'Positive current collector conductivity [S.m-1]': 36914000.0,
 'Positive current collector density [kg.m-3]': 2700.0,
 'Positive current collector specific heat capacity [J.kg-1.K-1]': 897.0,
 'Positive current collector thermal conductivity [W.m-1.K-1]': 237.0,
 'Positive current collector thickness [m]': 1.6e-05,
 'Positive electrode Bruggeman coefficient (electrode)': 1.5,
 'Positive electrode Bruggeman coefficient (electrolyte)': 1.5,
 'Positive electrode LAM constant exponential term': 2.0,
 'Positive electrode LAM constant proportional term [s-1]': 2.7778e-07,
 'Positive electrode OCP [V]': nmc_LGM50_ocp_Chen2020,
 'Positive electrode OCP entropic change [V.K-1]': 0.0,
 "Positive electrode Paris' law constant b": 1.12,
 "Positive electrode Paris' law constant m": 2.2,
 "Positive electrode Poisson's ratio": 0.2,
 "Positive electrode Young's modulus [Pa]": 375000000000.0,
 'Positive electrode active material volume fraction': 0.665,
 'Positive electrode cation signed stoichiometry': -1.0,
 'Positive electrode charge transfer coefficient': 0.5,
 'Positive electrode conductivity [S.m-1]': 0.18,
 'Positive electrode cracking rate': cracking_rate_Ai2020,
 'Positive electrode critical stress [Pa]': 375000000.0,
 'Positive electrode density [kg.m-3]': 3262.0,
 'Positive electrode diffusivity [m2.s-1]': nmc_LGM50_diffusivity_Chen2020,
 'Positive electrode double-layer capacity [F.m-2]': 0.2,
 'Positive electrode electrons in reaction': 1.0,
 'Positive electrode exchange-current density [A.m-2]': nmc_LGM50_electrolyte_exchange_current_density_Chen2020,
 'Positive electrode initial crack length [m]': 2e-08,
 'Positive electrode initial crack width [m]': 1.5e-08,
 'Positive electrode number of cracks per unit area [m-2]': 3180000000000000.0,
 'Positive electrode partial molar volume [m3.mol-1]': 1.25e-05,
 'Positive electrode porosity': 0.335,
 'Positive electrode reaction-driven LAM factor [m3.mol-1]': 0.0,
 'Positive electrode reference concentration for free of deformation [mol.m-3]': 0.0,
 'Positive electrode specific heat capacity [J.kg-1.K-1]': 700.0,
 'Positive electrode thermal conductivity [W.m-1.K-1]': 2.1,
 'Positive electrode thickness [m]': 7.56e-05,
 'Positive electrode volume change': volume_change_Ai2020,
 'Positive particle radius [m]': 5.22e-06,
 'Ratio of lithium moles to SEI moles': 1.0,
 'Reference temperature [K]': 298.15,
 'SEI growth activation energy [J.mol-1]': 38000.0,
 'SEI kinetic rate constant [m.s-1]': 1e-12,
 'SEI open-circuit potential [V]': 0.4,
 'SEI reaction exchange current density [A.m-2]': 1.5e-07,
 'SEI resistivity [Ohm.m]': 200000.0,
 'Separator Bruggeman coefficient (electrolyte)': 1.5,
 'Separator density [kg.m-3]': 397.0,
 'Separator porosity': 0.47,
 'Separator specific heat capacity [J.kg-1.K-1]': 700.0,
 'Separator thermal conductivity [W.m-1.K-1]': 0.16,
 'Separator thickness [m]': 1.2e-05,
 'Total heat transfer coefficient [W.m-2.K-1]': 10.0,
 'Typical current [A]': 5.0,
 'Typical electrolyte concentration [mol.m-3]': 1000.0,
 'Typical plated lithium concentration [mol.m-3]': 1000.0,
 'Upper voltage cut-off [V]': 4.2}