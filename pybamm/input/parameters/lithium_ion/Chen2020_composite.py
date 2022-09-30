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

def silicon_ocp_lithiation_Mark2016(sto):
    """
    silicon Open Circuit Potential (OCP) as a a function of the
    stochiometry. The fit is taken from the Enertech cell [1], which is only accurate
    for 0 < sto < 1.

    References
    ----------
    .. [1] Verbrugge M, Baker D, Xiao X. Formulation for the treatment of multiple
    electrochemical reactions and associated speciation for the Lithium-Silicon
    electrode[J]. Journal of The Electrochemical Society, 2015, 163(2): A262.

    Parameters
    ----------
    sto: double
       Stochiometry of material (li-fraction)

    Returns
    -------
    :class:`pybamm.Symbol`
        OCP [V]
    """
    p1 = -96.63
    p2 = 372.6
    p3 = -587.6
    p4 = 489.9
    p5 = -232.8
    p6 = 62.99
    p7 = -9.286
    p8 = 0.8633

    U_lithiation = (
        p1 * sto**7
        + p2 * sto**6
        + p3 * sto**5
        + p4 * sto**4
        + p5 * sto**3
        + p6 * sto**2
        + p7 * sto
        + p8
    )
    return U_lithiation

def silicon_ocp_delithiation_Mark2016(sto):
    """
    silicon Open Circuit Potential (OCP) as a a function of the
    stochiometry. The fit is taken from the Enertech cell [1], which is only accurate
    for 0 < sto < 1.

    References
    ----------
    .. [1] Verbrugge M, Baker D, Xiao X. Formulation for the treatment of multiple
    electrochemical reactions and associated speciation for the Lithium-Silicon
    electrode[J]. Journal of The Electrochemical Society, 2015, 163(2): A262.

    Parameters
    ----------
    sto: double
       Stochiometry of material (li-fraction)

    Returns
    -------
    :class:`pybamm.Symbol`
        OCP [V]
    """
    p1 = -51.02
    p2 = 161.3
    p3 = -205.7
    p4 = 140.2
    p5 = -58.76
    p6 = 16.87
    p7 = -3.792
    p8 = 0.9937

    U_delithiation = (
        p1 * sto**7
        + p2 * sto**6
        + p3 * sto**5
        + p4 * sto**4
        + p5 * sto**3
        + p6 * sto**2
        + p7 * sto
        + p8
    )
    return U_delithiation

def silicon_LGM50_electrolyte_exchange_current_density_Chen2020(
    c_e, c_s_surf, c_s_max, T
):
    """
    Exchange-current density for Butler-Volmer reactions between silicon and LiPF6 in
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

    m_ref = (
        6.48e-7 * 28700 / 278000
    )  # (A/m2)(mol/m3)**1.5 - includes ref concentrations
    E_r = 35000
    arrhenius = exp(E_r / constants.R * (1 / 298.15 - 1 / T))

    return (
        m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5
    )

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
    c_s_max : :class:`pybamm.Symbol`
        Maximum particle concentration [mol.m-3]
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

def electrolyte_diffusivity_Nyman2008(c_e, T):
    """
    Diffusivity of LiPF6 in EC:EMC (3:7) as a function of ion concentration. The data
    comes from [1]

    References
    ----------
    .. [1] A. Nyman, M. Behm, and G. Lindbergh, "Electrochemical characterisation and
    modelling of the mass transport phenomena in LiPF6-EC-EMC electrolyte,"
    Electrochim. Acta, vol. 53, no. 22, pp. 6356–6365, 2008.

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

    return D_c_e

def electrolyte_conductivity_Nyman2008(c_e, T):
    """
    Conductivity of LiPF6 in EC:EMC (3:7) as a function of ion concentration. The data
    comes from [1].

    References
    ----------
    .. [1] A. Nyman, M. Behm, and G. Lindbergh, "Electrochemical characterisation and
    modelling of the mass transport phenomena in LiPF6-EC-EMC electrolyte,"
    Electrochim. Acta, vol. 53, no. 22, pp. 6356–6365, 2008.

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

    return sigma_e

def get_parameter_values():
    return {'1 + dlnf/dlnc': 1.0,
 'Ambient temperature [K]': 298.15,
 'Cation transference number': 0.2594,
 'Cell cooling surface area [m2]': 0.00531,
 'Cell thermal expansion coefficient [m.K-1]': 1.1e-06,
 'Cell volume [m3]': 2.42e-05,
 'Current function [A]': 5.0,
 'Electrode height [m]': 0.065,
 'Electrode width [m]': 1.58,
 'Electrolyte conductivity [S.m-1]': electrolyte_conductivity_Nyman2008,
 'Electrolyte diffusivity [m2.s-1]': electrolyte_diffusivity_Nyman2008,
 'Initial concentration in electrolyte [mol.m-3]': 1000.0,
 'Initial concentration in negative electrode [mol.m-3]': 29866.0,
 'Initial concentration in positive electrode [mol.m-3]': 17038.0,
 'Initial temperature [K]': 298.15,
 'Lower voltage cut-off [V]': 2.5,
 'Maximum concentration in positive electrode [mol.m-3]': 63104.0,
 'Negative current collector conductivity [S.m-1]': 58411000.0,
 'Negative current collector density [kg.m-3]': 8960.0,
 'Negative current collector specific heat capacity [J.kg-1.K-1]': 385.0,
 'Negative current collector thermal conductivity [W.m-1.K-1]': 401.0,
 'Negative current collector thickness [m]': 1.2e-05,
 'Negative electrode Bruggeman coefficient (electrode)': 1.5,
 'Negative electrode Bruggeman coefficient (electrolyte)': 1.5,
 'Negative electrode cation signed stoichiometry': -1.0,
 'Negative electrode charge transfer coefficient': 0.5,
 'Negative electrode conductivity [S.m-1]': 215.0,
 'Negative electrode double-layer capacity [F.m-2]': 0.2,
 'Negative electrode porosity': 0.25,
 'Negative electrode specific heat capacity [J.kg-1.K-1]': 700.0,
 'Negative electrode thermal conductivity [W.m-1.K-1]': 1.7,
 'Negative electrode thickness [m]': 8.52e-05,
 'Nominal cell capacity [A.h]': 5.0,
 'Number of cells connected in series to make a battery': 1.0,
 'Number of electrodes connected in parallel to make a cell': 1.0,
 'Positive current collector conductivity [S.m-1]': 36914000.0,
 'Positive current collector density [kg.m-3]': 2700.0,
 'Positive current collector specific heat capacity [J.kg-1.K-1]': 897.0,
 'Positive current collector thermal conductivity [W.m-1.K-1]': 237.0,
 'Positive current collector thickness [m]': 1.6e-05,
 'Positive electrode Bruggeman coefficient (electrode)': 1.5,
 'Positive electrode Bruggeman coefficient (electrolyte)': 1.5,
 'Positive electrode OCP [V]': nmc_LGM50_ocp_Chen2020,
 'Positive electrode OCP entropic change [V.K-1]': 0.0,
 'Positive electrode active material volume fraction': 0.665,
 'Positive electrode cation signed stoichiometry': -1.0,
 'Positive electrode charge transfer coefficient': 0.5,
 'Positive electrode conductivity [S.m-1]': 0.18,
 'Positive electrode density [kg.m-3]': 3262.0,
 'Positive electrode diffusivity [m2.s-1]': 4e-15,
 'Positive electrode double-layer capacity [F.m-2]': 0.2,
 'Positive electrode electrons in reaction': 1.0,
 'Positive electrode exchange-current density [A.m-2]': nmc_LGM50_electrolyte_exchange_current_density_Chen2020,
 'Positive electrode porosity': 0.335,
 'Positive electrode specific heat capacity [J.kg-1.K-1]': 700.0,
 'Positive electrode thermal conductivity [W.m-1.K-1]': 2.1,
 'Positive electrode thickness [m]': 7.56e-05,
 'Positive particle radius [m]': 5.22e-06,
 'Primary: EC diffusivity [m2.s-1]': 2e-18,
 'Primary: EC initial concentration in electrolyte [mol.m-3]': 4541.0,
 'Primary: Initial concentration in negative electrode [mol.m-3]': 27700.0,
 'Primary: Initial inner SEI thickness [m]': 2.5e-09,
 'Primary: Initial outer SEI thickness [m]': 2.5e-09,
 'Primary: Inner SEI partial molar volume [m3.mol-1]': 9.585e-05,
 'Primary: Maximum concentration in negative electrode [mol.m-3]': 28700.0,
 'Primary: Negative electrode OCP [V]': ('graphite_ocp_Enertech_Ai2020,
                                         ([array([0.00000000e+00, 5.00000000e-04, 1.27041000e-03, 1.52479000e-03,
       1.90595000e-03, 2.22355800e-03, 4.06054700e-03, 4.82015100e-03,
       6.46394300e-03, 7.41337000e-03, 8.61650600e-03, 9.12341700e-03,
       1.07682260e-02, 1.26650460e-02, 1.41183440e-02, 1.77867520e-02,
       2.06946900e-02, 2.39837990e-02, 3.05021750e-02, 3.60011350e-02,
       3.96066620e-02, 5.91480830e-02, 6.12979420e-02, 7.13498330e-02,
       8.02655260e-02, 1.19208079e-01, 1.28120548e-01, 1.34253707e-01,
       1.41584594e-01, 1.50874177e-01, 1.60609131e-01, 1.70345957e-01,
       1.89747769e-01, 2.09222253e-01, 2.19017730e-01, 2.28756579e-01,
       2.38552575e-01, 2.48349231e-01, 2.58084023e-01, 2.67821184e-01,
       2.87415350e-01, 2.97209811e-01, 3.07004942e-01, 3.16798396e-01,
       3.26534032e-01, 3.36321558e-01, 3.46061758e-01, 3.55856392e-01,
       3.65593044e-01, 3.75388012e-01, 3.85120781e-01, 3.94915577e-01,
       4.04717479e-01, 4.14512102e-01, 4.24244871e-01, 4.34039331e-01,
       4.43770240e-01, 4.53564862e-01, 4.63298139e-01, 4.73034456e-01,
       4.82766544e-01, 4.92564552e-01, 5.02302892e-01, 5.12042595e-01,
       5.21833161e-01, 5.31572182e-01, 5.41369033e-01, 5.51104831e-01,
       5.60899800e-01, 5.70635608e-01, 5.80434806e-01, 5.90235692e-01,
       5.99977407e-01, 6.09716266e-01, 6.19517822e-01, 6.29313635e-01,
       6.39049108e-01, 6.48790152e-01, 6.58584104e-01, 6.68320248e-01,
       6.78055040e-01, 6.87851869e-01, 6.97649380e-01, 7.07389072e-01,
       7.17188097e-01, 7.26977148e-01, 7.36776336e-01, 7.46515866e-01,
       7.56259106e-01, 7.66055091e-01, 7.75789039e-01, 7.85537861e-01,
       7.95329790e-01, 8.05080646e-01, 8.14827099e-01, 8.24570003e-01,
       8.34370889e-01, 8.44173289e-01, 8.53913187e-01, 8.63650510e-01,
       8.73392073e-01, 8.83126865e-01, 8.92918286e-01, 9.02708516e-01,
       9.12443308e-01, 9.22232533e-01, 9.32019724e-01, 9.41812832e-01,
       9.51602392e-01, 9.61392795e-01, 9.70177652e-01, 9.76051358e-01,
       9.80413449e-01, 9.83887804e-01, 9.86792703e-01, 9.89255096e-01,
       9.91401407e-01, 9.93359929e-01, 9.95130154e-01, 9.96776304e-01,
       9.98229440e-01, 9.99241066e-01, 9.99746961e-01, 9.99936448e-01,
       1.00000000e+00])],
                                          array([3.5       , 3.        , 1.04      , 1.01      , 0.97265384,
       0.94249055, 0.81624059, 0.78028093, 0.71896262, 0.69137476,
       0.66139178, 0.64996223, 0.6165173 , 0.58331086, 0.56083078,
       0.51243948, 0.48025136, 0.44849587, 0.39598881, 0.35950768,
       0.33847798, 0.25631956, 0.25117361, 0.23605532, 0.23100922,
       0.2232966 , 0.21828424, 0.21327386, 0.20822836, 0.20320974,
       0.19862098, 0.19381638, 0.18416691, 0.17679053, 0.17383044,
       0.17096326, 0.1679035 , 0.16464998, 0.16149133, 0.15859383,
       0.15339916, 0.15100232, 0.14886213, 0.14691891, 0.14532814,
       0.14400211, 0.14290212, 0.14201426, 0.14131601, 0.1407591 ,
       0.14031432, 0.13994232, 0.13961785, 0.13932541, 0.13905101,
       0.1387793 , 0.13851741, 0.1382589 , 0.13798129, 0.13767223,
       0.13732933, 0.13690322, 0.13639024, 0.13575758, 0.1349471 ,
       0.13392324, 0.13262168, 0.13098947, 0.12896492, 0.12654999,
       0.12374288, 0.12077083, 0.11792963, 0.11537998, 0.11320542,
       0.11136648, 0.10985549, 0.10857895, 0.10752068, 0.10663254,
       0.10589376, 0.10526061, 0.10471319, 0.10425437, 0.10384562,
       0.10347712, 0.10315393, 0.10285654, 0.10258744, 0.10233828,
       0.10210199, 0.1018809 , 0.10167642, 0.10146588, 0.10126417,
       0.10106263, 0.10087041, 0.10068096, 0.10048922, 0.10030044,
       0.10009972, 0.0998771 , 0.09962899, 0.09933262, 0.09895842,
       0.09844254, 0.09768364, 0.096492  , 0.09451079, 0.09113682,
       0.08611519, 0.08107875, 0.07604037, 0.07099153, 0.06589833,
       0.06084405, 0.05581012, 0.0506707 , 0.0455624 , 0.04039266,
       0.03526127, 0.03024266, 0.02485077, 0.0192515 , 0.00499468]))),
 'Primary: Negative electrode OCP entropic change [V.K-1]': 0.0,
 'Primary: Negative electrode active material volume fraction': 0.735,
 'Primary: Negative electrode density [kg.m-3]': 1657.0,
 'Primary: Negative electrode diffusivity [m2.s-1]': 5.5e-14,
 'Primary: Negative electrode electrons in reaction': 1.0,
 'Primary: Negative electrode exchange-current density [A.m-2]': graphite_LGM50_electrolyte_exchange_current_density_Chen2020,
 'Primary: Negative particle radius [m]': 5.86e-06,
 'Primary: Outer SEI partial molar volume [m3.mol-1]': 9.585e-05,
 'Primary: Ratio of lithium moles to SEI moles': 2.0,
 'Primary: SEI growth activation energy [J.mol-1]': 0.0,
 'Primary: SEI kinetic rate constant [m.s-1]': 1e-12,
 'Primary: SEI open-circuit potential [V]': 0.4,
 'Primary: SEI resistivity [Ohm.m]': 200000.0,
 'Reference temperature [K]': 298.15,
 'Secondary: EC diffusivity [m2.s-1]': 2e-18,
 'Secondary: EC initial concentration in electrolyte [mol.m-3]': 4541.0,
 'Secondary: Initial concentration in negative electrode [mol.m-3]': 276610.0,
 'Secondary: Initial inner SEI thickness [m]': 2.5e-09,
 'Secondary: Initial outer SEI thickness [m]': 2.5e-09,
 'Secondary: Inner SEI partial molar volume [m3.mol-1]': 9.585e-05,
 'Secondary: Maximum concentration in negative electrode [mol.m-3]': 278000.0,
 'Secondary: Negative electrode OCP entropic change [V.K-1]': 0.0,
 'Secondary: Negative electrode active material volume fraction': 0.015,
 'Secondary: Negative electrode delithiation OCP [V]': silicon_ocp_delithiation_Mark2016,
 'Secondary: Negative electrode density [kg.m-3]': 2650.0,
 'Secondary: Negative electrode diffusivity [m2.s-1]': 1.67e-14,
 'Secondary: Negative electrode electrons in reaction': 1.0,
 'Secondary: Negative electrode exchange-current density [A.m-2]': silicon_LGM50_electrolyte_exchange_current_density_Chen2020,
 'Secondary: Negative electrode lithiation OCP [V]': silicon_ocp_lithiation_Mark2016,
 'Secondary: Negative particle radius [m]': 1.52e-06,
 'Secondary: Outer SEI partial molar volume [m3.mol-1]': 9.585e-05,
 'Secondary: Ratio of lithium moles to SEI moles': 2.0,
 'Secondary: SEI growth activation energy [J.mol-1]': 0.0,
 'Secondary: SEI kinetic rate constant [m.s-1]': 1e-12,
 'Secondary: SEI open-circuit potential [V]': 0.4,
 'Secondary: SEI resistivity [Ohm.m]': 200000.0,
 'Separator Bruggeman coefficient (electrolyte)': 1.5,
 'Separator density [kg.m-3]': 397.0,
 'Separator porosity': 0.47,
 'Separator specific heat capacity [J.kg-1.K-1]': 700.0,
 'Separator thermal conductivity [W.m-1.K-1]': 0.16,
 'Separator thickness [m]': 1.2e-05,
 'Total heat transfer coefficient [W.m-2.K-1]': 10.0,
 'Typical current [A]': 5.0,
 'Typical electrolyte concentration [mol.m-3]': 1000.0,
 'Upper voltage cut-off [V]': 4.2}