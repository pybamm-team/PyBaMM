def graphite_diffusivity_Dualfoil1998(sto, T):
    """
    Graphite diffusivity as a function of stochiometry [1, 2, 3].

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
        Electrode stochiometry
    T: :class:`pybamm.Symbol`
        Dimensional temperature, [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity [m2.s-1]
    """
    D_ref = 3.9 * 10 ** (-14)
    E_D_s = 5000
    T_ref = Parameter("Reference temperature [K]")
    arrhenius = exp(E_D_s / constants.R * (1 / T_ref - 1 / T))
    return D_ref * arrhenius

def graphite_electrolyte_exchange_current_density_Dualfoil1998(
    c_e, c_s_surf, c_s_max, T
):
    """
    Exchange-current density for Butler-Volmer reactions between graphite and LiPF6 in
    EC:DMC.

    References
    ----------
    .. [2] http://www.cchem.berkeley.edu/jsngrp/fortran.html

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
        1 * 10 ** (-11) * constants.F
    )  # (A/m2)(mol/m3)**1.5 - includes ref concentrations
    E_r = 5000  # activation energy for Temperature Dependent Reaction Constant [J/mol]
    arrhenius = exp(E_r / constants.R * (1 / 298.15 - 1 / T))

    return (
        m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5
    )

def graphite_entropy_Enertech_Ai2020_function(sto, c_s_max):
    """
    Lithium Cobalt Oxide (LiCO2) entropic change in open circuit potential (OCP) at
    a temperature of 298.15K as a function of the stochiometry. The fit is taken
    from Ref [1], which is only accurate
    for 0.43 < sto < 0.9936.

    References
    ----------
    .. [1] Ai, W., Kraft, L., Sturm, J., Jossen, A., & Wu, B. (2020).
    Electrochemical Thermal-Mechanical Modelling of Stress Inhomogeneity in Lithium-Ion Pouch Cells. # noqa
    Journal of The Electrochemical Society, 167(1), 013512. DOI: 10.1149/2.0122001JES # noqa

    Parameters
    ----------
    sto: double
       Stochiometry of material (li-fraction)

    Returns
    -------
    :class:`pybamm.Symbol`
        Entropic change [V.K-1]
    """

    du_dT = (
        0.001
        * (
            0.005269056
            + 3.299265709 * sto
            - 91.79325798 * sto**2
            + 1004.911008 * sto**3
            - 5812.278127 * sto**4
            + 19329.7549 * sto**5
            - 37147.8947 * sto**6
            + 38379.18127 * sto**7
            - 16515.05308 * sto**8
        )
        / (
            1
            - 48.09287227 * sto
            + 1017.234804 * sto**2
            - 10481.80419 * sto**3
            + 59431.3 * sto**4
            - 195881.6488 * sto**5
            + 374577.3152 * sto**6
            - 385821.1607 * sto**7
            + 165705.8597 * sto**8
        )
    )

    return du_dT

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
    c_s_max : :class:`pybamm.Symbol`
        Maximum particle concentration [mol.m-3]

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
    graphite particle cracking rate as a function of temperature [1, 2].

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
    T_ref = Parameter("Reference temperature [K]")
    Eac_cr = Parameter(
        "Negative electrode activation energy for cracking rate [J.mol-1]"
    )
    arrhenius = exp(Eac_cr / constants.R * (1 / T_dim - 1 / T_ref))
    return k_cr * arrhenius

def lico2_diffusivity_Dualfoil1998(sto, T):
    """
    LiCo2 diffusivity as a function of stochiometry, in this case the
    diffusivity is taken to be a constant. The value is taken from Dualfoil [1].

    References
    ----------
    .. [1] http://www.cchem.berkeley.edu/jsngrp/fortran.html

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry
    T: :class:`pybamm.Symbol`
        Dimensional temperature, [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity [m2.s-1]
    """
    D_ref = 5.387 * 10 ** (-15)
    E_D_s = 5000
    T_ref = Parameter("Reference temperature [K]")
    arrhenius = exp(E_D_s / constants.R * (1 / T_ref - 1 / T))
    return D_ref * arrhenius

def lico2_electrolyte_exchange_current_density_Dualfoil1998(c_e, c_s_surf, c_s_max, T):
    """
    Exchange-current density for Butler-Volmer reactions between lico2 and LiPF6 in
    EC:DMC.

    References
    ----------
    .. [2] http://www.cchem.berkeley.edu/jsngrp/fortran.html

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
    m_ref = 1 * 10 ** (-11) * constants.F  # need to match the unit from m/s
    # (A/m2)(mol/m3)**1.5 - includes ref concentrations
    E_r = 5000
    arrhenius = exp(E_r / constants.R * (1 / 298.15 - 1 / T))

    return (
        m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5
    )

def lico2_entropic_change_Ai2020_function(sto, c_s_max):
    """
    Lithium Cobalt Oxide (LiCO2) entropic change in open circuit potential (OCP) at
    a temperature of 298.15K as a function of the stochiometry. The fit is taken
    from Ref [1], which is only accurate
    for 0.43 < sto < 0.9936.

    References
    ----------
    .. [1] Ai, W., Kraft, L., Sturm, J., Jossen, A., & Wu, B. (2020).
    Electrochemical Thermal-Mechanical Modelling of Stress Inhomogeneity
    in Lithium-Ion Pouch Cells. Journal of The Electrochemical Society,
        167(1), 013512. DOI: 10.1149/2.0122001JES

    Parameters
    ----------
    sto: double
       Stochiometry of material (li-fraction)

    Returns
    -------
    :class:`pybamm.Symbol`
        Entropic change [V.K-1]
    """

    # Since the equation for LiCo2 from this ref. has the stretch factor,
    # should this too? If not, the "bumps" in the OCV don't line up.
    p1 = -3.20392657
    p2 = 14.5719049
    p3 = -27.9047599
    p4 = 29.1744564
    p5 = -17.992018
    p6 = 6.54799331
    p7 = -1.30382445
    p8 = 0.109667298

    du_dT = (
        p1 * sto**7
        + p2 * sto**6
        + p3 * sto**5
        + p4 * sto**4
        + p5 * sto**3
        + p6 * sto**2
        + p7 * sto
        + p8
    )

    return du_dT

def lico2_volume_change_Ai2020(sto, c_s_max):
    """
    lico2 particle volume change as a function of stochiometry [1, 2].

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
    c_s_max : :class:`pybamm.Symbol`
        Maximum particle concentration [mol.m-3]

    Returns
    -------
    t_change:class:`pybamm.Symbol`
        volume change, dimensionless, normalised by particle volume
    """
    omega = Parameter("Positive electrode partial molar volume [m3.mol-1]")
    t_change = omega * c_s_max * sto
    return t_change

def lico2_cracking_rate_Ai2020(T_dim):
    """
    lico2 particle cracking rate as a function of temperature [1, 2].

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
    T: :class:`pybamm.Symbol`
        temperature, [K]

    Returns
    -------
    k_cr: :class:`pybamm.Symbol`
        cracking rate, [m/(Pa.m0.5)^m_cr]
        where m_cr is another Paris' law constant
    """
    k_cr = 3.9e-20
    T_ref = Parameter("Reference temperature [K]")
    Eac_cr = Parameter(
        "Positive electrode activation energy for cracking rate [J.mol-1]"
    )
    arrhenius = exp(Eac_cr / constants.R * (1 / T_dim - 1 / T_ref))
    return k_cr * arrhenius

def dlnf_dlnc_Ai2020(c_e, T, T_ref=298.3, t_plus=0.38):
    """
    Activity dependence of LiPF6 in EC:DMC as a function of ion concentration.

    References
    ----------
    .. [1] Ai, W., Kraft, L., Sturm, J., Jossen, A., & Wu, B. (2020).
    Electrochemical Thermal-Mechanical Modelling of Stress Inhomogeneity
    in Lithium-Ion Pouch Cells. Journal of The Electrochemical Society,
    167(1), 013512. DOI: 10.1149/2.0122001JES.

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration, mol/m^3
    T: :class:`pybamm.Symbol`
        Dimensional temperature, K

    Returns
    -------
    :class:`pybamm.Symbol`
        1 + dlnf/dlnc
    """
    T_ref = Parameter("Reference temperature [K]")
    t_plus = Parameter("Cation transference number")
    dlnf_dlnc = (
        0.601
        - 0.24 * (c_e / 1000) ** 0.5
        + 0.982 * (1 - 0.0052 * (T - T_ref)) * (c_e / 1000) ** 1.5
    ) / (1 - t_plus)
    return dlnf_dlnc

def electrolyte_diffusivity_Ai2020(c_e, T):
    """
    Diffusivity of LiPF6 in EC:DMC as a function of ion concentration.

    References
    ----------
    .. [1] Ai, W., Kraft, L., Sturm, J., Jossen, A., & Wu, B. (2020).
    Electrochemical Thermal-Mechanical Modelling of Stress Inhomogeneity
    in Lithium-Ion Pouch Cells. Journal of The Electrochemical Society,
    167(1), 013512. DOI: 10.1149/2.0122001JES.

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration, mol/m^3
    T: :class:`pybamm.Symbol`
        Dimensional temperature, K


    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    D_c_e = 10 ** (-8.43 - 54 / (T - 229 - 5e-3 * c_e) - 0.22e-3 * c_e)

    return D_c_e

def electrolyte_conductivity_Ai2020(c_e, T):
    """
    Conductivity of LiPF6 in EC:DMC as a function of ion concentration.
    Concentration should be in dm3 in the function.

    References
    ----------
    .. [1] Ai, W., Kraft, L., Sturm, J., Jossen, A., & Wu, B. (2020).
    Electrochemical Thermal-Mechanical Modelling of Stress Inhomogeneity
    in Lithium-Ion Pouch Cells. Journal of The Electrochemical Society,
    167(1), 013512. DOI: 10.1149/2.0122001JES.
    .. [2] Torchio, Marcello, et al. "Lionsimba: a matlab framework based
    on a finite volume model suitable for li-ion battery design, simulation,
    and control." Journal of The Electrochemical Society 163.7 (2016): A1192.

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
        1e-4
        * c_e
        * (
            (-10.5 + 0.668 * 1e-3 * c_e + 0.494 * 1e-6 * c_e**2)
            + (0.074 - 1.78 * 1e-5 * c_e - 8.86 * 1e-10 * c_e**2) * T
            + (-6.96 * 1e-5 + 2.8 * 1e-8 * c_e) * T**2
        )
        ** 2
    )

    return sigma_e

def get_parameter_values():
    return {'1 + dlnf/dlnc': dlnf_dlnc_Ai2020,
 'Ambient temperature [K]': 298.15,
 'Bulk solvent concentration [mol.m-3]': 2636.0,
 'Cation transference number': 0.38,
 'Cell cooling surface area [m2]': 0.0060484,
 'Cell emissivity': 0.95,
 'Cell thermal expansion coefficient [m.K-1]': 1.1e-06,
 'Cell volume [m3]': 1.5341e-05,
 'Current function [A]': 2.28,
 'EC diffusivity [m2.s-1]': 2e-18,
 'EC initial concentration in electrolyte [mol.m-3]': 4541.0,
 'Electrode height [m]': 0.051,
 'Electrode width [m]': 0.047,
 'Electrolyte conductivity [S.m-1]': electrolyte_conductivity_Ai2020,
 'Electrolyte diffusivity [m2.s-1]': electrolyte_diffusivity_Ai2020,
 'Initial concentration in electrolyte [mol.m-3]': 1000.0,
 'Initial concentration in negative electrode [mol.m-3]': 24108.0,
 'Initial concentration in positive electrode [mol.m-3]': 21725.0,
 'Initial inner SEI thickness [m]': 2.5e-09,
 'Initial outer SEI thickness [m]': 2.5e-09,
 'Initial temperature [K]': 298.15,
 'Inner SEI electron conductivity [S.m-1]': 8.95e-14,
 'Inner SEI lithium interstitial diffusivity [m2.s-1]': 1e-20,
 'Inner SEI open-circuit potential [V]': 0.1,
 'Inner SEI partial molar volume [m3.mol-1]': 9.585e-05,
 'Inner SEI reaction proportion': 0.5,
 'Lithium interstitial reference concentration [mol.m-3]': 15.0,
 'Lower voltage cut-off [V]': 3.0,
 'Maximum concentration in negative electrode [mol.m-3]': 28700.0,
 'Maximum concentration in positive electrode [mol.m-3]': 49943.0,
 'Negative current collector conductivity [S.m-1]': 58411000.0,
 'Negative current collector density [kg.m-3]': 8960.0,
 'Negative current collector specific heat capacity [J.kg-1.K-1]': 385.0,
 'Negative current collector thermal conductivity [W.m-1.K-1]': 401.0,
 'Negative current collector thickness [m]': 1e-05,
 'Negative electrode Bruggeman coefficient (electrode)': 0.0,
 'Negative electrode Bruggeman coefficient (electrolyte)': 2.914,
 'Negative electrode LAM constant exponential term': 2.0,
 'Negative electrode LAM constant proportional term [s-1]': 0.0,
 'Negative electrode OCP [V]': ('graphite_ocp_Enertech_Ai2020,
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
 'Negative electrode OCP entropic change [V.K-1]': graphite_entropy_Enertech_Ai2020_function,
 "Negative electrode Paris' law constant b": 1.12,
 "Negative electrode Paris' law constant m": 2.2,
 "Negative electrode Poisson's ratio": 0.3,
 "Negative electrode Young's modulus [Pa]": 15000000000.0,
 'Negative electrode activation energy for cracking rate [J.mol-1]': 0.0,
 'Negative electrode active material volume fraction': 0.61,
 'Negative electrode cation signed stoichiometry': -1.0,
 'Negative electrode charge transfer coefficient': 0.5,
 'Negative electrode conductivity [S.m-1]': 100.0,
 'Negative electrode cracking rate': graphite_cracking_rate_Ai2020,
 'Negative electrode critical stress [Pa]': 60000000.0,
 'Negative electrode density [kg.m-3]': 2470.0,
 'Negative electrode diffusivity [m2.s-1]': graphite_diffusivity_Dualfoil1998,
 'Negative electrode double-layer capacity [F.m-2]': 0.2,
 'Negative electrode electrons in reaction': 1.0,
 'Negative electrode exchange-current density [A.m-2]': graphite_electrolyte_exchange_current_density_Dualfoil1998,
 'Negative electrode initial crack length [m]': 2e-08,
 'Negative electrode initial crack width [m]': 1.5e-08,
 'Negative electrode number of cracks per unit area [m-2]': 3180000000000000.0,
 'Negative electrode partial molar volume [m3.mol-1]': 3.1e-06,
 'Negative electrode porosity': 0.33,
 'Negative electrode reaction-driven LAM factor [m3.mol-1]': 0.0,
 'Negative electrode reference concentration for free of deformation [mol.m-3]': 0.0,
 'Negative electrode specific heat capacity [J.kg-1.K-1]': 1080.2,
 'Negative electrode thermal conductivity [W.m-1.K-1]': 1.04,
 'Negative electrode thickness [m]': 7.65e-05,
 'Negative electrode volume change': graphite_volume_change_Ai2020,
 'Negative particle radius [m]': 5e-06,
 'Nominal cell capacity [A.h]': 2.28,
 'Number of cells connected in series to make a battery': 1.0,
 'Number of electrodes connected in parallel to make a cell': 34.0,
 'Outer SEI open-circuit potential [V]': 0.8,
 'Outer SEI partial molar volume [m3.mol-1]': 9.585e-05,
 'Outer SEI solvent diffusivity [m2.s-1]': 2.5000000000000002e-22,
 'Positive current collector conductivity [S.m-1]': 36914000.0,
 'Positive current collector density [kg.m-3]': 2700.0,
 'Positive current collector specific heat capacity [J.kg-1.K-1]': 897.0,
 'Positive current collector thermal conductivity [W.m-1.K-1]': 237.0,
 'Positive current collector thickness [m]': 1.5e-05,
 'Positive electrode Bruggeman coefficient (electrode)': 0.0,
 'Positive electrode Bruggeman coefficient (electrolyte)': 1.83,
 'Positive electrode LAM constant exponential term': 2.0,
 'Positive electrode LAM constant proportional term [s-1]': 2.78e-13,
 'Positive electrode OCP [V]': ('lico2_ocp_Ai2020,
                                ([array([0.43      , 0.43663978, 0.43790614, 0.4391725 , 0.44043886,
       0.44170522, 0.44297158, 0.44423794, 0.4455043 , 0.44677066,
       0.44803701, 0.44930337, 0.45056973, 0.45183609, 0.45310245,
       0.45436881, 0.45563517, 0.45690153, 0.45816788, 0.45943424,
       0.4607006 , 0.46196696, 0.46323332, 0.46449968, 0.46576604,
       0.4670324 , 0.46829876, 0.46956511, 0.47083147, 0.47209783,
       0.47336419, 0.47463055, 0.47589691, 0.47716327, 0.47842963,
       0.47969599, 0.48096235, 0.4822287 , 0.48349506, 0.48476142,
       0.48602778, 0.48729414, 0.4885605 , 0.48982686, 0.49109322,
       0.49235957, 0.49362593, 0.49489229, 0.49615865, 0.49742501,
       0.49869137, 0.49995773, 0.50122409, 0.50249045, 0.5037568 ,
       0.50502316, 0.50628952, 0.50755588, 0.50882224, 0.5100886 ,
       0.51135496, 0.51262132, 0.51388768, 0.51515404, 0.51642039,
       0.51768675, 0.51895311, 0.52021947, 0.52148583, 0.52275219,
       0.52401855, 0.52528491, 0.52655126, 0.52781762, 0.52908398,
       0.53035034, 0.5316167 , 0.53288306, 0.53414942, 0.53541577,
       0.53668213, 0.53794849, 0.53921485, 0.54048121, 0.54174757,
       0.54301393, 0.54428029, 0.54554665, 0.546813  , 0.54807936,
       0.54934572, 0.55061208, 0.55187844, 0.5531448 , 0.55441116,
       0.55567752, 0.55694387, 0.55821023, 0.55947659, 0.56074295,
       0.56200931, 0.56327567, 0.56454203, 0.56580839, 0.56707475,
       0.56834111, 0.56960746, 0.57087382, 0.57214018, 0.57340654,
       0.5746729 , 0.57593926, 0.57720562, 0.57847198, 0.57973834,
       0.58100469, 0.58227105, 0.58353741, 0.58480377, 0.58607013,
       0.58733649, 0.58860285, 0.58986921, 0.59113556, 0.59240192,
       0.59366828, 0.59493464, 0.596201  , 0.59746736, 0.59873372,
       0.60000008, 0.60126644, 0.6025328 , 0.60379915, 0.60506551,
       0.60633187, 0.60759823, 0.60886459, 0.61013095, 0.61139731,
       0.61266367, 0.61393003, 0.61519638, 0.61646274, 0.6177291 ,
       0.61899546, 0.62026182, 0.62152818, 0.62279454, 0.6240609 ,
       0.62532725, 0.62659361, 0.62785997, 0.62912633, 0.63039269,
       0.63165905, 0.63292541, 0.63419177, 0.63545813, 0.63672449,
       0.63799084, 0.6392572 , 0.64052356, 0.64178992, 0.64305628,
       0.64432264, 0.645589  , 0.64685536, 0.64812172, 0.64938807,
       0.65065443, 0.65192079, 0.65318715, 0.65445351, 0.65571987,
       0.65698623, 0.65825259, 0.65951894, 0.6607853 , 0.66205166,
       0.66331802, 0.66458438, 0.66585074, 0.6671171 , 0.66838346,
       0.66964982, 0.67091618, 0.67218253, 0.67344889, 0.67471525,
       0.67598161, 0.67724797, 0.67851433, 0.67978069, 0.68104705,
       0.68231341, 0.68357976, 0.68484612, 0.68611248, 0.68737884,
       0.6886452 , 0.68991157, 0.69117793, 0.69244429, 0.69371065,
       0.694977  , 0.69624336, 0.69750972, 0.69877608, 0.70004244,
       0.7013088 , 0.70257516, 0.70384152, 0.70510788, 0.70637423,
       0.70764059, 0.70890695, 0.71017331, 0.71143967, 0.71270603,
       0.71397239, 0.71523875, 0.7165051 , 0.71777146, 0.71903782,
       0.72030418, 0.72157054, 0.7228369 , 0.72410326, 0.72536962,
       0.72663598, 0.72790234, 0.72916869, 0.73043505, 0.73170141,
       0.73296777, 0.73423413, 0.73550049, 0.73676685, 0.73803321,
       0.73929957, 0.74056592, 0.74183228, 0.74309864, 0.744365  ,
       0.74563136, 0.74689772, 0.74816408, 0.74943044, 0.75069679,
       0.75196315, 0.75322951, 0.75449587, 0.75576223, 0.75702859,
       0.75829495, 0.75956131, 0.76082767, 0.76209403, 0.76336038,
       0.76462674, 0.7658931 , 0.76715946, 0.76842582, 0.76969218,
       0.77095854, 0.7722249 , 0.77349126, 0.77475761, 0.77602397,
       0.77729033, 0.77855669, 0.77982305, 0.78108941, 0.78235577,
       0.78362213, 0.78488848, 0.78615484, 0.7874212 , 0.78868756,
       0.78995392, 0.79122028, 0.79248664, 0.793753  , 0.79501936,
       0.79628572, 0.79755207, 0.79881843, 0.80008479, 0.80135115,
       0.80261751, 0.80388387, 0.80515023, 0.80641659, 0.80768294,
       0.8089493 , 0.81021566, 0.81148202, 0.81274838, 0.81401474,
       0.8152811 , 0.81654746, 0.81781382, 0.81908017, 0.82034653,
       0.82161289, 0.82287925, 0.82414561, 0.82541197, 0.82667833,
       0.82794469, 0.82921104, 0.8304774 , 0.83174376, 0.83301012,
       0.83427648, 0.83554284, 0.8368092 , 0.83807556, 0.83934192,
       0.84060828, 0.84187463, 0.84314099, 0.84440735, 0.84567371,
       0.84694007, 0.84820643, 0.84947279, 0.85073915, 0.85200551,
       0.85327186, 0.85453822, 0.85580458, 0.85707094, 0.8583373 ,
       0.85960366, 0.86087002, 0.86213638, 0.86340273, 0.86466909,
       0.86593545, 0.86720181, 0.86846817, 0.86973453, 0.87100089,
       0.87226725, 0.87353361, 0.87479997, 0.87606632, 0.87733268,
       0.87859904, 0.8798654 , 0.88113176, 0.88239812, 0.88366448,
       0.88493084, 0.8861972 , 0.88746355, 0.88872991, 0.88999627,
       0.89126263, 0.89252899, 0.89379535, 0.89506171, 0.89632807,
       0.89759442, 0.89886078, 0.90012714, 0.9013935 , 0.90265986,
       0.90392622, 0.90519258, 0.90645894, 0.9077253 , 0.90899166,
       0.91025801, 0.91152437, 0.91279073, 0.91405709, 0.91532345,
       0.91658981, 0.91785617, 0.91912253, 0.92038889, 0.92165524,
       0.9229216 , 0.92418796, 0.92545432, 0.92672068, 0.92798704,
       0.9292534 , 0.93051976, 0.93178611, 0.93305247, 0.93431883,
       0.93558519, 0.93685155, 0.93811791, 0.93938427, 0.94065063,
       0.94191699, 0.94318335, 0.9444497 , 0.94571606, 0.94698242,
       0.94824878, 0.94951514, 0.9507815 , 0.95204786, 0.95331422,
       0.95458058, 0.95584693, 0.95711329, 0.95837965, 0.95964601,
       0.96091237, 0.96217873, 0.96344509, 0.96471145, 0.9659778 ,
       0.96724416, 0.96851052, 0.96977688, 0.97104324, 0.9723096 ,
       0.97357596, 0.97484232, 0.97610868, 0.97737504, 0.97864139,
       0.97990775, 0.98117411, 0.98244047, 0.98370683, 0.98497319,
       0.98623955, 0.98750591, 0.98877227, 0.99003862, 0.99130498,
       0.99257134, 0.9938377 , 0.99510406, 0.99637042, 0.99763678,
       0.99890314])],
                                 array([4.3       , 4.27990775, 4.27647267, 4.27380094, 4.27112921,
       4.26864831, 4.26616742, 4.26368653, 4.26120563, 4.25872474,
       4.25624385, 4.25376295, 4.25128206, 4.248992  , 4.24651111,
       4.24422105, 4.24174016, 4.23925927, 4.23696921, 4.23467916,
       4.23219826, 4.22971737, 4.22723647, 4.22513726, 4.22265636,
       4.22036631, 4.21788541, 4.21559536, 4.2133053 , 4.21101525,
       4.20853436, 4.2062443 , 4.20395424, 4.20166419, 4.19937413,
       4.19708408, 4.19479402, 4.19250397, 4.19021391, 4.18792386,
       4.18582464, 4.18353458, 4.18105369, 4.17895447, 4.17685525,
       4.1745652 , 4.17227514, 4.16998509, 4.16788587, 4.16578665,
       4.16368743, 4.16158822, 4.15929816, 4.15738978, 4.15529056,
       4.15319135, 4.15109213, 4.14899291, 4.14689369, 4.14479448,
       4.14250442, 4.1404052 , 4.13830599, 4.13620677, 4.13410755,
       4.13181749, 4.12971828, 4.12761906, 4.12551984, 4.12322979,
       4.12113057, 4.11922219, 4.11693213, 4.11502376, 4.1127337 ,
       4.11063448, 4.1087261 , 4.10681772, 4.10471851, 4.10261929,
       4.10090174, 4.09861169, 4.09670331, 4.09498577, 4.09288655,
       4.09097817, 4.08906979, 4.08735225, 4.08544387, 4.08353549,
       4.08181795, 4.07990957, 4.07819203, 4.07647449, 4.07456611,
       4.07284857, 4.07113102, 4.06941348, 4.06769594, 4.0659784 ,
       4.06445169, 4.06273415, 4.06120745, 4.05929907, 4.05777237,
       4.05605483, 4.05452812, 4.05281058, 4.05128388, 4.04956633,
       4.04803963, 4.04632209, 4.04479538, 4.04326868, 4.04155114,
       4.04002444, 4.03830689, 4.03697103, 4.03525349, 4.03372678,
       4.03220008, 4.03067338, 4.02914667, 4.02761997, 4.02609327,
       4.02456656, 4.02303986, 4.02170399, 4.02017729, 4.01884142,
       4.01731472, 4.01578802, 4.01426131, 4.01292544, 4.01158958,
       4.01025371, 4.00872701, 4.00739114, 4.00605528, 4.00471941,
       4.00338355, 4.00185684, 4.00071182, 3.99937595, 3.99784925,
       3.99670422, 3.99536835, 3.99403249, 3.99288746, 3.99155159,
       3.99021573, 3.9890707 , 3.98773483, 3.98639897, 3.98525394,
       3.98410891, 3.98277305, 3.98162802, 3.98029215, 3.97914713,
       3.97819294, 3.97685707, 3.97571204, 3.97456701, 3.97342199,
       3.97227696, 3.97094109, 3.9699869 , 3.96884188, 3.96769685,
       3.96674266, 3.96559763, 3.9644526 , 3.96349841, 3.96235339,
       3.96120836, 3.96025417, 3.95910914, 3.95815495, 3.95720076,
       3.95605573, 3.95510154, 3.95414735, 3.95300233, 3.95204814,
       3.95128478, 3.95013976, 3.94918557, 3.94823138, 3.94727719,
       3.94651384, 3.94536881, 3.94460546, 3.94365127, 3.94269708,
       3.94193372, 3.94097953, 3.94002534, 3.93926199, 3.9383078 ,
       3.93754445, 3.93659026, 3.93582691, 3.93487272, 3.93410937,
       3.93334602, 3.93258266, 3.93181931, 3.93086512, 3.93029261,
       3.92933842, 3.92857507, 3.92781171, 3.92704836, 3.92628501,
       3.92552166, 3.92494915, 3.92418579, 3.92361328, 3.92284993,
       3.92208658, 3.92132322, 3.92075071, 3.91998736, 3.91922401,
       3.91865149, 3.91807898, 3.91750647, 3.91693395, 3.91636144,
       3.91578892, 3.91502557, 3.9146439 , 3.91388054, 3.91330803,
       3.91273552, 3.912163  , 3.91178133, 3.91120881, 3.9106363 ,
       3.91006379, 3.90949127, 3.90891876, 3.90853708, 3.90815541,
       3.90758289, 3.90701038, 3.9066287 , 3.90605619, 3.90586535,
       3.90529284, 3.90491116, 3.90452948, 3.90395697, 3.90357529,
       3.90319362, 3.90281194, 3.9026211 , 3.90223943, 3.90185775,
       3.90147608, 3.9010944 , 3.90090356, 3.90052189, 3.90014021,
       3.89994937, 3.89975854, 3.89937686, 3.89918602, 3.89899518,
       3.89861351, 3.89842267, 3.89804099, 3.89785015, 3.89765932,
       3.89746848, 3.89727764, 3.8970868 , 3.89670513, 3.89670513,
       3.89651429, 3.89632345, 3.89632345, 3.89613261, 3.89575094,
       3.89575094, 3.8955601 , 3.89536926, 3.89517842, 3.89498759,
       3.89479675, 3.89460591, 3.89460591, 3.89441507, 3.89422423,
       3.8940334 , 3.8940334 , 3.89384256, 3.89365172, 3.89365172,
       3.89346088, 3.89327004, 3.89327004, 3.89307921, 3.89288837,
       3.89269753, 3.89269753, 3.89250669, 3.89231585, 3.89212502,
       3.89193418, 3.89193418, 3.89174334, 3.8915525 , 3.89136166,
       3.89117083, 3.89097999, 3.89097999, 3.89078915, 3.89059831,
       3.89040747, 3.89021664, 3.8900258 , 3.88983496, 3.88983496,
       3.88964412, 3.88926245, 3.88907161, 3.88888077, 3.88868993,
       3.88849909, 3.88830826, 3.88811742, 3.88773574, 3.8875449 ,
       3.88735407, 3.88716323, 3.88697239, 3.88659072, 3.88620904,
       3.8860182 , 3.88582736, 3.88544569, 3.88506401, 3.88487317,
       3.8844915 , 3.88430066, 3.88391898, 3.88372815, 3.88334647,
       3.88277396, 3.88258312, 3.88220144, 3.88181977, 3.88143809,
       3.88105642, 3.88067474, 3.88010222, 3.87952971, 3.87914804,
       3.87857552, 3.87819385, 3.87781217, 3.87723965, 3.87666714,
       3.87609463, 3.87552211, 3.8749496 , 3.87437709, 3.87361373,
       3.87304122, 3.87246871, 3.87170535, 3.87113284, 3.87036949,
       3.8694153 , 3.86884279, 3.86807943, 3.86712524, 3.86617105,
       3.8654077 , 3.86445351, 3.86349932, 3.86254513, 3.8614001 ,
       3.86025508, 3.85930089, 3.85815586, 3.85681999, 3.85567497,
       3.85414826, 3.85300323, 3.85147653, 3.84994983, 3.84842312,
       3.84689642, 3.84498804, 3.8432705 , 3.84136212, 3.8392629 ,
       3.83716368, 3.83506447, 3.83258357, 3.83010268, 3.82743095,
       3.82475922, 3.82189665, 3.81884324, 3.81559899, 3.81216391,
       3.80853799, 3.80491207, 3.80090447, 3.79689687, 3.79269844,
       3.78830917, 3.78391989, 3.77914894, 3.77456883, 3.76960704,
       3.76502693, 3.76006515, 3.75510336, 3.75014157, 3.74517979,
       3.740218  , 3.73506538, 3.72991275, 3.72476012, 3.71941666,
       3.71388236, 3.70815722, 3.70224125, 3.69613443, 3.6894551 ,
       3.6823941 , 3.67476058, 3.66636371, 3.65663097, 3.64556237,
       3.63220371, 3.61521913, 3.5919369 , 3.55720439, 3.50109804,
       3.4089233 ]))),
 'Positive electrode OCP entropic change [V.K-1]': lico2_entropic_change_Ai2020_function,
 "Positive electrode Paris' law constant b": 1.12,
 "Positive electrode Paris' law constant m": 2.2,
 "Positive electrode Poisson's ratio": 0.2,
 "Positive electrode Young's modulus [Pa]": 375000000000.0,
 'Positive electrode activation energy for cracking rate [J.mol-1]': 0.0,
 'Positive electrode active material volume fraction': 0.62,
 'Positive electrode cation signed stoichiometry': -1.0,
 'Positive electrode charge transfer coefficient': 0.5,
 'Positive electrode conductivity [S.m-1]': 10.0,
 'Positive electrode cracking rate': lico2_cracking_rate_Ai2020,
 'Positive electrode critical stress [Pa]': 375000000.0,
 'Positive electrode density [kg.m-3]': 2470.0,
 'Positive electrode diffusivity [m2.s-1]': lico2_diffusivity_Dualfoil1998,
 'Positive electrode double-layer capacity [F.m-2]': 0.2,
 'Positive electrode electrons in reaction': 1.0,
 'Positive electrode exchange-current density [A.m-2]': lico2_electrolyte_exchange_current_density_Dualfoil1998,
 'Positive electrode initial crack length [m]': 2e-08,
 'Positive electrode initial crack width [m]': 1.5e-08,
 'Positive electrode number of cracks per unit area [m-2]': 3180000000000000.0,
 'Positive electrode partial molar volume [m3.mol-1]': -7.28e-07,
 'Positive electrode porosity': 0.32,
 'Positive electrode reaction-driven LAM factor [m3.mol-1]': 0.0,
 'Positive electrode reference concentration for free of deformation [mol.m-3]': 0.0,
 'Positive electrode specific heat capacity [J.kg-1.K-1]': 1080.2,
 'Positive electrode surface area to volume ratio [m-1]': 620000.0,
 'Positive electrode thermal conductivity [W.m-1.K-1]': 1.58,
 'Positive electrode thickness [m]': 6.8e-05,
 'Positive electrode volume change': lico2_volume_change_Ai2020,
 'Positive particle radius [m]': 3e-06,
 'Ratio of lithium moles to SEI moles': 2.0,
 'Reference temperature [K]': 298.15,
 'SEI growth activation energy [J.mol-1]': 0.0,
 'SEI kinetic rate constant [m.s-1]': 1e-12,
 'SEI open-circuit potential [V]': 0.4,
 'SEI reaction exchange current density [A.m-2]': 1.5e-07,
 'SEI resistivity [Ohm.m]': 200000.0,
 'Separator Bruggeman coefficient (electrolyte)': 1.5,
 'Separator density [kg.m-3]': 2470.0,
 'Separator porosity': 0.5,
 'Separator specific heat capacity [J.kg-1.K-1]': 1080.2,
 'Separator thermal conductivity [W.m-1.K-1]': 0.334,
 'Separator thickness [m]': 2.5e-05,
 'Total heat transfer coefficient [W.m-2.K-1]': 35.0,
 'Typical current [A]': 2.28,
 'Typical electrolyte concentration [mol.m-3]': 1000.0,
 'Upper voltage cut-off [V]': 4.2}