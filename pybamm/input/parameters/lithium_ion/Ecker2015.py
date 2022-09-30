def graphite_diffusivity_Ecker2015(sto, T):
    """
    Graphite diffusivity as a function of stochiometry [1, 2, 3].

    References
    ----------
     .. [1] Ecker, Madeleine, et al. "Parameterization of a physico-chemical model of
    a lithium-ion battery i. determination of parameters." Journal of the
    Electrochemical Society 162.9 (2015): A1836-A1848.
    .. [2] Ecker, Madeleine, et al. "Parameterization of a physico-chemical model of
    a lithium-ion battery ii. model validation." Journal of The Electrochemical
    Society 162.9 (2015): A1849-A1857.
    .. [3] Richardson, Giles, et. al. "Generalised single particle models for
    high-rate operation of graded lithium-ion electrodes: Systematic derivation
    and validation." Electrochemica Acta 339 (2020): 135862

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

    D_ref = 8.4e-13 * exp(-11.3 * sto) + 8.2e-15
    E_D_s = 3.03e4
    arrhenius = exp(-E_D_s / (constants.R * T)) * exp(E_D_s / (constants.R * 296))

    return D_ref * arrhenius

def graphite_ocp_Ecker2015_function(sto):
    """
    Graphite OCP as a function of stochiometry [1, 2, 3].

    References
    ----------
     .. [1] Ecker, Madeleine, et al. "Parameterization of a physico-chemical model of
    a lithium-ion battery i. determination of parameters." Journal of the
    Electrochemical Society 162.9 (2015): A1836-A1848.
    .. [2] Ecker, Madeleine, et al. "Parameterization of a physico-chemical model of
    a lithium-ion battery ii. model validation." Journal of The Electrochemical
    Society 162.9 (2015): A1849-A1857.
    .. [3] Richardson, Giles, et. al. "Generalised single particle models for
    high-rate operation of graded lithium-ion electrodes: Systematic derivation
    and validation." Electrochemica Acta 339 (2020): 135862

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry

    Returns
    -------
    :class:`pybamm.Symbol`
        Open circuit potential
    """

    # Graphite negative electrode from Ecker, Kabitz, Laresgoiti et al.
    # Analytical fit (WebPlotDigitizer + gnuplot)
    a = 0.716502
    b = 369.028
    c = 0.12193
    d = 35.6478
    e = 0.0530947
    g = 0.0169644
    h = 27.1365
    i = 0.312832
    j = 0.0199313
    k = 28.5697
    m = 0.614221
    n = 0.931153
    o = 36.328
    p = 1.10743
    q = 0.140031
    r = 0.0189193
    s = 21.1967
    t = 0.196176

    u_eq = (
        a * exp(-b * sto)
        + c * exp(-d * (sto - e))
        - r * tanh(s * (sto - t))
        - g * tanh(h * (sto - i))
        - j * tanh(k * (sto - m))
        - n * exp(o * (sto - p))
        + q
    )

    return u_eq

def graphite_electrolyte_exchange_current_density_Ecker2015(c_e, c_s_surf, c_s_max, T):
    """
    Exchange-current density for Butler-Volmer reactions between graphite and LiPF6 in
    EC:DMC.

    References
    ----------
    .. [1] Ecker, Madeleine, et al. "Parameterization of a physico-chemical model of
    a lithium-ion battery i. determination of parameters." Journal of the
    Electrochemical Society 162.9 (2015): A1836-A1848.
    .. [2] Ecker, Madeleine, et al. "Parameterization of a physico-chemical model of
    a lithium-ion battery ii. model validation." Journal of The Electrochemical
    Society 162.9 (2015): A1849-A1857.
    .. [3] Richardson, Giles, et. al. "Generalised single particle models for
    high-rate operation of graded lithium-ion electrodes: Systematic derivation
    and validation." Electrochemica Acta 339 (2020): 135862

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

    k_ref = 1.11 * 1e-10

    # multiply by Faraday's constant to get correct units
    m_ref = constants.F * k_ref  # (A/m2)(mol/m3)**1.5 - includes ref concentrations
    E_r = 53400

    arrhenius = exp(-E_r / (constants.R * T)) * exp(E_r / (constants.R * 296.15))

    return (
        m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5
    )

def nco_diffusivity_Ecker2015(sto, T):
    """
    NCO diffusivity as a function of stochiometry [1, 2, 3].

    References
    ----------
    .. [1] Ecker, Madeleine, et al. "Parameterization of a physico-chemical model of
    a lithium-ion battery i. determination of parameters." Journal of the
    Electrochemical Society 162.9 (2015): A1836-A1848.
    .. [2] Ecker, Madeleine, et al. "Parameterization of a physico-chemical model of
    a lithium-ion battery ii. model validation." Journal of The Electrochemical
    Society 162.9 (2015): A1849-A1857.
    .. [3] Richardson, Giles, et. al. "Generalised single particle models for
    high-rate operation of graded lithium-ion electrodes: Systematic derivation
    and validation." Electrochemica Acta 339 (2020): 135862

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

    D_ref = 3.7e-13 - 3.4e-13 * exp(-12 * (sto - 0.62) * (sto - 0.62))
    E_D_s = 8.06e4
    arrhenius = exp(-E_D_s / (constants.R * T)) * exp(E_D_s / (constants.R * 296.15))

    return D_ref * arrhenius

def nco_ocp_Ecker2015_function(sto):
    """
    NCO OCP as a function of stochiometry [1, 2, 3].

    References
    ----------
    .. [1] Ecker, Madeleine, et al. "Parameterization of a physico-chemical model of
    a lithium-ion battery i. determination of parameters." Journal of the
    Electrochemical Society 162.9 (2015): A1836-A1848.
    .. [2] Ecker, Madeleine, et al. "Parameterization of a physico-chemical model of
    a lithium-ion battery ii. model validation." Journal of The Electrochemical
    Society 162.9 (2015): A1849-A1857.
    .. [3] Richardson, Giles, et. al. "Generalised single particle models for
    high-rate operation of graded lithium-ion electrodes: Systematic derivation
    and validation." Electrochemica Acta 339 (2020): 135862

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
       Stochiometry of material (li-fraction)

    """

    # LiNiCo from Ecker, Kabitz, Laresgoiti et al.
    # Analytical fit (WebPlotDigitizer + gnuplot)
    # Parameter m modified by Simon O'Kane to improve fit
    a = -2.35211
    c = 0.0747061
    d = 31.886
    e = 0.0219921
    g = 0.640243
    h = 5.48623
    i = 0.439245
    j = 3.82383
    k = 4.12167
    m = 0.176187
    n = 0.0542123
    o = 18.2919
    p = 0.762272
    q = 4.23285
    r = -6.34984
    s = 2.66395
    t = 0.174352

    u_eq = (
        a * sto
        - c * tanh(d * (sto - e))
        - r * tanh(s * (sto - t))
        - g * tanh(h * (sto - i))
        - j * tanh(k * (sto - m))
        - n * tanh(o * (sto - p))
        + q
    )
    return u_eq

def nco_electrolyte_exchange_current_density_Ecker2015(c_e, c_s_surf, c_s_max, T):
    """
    Exchange-current density for Butler-Volmer reactions between NCO and LiPF6 in
    EC:DMC [1, 2, 3].

    References
    ----------
    .. [1] Ecker, Madeleine, et al. "Parameterization of a physico-chemical model of
    a lithium-ion battery i. determination of parameters." Journal of the
    Electrochemical Society 162.9 (2015): A1836-A1848.
    .. [2] Ecker, Madeleine, et al. "Parameterization of a physico-chemical model of
    a lithium-ion battery ii. model validation." Journal of The Electrochemical
    Society 162.9 (2015): A1849-A1857.
    .. [3] Richardson, Giles, et. al. "Generalised single particle models for
    high-rate operation of graded lithium-ion electrodes: Systematic derivation
    and validation." Electrochemica Acta 339 (2020): 135862

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

    k_ref = 3.01e-11

    # multiply by Faraday's constant to get correct units
    m_ref = constants.F * k_ref  # (A/m2)(mol/m3)**1.5 - includes ref concentrations

    E_r = 4.36e4
    arrhenius = exp(-E_r / (constants.R * T)) * exp(E_r / (constants.R * 296.15))

    return (
        m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5
    )

def electrolyte_diffusivity_Ecker2015(c_e, T):
    """
    Diffusivity of LiPF6 in EC:DMC as a function of ion concentration [1, 2, 3].

    References
    ----------
    .. [1] Ecker, Madeleine, et al. "Parameterization of a physico-chemical model of
    a lithium-ion battery i. determination of parameters." Journal of the
    Electrochemical Society 162.9 (2015): A1836-A1848.
    .. [2] Ecker, Madeleine, et al. "Parameterization of a physico-chemical model of
    a lithium-ion battery ii. model validation." Journal of The Electrochemical
    Society 162.9 (2015): A1849-A1857.
    .. [3] Richardson, Giles, et. al. "Generalised single particle models for
    high-rate operation of graded lithium-ion electrodes: Systematic derivation
    and validation." Electrochemica Acta 339 (2020): 135862

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

    # The diffusivity epends on the electrolyte conductivity
    inputs = {"Electrolyte concentration [mol.m-3]": c_e, "Temperature [K]": T}
    sigma_e = pybamm.FunctionParameter("Electrolyte conductivity [S.m-1]", inputs)

    D_c_e = (constants.k_b / (constants.F * constants.q_e)) * sigma_e * T / c_e

    return D_c_e

def electrolyte_conductivity_Ecker2015(c_e, T):
    """
    Conductivity of LiPF6 in EC:DMC as a function of ion concentration [1, 2, 3].

    References
    ----------
    .. [1] Ecker, Madeleine, et al. "Parameterization of a physico-chemical model of
    a lithium-ion battery i. determination of parameters." Journal of the
    Electrochemical Society 162.9 (2015): A1836-A1848.
    .. [2] Ecker, Madeleine, et al. "Parameterization of a physico-chemical model of
    a lithium-ion battery ii. model validation." Journal of The Electrochemical
    Society 162.9 (2015): A1849-A1857.
    .. [3] Richardson, Giles, et. al. "Generalised single particle models for
    high-rate operation of graded lithium-ion electrodes: Systematic derivation
    and validation." Electrochemica Acta 339 (2020): 135862

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

    # mol/m^3 to mol/l
    cm = 1e-3 * c_e

    # value at T = 296K
    sigma_e_296 = 0.2667 * cm**3 - 1.2983 * cm**2 + 1.7919 * cm + 0.1726

    # add temperature dependence
    E_k_e = 1.71e4
    C = 296 * exp(E_k_e / (constants.R * 296))
    sigma_e = C * sigma_e_296 * exp(-E_k_e / (constants.R * T)) / T

    return sigma_e

def get_parameter_values():
    return {'1 + dlnf/dlnc': 1.0,
 'Ambient temperature [K]': 298.15,
 'Bulk solvent concentration [mol.m-3]': 2636.0,
 'Cation transference number': 0.26,
 'Cell cooling surface area [m2]': 0.0172,
 'Cell volume [m3]': 1.52e-06,
 'Current function [A]': 0.15652,
 'EC diffusivity [m2.s-1]': 2e-18,
 'EC initial concentration in electrolyte [mol.m-3]': 4541.0,
 'Edge heat transfer coefficient [W.m-2.K-1]': 10.0,
 'Electrode height [m]': 0.101,
 'Electrode width [m]': 0.085,
 'Electrolyte conductivity [S.m-1]': electrolyte_conductivity_Ecker2015,
 'Electrolyte diffusivity [m2.s-1]': electrolyte_diffusivity_Ecker2015,
 'Initial concentration in electrolyte [mol.m-3]': 1000.0,
 'Initial concentration in negative electrode [mol.m-3]': 26120.05,
 'Initial concentration in positive electrode [mol.m-3]': 12630.8,
 'Initial inner SEI thickness [m]': 2.5e-09,
 'Initial outer SEI thickness [m]': 2.5e-09,
 'Initial temperature [K]': 298.15,
 'Inner SEI electron conductivity [S.m-1]': 8.95e-14,
 'Inner SEI lithium interstitial diffusivity [m2.s-1]': 1e-20,
 'Inner SEI open-circuit potential [V]': 0.1,
 'Inner SEI partial molar volume [m3.mol-1]': 9.585e-05,
 'Inner SEI reaction proportion': 0.5,
 'Lithium interstitial reference concentration [mol.m-3]': 15.0,
 'Lower voltage cut-off [V]': 2.5,
 'Maximum concentration in negative electrode [mol.m-3]': 31920.0,
 'Maximum concentration in positive electrode [mol.m-3]': 48580.0,
 'Measured negative electrode OCP [V]': ('graphite_ocp_Ecker2015,
                                         ([array([0.00151515, 0.00606061, 0.01060606, 0.01666667, 0.02121212,
       0.02272727, 0.03030303, 0.03939394, 0.04545455, 0.0530303 ,
       0.06666667, 0.07878788, 0.08939394, 0.10151515, 0.12727273,
       0.14242424, 0.15909091, 0.17727273, 0.19393939, 0.21363636,
       0.23333333, 0.25757576, 0.27878788, 0.3030303 , 0.32878788,
       0.35151515, 0.37121212, 0.39242424, 0.56818182, 0.58787879,
       0.60606061, 0.62727273, 0.65454545, 0.67424242, 0.69393939,
       0.71818182, 0.73939394, 0.89090909, 0.95606061, 0.97727273,
       1.        ])],
                                          array([1.43251534, 0.86196319, 0.79141104, 0.6595092 , 0.5797546 ,
       0.52453988, 0.47546012, 0.41411043, 0.36809816, 0.33128834,
       0.28220859, 0.24846626, 0.22392638, 0.2208589 , 0.21165644,
       0.20245399, 0.1993865 , 0.19325153, 0.18404908, 0.1809816 ,
       0.17177914, 0.16564417, 0.16257669, 0.15337423, 0.14110429,
       0.13496933, 0.13190184, 0.12883436, 0.12576687, 0.12269939,
       0.11656442, 0.10122699, 0.09509202, 0.09509202, 0.08895706,
       0.08895706, 0.08588957, 0.08282209, 0.08272209, 0.0797546 ,
       0.07055215]))),
 'Measured negative electrode diffusivity [m2.s-1]': ('measured_graphite_diffusivity_Ecker2015,
                                                      ([array([0.04291659, 0.08025338, 0.12014957, 0.15796121, 0.19575227,
       0.23576503, 0.27144412, 0.31002889, 0.34841653, 0.38667045,
       0.42641578, 0.46418627, 0.50187791, 0.54121182, 0.57667491,
       0.61480369, 0.65458159, 0.76813993, 0.80894297, 0.88230341,
       0.92208646, 0.96075009, 0.99763602])],
                                                       array([2.53189836e-13, 4.43829239e-14, 3.19474263e-14, 2.60779630e-14,
       2.25590858e-14, 1.16865022e-14, 1.23869273e-14, 2.02919545e-14,
       3.26047164e-15, 7.64015664e-16, 8.41774737e-16, 7.71709353e-16,
       8.83792155e-16, 3.10917132e-15, 6.06218467e-15, 3.59593721e-14,
       2.03230938e-15, 7.80266422e-16, 7.72894292e-16, 9.65829674e-16,
       9.56699959e-16, 1.25457764e-15, 1.39568471e-14]))),
 'Measured positive electrode OCP [V]': ('nco_ocp_Ecker2015,
                                         ([array([0.00106608, 0.04127378, 0.0653984 , 0.08722544, 0.11154152,
       0.13547468, 0.15940784, 0.18174545, 0.2056786 , 0.22801622,
       0.2511516 , 0.27508476, 0.29822014, 0.32215329, 0.34528868,
       0.36922183, 0.39395276, 0.41469483, 0.44022353, 0.46176337,
       0.4864943 , 0.50883191, 0.53276506, 0.55669822, 0.57823806,
       0.59818236, 0.64844198, 0.67237514, 0.69710607, 0.71944368,
       0.74337683, 0.76491667, 0.78805206, 0.81278299, 0.8351206 ,
       0.85905375, 0.88218914, 0.90532452, 0.93005545, 0.95159529,
       0.9683485 , 0.98270839, 0.9994616 ])],
                                          array([4.58426321, 4.54243734, 4.52965007, 4.52332475, 4.49893236,
       4.46985871, 4.42953849, 4.3892071 , 4.34213893, 4.29955823,
       4.2547338 , 4.21666289, 4.1785864 , 4.14726344, 4.11143626,
       4.08236261, 4.05554386, 4.03094651, 4.00638266, 3.98404021,
       3.96846803, 3.9528791 , 3.94179997, 3.9239729 , 3.91962497,
       3.91301656, 3.91111887, 3.91128632, 3.89796345, 3.88237453,
       3.86004882, 3.81746254, 3.78838331, 3.75931524, 3.73247975,
       3.70790473, 3.68107481, 3.64974626, 3.61168094, 3.57809192,
       3.55796529, 3.54232055, 3.51994462]))),
 'Measured positive electrode diffusivity [m2.s-1]': ('measured_nco_diffusivity_Ecker2015,
                                                      ([array([0.13943218, 0.2       , 0.26182965, 0.32239748, 0.38675079,
       0.44605678, 0.50788644, 0.56845426, 0.63154574, 0.69337539,
       0.75268139, 0.81577287, 0.87507886, 0.94069401, 1.        ])],
                                                       array([1.82565403e-13, 3.32985856e-13, 3.08012285e-13, 2.63339203e-13,
       1.98119699e-13, 1.41887887e-13, 7.20118242e-14, 2.85770870e-14,
       4.54160840e-15, 5.47944475e-14, 2.02968867e-13, 1.58828651e-13,
       1.34460920e-13, 2.05450533e-14, 5.44629298e-15]))),
 'Negative current collector conductivity [S.m-1]': 58411000.0,
 'Negative current collector density [kg.m-3]': 8933.0,
 'Negative current collector specific heat capacity [J.kg-1.K-1]': 385.0,
 'Negative current collector surface heat transfer coefficient [W.m-2.K-1]': 10.0,
 'Negative current collector thermal conductivity [W.m-1.K-1]': 398.0,
 'Negative current collector thickness [m]': 1.4e-05,
 'Negative electrode Bruggeman coefficient (electrode)': 0.0,
 'Negative electrode Bruggeman coefficient (electrolyte)': 1.6372789338386007,
 'Negative electrode OCP [V]': graphite_ocp_Ecker2015_function,
 'Negative electrode OCP entropic change [V.K-1]': 0.0,
 'Negative electrode active material volume fraction': 0.372403,
 'Negative electrode cation signed stoichiometry': -1.0,
 'Negative electrode conductivity [S.m-1]': 14.0,
 'Negative electrode density [kg.m-3]': 1555.0,
 'Negative electrode diffusivity [m2.s-1]': graphite_diffusivity_Ecker2015,
 'Negative electrode electrons in reaction': 1.0,
 'Negative electrode exchange-current density [A.m-2]': graphite_electrolyte_exchange_current_density_Ecker2015,
 'Negative electrode porosity': 0.329,
 'Negative electrode reaction-driven LAM factor [m3.mol-1]': 0.0,
 'Negative electrode specific heat capacity [J.kg-1.K-1]': 1437.0,
 'Negative electrode thermal conductivity [W.m-1.K-1]': 1.58,
 'Negative electrode thickness [m]': 7.4e-05,
 'Negative particle radius [m]': 1.37e-05,
 'Negative tab centre y-coordinate [m]': 0.0045,
 'Negative tab centre z-coordinate [m]': 0.101,
 'Negative tab heat transfer coefficient [W.m-2.K-1]': 10.0,
 'Negative tab width [m]': 0.007,
 'Nominal cell capacity [A.h]': 0.15625,
 'Number of cells connected in series to make a battery': 1.0,
 'Number of electrodes connected in parallel to make a cell': 1.0,
 'Outer SEI open-circuit potential [V]': 0.8,
 'Outer SEI partial molar volume [m3.mol-1]': 9.585e-05,
 'Outer SEI solvent diffusivity [m2.s-1]': 2.5000000000000002e-22,
 'Positive current collector conductivity [S.m-1]': 36914000.0,
 'Positive current collector density [kg.m-3]': 2702.0,
 'Positive current collector specific heat capacity [J.kg-1.K-1]': 903.0,
 'Positive current collector surface heat transfer coefficient [W.m-2.K-1]': 10.0,
 'Positive current collector thermal conductivity [W.m-1.K-1]': 238.0,
 'Positive current collector thickness [m]': 1.5e-05,
 'Positive electrode Bruggeman coefficient (electrode)': 0.0,
 'Positive electrode Bruggeman coefficient (electrolyte)': 1.5442267190786427,
 'Positive electrode OCP [V]': nco_ocp_Ecker2015_function,
 'Positive electrode OCP entropic change [V.K-1]': 0.0,
 'Positive electrode active material volume fraction': 0.40832,
 'Positive electrode cation signed stoichiometry': -1.0,
 'Positive electrode conductivity [S.m-1]': 68.1,
 'Positive electrode density [kg.m-3]': 2895.0,
 'Positive electrode diffusivity [m2.s-1]': nco_diffusivity_Ecker2015,
 'Positive electrode electrons in reaction': 1.0,
 'Positive electrode exchange-current density [A.m-2]': nco_electrolyte_exchange_current_density_Ecker2015,
 'Positive electrode porosity': 0.296,
 'Positive electrode reaction-driven LAM factor [m3.mol-1]': 0.0,
 'Positive electrode specific heat capacity [J.kg-1.K-1]': 1270.0,
 'Positive electrode thermal conductivity [W.m-1.K-1]': 1.04,
 'Positive electrode thickness [m]': 5.4e-05,
 'Positive particle radius [m]': 6.5e-06,
 'Positive tab centre y-coordinate [m]': 0.0309,
 'Positive tab centre z-coordinate [m]': 0.101,
 'Positive tab heat transfer coefficient [W.m-2.K-1]': 10.0,
 'Positive tab width [m]': 0.0069,
 'Ratio of lithium moles to SEI moles': 2.0,
 'Reference temperature [K]': 296.15,
 'SEI growth activation energy [J.mol-1]': 0.0,
 'SEI kinetic rate constant [m.s-1]': 1e-12,
 'SEI open-circuit potential [V]': 0.4,
 'SEI reaction exchange current density [A.m-2]': 1.5e-07,
 'SEI resistivity [Ohm.m]': 200000.0,
 'Separator Bruggeman coefficient (electrolyte)': 1.9804586773134945,
 'Separator density [kg.m-3]': 1017.0,
 'Separator porosity': 0.508,
 'Separator specific heat capacity [J.kg-1.K-1]': 1978.0,
 'Separator thermal conductivity [W.m-1.K-1]': 0.34,
 'Separator thickness [m]': 2e-05,
 'Total heat transfer coefficient [W.m-2.K-1]': 10.0,
 'Typical current [A]': 0.15652,
 'Typical electrolyte concentration [mol.m-3]': 1000.0,
 'Upper voltage cut-off [V]': 4.2}