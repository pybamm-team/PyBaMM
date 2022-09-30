def graphite_diffusivity_Kim2011(sto, T):
    """
    Graphite diffusivity [1].

    References
    ----------
    .. [1] Kim, G. H., Smith, K., Lee, K. J., Santhanagopalan, S., & Pesaran, A.
    (2011). Multi-domain modeling of lithium-ion batteries encompassing
    multi-physics in varied length scales. Journal of The Electrochemical
    Society, 158(8), A955-A969.

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

    D_ref = 9 * 10 ** (-14)
    E_D_s = 4e3
    arrhenius = exp(E_D_s / constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius

def graphite_ocp_Kim2011(sto):
    """
    Graphite Open Circuit Potential (OCP) as a function of the stochiometry [1].

    References
    ----------
    .. [1] Kim, G. H., Smith, K., Lee, K. J., Santhanagopalan, S., & Pesaran, A.
    (2011). Multi-domain modeling of lithium-ion batteries encompassing
    multi-physics in varied length scales. Journal of The Electrochemical
    Society, 158(8), A955-A969.
    """

    u_eq = (
        0.124
        + 1.5 * exp(-70 * sto)
        - 0.0351 * tanh((sto - 0.286) / 0.083)
        - 0.0045 * tanh((sto - 0.9) / 0.119)
        - 0.035 * tanh((sto - 0.99) / 0.05)
        - 0.0147 * tanh((sto - 0.5) / 0.034)
        - 0.102 * tanh((sto - 0.194) / 0.142)
        - 0.022 * tanh((sto - 0.98) / 0.0164)
        - 0.011 * tanh((sto - 0.124) / 0.0226)
        + 0.0155 * tanh((sto - 0.105) / 0.029)
    )

    return u_eq

def graphite_electrolyte_exchange_current_density_Kim2011(c_e, c_s_surf, c_s_max, T):
    """
    Exchange-current density for Butler-Volmer reactions between graphite and LiPF6 in
    EC:DMC
    [1].

    References
    ----------
    .. [1] Kim, G. H., Smith, K., Lee, K. J., Santhanagopalan, S., & Pesaran, A.
    (2011). Multi-domain modeling of lithium-ion batteries encompassing
    multi-physics in varied length scales. Journal of The Electrochemical
    Society, 158(8), A955-A969.

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

    i0_ref = 36  # reference exchange current density at 100% SOC
    sto = 0.36  # stochiometry at 100% SOC
    c_s_n_ref = sto * c_s_max  # reference electrode concentration
    c_e_ref = Parameter("Typical electrolyte concentration [mol.m-3]")
    alpha = 0.5  # charge transfer coefficient

    m_ref = i0_ref / (
        c_e_ref**alpha * (c_s_max - c_s_n_ref) ** alpha * c_s_n_ref**alpha
    )

    E_r = 3e4
    arrhenius = exp(E_r / constants.R * (1 / 298.15 - 1 / T))

    return (
        m_ref
        * arrhenius
        * c_e**alpha
        * c_s_surf**alpha
        * (c_s_max - c_s_surf) ** alpha
    )

def nca_diffusivity_Kim2011(sto, T):
    """
    NCA diffusivity as a function of stochiometry [1].

    References
    ----------
    .. [1] Kim, G. H., Smith, K., Lee, K. J., Santhanagopalan, S., & Pesaran, A.
    (2011). Multi-domain modeling of lithium-ion batteries encompassing
    multi-physics in varied length scales. Journal of The Electrochemical
    Society, 158(8), A955-A969.

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
    D_ref = 3 * 10 ** (-15)
    E_D_s = 2e4
    arrhenius = exp(E_D_s / constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius

def nca_electrolyte_exchange_current_density_Kim2011(c_e, c_s_surf, c_s_max, T):
    """
    Exchange-current density for Butler-Volmer reactions between NCA and LiPF6 in EC:DMC
    [1].

    References
    ----------
    .. [1] Kim, G. H., Smith, K., Lee, K. J., Santhanagopalan, S., & Pesaran, A.
    (2011). Multi-domain modeling of lithium-ion batteries encompassing
    multi-physics in varied length scales. Journal of The Electrochemical
    Society, 158(8), A955-A969.

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
    i0_ref = 4  # reference exchange current density at 100% SOC
    sto = 0.41  # stochiometry at 100% SOC
    c_s_ref = sto * c_s_max  # reference electrode concentration
    c_e_ref = Parameter("Typical electrolyte concentration [mol.m-3]")
    alpha = 0.5  # charge transfer coefficient

    m_ref = i0_ref / (
        c_e_ref**alpha * (c_s_max - c_s_ref) ** alpha * c_s_ref**alpha
    )
    E_r = 3e4
    arrhenius = exp(E_r / constants.R * (1 / 298.15 - 1 / T))

    return (
        m_ref
        * arrhenius
        * c_e**alpha
        * c_s_surf**alpha
        * (c_s_max - c_s_surf) ** alpha
    )

def electrolyte_diffusivity_Kim2011(c_e, T):
    """
    Diffusivity of LiPF6 in EC as a function of ion concentration from [1].

     References
    ----------
    .. [1] Kim, G. H., Smith, K., Lee, K. J., Santhanagopalan, S., & Pesaran, A.
    (2011). Multi-domain modeling of lithium-ion batteries encompassing
    multi-physics in varied length scales. Journal of The Electrochemical
    Society, 158(8), A955-A969.

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

    D_c_e = (
        5.84 * 10 ** (-7) * exp(-2870 / T) * (c_e / 1000) ** 2
        - 33.9 * 10 ** (-7) * exp(-2920 / T) * (c_e / 1000)
        + 129 * 10 ** (-7) * exp(-3200 / T)
    )

    return D_c_e

def electrolyte_conductivity_Kim2011(c_e, T):
    """
    Conductivity of LiPF6 in EC as a function of ion concentration from [1].

    References
    ----------
    .. [1] Kim, G. H., Smith, K., Lee, K. J., Santhanagopalan, S., & Pesaran, A.
    (2011). Multi-domain modeling of lithium-ion batteries encompassing
    multi-physics in varied length scales. Journal of The Electrochemical
    Society, 158(8), A955-A969.

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
        3.45 * exp(-798 / T) * (c_e / 1000) ** 3
        - 48.5 * exp(-1080 / T) * (c_e / 1000) ** 2
        + 244 * exp(-1440 / T) * (c_e / 1000)
    )

    return sigma_e

def get_parameter_values():
    return {'1 + dlnf/dlnc': 1.0,
 'Ambient temperature [K]': 298.15,
 'Bulk solvent concentration [mol.m-3]': 2636.0,
 'Cation transference number': 0.4,
 'Cell cooling surface area [m2]': 0.0561,
 'Cell volume [m3]': 4.62e-06,
 'Current function [A]': 0.43,
 'EC diffusivity [m2.s-1]': 2e-18,
 'EC initial concentration in electrolyte [mol.m-3]': 4541.0,
 'Edge heat transfer coefficient [W.m-2.K-1]': 0.3,
 'Electrode height [m]': 0.2,
 'Electrode width [m]': 0.14,
 'Electrolyte conductivity [S.m-1]': electrolyte_conductivity_Kim2011,
 'Electrolyte diffusivity [m2.s-1]': electrolyte_diffusivity_Kim2011,
 'Initial concentration in electrolyte [mol.m-3]': 1200.0,
 'Initial concentration in negative electrode [mol.m-3]': 18081.0,
 'Initial concentration in positive electrode [mol.m-3]': 20090.0,
 'Initial inner SEI thickness [m]': 2.5e-09,
 'Initial outer SEI thickness [m]': 2.5e-09,
 'Initial temperature [K]': 298.15,
 'Inner SEI electron conductivity [S.m-1]': 8.95e-14,
 'Inner SEI lithium interstitial diffusivity [m2.s-1]': 1e-20,
 'Inner SEI open-circuit potential [V]': 0.1,
 'Inner SEI partial molar volume [m3.mol-1]': 9.585e-05,
 'Inner SEI reaction proportion': 0.5,
 'Lithium interstitial reference concentration [mol.m-3]': 15.0,
 'Lower voltage cut-off [V]': 2.7,
 'Maximum concentration in negative electrode [mol.m-3]': 28700.0,
 'Maximum concentration in positive electrode [mol.m-3]': 49000.0,
 'Negative current collector conductivity [S.m-1]': 59600000.0,
 'Negative current collector density [kg.m-3]': 11544.75,
 'Negative current collector specific heat capacity [J.kg-1.K-1]': 385.0,
 'Negative current collector surface heat transfer coefficient [W.m-2.K-1]': 0.0,
 'Negative current collector thermal conductivity [W.m-1.K-1]': 267.467,
 'Negative current collector thickness [m]': 1e-05,
 'Negative electrode Bruggeman coefficient (electrode)': 2.0,
 'Negative electrode Bruggeman coefficient (electrolyte)': 2.0,
 'Negative electrode OCP [V]': graphite_ocp_Kim2011,
 'Negative electrode OCP entropic change [V.K-1]': 0.0,
 'Negative electrode active material volume fraction': 0.51,
 'Negative electrode cation signed stoichiometry': -1.0,
 'Negative electrode charge transfer coefficient': 0.5,
 'Negative electrode conductivity [S.m-1]': 100.0,
 'Negative electrode density [kg.m-3]': 2136.43638,
 'Negative electrode diffusivity [m2.s-1]': graphite_diffusivity_Kim2011,
 'Negative electrode double-layer capacity [F.m-2]': 0.2,
 'Negative electrode electrons in reaction': 1.0,
 'Negative electrode exchange-current density [A.m-2]': graphite_electrolyte_exchange_current_density_Kim2011,
 'Negative electrode porosity': 0.4,
 'Negative electrode reaction-driven LAM factor [m3.mol-1]': 0.0,
 'Negative electrode specific heat capacity [J.kg-1.K-1]': 700.0,
 'Negative electrode thermal conductivity [W.m-1.K-1]': 1.1339,
 'Negative electrode thickness [m]': 7e-05,
 'Negative particle radius [m]': 5.083e-07,
 'Negative tab centre y-coordinate [m]': 0.013,
 'Negative tab centre z-coordinate [m]': 0.2,
 'Negative tab heat transfer coefficient [W.m-2.K-1]': 25.0,
 'Negative tab width [m]': 0.044,
 'Nominal cell capacity [A.h]': 0.43,
 'Number of cells connected in series to make a battery': 1.0,
 'Number of electrodes connected in parallel to make a cell': 1.0,
 'Outer SEI open-circuit potential [V]': 0.8,
 'Outer SEI partial molar volume [m3.mol-1]': 9.585e-05,
 'Outer SEI solvent diffusivity [m2.s-1]': 2.5000000000000002e-22,
 'Positive current collector conductivity [S.m-1]': 37800000.0,
 'Positive current collector density [kg.m-3]': 3490.24338,
 'Positive current collector specific heat capacity [J.kg-1.K-1]': 897.0,
 'Positive current collector surface heat transfer coefficient [W.m-2.K-1]': 0.0,
 'Positive current collector thermal conductivity [W.m-1.K-1]': 158.079,
 'Positive current collector thickness [m]': 1e-05,
 'Positive electrode Bruggeman coefficient (electrode)': 2.0,
 'Positive electrode Bruggeman coefficient (electrolyte)': 2.0,
 'Positive electrode OCP [V]': ('nca_ocp_Kim2011_data,
                                ([array([0.37021443, 0.37577436, 0.38369048, 0.39189598, 0.40106922,
       0.40686181, 0.41168962, 0.41941373, 0.42665544, 0.43293042,
       0.43969074, 0.44548431, 0.45321039, 0.45852157, 0.46286601,
       0.47107645, 0.47638764, 0.48363133, 0.4894249 , 0.49811871,
       0.50777631, 0.51647111, 0.52805924, 0.53771684, 0.54930891,
       0.55655162, 0.56717498, 0.57683455, 0.58697453, 0.59614876,
       0.6087066 , 0.6159493 , 0.62464311, 0.63526844, 0.64637813,
       0.66038609, 0.67632754, 0.69178562, 0.70386304, 0.72463723,
       0.73913054, 0.75314145, 0.76763475, 0.77971218, 0.79178861,
       0.80434842, 0.81449236, 0.82608542, 0.83574499, 0.84637328,
       0.85603187, 0.86521004, 0.87390286, 0.88404778, 0.89274258,
       0.90240313, 0.91254904, 0.92221058, 0.93380562, 0.94829596,
       0.95795159, 0.96519232, 0.97097405, 0.97434484, 0.97674494,
       0.98058923, 0.98201962, 0.983451  , 0.98488435, 0.98583235,
       0.9872588 , 0.98964212, 0.9905704 , 0.99150558, 0.99338284])],
                                 array([4.21044086, 4.19821487, 4.18214203, 4.16516313, 4.14960477,
       4.1382866 , 4.12979962, 4.11565356, 4.10292309, 4.09018933,
       4.08029169, 4.07039076, 4.05907917, 4.05059383, 4.04210521,
       4.0322125 , 4.02372716, 4.01383117, 4.00393024, 3.99403917,
       3.97989969, 3.97142586, 3.95304124, 3.93890175, 3.92618608,
       3.91487285, 3.89931941, 3.8880144 , 3.87245932, 3.85831819,
       3.84418857, 3.83287533, 3.82298427, 3.81026531, 3.79896523,
       3.78484054, 3.77072242, 3.75660267, 3.74672311, 3.7312042 ,
       3.71991563, 3.71004265, 3.69875408, 3.68887453, 3.67757773,
       3.66628259, 3.65639645, 3.64509802, 3.633793  , 3.62532575,
       3.6126035 , 3.60413132, 3.59282302, 3.58435413, 3.5758803 ,
       3.56599252, 3.55894086, 3.55047033, 3.54200636, 3.52646608,
       3.50949212, 3.49534441, 3.46843664, 3.45002407, 3.42593926,
       3.39335597, 3.36501616, 3.33809359, 3.3140055 , 3.28708128,
       3.25307253, 3.2048947 , 3.14962574, 3.10427745, 3.02350152]))),
 'Positive electrode OCP entropic change [V.K-1]': 0.0,
 'Positive electrode active material volume fraction': 0.41,
 'Positive electrode cation signed stoichiometry': -1.0,
 'Positive electrode charge transfer coefficient': 0.5,
 'Positive electrode conductivity [S.m-1]': 10.0,
 'Positive electrode density [kg.m-3]': 4205.82708,
 'Positive electrode diffusivity [m2.s-1]': nca_diffusivity_Kim2011,
 'Positive electrode double-layer capacity [F.m-2]': 0.2,
 'Positive electrode electrons in reaction': 1.0,
 'Positive electrode exchange-current density [A.m-2]': nca_electrolyte_exchange_current_density_Kim2011,
 'Positive electrode porosity': 0.4,
 'Positive electrode reaction-driven LAM factor [m3.mol-1]': 0.0,
 'Positive electrode specific heat capacity [J.kg-1.K-1]': 700.0,
 'Positive electrode thermal conductivity [W.m-1.K-1]': 1.4007,
 'Positive electrode thickness [m]': 5e-05,
 'Positive particle radius [m]': 1.633e-06,
 'Positive tab centre y-coordinate [m]': 0.137,
 'Positive tab centre z-coordinate [m]': 0.2,
 'Positive tab heat transfer coefficient [W.m-2.K-1]': 25.0,
 'Positive tab width [m]': 0.044,
 'Ratio of lithium moles to SEI moles': 2.0,
 'Reference temperature [K]': 298.15,
 'SEI growth activation energy [J.mol-1]': 0.0,
 'SEI kinetic rate constant [m.s-1]': 1e-12,
 'SEI open-circuit potential [V]': 0.4,
 'SEI reaction exchange current density [A.m-2]': 1.5e-07,
 'SEI resistivity [Ohm.m]': 200000.0,
 'Separator Bruggeman coefficient (electrolyte)': 2.0,
 'Separator density [kg.m-3]': 511.86798,
 'Separator porosity': 0.4,
 'Separator specific heat capacity [J.kg-1.K-1]': 700.0,
 'Separator thermal conductivity [W.m-1.K-1]': 0.10672,
 'Separator thickness [m]': 2.5e-05,
 'Total heat transfer coefficient [W.m-2.K-1]': 25.0,
 'Typical current [A]': 0.43,
 'Typical electrolyte concentration [mol.m-3]': 1200.0,
 'Upper voltage cut-off [V]': 4.2}