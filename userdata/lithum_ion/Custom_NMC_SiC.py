"""
================================================================================
自定义用户参数文件 — 消费锂离子电池 (NMC / Si-C 复合负极 体系)
================================================================================

文件位置: userdata/lithum_ion/Custom_NMC_SiC.py
用途:     三元(NMC)正极 + 石墨/硅碳(Si-C)复合负极 电池参数模板

适用场景:
    高比能消费电子电池 (如旗舰手机的 Si-C 负极电池)

复合负极说明:
    Primary   = 石墨相 (Graphite)
    Secondary = 硅/硅碳相 (Si / Si-C)

使用方式:
    import sys
    sys.path.insert(0, "path/to/pybamm/src")

    from userdata.lithum_ion.Custom_NMC_SiC import get_parameter_values
    params = pybamm.ParameterValues(get_parameter_values())
    model  = pybamm.lithium_ion.DFN(
        options={"particle phases": ("2", "1")}
    )
    sim    = pybamm.Simulation(model, parameter_values=params)
================================================================================
"""

import numpy as np

import pybamm

# =============================================================================
# 负极 Primary 相 — Graphite (石墨)
# =============================================================================


def graphite_ocp_Custom(sto):
    """
    石墨负极 OCP。

    【需要用户重新测定】半电池充放电/GITT 测定, numpy 函数拟合。

    Parameters
    ----------
    sto : pybamm.Symbol  嵌锂化学计量比

    Returns
    -------
    pybamm.Symbol  开路电位 [V]
    """
    # ---- 用户需要替换 ----
    u_eq = (
        1.9793 * np.exp(-39.3631 * sto)
        + 0.2482
        - 0.0909 * np.tanh(29.8538 * (sto - 0.1234))
        - 0.04478 * np.tanh(14.9159 * (sto - 0.2769))
        - 0.0205 * np.tanh(30.4444 * (sto - 0.6103))
    )
    return u_eq


def graphite_diffusivity_Custom(sto, T):
    """
    石墨固相锂扩散系数。

    【需要用户重新测定】GITT/EIS。

    Parameters
    ----------
    sto : pybamm.Symbol  化学计量比
    T   : pybamm.Symbol  温度 [K]

    Returns
    -------
    pybamm.Symbol  扩散系数 [m^2·s^-1]
    """
    D_ref = 3.3e-14  # [用户需要测定]
    E_D_s = 0.0  # [用户需要测定]
    arrhenius = np.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))
    return D_ref * arrhenius


def graphite_exchange_current_density_Custom(c_e, c_s_surf, c_s_max, T):
    """
    石墨 Butler-Volmer 交换电流密度。

    【需要用户重新测定】EIS。

    Parameters
    ----------
    c_e      : pybamm.Symbol  电解液 Li+ 浓度 [mol·m^-3]
    c_s_surf : pybamm.Symbol  颗粒表面 Li 浓度 [mol·m^-3]
    c_s_max  : pybamm.Symbol  最大 Li 浓度 [mol·m^-3]
    T        : pybamm.Symbol  温度 [K]

    Returns
    -------
    pybamm.Symbol  交换电流密度 [A·m^-2]
    """
    m_ref = 6.48e-7  # [用户需要测定]
    E_r = 35000  # [用户需要测定]
    arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))
    return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5


def graphite_entropic_change_Custom(sto):
    """
    石墨熵变 dU/dT。占位返回 0。

    Parameters
    ----------
    sto : pybamm.Symbol  化学计量比

    Returns
    -------
    pybamm.Symbol  熵变 [V·K^-1]
    """
    return pybamm.Scalar(0.0)


# =============================================================================
# 负极 Secondary 相 — Silicon / Si-C (硅/硅碳)
# =============================================================================


def silicon_ocp_lithiation_Custom(sto):
    """
    硅相嵌锂 (Lithiation) OCP。

    【需要用户重新测定】
    硅半电池准稳态 OCV 测试, 分离嵌锂/脱锂方向。

    Parameters
    ----------
    sto : pybamm.Symbol  硅相嵌锂化学计量比

    Returns
    -------
    pybamm.Symbol  嵌锂 OCP [V]
    """
    # ---- 用户需要替换 ----
    p1, p2, p3, p4 = -96.63, 372.6, -587.6, 489.9
    p5, p6, p7, p8 = -232.8, 62.99, -9.286, 0.8633
    U_lithiation = (
        p1 * sto**7
        + p2 * sto**6
        + p3 * sto**5
        + p4 * sto**4
        + p5 * sto**3
        + p6 * sto**2
        + p7 * sto
        + p8
    ) + 1e-4 * (1 / sto + 1 / (sto - 1))
    return U_lithiation


def silicon_ocp_delithiation_Custom(sto):
    """
    硅相脱锂 (Delithiation) OCP。

    【需要用户重新测定】

    Parameters
    ----------
    sto : pybamm.Symbol  硅相脱锂化学计量比

    Returns
    -------
    pybamm.Symbol  脱锂 OCP [V]
    """
    # ---- 用户需要替换 ----
    p1, p2, p3, p4 = -51.02, 161.3, -205.7, 140.2
    p5, p6, p7, p8 = -58.76, 16.87, -3.792, 0.9937
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


def silicon_ocp_average_Custom(sto):
    """
    硅相平均 OCP。

    Parameters
    ----------
    sto : pybamm.Symbol  化学计量比

    Returns
    -------
    pybamm.Symbol  平均 OCP [V]
    """
    return (
        silicon_ocp_lithiation_Custom(sto) + silicon_ocp_delithiation_Custom(sto)
    ) / 2


def silicon_diffusivity_Custom(sto, T):
    """
    硅/硅碳相固相锂扩散系数。

    【需要用户重新测定】
    硅相扩散系数 ~10^-15 ~ 10^-16 m^2/s, 随嵌锂量变化。
    GITT 测量 (注意大体积膨胀可能影响接触)。

    Parameters
    ----------
    sto : pybamm.Symbol  化学计量比
    T   : pybamm.Symbol  温度 [K]

    Returns
    -------
    pybamm.Symbol  扩散系数 [m^2·s^-1]
    """
    D_ref = 1.0e-15  # [用户需要测定]
    E_D_s = 0.0  # [用户需要测定]
    arrhenius = np.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))
    return D_ref * arrhenius


def silicon_exchange_current_density_Custom(c_e, c_s_surf, c_s_max, T):
    """
    硅/硅碳相交换电流密度。

    【需要用户重新测定】EIS 测量拟合。

    Parameters
    ----------
    c_e      : pybamm.Symbol  电解液 Li+ 浓度 [mol·m^-3]
    c_s_surf : pybamm.Symbol  颗粒表面 Li 浓度 [mol·m^-3]
    c_s_max  : pybamm.Symbol  最大 Li 浓度 [mol·m^-3]
    T        : pybamm.Symbol  温度 [K]

    Returns
    -------
    pybamm.Symbol  交换电流密度 [A·m^-2]
    """
    m_ref = 6.48e-7 * 28700 / 278000  # [用户需要测定]
    E_r = 35000  # [用户需要测定]
    arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))
    return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5


def silicon_entropic_change_Custom(sto):
    """
    硅相熵变 dU/dT。占位返回 0。

    Parameters
    ----------
    sto : pybamm.Symbol  化学计量比

    Returns
    -------
    pybamm.Symbol  熵变 [V·K^-1]
    """
    return pybamm.Scalar(0.0)


# =============================================================================
# 正极 — NMC (三元镍钴锰酸锂)
# =============================================================================


def nmc_ocp_Custom(sto):
    """
    NMC 正极 OCP。

    【需要用户重新测定】
    OCP 高度依赖 NMC 型号 (811/622/532/111)。
    半电池充放电测定。

    Parameters
    ----------
    sto : pybamm.Symbol  嵌锂化学计量比

    Returns
    -------
    pybamm.Symbol  开路电位 [V]
    """
    # ---- 用户需要替换 ----
    u_eq = (
        -0.8090 * sto
        + 4.4875
        - 0.0428 * np.tanh(18.5138 * (sto - 0.5542))
        - 17.7326 * np.tanh(15.7890 * (sto - 0.3117))
        + 17.5842 * np.tanh(15.9308 * (sto - 0.3120))
    )
    return u_eq


def nmc_diffusivity_Custom(sto, T):
    """
    NMC 正极固相扩散系数。

    【需要用户重新测定】GITT/EIS。

    Parameters
    ----------
    sto : pybamm.Symbol  化学计量比
    T   : pybamm.Symbol  温度 [K]

    Returns
    -------
    pybamm.Symbol  扩散系数 [m^2·s^-1]
    """
    D_ref = 4.0e-15  # [用户需要测定]
    E_D_s = 0.0  # [用户需要测定]
    arrhenius = np.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))
    return D_ref * arrhenius


def nmc_exchange_current_density_Custom(c_e, c_s_surf, c_s_max, T):
    """
    NMC 正极交换电流密度。

    【需要用户重新测定】

    Parameters
    ----------
    c_e      : pybamm.Symbol  电解液 Li+ 浓度 [mol·m^-3]
    c_s_surf : pybamm.Symbol  颗粒表面 Li 浓度 [mol·m^-3]
    c_s_max  : pybamm.Symbol  最大 Li 浓度 [mol·m^-3]
    T        : pybamm.Symbol  温度 [K]

    Returns
    -------
    pybamm.Symbol  交换电流密度 [A·m^-2]
    """
    m_ref = 3.42e-6  # [用户需要测定]
    E_r = 17800  # [用户需要测定]
    arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))
    return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5


def nmc_entropic_change_Custom(sto):
    """
    NMC 正极熵变 dU/dT。占位返回 0。

    Parameters
    ----------
    sto : pybamm.Symbol  化学计量比

    Returns
    -------
    pybamm.Symbol  熵变 [V·K^-1]
    """
    return pybamm.Scalar(0.0)


# =============================================================================
# 电解液
# =============================================================================


def electrolyte_diffusivity_Custom(c_e, T):
    """
    电解液 Li+ 扩散系数。

    【需要用户重新测定】占位使用 Nyman2008。

    Parameters
    ----------
    c_e : pybamm.Symbol  电解液浓度 [mol·m^-3]
    T   : pybamm.Symbol  温度 [K]

    Returns
    -------
    pybamm.Symbol  扩散系数 [m^2·s^-1]
    """
    D_c_e = 8.794e-11 * (c_e / 1000) ** 2 - 3.972e-10 * (c_e / 1000) + 4.862e-10
    return D_c_e


def electrolyte_conductivity_Custom(c_e, T):
    """
    电解液离子电导率。

    【需要用户重新测定】占位使用 Nyman2008。

    Parameters
    ----------
    c_e : pybamm.Symbol  电解液浓度 [mol·m^-3]
    T   : pybamm.Symbol  温度 [K]

    Returns
    -------
    pybamm.Symbol  电导率 [S·m^-1]
    """
    sigma_e = (
        0.1297 * (c_e / 1000) ** 3 - 2.51 * (c_e / 1000) ** 1.5 + 3.329 * (c_e / 1000)
    )
    return sigma_e


# =============================================================================
# 参数字典
# =============================================================================


def get_parameter_values():
    """
    返回 NMC / Si-C 复合负极电池的完整参数字典 (120 个参数)。

    ================  ===============================================
    类别                说明
    ================  ===============================================
    SEI (Primary)      石墨相 SEI (15 个)
    SEI (Secondary)    硅相 SEI (15 个)
    电芯几何            18 个 [需要测定]
    负极 Primary (石墨)  9 个 [需要测定]
    负极 Secondary (硅)  12 个 [需要测定]
    负极 共享            8 个 [需要测定]
    正极 NMC             16 个 [需要测定]
    隔膜                 5 个 [需要测定]
    电解液               5 个 [需要测定]
    实验条件             14 个 [需要设定]
    ================  ===============================================
    """
    return {
        "chemistry": "lithium_ion",
        # ==================================================================
        # SEI — Primary (石墨相)
        # ==================================================================
        "Primary: Ratio of lithium moles to SEI moles": 2.0,
        "Primary: SEI partial molar volume [m3.mol-1]": 9.585e-05,
        "Primary: SEI reaction exchange current density [A.m-2]": 1.5e-07,
        "Primary: SEI resistivity [Ohm.m]": 200000.0,
        "Primary: SEI solvent diffusivity [m2.s-1]": 2.5e-22,
        "Primary: Bulk solvent concentration [mol.m-3]": 2636.0,
        "Primary: SEI open-circuit potential [V]": 0.4,
        "Primary: SEI electron conductivity [S.m-1]": 8.95e-14,
        "Primary: SEI lithium interstitial diffusivity [m2.s-1]": 1e-20,
        "Primary: Lithium interstitial reference concentration [mol.m-3]": 15.0,
        "Primary: Initial SEI thickness [m]": 5e-09,
        "Primary: EC initial concentration in electrolyte [mol.m-3]": 4541.0,
        "Primary: EC diffusivity [m2.s-1]": 2e-18,
        "Primary: SEI kinetic rate constant [m.s-1]": 1e-12,
        "Primary: SEI growth activation energy [J.mol-1]": 0.0,
        # ==================================================================
        # SEI — Secondary (硅相)
        # ==================================================================
        "Secondary: Ratio of lithium moles to SEI moles": 2.0,
        "Secondary: SEI partial molar volume [m3.mol-1]": 9.585e-05,
        "Secondary: SEI reaction exchange current density [A.m-2]": 1.5e-07,
        "Secondary: SEI resistivity [Ohm.m]": 200000.0,
        "Secondary: SEI solvent diffusivity [m2.s-1]": 2.5e-22,
        "Secondary: Bulk solvent concentration [mol.m-3]": 2636.0,
        "Secondary: SEI open-circuit potential [V]": 0.4,
        "Secondary: SEI electron conductivity [S.m-1]": 8.95e-14,
        "Secondary: SEI lithium interstitial diffusivity [m2.s-1]": 1e-20,
        "Secondary: Lithium interstitial reference concentration [mol.m-3]": 15.0,
        "Secondary: Initial SEI thickness [m]": 5e-09,
        "Secondary: EC initial concentration in electrolyte [mol.m-3]": 4541.0,
        "Secondary: EC diffusivity [m2.s-1]": 2e-18,
        "Secondary: SEI kinetic rate constant [m.s-1]": 1e-12,
        "Secondary: SEI growth activation energy [J.mol-1]": 0.0,
        "Positive electrode reaction-driven LAM factor [m3.mol-1]": 0.0,
        # ==================================================================
        # 电芯几何尺寸 [需要用户测定]
        # ==================================================================
        "Negative current collector thickness [m]": 8.0e-06,
        "Negative electrode thickness [m]": 6.5e-05,  # Si-C 涂层
        "Separator thickness [m]": 1.6e-05,
        "Positive electrode thickness [m]": 6.0e-05,
        "Positive current collector thickness [m]": 1.2e-05,
        "Electrode height [m]": 0.06,  # [测定]
        "Electrode width [m]": 0.8,  # [测定]
        "Cell cooling surface area [m2]": 0.004,  # [测定]
        "Cell volume [m3]": 6.0e-06,  # [测定]
        "Cell thermal expansion coefficient [m.K-1]": 1.1e-06,
        # 集流体
        "Negative current collector conductivity [S.m-1]": 58411000.0,
        "Positive current collector conductivity [S.m-1]": 36914000.0,
        "Negative current collector density [kg.m-3]": 8960.0,
        "Positive current collector density [kg.m-3]": 2700.0,
        "Negative current collector specific heat capacity [J.kg-1.K-1]": 385.0,
        "Positive current collector specific heat capacity [J.kg-1.K-1]": 897.0,
        "Negative current collector thermal conductivity [W.m-1.K-1]": 401.0,
        "Positive current collector thermal conductivity [W.m-1.K-1]": 237.0,
        "Nominal cell capacity [A.h]": 3.0,  # [测定]
        "Current function [A]": 3.0,
        "Contact resistance [Ohm]": 0,
        # ==================================================================
        # 负极 Primary 相 — 石墨 [需要测定]
        # ==================================================================
        "Primary: Maximum concentration in negative electrode [mol.m-3]": 28700.0,
        "Primary: Initial concentration in negative electrode [mol.m-3]": 27700.0,
        "Primary: Negative particle diffusivity [m2.s-1]": graphite_diffusivity_Custom,
        "Primary: Negative electrode OCP [V]": graphite_ocp_Custom,
        "Primary: Negative electrode active material volume fraction": 0.72,
        # [测定] 石墨体积分数 (扣除硅后)
        "Primary: Negative particle radius [m]": 5.86e-06,  # [测定]
        "Primary: Negative electrode exchange-current density [A.m-2]"
        "": graphite_exchange_current_density_Custom,
        "Primary: Negative electrode density [kg.m-3]": 1657.0,  # [测定]
        "Primary: Negative electrode OCP entropic change [V.K-1]": graphite_entropic_change_Custom,
        # ==================================================================
        # 负极 Secondary 相 — 硅/硅碳 [需要测定]
        # ==================================================================
        "Secondary: Maximum concentration in negative electrode [mol.m-3]": 278000.0,
        # [测定] 按硅含量比例计算
        "Secondary: Initial concentration in negative electrode [mol.m-3]": 276610.0,
        # [测定]
        "Secondary: Negative particle diffusivity [m2.s-1]": silicon_diffusivity_Custom,
        "Secondary: Negative electrode lithiation OCP [V]"
        "": silicon_ocp_lithiation_Custom,
        "Secondary: Negative electrode delithiation OCP [V]"
        "": silicon_ocp_delithiation_Custom,
        "Secondary: Negative electrode OCP [V]": silicon_ocp_average_Custom,
        "Secondary: Negative electrode active material volume fraction": 0.03,
        # [测定] 硅相体积分数 (例如 3% = 约 10wt% Si)
        "Secondary: Negative particle radius [m]": 1.5e-06,  # [测定] 纳米硅/硅碳 PSD D50
        "Secondary: Negative electrode exchange-current density [A.m-2]"
        "": silicon_exchange_current_density_Custom,
        "Secondary: Negative electrode density [kg.m-3]": 2650.0,  # [测定]
        "Secondary: Negative electrode OCP entropic change [V.K-1]": silicon_entropic_change_Custom,
        # ==================================================================
        # 负极 共享参数 [需要测定]
        # ==================================================================
        "Negative electrode conductivity [S.m-1]": 215.0,
        "Negative electrode porosity": 0.26,  # [测定]
        "Negative electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Negative electrode Bruggeman coefficient (electrode)": 0,
        "Negative electrode charge transfer coefficient": 0.5,
        "Negative electrode double-layer capacity [F.m-2]": 0.2,
        "Negative electrode specific heat capacity [J.kg-1.K-1]": 700.0,
        "Negative electrode thermal conductivity [W.m-1.K-1]": 1.7,
        # ==================================================================
        # 正极 — NMC [需要测定]
        # ==================================================================
        "Positive electrode conductivity [S.m-1]": 0.18,
        "Maximum concentration in positive electrode [mol.m-3]": 63104.0,  # [测定]
        "Positive particle diffusivity [m2.s-1]": nmc_diffusivity_Custom,
        "Positive electrode OCP [V]": nmc_ocp_Custom,
        "Positive electrode porosity": 0.335,  # [测定]
        "Positive electrode active material volume fraction": 0.665,  # [测定]
        "Positive particle radius [m]": 5.22e-06,  # [测定]
        "Positive electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Positive electrode Bruggeman coefficient (electrode)": 0,
        "Positive electrode charge transfer coefficient": 0.5,
        "Positive electrode double-layer capacity [F.m-2]": 0.2,
        "Positive electrode exchange-current density [A.m-2]"
        "": nmc_exchange_current_density_Custom,
        "Positive electrode density [kg.m-3]": 3262.0,  # [测定]
        "Positive electrode specific heat capacity [J.kg-1.K-1]": 700.0,
        "Positive electrode thermal conductivity [W.m-1.K-1]": 2.1,
        "Positive electrode OCP entropic change [V.K-1]": nmc_entropic_change_Custom,
        # ==================================================================
        # 隔膜 [需要测定]
        # ==================================================================
        "Separator porosity": 0.45,  # [测定]
        "Separator Bruggeman coefficient (electrolyte)": 1.5,
        "Separator density [kg.m-3]": 450.0,  # [测定]
        "Separator specific heat capacity [J.kg-1.K-1]": 700.0,
        "Separator thermal conductivity [W.m-1.K-1]": 0.2,
        # ==================================================================
        # 电解液 [需要测定]
        # ==================================================================
        "Initial concentration in electrolyte [mol.m-3]": 1000.0,
        "Cation transference number": 0.2594,  # [测定]
        "Thermodynamic factor": 1.0,
        "Electrolyte diffusivity [m2.s-1]": electrolyte_diffusivity_Custom,
        "Electrolyte conductivity [S.m-1]": electrolyte_conductivity_Custom,
        # ==================================================================
        # 实验 / 仿真条件 [需要设定]
        # ==================================================================
        "Reference temperature [K]": 298.15,
        "Total heat transfer coefficient [W.m-2.K-1]": 10.0,
        "Ambient temperature [K]": 298.15,
        "Number of electrodes connected in parallel to make a cell": 1.0,
        "Number of cells connected in series to make a battery": 1.0,
        "Lower voltage cut-off [V]": 2.8,  # NMC/Si-C 体系
        "Upper voltage cut-off [V]": 4.25,  # 高电压 NMC
        "Open-circuit voltage at 0% SOC [V]": 2.8,
        "Open-circuit voltage at 100% SOC [V]": 4.25,
        "Initial concentration in negative electrode [mol.m-3]": 29866.0,  # [测定]
        "Initial concentration in positive electrode [mol.m-3]": 17038.0,  # [测定]
        "Initial temperature [K]": 298.15,
        "citations": ["Chen2020", "Ai2022"],
    }
