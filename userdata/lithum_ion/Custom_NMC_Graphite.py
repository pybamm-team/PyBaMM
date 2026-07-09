# -*- coding: utf-8 -*-
"""
=============================================================================
自定义用户参数文件 — 消费锂离子电池 (NMC / Graphite 体系)
=============================================================================

文件位置: userdata/lithum_ion/Custom_NMC_Graphite.py
用途:     为用户自定义 NMC/石墨 电池配方提供可编辑的参数模板

使用方式:
    import sys
    sys.path.insert(0, "path/to/pybamm/src")

    from userdata.lithum_ion.Custom_NMC_Graphite import get_parameter_values
    params = pybamm.ParameterValues(get_parameter_values())
    model  = pybamm.lithium_ion.SPM()
    sim    = pybamm.Simulation(model, parameter_values=params)

说明:
    此模板基于 Chen2020 等文献数据，所有标有【需要用户重新测定】的参数
    均为占位值，用户需要用自己的实验数据替换。
=============================================================================
"""

import numpy as np
import pybamm


# =============================================================================
# 负极 — Graphite (石墨)
# =============================================================================

def graphite_ocp_Custom(sto):
    """
    石墨负极开路电位 (OCP) vs 化学计量比 sto。

    【需要用户重新测定】
    通过 GITT 或半电池充放电获取 OCP 曲线，用 numpy 函数拟合。
    常用形式: 指数+双曲正切, 多项式, 或 CSV 插值 (pybamm.Interpolant)。

    Parameters
    ----------
    sto : pybamm.Symbol  电极嵌锂化学计量比 (0 ≤ sto ≤ 1)

    Returns
    -------
    pybamm.Symbol  开路电位 [V]
    """
    # ---- 用户需要替换为实测数据拟合公式 ----
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
    石墨负极固相锂扩散系数 D_s(sto, T)。

    【需要用户重新测定】
    GITT / EIS 在不同 SOC 下测定, Arrhenius 拟合温度效应。
    典型形式: D = D_ref(sto) * exp(-Ea/R * (1/T - 1/T_ref))

    Parameters
    ----------
    sto : pybamm.Symbol  化学计量比
    T   : pybamm.Symbol  温度 [K]

    Returns
    -------
    pybamm.Symbol  扩散系数 [m^2·s^-1]
    """
    # ---- 用户需要替换 ----
    D_ref = 3.3e-14          # [用户需要测定] 参考扩散系数 [m^2·s^-1]
    E_D_s = 0.0              # [用户需要测定] 扩散活化能 [J·mol^-1]; 0=无温依赖
    arrhenius = np.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))
    return D_ref * arrhenius


def graphite_exchange_current_density_Custom(c_e, c_s_surf, c_s_max, T):
    """
    石墨负极 Butler-Volmer 交换电流密度。

    【需要用户重新测定】
    标准形式: j0 = m_ref * arrhenius * c_e^0.5 * c_s^0.5 * (c_max - c_s)^0.5
    通过 EIS 在不同温度和 SOC 下测量, 拟合 m_ref 与活化能 E_r。

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
    # ---- 用户需要替换 ----
    m_ref = 6.48e-7            # [用户需要测定] (A/m^2)(m^3/mol)^1.5
    E_r   = 35000              # [用户需要测定] 反应活化能 [J·mol^-1]
    arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))
    return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf)**0.5


def graphite_entropic_change_Custom(sto):
    """
    石墨负极熵变 dU/dT (可逆热效应)。

    【需要用户重新测定】
    在不同温度下测量 OCP, 计算 OCP 对温度的偏导数。
    占位返回 0 (忽略可逆热)。

    Parameters
    ----------
    sto : pybamm.Symbol  化学计量比

    Returns
    -------
    pybamm.Symbol  熵变 [V·K^-1]
    """
    return pybamm.Scalar(0.0)


# =============================================================================
# 正极 — NMC (镍钴锰酸锂)
# =============================================================================

def nmc_ocp_Custom(sto):
    """
    NMC 正极开路电位 vs 化学计量比。

    【需要用户重新测定】
    OCP 曲线高度依赖镍钴锰比例 (NMC811 / 622 / 532 / 111)。
    需通过半电池充放电或 GITT 测定。

    Parameters
    ----------
    sto : pybamm.Symbol  嵌锂化学计量比

    Returns
    -------
    pybamm.Symbol  开路电位 [V]
    """
    # ---- 用户需要替换为实测数据拟合公式 ----
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

    【需要用户重新测定】
    GITT / EIS / PITT 在不同 SOC 下测定。

    Parameters
    ----------
    sto : pybamm.Symbol  化学计量比
    T   : pybamm.Symbol  温度 [K]

    Returns
    -------
    pybamm.Symbol  扩散系数 [m^2·s^-1]
    """
    # ---- 用户需要替换 ----
    D_ref = 4.0e-15          # [用户需要测定] 参考扩散系数 [m^2·s^-1]
    E_D_s = 0.0              # [用户需要测定] 扩散活化能 [J·mol^-1]
    arrhenius = np.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))
    return D_ref * arrhenius


def nmc_exchange_current_density_Custom(c_e, c_s_surf, c_s_max, T):
    """
    NMC 正极 Butler-Volmer 交换电流密度。

    【需要用户重新测定】
    通过 EIS 在不同温度/SOC 下测量, 拟合动力学参数。

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
    # ---- 用户需要替换 ----
    m_ref = 3.42e-6            # [用户需要测定] (A/m^2)(m^3/mol)^1.5
    E_r   = 17800              # [用户需要测定] 反应活化能 [J·mol^-1]
    arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))
    return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf)**0.5


def nmc_entropic_change_Custom(sto):
    """
    NMC 正极熵变 dU/dT。

    【需要用户重新测定】
    变温 OCP 实验获取。占位返回 0。

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
    电解液 Li+ 扩散系数 D_e(c_e, T)。

    【需要用户重新测定】
    取决于电解液配方 (锂盐种类、溶剂比例、添加剂)。
    示例使用 Nyman2008 (1M LiPF6 in EC:EMC 3:7) 作为参考。

    常见电解液配方:
      - 1M LiPF6 in EC:DMC (1:1)
      - 1M LiPF6 in EC:EMC (3:7)
      - 1M LiPF6 in EC:DMC:EMC (1:1:1)
      - 含添加剂 (FEC, VC) 的配方

    Parameters
    ----------
    c_e : pybamm.Symbol  电解液浓度 [mol·m^-3]
    T   : pybamm.Symbol  温度 [K]

    Returns
    -------
    pybamm.Symbol  扩散系数 [m^2·s^-1]
    """
    # ---- 用户需要根据电解液配方替换 ----
    D_c_e = (8.794e-11 * (c_e / 1000)**2
             - 3.972e-10 * (c_e / 1000)
             + 4.862e-10)
    return D_c_e


def electrolyte_conductivity_Custom(c_e, T):
    """
    电解液离子电导率 σ_e(c_e, T)。

    【需要用户重新测定】
    取决于电解液配方。示例使用 Nyman2008 (EC:EMC 3:7)。

    Parameters
    ----------
    c_e : pybamm.Symbol  电解液浓度 [mol·m^-3]
    T   : pybamm.Symbol  温度 [K]

    Returns
    -------
    pybamm.Symbol  电导率 [S·m^-1]
    """
    # ---- 用户需要根据电解液配方替换 ----
    sigma_e = (0.1297 * (c_e / 1000)**3
               - 2.51 * (c_e / 1000)**1.5
               + 3.329 * (c_e / 1000))
    return sigma_e


# =============================================================================
# 参数字典 — 通过函数返回, 避免原地编辑带来的副作用
# =============================================================================

def get_parameter_values():
    """
    返回 NMC / Graphite 电池的完整参数字典。

    ============  ==========================================================
    类别            说明
    ============  ==========================================================
    SEI            固态电解质界面膜参数 (可保留默认值)
    电芯几何       集流体 / 涂层 / 隔膜厚度, 面积, 体积 [需要测定]
    负极 (石墨)    电导率, 最大浓度, OCP, 扩散, 孔隙率, 粒径 [需要测定]
    正极 (NMC)     电导率, 最大浓度, OCP, 扩散, 孔隙率, 粒径 [需要测定]
    隔膜           孔隙率, 密度, 热物性 [需要测定]
    电解液         初始浓度, 迁移数, 扩散/电导率函数 [需要测定]
    实验条件       参考温度, 换热系数, 截止电压, 初始 SOC [需要设定]
    ============  ==========================================================
    """
    return {
        "chemistry": "lithium_ion",

        # ==================================================================
        # SEI (可保留默认值; 研究老化/衰减时需校准)
        # ==================================================================
        "Ratio of lithium moles to SEI moles": 2.0,
        "SEI partial molar volume [m3.mol-1]": 9.585e-05,
        "SEI reaction exchange current density [A.m-2]": 1.5e-07,
        "SEI resistivity [Ohm.m]": 200000.0,
        "SEI solvent diffusivity [m2.s-1]": 2.5e-22,
        "Bulk solvent concentration [mol.m-3]": 2636.0,
        "SEI open-circuit potential [V]": 0.4,
        "SEI electron conductivity [S.m-1]": 8.95e-14,
        "SEI lithium interstitial diffusivity [m2.s-1]": 1e-20,
        "Lithium interstitial reference concentration [mol.m-3]": 15.0,
        "Initial SEI thickness [m]": 5e-09,
        "EC initial concentration in electrolyte [mol.m-3]": 4541.0,
        "EC diffusivity [m2.s-1]": 2e-18,
        "SEI kinetic rate constant [m.s-1]": 1e-12,
        "SEI growth activation energy [J.mol-1]": 0.0,
        "Negative electrode reaction-driven LAM factor [m3.mol-1]": 0.0,
        "Positive electrode reaction-driven LAM factor [m3.mol-1]": 0.0,

        # ==================================================================
        # 电芯几何尺寸  [需要用户测定] 游标卡尺 / SEM 截面
        # ==================================================================
        "Negative current collector thickness [m]": 1.2e-05,   # 铜箔 ~6-12 μm
        "Negative electrode thickness [m]": 8.52e-05,          # 负极涂层
        "Separator thickness [m]": 1.2e-05,                    # 隔膜
        "Positive electrode thickness [m]": 7.56e-05,          # 正极涂层
        "Positive current collector thickness [m]": 1.6e-05,   # 铝箔 ~12-20 μm
        "Electrode height [m]": 0.065,                         # [测定]
        "Electrode width [m]": 1.58,                           # [测定] 展开长度
        "Cell cooling surface area [m2]": 0.00531,             # [测定] 外表面积
        "Cell volume [m3]": 2.42e-05,                          # [测定]
        "Cell thermal expansion coefficient [m.K-1]": 1.1e-06,

        # 集流体 (标准值, 通常不需修改)
        "Negative current collector conductivity [S.m-1]": 58411000.0,
        "Positive current collector conductivity [S.m-1]": 36914000.0,
        "Negative current collector density [kg.m-3]": 8960.0,
        "Positive current collector density [kg.m-3]": 2700.0,
        "Negative current collector specific heat capacity [J.kg-1.K-1]": 385.0,
        "Positive current collector specific heat capacity [J.kg-1.K-1]": 897.0,
        "Negative current collector thermal conductivity [W.m-1.K-1]": 401.0,
        "Positive current collector thermal conductivity [W.m-1.K-1]": 237.0,
        "Nominal cell capacity [A.h]": 5.0,                    # [测定] 0.2C 放电
        "Current function [A]": 5.0,                            # [设定] 测试电流
        "Contact resistance [Ohm]": 0,

        # ==================================================================
        # 负极 — Graphite  [需要测定]
        # ==================================================================
        "Negative electrode conductivity [S.m-1]": 215.0,      # 四探针 / 配方估算
        "Maximum concentration in negative electrode [mol.m-3]": 33133.0,
        # c_max 需根据活性材料理论容量和密度计算
        "Negative particle diffusivity [m2.s-1]": graphite_diffusivity_Custom,
        "Negative electrode OCP [V]": graphite_ocp_Custom,
        "Negative electrode porosity": 0.25,                   # 压汞法 / 密度法
        "Negative electrode active material volume fraction": 0.75,
        "Negative particle radius [m]": 5.86e-06,              # PSD D50
        "Negative electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Negative electrode Bruggeman coefficient (electrode)": 0,
        "Negative electrode charge transfer coefficient": 0.5,
        "Negative electrode double-layer capacity [F.m-2]": 0.2,
        "Negative electrode exchange-current density [A.m-2]"
        "": graphite_exchange_current_density_Custom,
        "Negative electrode density [kg.m-3]": 1657.0,         # 称重/体积
        "Negative electrode specific heat capacity [J.kg-1.K-1]": 700.0,
        "Negative electrode thermal conductivity [W.m-1.K-1]": 1.7,
        "Negative electrode OCP entropic change [V.K-1]": graphite_entropic_change_Custom,

        # ==================================================================
        # 正极 — NMC  [需要测定]
        # ==================================================================
        "Positive electrode conductivity [S.m-1]": 0.18,       # NMC 较低, 依赖导电剂
        "Maximum concentration in positive electrode [mol.m-3]": 63104.0,
        # c_max = ρ * Q_theoretical / (F * 3.6), NMC811 ~50000-63000
        "Positive particle diffusivity [m2.s-1]": nmc_diffusivity_Custom,
        "Positive electrode OCP [V]": nmc_ocp_Custom,
        "Positive electrode porosity": 0.335,
        "Positive electrode active material volume fraction": 0.665,
        "Positive particle radius [m]": 5.22e-06,              # PSD D50 (NMC二次颗粒)
        "Positive electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Positive electrode Bruggeman coefficient (electrode)": 0,
        "Positive electrode charge transfer coefficient": 0.5,
        "Positive electrode double-layer capacity [F.m-2]": 0.2,
        "Positive electrode exchange-current density [A.m-2]"
        "": nmc_exchange_current_density_Custom,
        "Positive electrode density [kg.m-3]": 3262.0,
        "Positive electrode specific heat capacity [J.kg-1.K-1]": 700.0,
        "Positive electrode thermal conductivity [W.m-1.K-1]": 2.1,
        "Positive electrode OCP entropic change [V.K-1]": nmc_entropic_change_Custom,

        # ==================================================================
        # 隔膜  [需要测定] 查询供应商参数表
        # ==================================================================
        "Separator porosity": 0.47,
        "Separator Bruggeman coefficient (electrolyte)": 1.5,
        "Separator density [kg.m-3]": 397.0,
        "Separator specific heat capacity [J.kg-1.K-1]": 700.0,
        "Separator thermal conductivity [W.m-1.K-1]": 0.16,

        # ==================================================================
        # 电解液  [需要测定]
        # ==================================================================
        "Initial concentration in electrolyte [mol.m-3]": 1000.0,  # 1M = 1000
        "Cation transference number": 0.2594,                       # Bruce-Vincent法
        "Thermodynamic factor": 1.0,                                # 稀释近似
        "Electrolyte diffusivity [m2.s-1]": electrolyte_diffusivity_Custom,
        "Electrolyte conductivity [S.m-1]": electrolyte_conductivity_Custom,

        # ==================================================================
        # 实验 / 仿真条件  [需要设定]
        # ==================================================================
        "Reference temperature [K]": 298.15,
        "Total heat transfer coefficient [W.m-2.K-1]": 10.0,        # 自然对流5-15
        "Ambient temperature [K]": 298.15,
        "Number of electrodes connected in parallel to make a cell": 1.0,
        "Number of cells connected in series to make a battery": 1.0,
        "Lower voltage cut-off [V]": 2.5,                           # [测定]
        "Upper voltage cut-off [V]": 4.2,                           # [测定]
        "Open-circuit voltage at 0% SOC [V]": 2.5,
        "Open-circuit voltage at 100% SOC [V]": 4.2,
        "Initial concentration in negative electrode [mol.m-3]": 29866.0,  # 由正负匹配决定
        "Initial concentration in positive electrode [mol.m-3]": 17038.0,  # 由正负匹配决定
        "Initial temperature [K]": 298.15,

        "citations": ["Chen2020"],
    }
