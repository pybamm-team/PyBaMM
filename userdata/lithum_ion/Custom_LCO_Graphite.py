# -*- coding: utf-8 -*-
"""
=============================================================================
自定义用户参数文件 — 消费锂离子电池 (LCO / Graphite 体系)  ★ 推荐用于消费电子
=============================================================================

文件位置: userdata/lithum_ion/Custom_LCO_Graphite.py
用途:     为用户自定义钴酸锂(LiCoO₂)/石墨电池配方提供可编辑的参数模板

适用场景:
    - 智能手机 / 平板电脑 / 笔记本电脑 / 蓝牙耳机 电池
    - 无人机 / 数码相机 / 手持设备 电池

LCO 体系特点:
    ◈ 压实密度高 (~4.1 g/cm³) → 体积能量密度最优
    ◈ 电压平台高 (~3.9V vs Li/Li⁺)
    ◈ 多平台 OCP, 与 O3/H1-3/O1 相变相关
    ◈ 热稳定性较差, 需严格控制过充和温度
    ◈ 钴含量高, 成本最高
    ◈ 标准充电截止: 4.2V (~140 mAh/g)
    ◈ 高电压 LCO (HVLCO): 4.35-4.5V (~180-190 mAh/g), 需特殊电解液+包覆

使用方式:
    import sys
    sys.path.insert(0, "path/to/pybamm/src")

    from userdata.lithum_ion.Custom_LCO_Graphite import get_parameter_values
    params = pybamm.ParameterValues(get_parameter_values())
    model  = pybamm.lithium_ion.SPM()
    sim    = pybamm.Simulation(model, parameter_values=params)

说明:
    所有标有【需要用户重新测定】的参数均为占位值, 请用实测数据替换。
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
    通过 GITT 或半电池充放电获取 OCP 曲线, 用 numpy 函数拟合。
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
    E_D_s = 0.0              # [用户需要测定] 扩散活化能 [J·mol^-1]
    arrhenius = np.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))
    return D_ref * arrhenius


def graphite_exchange_current_density_Custom(c_e, c_s_surf, c_s_max, T):
    """
    石墨负极 Butler-Volmer 交换电流密度。

    【需要用户重新测定】
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

    【需要用户重新测定】占位返回 0 (忽略可逆热)。

    Parameters
    ----------
    sto : pybamm.Symbol  化学计量比

    Returns
    -------
    pybamm.Symbol  熵变 [V·K^-1]
    """
    return pybamm.Scalar(0.0)


# =============================================================================
# 正极 — LCO (钴酸锂, LiCoO₂)
# =============================================================================

def lco_ocp_Custom(sto):
    """
    LCO 正极开路电位 vs 化学计量比。

    【需要用户重新测定】
    LCO OCP 具有多平台特征，对应不同晶体相变 (O3, H1-3, O1)。
    sto=0 对应 CoO₂, sto=1 对应 LiCoO₂。

    此示例使用简化的多台阶拟合公式近似 LCO 行为。
    更好的方案: 用半电池实测数据 + pybamm.Interpolant 插值。

    典型 LCO 电压特征:
      - 约 3.9V  主平台 (O3 相)
      - 约 4.1V  次平台 (H1-3 相)
      - 约 4.2V  小平台 (O1 相)

    Parameters
    ----------
    sto : pybamm.Symbol  嵌锂化学计量比

    Returns
    -------
    pybamm.Symbol  开路电位 [V]
    """
    # ---- 用户需要替换为实测 OCP 数据拟合公式 ----
    U = (
        3.85
        + 0.05 * sto
        - 0.02 * np.tanh(30.0 * (sto - 0.2))
        + 0.03 * np.tanh(40.0 * (sto - 0.4))
        + 0.15 * np.tanh(50.0 * (sto - 0.75))
    )
    return U


def lco_diffusivity_Custom(sto, T):
    """
    LCO 正极固相扩散系数。

    【需要用户重新测定】
    LCO 属于层状氧化物, 扩散系数通常 10^-13 ~ 10^-14 m^2/s 量级,
    随脱锂量 (sto) 和相变显著变化。GITT / EIS / PITT 测定。

    Parameters
    ----------
    sto : pybamm.Symbol  化学计量比
    T   : pybamm.Symbol  温度 [K]

    Returns
    -------
    pybamm.Symbol  扩散系数 [m^2·s^-1]
    """
    # ---- 用户需要替换 ----
    D_ref = 5.0e-14          # [用户需要测定] 参考扩散系数 [m^2·s^-1]
    E_D_s = 0.0              # [用户需要测定] 扩散活化能 [J·mol^-1]
    arrhenius = np.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))
    return D_ref * arrhenius


def lco_exchange_current_density_Custom(c_e, c_s_surf, c_s_max, T):
    """
    LCO 正极 Butler-Volmer 交换电流密度。

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
    m_ref = 5.0e-6             # [用户需要测定] (A/m^2)(m^3/mol)^1.5
    E_r   = 20000              # [用户需要测定] 反应活化能 [J·mol^-1]
    arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))
    return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf)**0.5


def lco_entropic_change_Custom(sto):
    """
    LCO 正极熵变 dU/dT。

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
    高电压 LCO 常需特殊电解液 (含 FEC 添加剂等)，扩散/电导率可能
    与标准 LiPF6 碳酸酯体系不同。示例使用 Nyman2008 作为占位参考。

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

    【需要用户重新测定】示例使用 Nyman2008。

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
# 参数字典
# =============================================================================

def get_parameter_values():
    """
    返回 LCO / Graphite 消费电子电池的完整参数字典。

    消费电子电芯特点:
      - 容量较小 (1~5 Ah), 单个或双电芯串联
      - 涂层较薄, 铜/铝箔较薄 (Cu 6-8 μm, Al 10-12 μm)
      - 压实密度高 (体积能量密度优先)
      - 充电截止电压 4.2V (标准) 或 4.35-4.45V (HVLCO)
      - 对体积和厚度要求严格

    ============  ==========================================================
    类别            参数数
    ============  ==========================================================
    SEI            15 个 (可保留默认值)
    电芯几何        18 个 [需要测定]
    负极            16 个 [需要测定]
    正极 (LCO)      16 个 [需要测定]
    隔膜             5 个 [需要测定]
    电解液           5 个 [需要测定]
    实验条件        14 个 [需要设定]
    ============  ==========================================================
    """
    return {
        "chemistry": "lithium_ion",

        # ==================================================================
        # SEI — 可保留默认值; 研究老化/衰减时需校准
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
        # 电芯几何尺寸  [需要用户测定] 消费电子: 更小/更薄
        # ==================================================================
        "Negative current collector thickness [m]": 8.0e-06,    # 铜箔 6-8 μm
        "Negative electrode thickness [m]": 6.0e-05,            # 负极涂层 50-70 μm
        "Separator thickness [m]": 1.6e-05,                     # 隔膜 12-20 μm
        "Positive electrode thickness [m]": 5.5e-05,            # 正极涂层 50-65 μm
        "Positive current collector thickness [m]": 1.2e-05,    # 铝箔 10-15 μm
        "Electrode height [m]": 0.05,                           # [测定]
        "Electrode width [m]": 0.5,                             # [测定]
        "Cell cooling surface area [m2]": 0.003,                # [测定]
        "Cell volume [m3]": 5.0e-06,                            # [测定]
        "Cell thermal expansion coefficient [m.K-1]": 1.1e-06,

        # 集流体 — 标准值, 通常不需修改
        "Negative current collector conductivity [S.m-1]": 58411000.0,
        "Positive current collector conductivity [S.m-1]": 36914000.0,
        "Negative current collector density [kg.m-3]": 8960.0,
        "Positive current collector density [kg.m-3]": 2700.0,
        "Negative current collector specific heat capacity [J.kg-1.K-1]": 385.0,
        "Positive current collector specific heat capacity [J.kg-1.K-1]": 897.0,
        "Negative current collector thermal conductivity [W.m-1.K-1]": 401.0,
        "Positive current collector thermal conductivity [W.m-1.K-1]": 237.0,
        "Nominal cell capacity [A.h]": 2.0,                     # [测定] 消费品典型 1-5 Ah
        "Current function [A]": 2.0,                            # [设定]
        "Contact resistance [Ohm]": 0,

        # ==================================================================
        # 负极 — Graphite  [需要测定]
        # ==================================================================
        "Negative electrode conductivity [S.m-1]": 215.0,       # 典型 100-500
        "Maximum concentration in negative electrode [mol.m-3]": 33133.0,  # [测定]
        "Negative particle diffusivity [m2.s-1]": graphite_diffusivity_Custom,
        "Negative electrode OCP [V]": graphite_ocp_Custom,
        "Negative electrode porosity": 0.26,                    # [测定] 典型 0.2-0.35
        "Negative electrode active material volume fraction": 0.74,         # [测定]
        "Negative particle radius [m]": 7.0e-06,                # [测定] PSD D50
        "Negative electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Negative electrode Bruggeman coefficient (electrode)": 0,
        "Negative electrode charge transfer coefficient": 0.5,
        "Negative electrode double-layer capacity [F.m-2]": 0.2,
        "Negative electrode exchange-current density [A.m-2]"
        "": graphite_exchange_current_density_Custom,
        "Negative electrode density [kg.m-3]": 1650.0,          # [测定]
        "Negative electrode specific heat capacity [J.kg-1.K-1]": 700.0,
        "Negative electrode thermal conductivity [W.m-1.K-1]": 1.7,
        "Negative electrode OCP entropic change [V.K-1]": graphite_entropic_change_Custom,

        # ==================================================================
        # 正极 — LCO  [需要测定]
        # LCO 特殊性: 压实密度高 (3.8-4.1 g/cm³), 电压平台高 (~3.9V)
        #             热稳定性差, 过充危险
        # ==================================================================
        "Positive electrode conductivity [S.m-1]": 1.0,         # LCO 电导率较 NMC 高
        "Maximum concentration in positive electrode [mol.m-3]": 49900.0,
        # c_max = ρ * Q_theoretical / (F * 3.6)
        # LCO 理论容量 274 mAh/g, 密度 ~5.06 g/cm³ → 理论 ~40000
        # 实际二次颗粒密度 ~4.1, Q_practical ~140-190 (取决于截止电压)
        "Positive particle diffusivity [m2.s-1]": lco_diffusivity_Custom,
        "Positive electrode OCP [V]": lco_ocp_Custom,
        "Positive electrode porosity": 0.28,                    # [测定] 高压实 → 低孔隙
        "Positive electrode active material volume fraction": 0.62,         # [测定]
        "Positive particle radius [m]": 5.0e-06,                # [测定] PSD D50
        "Positive electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Positive electrode Bruggeman coefficient (electrode)": 0,
        "Positive electrode charge transfer coefficient": 0.5,
        "Positive electrode double-layer capacity [F.m-2]": 0.2,
        "Positive electrode exchange-current density [A.m-2]"
        "": lco_exchange_current_density_Custom,
        "Positive electrode density [kg.m-3]": 3800.0,          # [测定] 压实密度
        "Positive electrode specific heat capacity [J.kg-1.K-1]": 650.0,
        "Positive electrode thermal conductivity [W.m-1.K-1]": 1.5,
        "Positive electrode OCP entropic change [V.K-1]": lco_entropic_change_Custom,

        # ==================================================================
        # 隔膜  [需要测定] 消费电子常用: PE / PP / 陶瓷涂覆隔膜
        # ==================================================================
        "Separator porosity": 0.42,                             # [测定]
        "Separator Bruggeman coefficient (electrolyte)": 1.5,
        "Separator density [kg.m-3]": 450.0,                    # [测定]
        "Separator specific heat capacity [J.kg-1.K-1]": 700.0,
        "Separator thermal conductivity [W.m-1.K-1]": 0.2,

        # ==================================================================
        # 电解液  [需要测定]
        # HVLCO 需要抗氧化电解液 (氟化溶剂, 添加剂如 FEC/PES/TTSPi)
        # ==================================================================
        "Initial concentration in electrolyte [mol.m-3]": 1000.0,   # 1M
        "Cation transference number": 0.26,                         # [测定]
        "Thermodynamic factor": 1.0,
        "Electrolyte diffusivity [m2.s-1]": electrolyte_diffusivity_Custom,
        "Electrolyte conductivity [S.m-1]": electrolyte_conductivity_Custom,

        # ==================================================================
        # 实验 / 仿真条件  [需要设定]
        # ==================================================================
        "Reference temperature [K]": 298.15,
        "Total heat transfer coefficient [W.m-2.K-1]": 10.0,        # 消费电子散热条件差, 可调低
        "Ambient temperature [K]": 298.15,
        "Number of electrodes connected in parallel to make a cell": 1.0,
        "Number of cells connected in series to make a battery": 1.0,
        "Lower voltage cut-off [V]": 3.0,                           # LCO 通常 3.0V
        "Upper voltage cut-off [V]": 4.2,                           # 标准 LCO; HVLCO 可设 4.35-4.45
        "Open-circuit voltage at 0% SOC [V]": 3.0,
        "Open-circuit voltage at 100% SOC [V]": 4.2,
        "Initial concentration in negative electrode [mol.m-3]": 29866.0,   # [测定]
        "Initial concentration in positive electrode [mol.m-3]": 22000.0,   # [测定]
        "Initial temperature [K]": 298.15,

        "citations": ["Chen2020"],
    }
