"""
================================================================================
自定义用户参数文件 — 消费锂离子电池 (LCO / Si-C 复合负极 体系)
================================================================================

文件位置: userdata/lithum_ion/Custom_LCO_SiC.py
用途:     钴酸锂(LiCoO₂)正极 + 石墨/硅碳(Si-C)复合负极 电池参数模板

复合负极说明:
    硅的比容量 (~3579 mAh/g for Li₁₅Si₄) 远高于石墨 (~372 mAh/g),
    但嵌锂过程中体积膨胀 ~300%, 导致颗粒破裂和 SEI 持续生长。
    商业 Si-C 负极通常将少量硅(5-20wt%)与石墨混合,
    兼具高比能与长循环寿命。

    本文件使用 pybamm 的双相复合电极模型 ("particle phases": ("2", "1")),
    Primary   = 石墨相 (Graphite)
    Secondary = 硅/硅碳相 (Si / Si-C)

硅相 OCP 的特殊性:
    硅的嵌锂/脱锂过程存在显著的电压滞后 (hysteresis),
    因此需要三组 OCP 函数:
      1. lithiation OCP   — 嵌锂方向 (放电) 开路电位
      2. delithiation OCP — 脱锂方向 (充电) 开路电位
      3. average OCP      — 平均值, 用于初始 SOC 计算

使用方式:
    import sys
    sys.path.insert(0, "path/to/pybamm/src")

    from userdata.lithum_ion.Custom_LCO_SiC import get_parameter_values
    params = pybamm.ParameterValues(get_parameter_values())
    # 注意: 需要选择支持复合电极的模型
    model  = pybamm.lithium_ion.DFN(
        options={"particle phases": ("2", "1")}
    )
    sim    = pybamm.Simulation(model, parameter_values=params)

参考:
    硅 OCP 函数参考 Verbrugge et al., JES 163(2): A262 (2015)
    复合电极参数结构参考 Chen2020_composite/Ai2022
================================================================================
"""

import numpy as np

import pybamm

# =============================================================================
# 负极 Primary 相 — Graphite (石墨)
# =============================================================================


def graphite_ocp_Custom(sto):
    """
    石墨负极开路电位 (OCP) vs 化学计量比。

    Chen2020 石墨 OCP 多项式 + tanh 拟合。
    在 sto 边界裁剪到 [0.01, 0.99] 防止数值溢出。

    Parameters
    ----------
    sto : pybamm.Symbol  嵌锂化学计量比 (0 ≤ sto ≤ 1)

    Returns
    -------
    pybamm.Symbol  开路电位 [V]
    """
    if isinstance(sto, pybamm.Symbol):
        sto = pybamm.maximum(pybamm.minimum(sto, 0.99), 0.01)
    else:
        sto = np.clip(sto, 0.01, 0.99)

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

    【需要用户重新测定】GITT/EIS 测定。

    Parameters
    ----------
    sto : pybamm.Symbol  化学计量比
    T   : pybamm.Symbol  温度 [K]

    Returns
    -------
    pybamm.Symbol  扩散系数 [m^2·s^-1]
    """
    D_ref = 3.3e-14  # [用户需要测定]
    E_D_s = 0.0  # [用户需要测定] 活化能
    arrhenius = np.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))
    return D_ref * arrhenius


def graphite_exchange_current_density_Custom(c_e, c_s_surf, c_s_max, T):
    """
    石墨 Butler-Volmer 交换电流密度。

    【需要用户重新测定】EIS 在不同温度/SOC 下测量拟合。

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
    石墨熵变 dU/dT。占位返回 0 (忽略可逆热)。

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
    硅相嵌锂 (Lithiation) 开路电位。

    使用平滑 tanh 函数构建硅 OCP 曲线:
        - 满电态 sto≈1 → O≈0.1 V (vs Li/Li⁺)
        - 空态 sto≈0   → O≈0.8 V
        - 中间有 ~0.3V 的平台 (对应两相反应区)

    Parameters
    ----------
    sto : pybamm.Symbol  硅相嵌锂化学计量比

    Returns
    -------
    pybamm.Symbol  嵌锂 OCP [V]
    """
    if isinstance(sto, pybamm.Symbol):
        sto = pybamm.maximum(pybamm.minimum(sto, 0.99), 0.01)
    else:
        sto = np.clip(sto, 0.01, 0.99)

    U = (
        0.05
        + 0.70 * (1 - sto)
        - 0.30 * np.tanh(20.0 * (sto - 0.5))
        - 0.05 * np.tanh(15.0 * (sto - 0.15))
        - 0.03 * np.tanh(15.0 * (sto - 0.85))
    )
    return U


def silicon_ocp_delithiation_Custom(sto):
    """
    硅相脱锂 (Delithiation) 开路电位。

    与嵌锂曲线形态类似但上移 ~0.1V 以模拟滞后效应。

    Parameters
    ----------
    sto : pybamm.Symbol  硅相脱锂化学计量比

    Returns
    -------
    pybamm.Symbol  脱锂 OCP [V]
    """
    if isinstance(sto, pybamm.Symbol):
        sto = pybamm.maximum(pybamm.minimum(sto, 0.99), 0.01)
    else:
        sto = np.clip(sto, 0.01, 0.99)

    U = (
        0.15
        + 0.70 * (1 - sto)
        - 0.30 * np.tanh(20.0 * (sto - 0.5))
        - 0.05 * np.tanh(15.0 * (sto - 0.15))
        - 0.03 * np.tanh(15.0 * (sto - 0.85))
    )
    return U


def silicon_ocp_average_Custom(sto):
    """
    硅相平均 OCP (用于初始 SOC 估算)。

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
    硅的锂扩散系数通常比石墨低 1-2 个数量级 (~10^-15 ~ 10^-16 m^2/s),
    且随嵌锂量剧烈变化 (非晶态 Li_xSi 相的 D 值与 x 有关)。
    硅碳复合材料的表观扩散系数还受碳基体影响。

    测量方法: GITT (需注意硅的大体积变化会影响接触) 或 EIS。
    若无法区分, 可使用常数占位。

    Parameters
    ----------
    sto : pybamm.Symbol  硅相化学计量比
    T   : pybamm.Symbol  温度 [K]

    Returns
    -------
    pybamm.Symbol  扩散系数 [m^2·s^-1]
    """
    # ---- 用户需要替换 ----
    D_ref = 1.0e-14  # [用户需要测定] 硅相参考扩散系数 (纳米硅偏高)
    E_D_s = 0.0  # [用户需要测定] 活化能
    arrhenius = np.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))
    return D_ref * arrhenius


def silicon_exchange_current_density_Custom(c_e, c_s_surf, c_s_max, T):
    """
    硅/硅碳相 Butler-Volmer 交换电流密度。

    【需要用户重新测定】
    硅的反应动力学参数与石墨不同。此处使用缩放后的石墨参数作为占位。
    实际需通过 Si-C 半电池的 EIS 测量拟合。

    注: m_ref 可以通过石墨值按 c_max 比例缩放得到初始估计:
         m_ref_Si = m_ref_graphite * (c_max_graphite / c_max_Si)

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
    # m_ref 按石墨(28700)与硅(278000)的 c_max 比值缩放作为初始估计
    m_ref = 6.48e-7 * 28700 / 278000 * 5  # [用户需要测定] 放大5倍缓解刚性问题
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
# 正极 — LCO (钴酸锂)
# =============================================================================


def lco_ocp_Custom(sto):
    """
    LCO 正极开路电位 vs 化学计量比。

    使用平滑 tanh 函数模拟 LCO 多平台 OCP 特征:
        - 满电态 sto≈1 → ~4.2 V
        - 空态 sto≈0   → ~3.9 V
        - 中间有平台过渡 (对应 O3/H1-3 相变)

    Parameters
    ----------
    sto : pybamm.Symbol  嵌锂化学计量比

    Returns
    -------
    pybamm.Symbol  开路电位 [V]
    """
    if isinstance(sto, pybamm.Symbol):
        sto = pybamm.maximum(pybamm.minimum(sto, 0.99), 0.01)
    else:
        sto = np.clip(sto, 0.01, 0.99)

    U = (
        3.9
        + 0.18 * sto
        - 0.04 * np.tanh(25.0 * (sto - 0.2))
        + 0.06 * np.tanh(30.0 * (sto - 0.45))
        + 0.12 * np.tanh(40.0 * (sto - 0.75))
    )
    return U


def lco_diffusivity_Custom(sto, T):
    """
    LCO 正极固相扩散系数。

    【需要用户重新测定】GITT/EIS 测定。

    Parameters
    ----------
    sto : pybamm.Symbol  化学计量比
    T   : pybamm.Symbol  温度 [K]

    Returns
    -------
    pybamm.Symbol  扩散系数 [m^2·s^-1]
    """
    D_ref = 5.0e-14  # [用户需要测定]
    E_D_s = 0.0  # [用户需要测定]
    arrhenius = np.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))
    return D_ref * arrhenius


def lco_exchange_current_density_Custom(c_e, c_s_surf, c_s_max, T):
    """
    LCO 正极交换电流密度。

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
    m_ref = 5.0e-6  # [用户需要测定]
    E_r = 20000  # [用户需要测定]
    arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))
    return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5


def lco_entropic_change_Custom(sto):
    """
    LCO 正极熵变 dU/dT。占位返回 0。

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

    【需要用户重新测定】
    Si-C 负极体系可能使用含 FEC 添加剂的电解液以稳定 SEI,
    扩散/电导率可能与标准体系不同。占位使用 Nyman2008。

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
    返回 LCO / Si-C 复合负极电池的完整参数字典 (120 个参数)。

    ================  ===============================================
    类别                说明
    ================  ===============================================
    SEI (Primary)      石墨相 SEI (15 个)
    SEI (Secondary)    硅相 SEI (15 个)
    电芯几何            集流体/涂层/隔膜厚度等 (18 个)
    负极 Primary (石墨)  9 个相专属参数 [需要测定]
    负极 Secondary (硅)  12 个相专属参数 [需要测定]
    负极 共享            8 个两相共享参数 [需要测定]
    正极 LCO             16 个 [需要测定]
    隔膜                 5 个 [需要测定]
    电解液               5 个 [需要测定]
    实验条件             14 个 [需要设定]
    ================  ===============================================
    """
    return {
        "chemistry": "lithium_ion",
        # ==================================================================
        # SEI 参数 — Primary (石墨相)
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
        # SEI 参数 — Secondary (硅相)
        # 硅表面的 SEI 特性与石墨不同, 通常更厚、更不稳定 [需要测定]
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
        # 正极 LAM
        "Positive electrode reaction-driven LAM factor [m3.mol-1]": 0.0,
        # ==================================================================
        # 电芯几何尺寸 [需要用户测定] 消费电子
        # ==================================================================
        "Negative current collector thickness [m]": 8.0e-06,
        "Negative electrode thickness [m]": 6.0e-05,  # Si-C 涂层, 铜箔侧
        "Separator thickness [m]": 1.6e-05,
        "Positive electrode thickness [m]": 5.5e-05,
        "Positive current collector thickness [m]": 1.2e-05,
        "Electrode height [m]": 0.05,  # [测定]
        "Electrode width [m]": 0.5,  # [测定]
        "Cell cooling surface area [m2]": 0.003,  # [测定]
        "Cell volume [m3]": 5.0e-06,  # [测定]
        "Cell thermal expansion coefficient [m.K-1]": 1.1e-06,
        # 集流体 (标准值)
        "Negative current collector conductivity [S.m-1]": 58411000.0,
        "Positive current collector conductivity [S.m-1]": 36914000.0,
        "Negative current collector density [kg.m-3]": 8960.0,
        "Positive current collector density [kg.m-3]": 2700.0,
        "Negative current collector specific heat capacity [J.kg-1.K-1]": 385.0,
        "Positive current collector specific heat capacity [J.kg-1.K-1]": 897.0,
        "Negative current collector thermal conductivity [W.m-1.K-1]": 401.0,
        "Positive current collector thermal conductivity [W.m-1.K-1]": 237.0,
        "Nominal cell capacity [A.h]": 2.0,  # [测定]
        "Current function [A]": 2.0,
        "Contact resistance [Ohm]": 0,
        # ==================================================================
        # 负极 Primary 相 — 石墨  [需要测定]
        # ==================================================================
        "Primary: Maximum concentration in negative electrode [mol.m-3]": 28700.0,
        # 石墨理论 c_max: 26390 (LiC6)
        "Primary: Initial concentration in negative electrode [mol.m-3]": 27700.0,
        # 对应高 SOC 状态 (满电时石墨接近 LiC6)
        "Primary: Negative particle diffusivity [m2.s-1]": graphite_diffusivity_Custom,
        "Primary: Negative electrode OCP [V]": graphite_ocp_Custom,
        "Primary: Negative electrode active material volume fraction": 0.735,
        # [测定] 石墨占负极活性材料的体积分数 (含硅时为 1 - 硅体积分数)
        "Primary: Negative particle radius [m]": 5.86e-06,  # [测定] PSD D50
        "Primary: Negative electrode exchange-current density [A.m-2]"
        "": graphite_exchange_current_density_Custom,
        "Primary: Negative electrode density [kg.m-3]": 1657.0,  # [测定]
        "Primary: Negative electrode OCP entropic change [V.K-1]": graphite_entropic_change_Custom,
        # ==================================================================
        # 负极 Secondary 相 — 硅/硅碳  [需要测定]
        # 硅的特性: 高容量 (~3579 mAh/g), 大体积膨胀 (~300%)
        #           低 OCP (~0.3-0.5V vs Li/Li⁺), 显著滞后
        # 硅碳复合材料: 容量和膨胀率按硅含量比例降低
        # ==================================================================
        "Secondary: Maximum concentration in negative electrode [mol.m-3]": 278000.0,
        # [测定] 硅 c_max: 纯硅 ~278000 (Li₁₅Si₄), 硅碳按硅含量比例折算
        # 计算公式: c_max_SiC = c_max_Si * wt%_Si / (wt%_Si + wt%_C * ρ_Si/ρ_C)
        "Secondary: Initial concentration in negative electrode [mol.m-3]": 276610.0,
        # [测定] 硅相初始 Li 浓度 (满电态), 可用 sto_initial * c_max 计算
        "Secondary: Negative particle diffusivity [m2.s-1]": silicon_diffusivity_Custom,
        "Secondary: Negative electrode lithiation OCP [V]"
        "": silicon_ocp_lithiation_Custom,
        # 硅嵌锂 OCP [需要测定]
        "Secondary: Negative electrode delithiation OCP [V]"
        "": silicon_ocp_delithiation_Custom,
        # 硅脱锂 OCP [需要测定]
        "Secondary: Negative electrode OCP [V]": silicon_ocp_average_Custom,
        # 硅平均 OCP (用于初始 SOC 计算)
        "Secondary: Negative electrode active material volume fraction": 0.015,
        # [测定] 硅相在负极涂层中的体积分数 (例如 1.5% = 约 5wt% Si)
        "Secondary: Negative particle radius [m]": 1.52e-06,  # [测定] PSD D50 (硅纳米颗粒)
        "Secondary: Negative electrode exchange-current density [A.m-2]"
        "": silicon_exchange_current_density_Custom,
        "Secondary: Negative electrode density [kg.m-3]": 2650.0,  # [测定] 硅相材料密度
        "Secondary: Negative electrode OCP entropic change [V.K-1]": silicon_entropic_change_Custom,
        # ==================================================================
        # 负极 共享参数 (两相共用) [需要测定]
        # ==================================================================
        "Negative electrode conductivity [S.m-1]": 215.0,
        "Negative electrode porosity": 0.25,  # [测定]
        "Negative electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Negative electrode Bruggeman coefficient (electrode)": 0,
        "Negative electrode charge transfer coefficient": 0.5,
        "Negative electrode double-layer capacity [F.m-2]": 0.2,
        "Negative electrode specific heat capacity [J.kg-1.K-1]": 700.0,
        "Negative electrode thermal conductivity [W.m-1.K-1]": 1.7,
        # ==================================================================
        # 正极 — LCO  [需要测定]
        # ==================================================================
        "Positive electrode conductivity [S.m-1]": 1.0,
        "Maximum concentration in positive electrode [mol.m-3]": 49900.0,  # [测定]
        "Positive particle diffusivity [m2.s-1]": lco_diffusivity_Custom,
        "Positive electrode OCP [V]": lco_ocp_Custom,
        "Positive electrode porosity": 0.28,  # [测定]
        "Positive electrode active material volume fraction": 0.62,  # [测定]
        "Positive particle radius [m]": 5.0e-06,  # [测定]
        "Positive electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Positive electrode Bruggeman coefficient (electrode)": 0,
        "Positive electrode charge transfer coefficient": 0.5,
        "Positive electrode double-layer capacity [F.m-2]": 0.2,
        "Positive electrode exchange-current density [A.m-2]"
        "": lco_exchange_current_density_Custom,
        "Positive electrode density [kg.m-3]": 3800.0,  # [测定]
        "Positive electrode specific heat capacity [J.kg-1.K-1]": 650.0,
        "Positive electrode thermal conductivity [W.m-1.K-1]": 1.5,
        "Positive electrode OCP entropic change [V.K-1]": lco_entropic_change_Custom,
        # ==================================================================
        # 隔膜 [需要测定]
        # ==================================================================
        "Separator porosity": 0.42,  # [测定]
        "Separator Bruggeman coefficient (electrolyte)": 1.5,
        "Separator density [kg.m-3]": 450.0,  # [测定]
        "Separator specific heat capacity [J.kg-1.K-1]": 700.0,
        "Separator thermal conductivity [W.m-1.K-1]": 0.2,
        # ==================================================================
        # 电解液 [需要测定]
        # Si-C 体系推荐使用含 FEC (氟代碳酸乙烯酯) 添加剂的电解液
        # ==================================================================
        "Initial concentration in electrolyte [mol.m-3]": 1000.0,
        "Cation transference number": 0.26,  # [测定]
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
        "Lower voltage cut-off [V]": 3.0,  # LCO 体系
        "Upper voltage cut-off [V]": 4.2,  # 标准 LCO
        "Open-circuit voltage at 0% SOC [V]": 3.0,
        "Open-circuit voltage at 100% SOC [V]": 4.2,
        "Initial concentration in negative electrode [mol.m-3]": 29866.0,
        # [测定] 负极总初始浓度 (两相加权平均)
        "Initial concentration in positive electrode [mol.m-3]": 22000.0,  # [测定]
        "Initial temperature [K]": 298.15,
        "citations": ["Chen2020", "Ai2022"],
    }
