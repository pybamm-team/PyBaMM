"""
=============================================================================
自定义用户参数文件 — 消费锂离子电池 (LFP / Graphite 体系)
=============================================================================

文件位置: userdata/lithum_ion/Custom_LFP_Graphite.py
用途:     为用户自定义磷酸铁锂(LiFePO₄)/石墨电池配方提供可编辑的参数模板

适用场景:
    - 便携式储能电源 / 户外电源
    - 电动工具 (电钻、电锯等)
    - 部分消费无人机 / 模型
    - 高安全需求设备

LFP 体系特点:
    ◈ 电压平台 ~3.4V (较 NMC/LCO 低) → 能量密度较低
    ◈ OCP 极平坦 (两相反应), 与 NMC/LCO 的倾斜曲线截然不同
    ◈ 本征电导率极低 (~10^-9 S/cm), 需碳包覆 + 纳米化
    ◈ 固相扩散系数低 (~10^-15 m^2/s)
    ◈ 热稳定性最优 (热失控温度 > 250°C), 安全性最高
    ◈ 循环寿命最长, 成本较低 (无钴)
    ◈ 理论容量 170 mAh/g

使用方式:
    import sys
    sys.path.insert(0, "path/to/pybamm/src")

    from userdata.lithum_ion.Custom_LFP_Graphite import get_parameter_values
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
    GITT / EIS 在不同 SOC 下测定。

    Parameters
    ----------
    sto : pybamm.Symbol  化学计量比
    T   : pybamm.Symbol  温度 [K]

    Returns
    -------
    pybamm.Symbol  扩散系数 [m^2·s^-1]
    """
    # ---- 用户需要替换 ----
    D_ref = 3.3e-14  # [用户需要测定]
    E_D_s = 0.0  # [用户需要测定]
    arrhenius = np.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))
    return D_ref * arrhenius


def graphite_exchange_current_density_Custom(c_e, c_s_surf, c_s_max, T):
    """
    石墨负极 Butler-Volmer 交换电流密度。

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
    # ---- 用户需要替换 ----
    m_ref = 6.48e-7  # [用户需要测定]
    E_r = 35000  # [用户需要测定]
    arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))
    return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5


def graphite_entropic_change_Custom(sto):
    """
    石墨负极熵变 dU/dT。 占位返回 0。

    Parameters
    ----------
    sto : pybamm.Symbol  化学计量比

    Returns
    -------
    pybamm.Symbol  熵变 [V·K^-1]
    """
    return pybamm.Scalar(0.0)


# =============================================================================
# 正极 — LFP (磷酸铁锂, LiFePO₄)
# =============================================================================


def lfp_ocp_Custom(sto):
    """
    LFP 正极开路电位 vs 化学计量比。

    【需要用户重新测定】
    LFP OCP 近似为水平平台 (两相反应, LiFePO₄ ⇌ FePO₄),
    平台两端各有一个很窄的单相固溶体区域。

    此示例使用 tanh 函数近似平台行为。
    GITT 测量 LFP 的 OCP 时需注意: 平台区域 dU/dsto ≈ 0,
    导致 GITT 扩散系数计算不准确, 推荐使用 EIS 辅助。

    Parameters
    ----------
    sto : pybamm.Symbol  嵌锂化学计量比 (sto=0 → FePO₄, sto=1 → LiFePO₄)

    Returns
    -------
    pybamm.Symbol  开路电位 [V]
    """
    # ---- 用户需要替换为实测 OCP 数据拟合公式 ----
    U = (
        3.42
        - 0.05 * np.tanh(50.0 * (sto - 0.02))  # 低 sto 端单相区
        + 0.05 * np.tanh(50.0 * (sto - 0.98))  # 高 sto 端单相区
    )
    return U


def lfp_diffusivity_Custom(sto, T):
    """
    LFP 正极固相扩散系数。

    【需要用户重新测定】
    LFP 固相扩散系数比 NMC 低 1-2 个数量级 (~10^-15 m^2/s)。
    且单相/两相区域差异大。

    测量建议: 优先在单相区 (sto 接近 0 或 1) 做 GITT。
    平台区域 (0.05 < sto < 0.95) GITT 不可靠, 改用 EIS + 等效电路拟合。

    Parameters
    ----------
    sto : pybamm.Symbol  化学计量比
    T   : pybamm.Symbol  温度 [K]

    Returns
    -------
    pybamm.Symbol  扩散系数 [m^2·s^-1]
    """
    # ---- 用户需要替换 ----
    D_ref = 1.0e-15  # [用户需要测定] LFP 典型 10^-14 ~ 10^-16 m^2/s
    E_D_s = 0.0  # [用户需要测定]
    arrhenius = np.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))
    return D_ref * arrhenius


def lfp_exchange_current_density_Custom(c_e, c_s_surf, c_s_max, T):
    """
    LFP 正极 Butler-Volmer 交换电流密度。

    【需要用户重新测定】
    LFP 反应动力学通常比 NMC 快 (交换电流密度更大),
    因为其纳米化 + 碳包覆结构提供了丰富的反应活性位点。

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
    m_ref = 1.0e-5  # [用户需要测定] LFP 通常 > NMC
    E_r = 20000  # [用户需要测定]
    arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))
    return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5


def lfp_entropic_change_Custom(sto):
    """
    LFP 正极熵变 dU/dT。 占位返回 0。

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

    【需要用户重新测定】示例使用 Nyman2008 作为占位参考。

    Parameters
    ----------
    c_e : pybamm.Symbol  电解液浓度 [mol·m^-3]
    T   : pybamm.Symbol  温度 [K]

    Returns
    -------
    pybamm.Symbol  扩散系数 [m^2·s^-1]
    """
    # ---- 用户需要根据电解液配方替换 ----
    D_c_e = 8.794e-11 * (c_e / 1000) ** 2 - 3.972e-10 * (c_e / 1000) + 4.862e-10
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
    sigma_e = (
        0.1297 * (c_e / 1000) ** 3 - 2.51 * (c_e / 1000) ** 1.5 + 3.329 * (c_e / 1000)
    )
    return sigma_e


# =============================================================================
# 参数字典
# =============================================================================


def get_parameter_values():
    """
    返回 LFP / Graphite 电池的完整参数字典。

    LFP 与消费电子主流 LCO/NMC 的主要差异:
      - 截止电压低 (3.6-3.7V)
      - 更多导电剂 (弥补低本征电导率)
      - 压实密度低 (2.0-2.4 g/cm³)
      - 纳米级一次颗粒 (100-500 nm) → 粒径参数需特别注意

    ============  ==========================================================
    类别            说明
    ============  ==========================================================
    SEI            可保留默认值
    电芯几何        [需要测定]
    负极 (石墨)     [需要测定]
    正极 (LFP)      [需要测定]
    隔膜            [需要测定]
    电解液          [需要测定]
    实验条件        [需要设定]
    ============  ==========================================================
    """
    return {
        "chemistry": "lithium_ion",
        # ==================================================================
        # SEI — 可保留默认值
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
        # 电芯几何尺寸  [需要用户测定]
        # ==================================================================
        "Negative current collector thickness [m]": 1.0e-05,
        "Negative electrode thickness [m]": 7.0e-05,
        "Separator thickness [m]": 2.0e-05,
        "Positive electrode thickness [m]": 6.5e-05,
        "Positive current collector thickness [m]": 1.5e-05,
        "Electrode height [m]": 0.1,  # [测定]
        "Electrode width [m]": 1.0,  # [测定]
        "Cell cooling surface area [m2]": 0.01,  # [测定]
        "Cell volume [m3]": 1.5e-05,  # [测定]
        "Cell thermal expansion coefficient [m.K-1]": 1.1e-06,
        # 集流体 — 标准值
        "Negative current collector conductivity [S.m-1]": 58411000.0,
        "Positive current collector conductivity [S.m-1]": 36914000.0,
        "Negative current collector density [kg.m-3]": 8960.0,
        "Positive current collector density [kg.m-3]": 2700.0,
        "Negative current collector specific heat capacity [J.kg-1.K-1]": 385.0,
        "Positive current collector specific heat capacity [J.kg-1.K-1]": 897.0,
        "Negative current collector thermal conductivity [W.m-1.K-1]": 401.0,
        "Positive current collector thermal conductivity [W.m-1.K-1]": 237.0,
        "Nominal cell capacity [A.h]": 2.5,  # [测定]
        "Current function [A]": 2.5,  # [设定]
        "Contact resistance [Ohm]": 0,
        # ==================================================================
        # 负极 — Graphite  [需要测定]
        # ==================================================================
        "Negative electrode conductivity [S.m-1]": 215.0,
        "Maximum concentration in negative electrode [mol.m-3]": 33133.0,  # [测定]
        "Negative particle diffusivity [m2.s-1]": graphite_diffusivity_Custom,
        "Negative electrode OCP [V]": graphite_ocp_Custom,
        "Negative electrode porosity": 0.28,  # [测定]
        "Negative electrode active material volume fraction": 0.72,  # [测定]
        "Negative particle radius [m]": 8.0e-06,  # [测定] PSD D50
        "Negative electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Negative electrode Bruggeman coefficient (electrode)": 0,
        "Negative electrode charge transfer coefficient": 0.5,
        "Negative electrode double-layer capacity [F.m-2]": 0.2,
        "Negative electrode exchange-current density [A.m-2]"
        "": graphite_exchange_current_density_Custom,
        "Negative electrode density [kg.m-3]": 1600.0,  # [测定]
        "Negative electrode specific heat capacity [J.kg-1.K-1]": 700.0,
        "Negative electrode thermal conductivity [W.m-1.K-1]": 1.7,
        "Negative electrode OCP entropic change [V.K-1]": graphite_entropic_change_Custom,
        # ==================================================================
        # 正极 — LFP  [需要测定]
        # LFP: 低电导率 → 大量导电剂; 低压实 → 高孔隙; 纳米颗粒
        # ==================================================================
        "Positive electrode conductivity [S.m-1]": 0.5,  # [测定] 碳包覆 + 导电剂
        "Maximum concentration in positive electrode [mol.m-3]": 22806.0,
        # c_max = ρ * Q_theoretical / (F * 3.6)
        # LFP: ρ≈3.6 (晶体), Q_theo=170 → 理论 ~17600
        # 实际二次颗粒 ρ≈2.2-2.3 (含碳包覆) → c_max ≈ 11000-12000
        "Positive particle diffusivity [m2.s-1]": lfp_diffusivity_Custom,
        "Positive electrode OCP [V]": lfp_ocp_Custom,
        "Positive electrode porosity": 0.30,  # [测定]
        "Positive electrode active material volume fraction": 0.55,  # [测定] 较低, 导电剂占体积
        "Positive particle radius [m]": 1.0e-06,  # [测定] 一次颗粒 100-500 nm
        "Positive electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Positive electrode Bruggeman coefficient (electrode)": 0,
        "Positive electrode charge transfer coefficient": 0.5,
        "Positive electrode double-layer capacity [F.m-2]": 0.2,
        "Positive electrode exchange-current density [A.m-2]"
        "": lfp_exchange_current_density_Custom,
        "Positive electrode density [kg.m-3]": 2200.0,  # [测定] 压实密度 2.0-2.4
        "Positive electrode specific heat capacity [J.kg-1.K-1]": 800.0,
        "Positive electrode thermal conductivity [W.m-1.K-1]": 1.2,
        "Positive electrode OCP entropic change [V.K-1]": lfp_entropic_change_Custom,
        # ==================================================================
        # 隔膜  [需要测定]
        # ==================================================================
        "Separator porosity": 0.45,  # [测定]
        "Separator Bruggeman coefficient (electrolyte)": 1.5,
        "Separator density [kg.m-3]": 450.0,  # [测定]
        "Separator specific heat capacity [J.kg-1.K-1]": 700.0,
        "Separator thermal conductivity [W.m-1.K-1]": 0.2,
        # ==================================================================
        # 电解液  [需要测定]
        # ==================================================================
        "Initial concentration in electrolyte [mol.m-3]": 1000.0,
        "Cation transference number": 0.26,  # [测定]
        "Thermodynamic factor": 1.0,
        "Electrolyte diffusivity [m2.s-1]": electrolyte_diffusivity_Custom,
        "Electrolyte conductivity [S.m-1]": electrolyte_conductivity_Custom,
        # ==================================================================
        # 实验 / 仿真条件  [需要设定]
        # ==================================================================
        "Reference temperature [K]": 298.15,
        "Total heat transfer coefficient [W.m-2.K-1]": 10.0,
        "Ambient temperature [K]": 298.15,
        "Number of electrodes connected in parallel to make a cell": 1.0,
        "Number of cells connected in series to make a battery": 1.0,
        "Lower voltage cut-off [V]": 2.5,  # LFP 通常 2.5-2.8V
        "Upper voltage cut-off [V]": 3.65,  # LFP 充电截止 ~3.6-3.7V
        "Open-circuit voltage at 0% SOC [V]": 2.5,
        "Open-circuit voltage at 100% SOC [V]": 3.45,
        "Initial concentration in negative electrode [mol.m-3]": 29866.0,  # [测定]
        "Initial concentration in positive electrode [mol.m-3]": 8000.0,  # [测定]
        "Initial temperature [K]": 298.15,
        "citations": ["Chen2020"],
    }
