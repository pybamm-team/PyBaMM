# PyBaMM 电池仿真学习指南

> 本文档基于 PyBaMM 项目源码整理，帮助快速上手电池仿真。

---

## 目录

0. [环境准备与快速入门](#零环境准备与快速入门)
1. [电池模型与选择](#一电池模型与选择)
2. [examples 示例学习路径](#二examples-示例学习路径)
3. [求解器](#三求解器)
4. [为实际电池做仿真](#四为实际电池做仿真)
5. [结果提取与可视化](#五结果提取与可视化)
6. [对比仿真](#六对比仿真)
7. [参数速查表](#七参数速查表)
8. [常见问题与报错处理](#八常见问题与报错处理)

---

## 零、环境准备与快速入门

### 0.1 环境准备

本项目已配置好 conda 环境 `pybamm`，包含 pybamm v26.5.1、matplotlib、jupyter 等所有必需依赖。

```bash
# 激活环境
conda activate pybamm

# 启动 Jupyter Notebook（在项目根目录下）
cd D:\PythonProject\LiBatteryProjects\pybamm
jupyter notebook
```

然后在浏览器中打开 `notebooks/pybamm_simulation_demo.ipynb` 即可开始交互式学习。

### 0.2 仿真三步曲（速览）

所有 PyBaMM 仿真都遵循 **模型 → 仿真 → 求解 → 可视化** 的流程：

```python
import pybamm

# 第一步：选择电池模型
model = pybamm.lithium_ion.SPM()

# 第二步：创建仿真对象
sim = pybamm.Simulation(model)

# 第三步：求解
sim.solve([0, 3600])  # 0 到 3600 秒

# 第四步：绘图查看结果
sim.plot()
```

---

## 一、电池模型与选择

### 1.1 核心电化学模型

| 模型 | 全称 | 相对速度 | 适用场景 |
|------|------|---------|---------|
| **SPM** | Single Particle Model | 1× (最快) | 快速仿真、SOC估算、初步验证 |
| **SPMe** | SPM with electrolyte | ~1.5× | 大倍率(>1C)、电解液浓差效应 |
| **MPM** | Multi-Particle Model | ~2-3× | 粒径分布影响大、商业化电池 |
| **DFN** | Doyle-Fuller-Newman | ~5× | 高精度仿真、电极厚度方向空间分辨 |
| **MP-DFN** | DFN + particle distribution | ~4× | 空间分辨 + 粒径分布（需选项开启）|
| **NewmanTobias** | Newman-Tobias 多孔电极模型 | ~4× | DFN 简化变体，恒电解液浓度 + 非线性动力学 |
| **MSMR** | Multi-Scale Multi-phase Reaction | ~7× (最慢) | 多位点化学体系、温度依赖热力学 |
| **Thevenin** | Thevenin 等效电路 | 毫秒级 | BMS 算法开发、快速原型验证 |

### 1.2 各模型详细说明

#### SPM — 单颗粒模型

最简化的电化学模型，将每个电极简化为单个代表性颗粒。**假设电解液浓度处处恒定**。

- ✅ 速度最快，适合参数扫描、初步验证
- ⚠️ 忽略了颗粒内部浓度梯度和电解液浓差

```python
model = pybamm.lithium_ion.SPM()
```

#### SPMe — 带电解液的单颗粒模型

在 SPM 基础上**加入电解液浓度扩散方程**，求解 ∂cₑ/∂t = -∇·Nₑ + sources，包含扩散、离子迁移和对流效应。

| 子模型层级 | SPM | SPMe |
|-----------|-----|------|
| 电解液扩散 | 常浓度（恒为零通量） | **Full（PDE 求解 cₑ）** |
| 电解液电导率 | LeadingOrder | Composite / Integrated |
| 电极电势 | LeadingOrder | Composite |

**什么时候选 SPMe？**

| 场景 | 原因 |
|------|------|
| **倍率 > 1C** | 大电流下锂离子局部消耗，产生浓度梯度，内阻增大、电压骤降 |
| **深充放（SOC 变化大）** | 电解液浓度全局变化显著，恒浓度假设不成立 |
| **仿真锂枝晶/锂沉积** | 局部电流密度分布与电解液浓度直接相关 |
| **温度建模** | 液相电导率 κ(cₑ, T) 随位置变化，需电解液浓度场 |

> **经验法则**：关心 C-rate > 1C 的电压平台和极化 → 选 SPMe；小倍率慢充慢放 → SPM 足够。

```python
# 大倍率放电 → 选 SPMe
model = pybamm.lithium_ion.SPMe()
experiment = pybamm.Experiment(["Discharge at 2C until 2.5V"])
```

#### MPM — 多颗粒模型（粒径分布）

现实中的电极颗粒**不是一样大的**。SPM/DFN 假设所有颗粒大小一致，MPM 引入粒径分布（默认为对数正态分布，标准差为平均半径的 30%）。

粒径分布对物理的影响：

| 影响方面 | 物理机理 |
|---------|---------|
| **扩散时间** ∝ R² | 小颗粒平衡速度快 4-16 倍 |
| **电流分配不均** | 小颗粒优先嵌入锂，大颗粒滞后 |
| **放电曲线变"圆"** | 电压平台不像单粒径那么尖锐 |
| **锂沉积风险** | 小颗粒先满 → 局部电流密度集中 → 析锂 |
| **循环寿命** | SEI 生长速率因颗粒大小而异 |

**什么时候选 MPM？**

- **商业化生产电池**（球磨石墨、大规模 NMC 制造）→ 粒径分布宽
- **LFP 电池** → 活性材料颗粒大且多分散
- **涉及退化/寿命预测** → 即使粒径分布窄，SEI 生长也跟粒径敏感

**如何判断粒径分布影响大不大？**

用 `DFN_size_distributions.py` 示例跑一下分布式 vs 非分布式的对比，看电压曲线差异是否在可接受范围内。也可以尝试不同标准差：

```python
model = pybamm.lithium_ion.MPM(
    options={
        "particle size": "distribution",
        "surface form": "algebraic",  # MPM 必选
    }
)
# 对比 MP-DFN（更精确的分布式版本）
model = pybamm.lithium_ion.DFN(options={"particle size": "distribution"})
```

#### DFN — 全 Newman 模型（Doyle-Fuller-Newman）

最完整的单粒径电化学模型，在颗粒径向**和**电极厚度方向都做空间离散化。

- ✅ 最高精度，颗粒内浓度梯度、液相浓差、界面过电位全部求解
- ⚠️ 计算量最大（~5× SPM），适合发表用和高精度验证

```python
model = pybamm.lithium_ion.DFN()
```

#### NewmanTobias — 多孔电极简化模型

基于 Newman & Tobias (1962) 多孔电极理论，在 PyBaMM 中继承自 DFN，是 **SPMe 与 DFN 之间的中间模型**。

| 特征 | DFN | NewmanTobias |
|------|-----|-------------|
| 电解液浓度 | 空间求解（PDE） | 恒定/均匀（简化） |
| 颗粒内浓度 | 空间求解 | 电极平均（XAveraged） |
| 界面动力学 | Butler-Volmer | Butler-Volmer |
| 电极厚度方向 | 空间离散 | 不做空间离散 |

**什么时候选 NewmanTobias？**

- 需要比 SPMe 更精确但 DFN 太慢时
- 关注**非线性动力学**效应但可忽略电解液空间浓度变化时
- 介于 SPMe 和 DFN 之间的精度/速度折中

```python
model = pybamm.lithium_ion.NewmanTobias()
```

#### MSMR — 多尺度多相反应模型

在 DFN 基础上引入**每个电极多个独立的嵌入位点**，每个位点有自己的热力学电位 U₀ᵢ、occupancy 份额 Xᵢ 和交换电流密度 j₀,ᵢ。

**与 DFN 的核心区别：**

| | DFN | MSMR |
|--|-----|------|
| 每个电极 OCV 曲线数 | 1 条 | N 条（不同位点） |
| 温度依赖 dU/dT | 需拟合 | 自然产生（Boltzmann 分布推导） |
| 参数结构 | 简单 | 按位点编号（X_n_l_0, U0_p_d_2 等） |

**什么时候选 MSMR？**

| 场景 | 原因 |
|------|------|
| **多价态正极**（NCA/NMC 中 Ni²⁺/Ni³⁺/Ni⁴⁺） | 不同价态 = 不同嵌入位点 |
| **温度效应研究/热失控预测** | dU/dT 自然 emerge，不需拟合 |
| **多电压平台材料** | 位点分化反映不同平台 |
| **选择性位点老化** | 某些位点先退化，影响容量衰减 |
| **钠离子电池（P2-O3 相）** | 多个棱柱位点 |

> 代价：计算量 ~7× SPM，适合机理研究，不适合快速迭代。

```python
# 6 个负极位点 + 4 个正极位点
model = pybamm.lithium_ion.MSMR(options={"number of MSMR reactions": ("6", "4")})
param = pybamm.ParameterValues("MSMR_Example")
```

#### Thevenin — 等效电路模型

用 OCV + 电阻 + RC 环节拟合电压响应，**不含电化学场方程**。

- ✅ 毫秒级求解，适合 BMS 控制算法开发
- ⚠️ 无内部物理场信息

```python
model = pybamm.equivalent_circuit.Thevenin()
```

### 1.3 电池体系支持

PyBaMM 支持多种电池体系：

| 体系 | 模块/源码位置 | 模型 |
|------|---------|------|
| 锂离子电池 | `pybamm.lithium_ion` | SPM, SPMe, DFN, NewmanTobias, MPM, MSMR 等 |
| 铅酸电池 | `pybamm.lead_acid` | LOQS, Full, BasicFull |
| 等效电路 | `pybamm.equivalent_circuit` | Thevenin |
| 钠离子电池 | `pybamm.sodium_ion` | BasicDFN |
| 锂金属电池 | `src/pybamm/models/full_battery_models/lithium_metal/dfn.py` | DFN |

### 1.4 PyBaMM 仿真领域总览

PyBaMM 的主战场是"电芯级"和"机理驱动"的电池仿真。它不是整车动力学软件，也不是整包 CFD 平台，但非常适合下面这些领域。

#### 子模型"积木"架构

PyBaMM 的电池模型不是一块铁板，而是由 ~17 个物理子模型像积木一样**组合装配**而成的。你在 1.6 节看到的 `options` 字典，本质上就是在选择每个子模型用什么配方。源码中，这些子模型的实现位于 `src/pybamm/models/submodels/` 下。

| 子模型分类 | 源码目录 | 控制的 options 键 | 典型选项 | 物理描述 |
|-----------|---------|------------------|---------|---------|
| **颗粒固相扩散** | `submodels/particle/` | `"particle"` | `Fickian diffusion`, `quadratic profile`, `MSMR` | 活性材料颗粒内部的锂离子扩散。理论上用 Fick 第二定律 $\partial c/\partial t = D \nabla^2 c$；`quadratic profile` 假设径向为抛物线分布，无需 PDE 求解，计算更快 |
| **界面反应动力学** | `submodels/interface/kinetics/` | `"intercalation kinetics"` | `Butler-Volmer`, `asymmetric Butler-Volmer`, `Marcus`, `Tafel`, `diffusion-limited` | 电极/电解液界面上的电化学反应速率。经典 Butler-Volmer 方程 $j = j_0 [\exp(\alpha_a F\eta/RT) - \exp(-\alpha_c F\eta/RT)]$；Marcus 理论则提供了强电场下的修正 |
| **电解液扩散** | `submodels/electrolyte_diffusion/` | （随模型层级自动选择） | `full`, `leading order`, `constant concentration` | 电解液相中锂盐浓度的空间分布。SPM 默认 `constant`（恒浓）——速度快但忽略浓差扩散；SPMe 升级为 `full`（完整 PDE）。高 C-rate 下浓差极化显著 |
| **电解液电导率** | `submodels/electrolyte_conductivity/` | （随模型层级自动选择） | `full`, `composite`, `integrated`, `leading order` | 电解液相的离子电导率和电位分布。`full` 求解 Ohm 定律获取液相电位 $\phi_e(x,t)$；`leading order` 用零阶近似 |
| **SEI 膜生长** | `submodels/interface/sei/` | `"SEI"` | `none`, `constant`, `reaction limited`, `solvent-diffusion limited`, `stress-driven` | 模拟固体电解质界面膜的生长。`reaction limited` 考虑电子隧穿限制；`stress-driven` 与颗粒机械应力耦合；`solvent-diffusion limited` 考虑溶剂扩散到 SEI 内部的限制 |
| **锂沉积/析锂** | `submodels/interface/lithium_plating/` | `"lithium plating"` | `none`, `reversible`, `irreversible` | 模拟充电时锂金属在负极表面的沉积。`reversible` 沉积锂可逆向剥离回电解液；`irreversible` 沉积为永久死锂 |
| **机械应力与裂纹** | `submodels/particle_mechanics/` | `"particle mechanics"` | `none`, `swelling only`, `crack propagation` | 颗粒体积膨胀/收缩产生的应力。`crack propagation` 会产生新表面 → 新 SEI 生长 → 加速容量损失，是循环老化的关键路径之一 |
| **活性材料损失** | `submodels/active_material/` | `"loss of active material"` | `none`, `stress-driven`, `reaction-driven` | 活性材料不可逆减少导致的容量损失。可由应力驱动（颗粒破碎失去电接触）或反应驱动（直接化学消耗） |
| **孔隙率演变** | `submodels/porosity/` | `"SEI porosity change"` 等 | `none`, `reaction driven`（ODE / PDE） | SEI 生长或反应产物占据孔隙空间导致孔隙率降低，进而影响传输效率和阻抗 |
| **热模型** | `submodels/thermal/` | `"thermal"` | `isothermal`, `lumped`, `x-lumped`, `x-full`, `3D` | 从 0D（等温/集总）到 3D（FEM，scikit-fem 后端）、从单温度到包含面内温度梯度的多层级热模型。3D 模型支持方柱（pouch）和圆柱几何 |
| **集流体与电流分配** | `submodels/current_collector/` | `"current collector"` + `"dimensionality"` | `homogeneous`, `potential pair`, `effective resistance` | `homogeneous` 假设电流均匀分布（0D）；`potential pair` 求解集流体上的电位分布（2D/3D），用于分析空间不均匀性、涂布缺陷、tab 设计等 |
| **OCV / 开路电位** | `submodels/interface/open_circuit_potential/` | `"open-circuit potential"` | `single`, `MSMR`, `hysteresis` | OCV vs SOC 曲线。`single` 为标准单条 OCP 曲线；`MSMR` 为多位点模型独立 OCP；`hysteresis` 模拟充放电 OCV 不重合的滞回效应 |
| **界面利用率** | `submodels/interface/interface_utilisation/` | `"interface utilisation"` | `constant`, `full`, `current driven` | 控制有益的界面面积比例（区别于孔面积和活性面积） |
| **外部电路控制** | `submodels/external_circuit/` | `"operating mode"` | `current`, `voltage`, `power`, `CCCV`, `resistance`, `function` | 电池外部的工作模式：恒流、恒压、恒功率、CCCV 充电、恒电阻、自定义函数 |
| **对流** | `submodels/convection/` | `"convection"` | `none`, `through-cell`, `transverse` | 电解液流动效应。`through-cell` 为全电池对流；`transverse` 为横向对流。一般锂离子电池对流传导弱，铅酸电池中更常用 |
| **传输效率 / 曲折因子** | `submodels/transport_efficiency/` | （参数级配置，非 options 键） | `Bruggeman`, `ordered packing`, `overlapping spheres`, `random overlapping cylinders`, `tortuosity factor`, `hyperbola of revolution`, `cation exchange membrane`, `heterogeneous catalyst` | 孔隙传输效率模型，将有效传输系数与孔隙率、曲折因子关联。`Bruggeman`（$\varepsilon^{1.5}$）最常用 |
| **电极欧姆** | `submodels/electrode/ohm/` | （随模型层级自动选择） | 固相电位求解 | 活性材料固相中的电子传导方程，计算固相电位 $\phi_s(x,t)$ |

> **理解这个架构的意义**：当你需要某个仿真能力时，可以先定位它属于哪个子模型，再查找对应的 options 键名。例如"想仿真循环老化"→ 需要开启 `"SEI"`（SEI 生长）、`"lithium plating"`（析锂）、`"particle mechanics"`（应力→裂纹→新 SEI）这三个子模型。

#### 应用领域一览（8 大方向）

| 仿真领域 | 典型问题 | 在 PyBaMM 中对应的能力 |
|---------|---------|------------------------|
| **电芯性能与倍率能力** | 不同 C-rate 下的容量、极化、电压平台、倍率性能对比 | `SPM/SPMe/DFN/MPM`，`rate_capability.py`，`compare_lithium_ion.py` |
| **充放电策略与实验协议设计** | CCCV、GITT、脉冲工况、formation、storage loss 等实验流程 | `pybamm.Experiment`，`experimental_protocols/cccv.py`，`experimental_protocols/gitt.py` |
| **工况复现与控制导向仿真** | Drive cycle、电流/功率跟踪、BMS 原型算法验证 | `drive_cycle.py`，`experiment_drive_cycle.py`，`pybamm.equivalent_circuit.Thevenin()`，`run_ecm.py` |
| **热管理与温升评估** | 环境温度变化、散热边界、软包冷却、热源分解 | 热模型选项 `thermal=*`，`DFN_ambient_temperature.py`，`pouch_cell_cooling.py`，`plot_thermal_components` |
| **老化与寿命预测** | SEI 生长、锂沉积、日历老化、循环老化、容量衰减 | `"SEI"` / `"lithium plating"` / `"loss of active material"` / `"particle mechanics"` 相关选项，`calendar_ageing.py`，`cycling_ageing.py` |
| **诊断与状态分析** | EIS 阻抗谱、电压损失拆解、半电池/电极状态分析 | `pybamm.EISSimulation`，`plot_voltage_components`，半电池配置，相关 notebooks |
| **参数化、实验校验与批量研究** | 内置参数集替换、BPX 导入、实验数据对比、批量扫参 | `ParameterValues`，`create_from_bpx`，`InputParameter`，`BatchStudy` |
| **新机理与学术建模** | 粒径分布、多位点反应、不同化学体系、自定义 PDE / submodel | `MPM`，`MSMR`，`lead_acid`，`sodium_ion`，锂金属模型，`create_model.py`，`custom_model.py` |

#### 应用领域 ↔ 子模型映射速查

当你有一个具体的仿真需求时，下面的映射可以帮你快速定位需要开启哪些子模型（options 键）：

| 你想做什么... | 需要关注的子模型 | 对应 options 键 | 说明 |
|-------------|----------------|----------------|------|
| 仿真基础电压/充放电曲线 | 固相扩散 + 动力学 + 电解液（自动） | `"particle"` + `"intercalation kinetics"` | 任何模型默认就包含这些，无需额外配置 |
| 大倍率/高精度电化学分析 | ↑ 基础 + 电解液浓差 + 电极厚度方向空间分辨 | 直接选 **DFN**（自动包含 Full diffusion + Full conductivity） | SPM 忽略浓差，SPMe 有浓差但无厚度方向空间分辨 |
| 考虑粒径分布的影响 | ↑ 基础 + 多颗粒粒径分布 | `"particle size": "distribution"` + `"surface form": "algebraic"` | 或直接选 MPM（更简单），MP-DFN（更精确） |
| 仿真循环老化/容量衰减 | ↑ 基础 + SEI + [析锂] + [应力裂纹] + [活性材料损失] | `"SEI"` + `"SEI porosity change"` + `"lithium plating"` + `"particle mechanics"` + `"loss of active material"` | SEI 是最常见的退化机制，其余可选配 |
| 做热管理/冷却方案设计 | ↑ 基础 + 热方程 + [空间维度] | `"thermal": "isothermal"/"lumped"/"x-full"` + `"dimensionality": 0/2/3` | 0D 集总最快，2D/3D 能捕捉空间温度梯度 |
| 分析集流体电流分配/涂布缺陷 | ↑ 基础 + 集流体电位对 | `"current collector": "potential pair"` + `"dimensionality": 2` | `potential pair` 求解 y-z 面的电位和电流密度分布 |
| 3D 热管理（软包/圆柱电池） | ↑ 基础 + 3D 热方程 + `Basic3DThermalSPM` 模型 | `"cell geometry": "pouch"/"cylindrical"` + `"dimensionality": 3` | 需安装 scikit-fem，6 面独立设置换热系数 |
| EIS 阻抗谱分析 | ↑ 基础 + 表面形式 | `"surface form": "differential"` + `pybamm.EISSimulation` | 时域仿真不需要 `surface form`，仅 EIS 需要 |
| 多电压平台/温度依赖热力学 | ↑ 基础 + 多位点 OCP | 直接选 **MSMR** + `"number of MSMR reactions": ("6", "4")` | dU/dT 从 Boltzmann 分布自然产生 |
| BMS 快速外特性原型 | 不需要电化学场方程 | 直接用 `pybamm.equivalent_circuit.Thevenin()` | 毫秒级求解，无内部物理场 |
| 模拟制造缺陷（烘干/涂布不均） | ↑ 基础 + 空间集流体分布 + 空间参数函数 | `"current collector": "potential pair"` + `dimensionality: 2` + 空间孔隙率函数 | 参见 `dried_out_pouch.py` 示例 |
| 半电池/电极状态分析 | ↑ 基础 + 工作电极配置 | `"working electrode": "positive"` 或 `"working electrode": "negative"` | 参考 `compare_lithium_ion_half_cell.py` |

上述子模型可以自由组合。例如一个完整的 **"大倍率 DFN + 循环老化 + 集总热"** 配置如下：

```python
model = pybamm.lithium_ion.DFN(
    options={
        # —— 颗粒与动力学（DFN 默认已是 Fickian + Butler-Volmer，这里显式写出来）——
        "particle": "Fickian diffusion",
        "intercalation kinetics": "asymmetric Butler-Volmer",
        # —— 退化路径 1：SEI 生长（反应速率限制，考虑溶剂扩散和电子隧穿）——
        "SEI": "reaction limited",
        "SEI porosity change": "true",  # SEI 占据孔隙 → 孔隙率下降 → 传输变差 → 阻抗增大
        # —— 退化路径 2：可逆锂沉积（快充时负极表面锂堆积）——
        "lithium plating": "reversible",
        # —— 退化路径 3：应力 → 裂纹 → 新 SEI（加速容量损失）——
        "particle mechanics": "crack propagation",
        "loss of active material": "stress-driven",
        # —— 热模型（集总参数，适合系统级分析）——
        "thermal": "lumped",
    }
)
```

#### 适用边界

如果按应用边界来概括，PyBaMM 最擅长的是"**电芯内部机理 + 实验工况 + 参数化验证**"的闭环研究：

- ✅ **擅长**：电芯级电化学仿真、老化机理研究、热管理设计、充放电协议优化、EIS 诊断、参数化与实验校验
- ⚠️ **可做但非专长**：整包热网络（可作为电芯子模型嵌入）、结构力学（仅颗粒级别的应力模型）
- ❌ **不适合**：整车动力学、结构碰撞、流场 CFD、电机控制、电池包级布置优化

对于整包热网络、结构力学、整车能量管理这类更高层系统问题，PyBaMM 更适合作为其中的电芯子模型，而不是唯一仿真平台。

### 1.5 模型选择决策图

```
你的需求是什么？
│
├─ 快速原型 / 参数探索 ──────→ SPM
│
├─ C-rate > 1C / 电解液效应 ──→ SPMe
│
├─ 非线性动力学 / 不需液体浓度场 → NewmanTobias（SPMe 与 DFN 折中）
│
├─ 粒径分布影响 ─────────────→ MPM（快）或 DFN{"particle size":"distribution"}（准）
│
├─ 高精度 / 大倍率空间分辨 ──→ DFN
│
├─ 多位点复杂化学体系 ───────→ MSMR
│
└─ 只需外特性 / BMS ─────────→ Thevenin
```

### 1.6 模型配置选项 (options)

模型通过 `options` 字典进行细粒度配置：

```python
import pybamm

# 基本用法
model = pybamm.lithium_ion.DFN()

# 带配置选项
options = {
    "thermal": "lumped",  # 热模型: isothermal, lumped, x-lumped, x-full
    "SEI": "reaction limited",  # SEI 膜: none, constant, reaction limited, stress-driven
    "lithium plating": "reversible",  # 锂沉积: none, reversible, irreversible
    "particle": "quadratic profile",  # 颗粒模型: Fickian diffusion, quadratic profile
    "intercalation kinetics": "asymmetric Butler-Volmer",  # 动力学模型
    "dimensionality": 0,  # 维度: 0D, 1D, 2D, 3D
}
model = pybamm.lithium_ion.DFN(options=options)
```

#### 主要配置类别速查

| 类别 | 选项示例 | 说明 |
|------|---------|------|
| **operating mode** | current, voltage, power, CCCV, resistance | 控制模式 |
| **thermal** | isothermal, lumped, x-lumped, x-full | 热模型 |
| **particle** | Fickian diffusion, quadratic profile, MSMR | 颗粒扩散 |
| **SEI** | none, constant, reaction limited, solvent-diffusion limited, etc. | SEI 生长 |
| **lithium plating** | none, reversible, irreversible | 锂沉积 |
| **intercalation kinetics** | Butler-Volmer, asymmetric Butler-Volmer, Marcus | 反应动力学 |
| **dimensionality** | 0, 1, 2, 3 | 几何维度 |
| **surface form** | full, differential, algebraic | 表面形式 |

#### 不同电极可配置不同选项（高级用法）

```python
options = {
    "particle": ("Fickian diffusion", "quadratic profile"),  # (负极, 正极)
    "SEI": ("reaction limited", "none"),  # 仅负极有 SEI
}
```

#### 查看所有可用选项

```python
all_options = pybamm.BatteryModelOptions({}).possible_options
```

> 💡 上表中每个选项的源码实现位于 `src/pybamm/models/submodels/` 下对应的子目录，完整映射关系见 1.4 节"子模型积木架构"表格。

### 1.7 代码示例：选择并创建模型

```python
import pybamm

# ====== 锂离子电池 ======
model_spm = pybamm.lithium_ion.SPM()  # 单颗粒模型
model_spme = pybamm.lithium_ion.SPMe()  # 带电解液的 SPM
model_dfn = pybamm.lithium_ion.DFN()  # 全 Newman 模型
model_mpm = pybamm.lithium_ion.MPM()  # 多颗粒模型
model_msmr = pybamm.lithium_ion.MSMR()  # 多尺度多相反应模型

# ====== 铅酸电池 ======
model_la = pybamm.lead_acid.Full()  # 完整铅酸模型
model_loqs = pybamm.lead_acid.LOQS()  # 铅酸简化模型

# ====== 等效电路 ======
model_ecm = pybamm.equivalent_circuit.Thevenin()  # Thevenin 模型

# ====== 创建仿真并求解 ======
sim = pybamm.Simulation(model_spm)
sim.solve([0, 3600])  # 仿真 0 到 3600 秒 (1小时)
sim.plot()
```

---

## 二、examples 示例学习路径

除了 `examples/scripts/` 下的可直接运行脚本外，PyBaMM 当前更系统的教学材料主要放在 `docs/source/examples/notebooks/`。可以把两者理解为：

- `examples/scripts/`：适合快速照着跑，确认某种功能是否存在。
- `docs/source/examples/notebooks/`：适合理解原理和底层流程，尤其是参数化、网格/离散化、性能优化、BatchStudy 等主题。

如果你读完本文后还想继续深挖，优先参考下面这些 notebook：

| 主题 | 推荐 notebook | 用途 |
|------|---------------|------|
| 入门运行流程 | `getting_started/` | `Simulation`、`Experiment`、参数集的基本用法 |
| 参数化与 BPX | `parameterization/parameter-values.ipynb`、`parameterization/bpx.ipynb` | 参数替换、输入参数、BPX 文件导入 |
| 网格与离散化 | `spatial_methods/finite-volumes.ipynb` | `geometry`、`Mesh`、`Discretisation` 的底层流程 |
| 批量对比 | `batch_study.ipynb` | 多模型、多参数、多实验工况的系统比较 |
| 性能优化 | `performance/02-input-parameters.ipynb` | 用 `InputParameter` 避免重复建模 |

### 2.1 第一阶段：入门必读（3 个）

| 顺序 | 脚本文件 | 学到什么 |
|------|---------|---------|
| 1 | `compare_lithium_ion.py` | 模型选择：对比 SPM/SPMe/DFN/NewmanTobias 的差异 |
| 2 | `print_parameters.py` | 参数系统：查看所有内置参数及其含义和单位 |
| 3 | `experiment_drive_cycle.py` | 工况设置：学会用 `pybamm.Experiment` 定义充放电协议和驱动循环 |

### 2.2 第二阶段：体系与参数（3 个）

| 顺序 | 脚本文件 | 学到什么 |
|------|---------|---------|
| 4 | `nca_parameters.py` | 自定义化学体系：NCA 正极参数配置方法 |
| 5 | `rate_capability.py` | 倍率性能：不同 C-rate 对比仿真 |
| 6 | `compare_lead_acid.py` | 换体系：铅酸电池仿真怎么做 |

### 2.3 第三阶段：根据实际需求选择性学习

| 方向 | 推荐脚本 | 说明 |
|------|---------|------|
| 🔬 **热管理** | `DFN_ambient_temperature.py` → `pouch_cell_cooling.py` | 温度变化、冷却策略 |
| 📉 **老化仿真** | `cycling_ageing.py` → `calendar_ageing.py` | 循环老化、日历老化 |
| ⚡ **特殊模型** | `MSMR.py` → `DFN_size_distributions.py` | 反应动力学、粒径分布 |
| 🔋 **等效电路** | `run_ecm.py` | Thevenin 等效电路快速仿真 |
| 📐 **3D 仿真** | `3d_examples/` 目录 | 三维几何、圆柱/软包/方形电池——涉及 thermal（3D 热方程）和 current_collector（集流体电位对）两个子模型，需 ScikitFEM 后端 |
| 🧪 **实验协议** | `experimental_protocols/cccv.py` → `gitt.py` | CCCV 循环、GITT 脉冲 |
| 📊 **模型对比** | `compare_dae_solver.py` → `compare_intercalation_kinetics.py` | 求解器、动力学模型对比 |
| 🏗️ **自定义模型** | `create_model.py` → `custom_model.py` | 从零搭建自定义物理模型 |
| 🧮 **EIS 仿真** | `run_eis_simulation.py` | 电化学阻抗谱 |
| 🔗 **数据耦合** | `minimal_example_of_lookup_tables.py` → `minimal_interp3d_example.py` | 查找表、三维插值 |

### 2.4 完整脚本列表

#### 模型对比类

| 脚本 | 内容 |
|------|------|
| `compare_lithium_ion.py` | 对比 SPM/SPMe/DFN/NewmanTobias |
| `compare_lithium_ion_2D.py` | 2D 模型对比 |
| `compare_lithium_ion_half_cell.py` | 半电池配置对比 |
| `compare_lithium_ion_heat_of_mixing.py` | 混合热效应 |
| `compare_lithium_ion_two_phase.py` | 复合电极 |
| `compare_lead_acid.py` | 铅酸模型对比 |
| `compare_intercalation_kinetics.py` | 动力学模型对比 |
| `compare_particle_models.py` | 颗粒模型对比 |
| `compare_spectral_volume.py` | 谱体积离散化 |
| `compare_extrapolations.py` | 外推方法 |
| `compare_interface_utilisation.py` | 界面利用率 |
| `compare_dae_solver.py` | DAE 求解器对比 |
| `compare_surface_temperature.py` | 表面温度 |

#### 参数与化学体系类

| 脚本 | 内容 |
|------|------|
| `nca_parameters.py` | NCA 化学体系参数 |
| `print_parameters.py` | 打印所有参数值 |
| `print_model_parameter_combinations.py` | 模型-参数组合兼容性 |
| `create_model.py` | 自定义参数模型 |
| `custom_model.py` | 完全自定义模型 |

#### 热仿真类

| 脚本 | 内容 |
|------|------|
| `thermal_lithium_ion.py` | 完整 3D 热仿真 |
| `heat_equation.py` | 简单热方程 |
| `DFN_ambient_temperature.py` | 环境温度变化 |
| `pouch_cell_cooling.py` | 软包冷却 |
| `compare_surface_temperature.py` | 表面温度对比 |

#### 老化与退化类

| 脚本 | 内容 |
|------|------|
| `calendar_ageing.py` | 日历老化 |
| `cycling_ageing.py` | 循环老化(SEI 生长) |
| `empirical_hysteresis.py` | 滞回效应 |

#### 驱动循环与工况类

| 脚本 | 内容 |
|------|------|
| `drive_cycle.py` | US06 驱动循环 |
| `experiment_drive_cycle.py` | 实验协议+驱动循环+功率映射 |
| `rate_capability.py` | C-rate 扫描 |
| `experimental_protocols/cccv.py` | CCCV 充放电循环 |
| `experimental_protocols/gitt.py` | GITT 脉冲实验 |

#### 基础模型示例

| 脚本 | 内容 |
|------|------|
| `DFN.py` | 基础 DFN 仿真 |
| `SPMe.py` | SPMe 仿真 |
| `SPMe_SOC.py` | SOC 依赖参数 |
| `SPMe_step.py` | 阶跃响应 |
| `MSMR.py` | 多尺度多相反应模型 |
| `DFN_size_distributions.py` | 颗粒粒径分布 |
| `DFN_ambient_temperature.py` | 环境温度影响 |
| `conservation_lithium.py` | 锂守恒验证 |

#### 等效电路

| 脚本 | 内容 |
|------|------|
| `run_ecm.py` | Thevenin ECM 模型 |
| `run_ecmd.py` | Thevenin + 扩散 |

#### 其他

| 脚本 | 内容 |
|------|------|
| `run_simulation.py` | 基础仿真模板 |
| `run_eis_simulation.py` | EIS 阻抗谱 |
| `multiprocess_inputs.py` | 多进程输入 |
| `multiprocess_jax_solver.py` | JAX 求解器多进程 |
| `coupled_variable_example.py` | 耦合变量示例 |
| `minimal_interp3d_example.py` | 3D 插值 |
| `minimal_example_of_lookup_tables.py` | 查找表使用 |

---

## 三、求解器

PyBaMM 支持多种求解器，按需选择：

| 求解器 | 创建方式 | 说明 |
|--------|---------|------|
| **CasADiSolver** | `pybamm.CasadiSolver()` | 默认求解器，平衡速度与精度 |
| **IDAKLUSolver** | `pybamm.IDAKLUSolver()` | C++ 后端，适合大型 stiff 问题 |
| **ScipySolver** | `pybamm.ScipySolver()` | 基于 SciPy，备选 |
| **JaxSolver** | `pybamm.JaxSolver()` | GPU/TPU 加速，需要 JAX |

```python
# 指定求解器
sim = pybamm.Simulation(model, solver=pybamm.CasadiSolver(mode="safe"))
```

**性能提示：**
- DFN 模型建议用 `IDAKLUSolver`，大型 stiff 系统更快
- 需要加速批量仿真时考虑 `JaxSolver`
- 遇到数值不稳定时切换 `CasadiSolver(mode="safe")`

---

## 四、为实际电池做仿真

### 3.1 整体工作流

```
┌─────────────────────────────────────────────────────────┐
│                    你的实际电池                            │
└──────────┬──────────────────────────────┬─────────────────┘
           │                              │
           ▼                              ▼
   ┌───────────────┐           ┌─────────────────────┐
   │ 选择模型       │           │ 提取电池参数          │
   │ SPM/SPMe/DFN   │           │ 几何/电化学/OCV/     │
   │ MPM/MSMR/ECM   │           │ 动力学/热/传输       │
   └───────┬───────┘           └──────────┬──────────┘
           │                              │
           ▼                              ▼
   ┌───────────────┐           ┌─────────────────────┐
   │ 配置 options   │           │ 构建 ParameterValues │
   │ thermal/SEI/   │           │ 内置集+自定义覆盖     │
   │ particle/...   │           │ 或完全自定义          │
   └───────┬───────┘           └──────────┬──────────┘
           │                              │
           └──────────┬───────────────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │ 定义充放电工况        │
           │ pybamm.Experiment   │
           │ CCCV / 驱动循环 /   │
           │ 恒流 / 功率 / GITT  │
           └──────────┬──────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │ 仿真求解              │
           │ sim.solve()          │
           └──────────┬──────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │ 对比实验数据验证      │
           │ sim.plot()          │
           │ sim.solution.save()  │
           └──────────┬──────────┘
                      │
              ┌───────┴───────┐
              ▼               ▼
         不匹配→ 调参       匹配→ 预测/优化
```

#### `Simulation` 默认会自动完成什么

大多数情况下，`pybamm.Simulation(...)` 会自动帮你完成参数替换、几何参数处理、网格生成、离散化和求解器调用，所以入门时不需要自己手动搭整条数值管线。

```python
model = pybamm.lithium_ion.DFN()
param = pybamm.ParameterValues("Chen2020")

var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 30, var.x_s: 30, var.x_p: 30, var.r_n: 10, var.r_p: 10}

sim = pybamm.Simulation(model, parameter_values=param, var_pts=var_pts)
solution = sim.solve([0, 3600])
```

#### 什么时候需要显式处理 geometry / mesh / discretisation

当你要比较不同网格、修改子网格类型、搭建自定义模型，或研究数值精度时，才需要显式走底层流程：

```python
model = pybamm.lithium_ion.DFN()
param = pybamm.ParameterValues("Chen2020")

var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 30, var.x_s: 30, var.x_p: 30, var.r_n: 10, var.r_p: 10}

geometry = model.default_geometry
param.process_geometry(geometry)

mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)

processed_model = param.process_model(model, inplace=False)
disc.process_model(processed_model)
```

这条底层路径可以结合 `examples/scripts/DFN.py`、`examples/scripts/SPMe.py`、`examples/scripts/create_model.py`、`examples/scripts/compare_spectral_volume.py` 一起看。

### 3.2 Step 1 — 选择模型

```python
import pybamm

# 根据需求选择（参考 1.5 节的决策图）
model = pybamm.lithium_ion.DFN()
```

### 3.3 Step 2 — 自定义参数

#### 方式 A：基于内置参数集修改（推荐入门方式）

```python
import pybamm

model = pybamm.lithium_ion.DFN()

# 先加载一个接近你体系的内置参数集
param = pybamm.ParameterValues(
    "Chen2020"
)  # 常用集: "Chen2020", "Marquis2019", "NCA_Kim2011"

# 用你的实际参数覆盖
param.update(
    {
        # ===== 几何参数（用你电池的实际尺寸）=====
        "Negative electrode thickness [m]": 50e-6,  # 负极厚度
        "Positive electrode thickness [m]": 80e-6,  # 正极厚度
        "Separator thickness [m]": 25e-6,  # 隔膜厚度
        "Electrode height [m]": 0.065,  # 电极高度
        "Electrode width [m]": 1.6,  # 电极宽度（决定容量）
        # ===== 电化学参数 =====
        "Initial concentration in negative electrode [mol.m-3]": 30000,  # x100 in LiyC6
        "Initial concentration in positive electrode [mol.m-3]": 49000,  # x100 in LiyNiMnCoO2
        "Maximum concentration in negative electrode [mol.m-3]": 31000,
        "Maximum concentration in positive electrode [mol.m-3]": 51555,
        # ===== 颗粒尺寸 =====
        "Negative particle radius [m]": 5e-6,  # 负极颗粒半径
        "Positive particle radius [m]": 5e-6,  # 正极颗粒半径
        # ===== 容量 =====
        "Nominal cell capacity [A.h]": 5.0,  # 名义容量
        # ===== 扩散系数 =====
        "Negative electrode diffusivity [m2.s-1]": 3.9e-14,
        "Positive electrode diffusivity [m2.s-1]": 1.0e-13,
        # ===== OCV 函数（可以用插值函数）=====
        # 参见 3.5 节
    }
)
```

#### 方式 B：完全自定义（从零构建）

```python
param = pybamm.ParameterValues(
    {
        "Nominal cell capacity [A.h]": 5.0,
        "Current function [A]": 5.0,
        "Negative electrode thickness [m]": 50e-6,
        "Positive electrode thickness [m]": 80e-6,
        "Separator thickness [m]": 25e-6,
        "Negative particle radius [m]": 5e-6,
        "Positive particle radius [m]": 5e-6,
        # ... 更多参数（共需 30+ 个）
    }
)
```

#### 方式 C：参数扫描（对比不同参数值）

```python
base_param = pybamm.ParameterValues("Chen2020")

for h_value in [5, 10, 20]:
    param = base_param.copy()
    param.update({"Total heat transfer coefficient [W.m-2.K-1]": h_value})
    sim = pybamm.Simulation(model, parameter_values=param)
    sim.solve()
    sim.plot()
```

#### 方式 D：从 BPX 文件导入参数

PyBaMM 支持把 BPX 参数文件直接转换为 `ParameterValues`。这在你已经有标准化电池参数文件时很省事。

```python
import pybamm

bpx_path = "your_battery_parameters.json"
param = pybamm.ParameterValues.create_from_bpx(bpx_path)

# 如需改变初始 SOC，导入后再设置
param.set_initial_state(0.6)
```

如果你已经把 BPX 文件读成 Python 字典对象，也可以用 `pybamm.ParameterValues.create_from_bpx_obj(...)`。

#### 方式 E：把频繁变化的量设为输入参数

如果你要反复改变电流、换热系数或某些动力学参数，不要每次都新建 `Simulation`。把该参数设为 `"[input]"`，然后在 `solve(..., inputs=...)` 时传值，能避免重复建模和离散化。

```python
model = pybamm.lithium_ion.SPM()
param = model.default_parameter_values.copy()
param["Current function [A]"] = "[input]"

sim = pybamm.Simulation(
    model,
    parameter_values=param,
    solver=pybamm.IDAKLUSolver(),
)

for current in [2.5, 5.0, 7.5]:
    sol = sim.solve([0, 3600], inputs={"Current function [A]": current})
```

注意：影响 geometry / mesh 的参数通常不能直接设为输入参数，例如电极厚度、隔膜厚度、颗粒半径等，因为它们会改变离散化。

#### 方式 F：设置初始 SOC / 初始状态

对实际电池而言，初始 SOC 往往和参数值本身同样重要。常用有两种做法：

```python
param = pybamm.ParameterValues("Chen2020")
param.set_initial_state(0.8)  # 80% SOC

sim = pybamm.Simulation(model, parameter_values=param)
solution = sim.solve([0, 3600])

# 或者不改 ParameterValues，直接在 solve 时临时指定
sim = pybamm.Simulation(
    model,
    parameter_values=pybamm.ParameterValues("Chen2020"),
)
solution = sim.solve([0, 3600], initial_soc=0.8)
```

### 3.4 Step 3 — 定义充放电工况

#### 基础语法

```python
experiment = pybamm.Experiment(
    [
        ("Discharge at 1C until 2.5V",),  # C-rate 放电到截止电压
        ("Charge at 5A until 4.2V",),  # 恒流充电到截止电压
        ("Hold at 4.2V until 10mA",),  # 恒压充电到截止电流
        ("Rest for 1 hour",),  # 静置
        ("Discharge at 10W until 2.5V",),  # 恒功率放电
        ("Discharge at C/20 for 1 hour",),  # 按时间放电
    ]
)
```

#### CCCV 充放电循环

```python
experiment = pybamm.Experiment(
    [
        (
            "Charge at 1A until 4.2V",
            "Hold at 4.2V until 50mA",  # CC-CV 充电
            "Rest for 30 minutes",
            "Discharge at 5A until 2.8V",  # 恒流放电
            "Rest for 30 minutes",
        ),
    ]
    * 100
)  # 循环 100 次
```

#### 自定义驱动循环（从 CSV 加载）

```python
import pandas as pd

# 加载你的驱动循环数据 (时间, 电流)
data_loader = pybamm.DataLoader()
your_cycle = pd.read_csv("your_drive_cycle.csv").to_numpy()

experiment = pybamm.Experiment([pybamm.step.current(your_cycle)])
```

#### 混合工况

```python
experiment = pybamm.Experiment(
    [
        (
            "Charge at 1A until 4.0V",
            "Hold at 4.0V until 50mA",
            "Rest for 30 minutes",
            pybamm.step.current(drive_cycle_data),  # 驱动循环
            "Rest for 30 minutes",
        ),
    ]
)
```

#### 功率映射驱动循环

```python
def map_drive_cycle(x, min_op, max_op):
    """将驱动循环映射到指定范围"""
    min_val = x[:, 1].min()
    max_val = x[:, 1].max()
    x[:, 1] = (x[:, 1] - min_val) / (max_val - min_val) * (max_op - min_op) + min_op
    return x


# 映射为功率曲线
drive_cycle_power = map_drive_cycle(drive_cycle_current, 1.5, 3.5)

experiment = pybamm.Experiment(
    [
        (
            pybamm.step.current(drive_cycle_current),
            pybamm.step.power(drive_cycle_power),
        ),
    ]
)
```

#### GITT 脉冲实验

```python
experiment = pybamm.Experiment(
    [("Discharge at C/20 for 1 hour", "Rest for 1 hour")] * 20
)
```

### 3.5 OCV 函数的定义方法

OCV 数据通常通过 GITT 实验获取；在 PyBaMM 参数中应使用 OCP 键名，并将插值包装成接收 `sto` 的函数：

```python
import pybamm
import numpy as np

# 你的 GITT 实验数据
sto_data = np.array([...])
pos_ocp_data = np.array([...])
neg_sto_data = np.array([...])
neg_ocp_data = np.array([...])


def positive_ocp(sto):
    return pybamm.Interpolant(
        sto_data, pos_ocp_data, sto, name="Positive electrode OCP [V]"
    )


def negative_ocp(sto):
    return pybamm.Interpolant(
        neg_sto_data, neg_ocp_data, sto, name="Negative electrode OCP [V]"
    )


param.update(
    {
        "Positive electrode OCP [V]": positive_ocp,
        "Negative electrode OCP [V]": negative_ocp,
    }
)
```

### 3.6 温度相关参数

```python
# 环境温度随时间变化
def ambient_temperature(y, z, t):
    return 300 + t * 100 / 3600  # 线性升温


param.update({"Ambient temperature [K]": ambient_temperature})


# 依赖温度的函数参数
def diffusivity(cc):
    return cc * 10 ** (-5)


param.update({"Diffusivity": diffusivity})
```

### 3.7 完整仿真脚本模板

以下是一个完整的**为实际电池仿真**的模板：

```python
import pybamm
import numpy as np

# ============ 1. 选择模型 ============
model = pybamm.lithium_ion.DFN()
# model = pybamm.lithium_ion.SPM()     # 或根据需求选择
# model = pybamm.lithium_ion.SPMe()

# ============ 2. 配置选项（可选）============
options = {
    "thermal": "lumped",
    "SEI": "none",
    "lithium plating": "none",
}
model = pybamm.lithium_ion.DFN(options=options)

# ============ 3. 设置参数 ============
param = pybamm.ParameterValues("Chen2020")  # 选一个接近的内置参数集

# >>> 在这里用你的实际参数覆盖 <<<
param.update(
    {
        "Nominal cell capacity [A.h]": 5.0,
        "Negative electrode thickness [m]": 50e-6,
        "Positive electrode thickness [m]": 80e-6,
        # ... 添加你的参数
        # OCP 函数（如有实测数据，见 3.5 节）
        # "Positive electrode OCP [V]": positive_ocp,
    }
)

# ============ 4. 定义工况 ============
experiment = pybamm.Experiment(
    [
        (
            "Charge at 1A until 4.2V",
            "Hold at 4.2V until 50mA",  # CC-CV 充电
            "Rest for 30 minutes",
            "Discharge at 5A until 2.8V",  # 恒流放电
            "Rest for 30 minutes",
        ),
    ]
)

# ============ 5. 仿真求解 ============
sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
sim.solve()

# ============ 6. 结果输出 ============
sim.plot()

# 导出数据
sim.solution.save_data(
    "simulation_result.csv",
    [
        "Time [h]",
        "Voltage [V]",
        "Current [A]",
        "Discharge capacity [A.h]",
        "Terminal voltage [V]",
    ],
    to_format="csv",
)

# 访问特定变量
time = sim.solution["Time [h]"].entries
voltage = sim.solution["Voltage [V]"].entries
current = sim.solution["Current [A]"].entries
```

### 3.8 与实验数据对比验证

仿真完成后，导出数据并与你的实测数据对比：

```python
import pandas as pd
import matplotlib.pyplot as plt

# 获取仿真数据
sim_time = sim.solution["Time [h]"].entries
sim_voltage = sim.solution["Voltage [V]"].entries

# 读取实验数据
exp_data = pd.read_csv("your_experiment_data.csv")
exp_time = exp_data["Time [h]"]
exp_voltage = exp_data["Voltage [V]"]

# 绘制对比图
plt.figure()
plt.plot(exp_time, exp_voltage, "b-", label="Experiment")
plt.plot(sim_time, sim_voltage, "r--", label="Simulation")
plt.xlabel("Time [h]")
plt.ylabel("Voltage [V]")
plt.legend()
plt.title("Experiment vs Simulation")
plt.show()
```

如果仿真与实验差异较大，回到 Step 2 调整参数。

### 3.9 EIS 阻抗谱仿真

PyBaMM 不只支持时域充放电，也内置了频域阻抗谱接口 `pybamm.EISSimulation`。如果你需要做 Nyquist / Bode 分析，可直接走下面的最小流程：

```python
import numpy as np
import pybamm

model = pybamm.lithium_ion.SPM(options={"surface form": "differential"})
param = pybamm.ParameterValues("Chen2020")

eis_sim = pybamm.EISSimulation(model, parameter_values=param)
frequencies = np.logspace(-4, 4, 30)

eis_sim.solve(frequencies, initial_soc=0.5)
eis_sim.nyquist_plot()
```

注意：`EISSimulation` 要求模型的 `surface form` 选项为 `"differential"` 或 `"algebraic"`；如果只是普通时域充放电，不需要开启这个选项。

---

## 五、结果提取与可视化

### 5.1 获取仿真变量

```python
sim.solve([0, 3600])
solution = sim.solution

# 获取电压、时间
voltage = solution["Terminal voltage [V]"]
time = solution["Time [s]"]

# 转为 numpy 数组
v = voltage.entries  # 电压值数组
t = time.entries  # 时间值数组
```

### 5.2 常用输出变量

| 变量名 | 获取方式 |
|--------|---------|
| 终端电压 | `solution["Terminal voltage [V]"]` |
| 电流 | `solution["Current [A]"]` |
| 负极颗粒表面浓度 | `solution["Negative particle surface concentration [mol.m-3]"]` |
| 正极颗粒表面浓度 | `solution["Positive particle surface concentration [mol.m-3]"]` |
| 电解液浓度 | `solution["Electrolyte concentration [mol.m-3]"]` |
| 内部温度 | `solution["Cell temperature [K]"]` |
| 反应过电位 | `solution["X-averaged reaction overpotential [V]"]` |
| 颗粒浓度过电位 | `solution["Particle concentration overpotential [V]"]` |

### 5.3 导出数据

```python
# 导出为 CSV
solution.save_data(
    "output.csv",
    [
        "Time [s]",
        "Terminal voltage [V]",
        "Current [A]",
        "Discharge capacity [A.h]",
    ],
    to_format="csv",
)

# 转为 xarray Dataset（高级分析用）
ds = solution.xarray()
```

### 5.4 进阶图：电压分解与热源分解

除了 `sim.plot()` 和 `dynamic_plot()`，PyBaMM 还内置了更面向诊断的可视化函数：

```python
# 电压由哪些分量组成
pybamm.plot_voltage_components(sim)

# 热源分解图，仅适用于 lumped thermal 模型
pybamm.plot_thermal_components(sim)
```

这两类图非常适合定位倍率损失、欧姆损失、反应过电位和热源贡献。

---

## 六、对比仿真

### 6.1 对比不同模型

```python
import pybamm

models = [
    pybamm.lithium_ion.SPM(),
    pybamm.lithium_ion.SPMe(),
    pybamm.lithium_ion.DFN(),
]
sims = [pybamm.Simulation(m) for m in models]

for sim in sims:
    sim.solve([0, 3600])

# 绘制对比图
pybamm.dynamic_plot(sims, labels=["SPM", "SPMe", "DFN"])
```

### 6.2 参数扫描（不同 C-rate）

```python
C_rates = [0.5, 1, 2]
experiments = [
    pybamm.Experiment([("Discharge at {}C for 10 hours or until 2.5 V".format(C))])
    for C in C_rates
]

sims = [pybamm.Simulation(model) for _ in C_rates]
for sim, exp in zip(sims, experiments):
    sim.experiment = exp
    sim.solve()

pybamm.dynamic_plot(sims, labels=[f"{C}C" for C in C_rates])
```

### 6.3 用 BatchStudy 做系统比较

如果你只是比较 2 到 3 组简单结果，手写循环就够了；如果要组织“多个模型 × 多个参数集 × 多个实验工况”的批量对比，项目内置了 `pybamm.BatchStudy`。

```python
models = {
    "SPM": pybamm.lithium_ion.SPM(),
    "SPMe": pybamm.lithium_ion.SPMe(),
}

parameter_values = {
    "Chen2020": pybamm.ParameterValues("Chen2020"),
}

study = pybamm.BatchStudy(
    models=models,
    parameter_values=parameter_values,
    permutations=True,
)

study.solve(t_eval=[0, 3600])
study.plot()
```

`permutations=False` 时，各字典会按顺序一一配对；`permutations=True` 时会做笛卡尔积组合，更适合系统扫参。求解后还可以用 `study.create_gif(...)` 导出动画。

---

## 七、参数速查表

### 7.1 必须提供的关键参数清单

为实际电池参数化时，至少需要以下参数：

| 参数类别 | 代表性参数 | 单位 | 获取方式 |
|---------|-----------|------|---------|
| **几何** | 极片厚度 (thickness) | m | SEM/电镜、图纸 |
| | 极片宽高 (width, height) | m | 图纸、测量 |
| | 颗粒半径 (particle radius) | m | SEM/文献 |
| | 隔膜厚度 (separator thickness) | m | 测量 |
| **电化学** | 活性材料初始浓度 | mol/m³ | 材料物性 |
| | 最大固相浓度 | mol/m³ | 材料物性 |
| **OCV** | 开路电压 vs SOC 曲线 | V | GITT 实验 |
| **动力学** | 交换电流密度 | A/m² | EIS 拟合 |
| | 反应速率常数 | mol/(m²·s) | EIS 拟合 |
| **传输** | 固相扩散系数 | m²/s | GITT/PITT |
| | 液相扩散系数 | m²/s | 文献 |
| | 液相电导率 | S/m | 文献 |
| **热** | 比热容 | J/(kg·K) | DSC |
| | 导热系数 | W/(m·K) | 激光闪射法 |
| | 密度 | kg/m³ | 称重/体积 |
| **容量** | 名义容量 | A·h | 恒流放电测试 |
| | 电解液初始浓度 | mol/m³ | 电解液规格 |
| **孔隙率** | 电极孔隙率 | — | 压汞法/计算 |

### 7.2 内置参数集

| 参数集名 | 适用体系 | 说明 |
|---------|---------|------|
| `"Chen2020"` | 锂离子 (LG M50, graphite/NMC) | Chen et al. 2020 参数集 |
| `"Marquis2019"` | 锂离子 | 默认参数集 |
| `"NCA_Kim2011"` | 锂离子 (NCA) | Kim et al. 2011 NCA 参数集 |
| `"Ecker2015"` | 锂离子 | Ecker et al. 2015 |
| `"MSMR_Example"` | 锂离子 | MSMR 模型示例参数集 |
| `"Sulzer2019"` | 铅酸 | 铅酸电池默认参数集 |

**查看参数集的方法：**
```python
import pybamm

# 列出所有可用参数集
print(pybamm.parameter_sets)

# 打印某一参数集的所有参数
param = pybamm.ParameterValues("Chen2020")
parameters = pybamm.LithiumIonParameters()
param.print_parameters(parameters, "chen2020_params.txt")
```

### 7.3 指定 C-rate 的方法

```python
# 方法 1：在 Simulation 中指定
sim = pybamm.Simulation(model, parameter_values=param, C_rate=1.0)

# 方法 2：修改 Current function
param.update({"Current function [A]": 5.0})  # 1C = 5A for 5Ah 电池

# 方法 3：通过 Experiment 定义
experiment = pybamm.Experiment(["Discharge at 1C until 2.5V"])
```

---

## 八、常见问题与报错处理

### Q1：仿真很慢怎么办？
- 降低模型复杂度：DFN → SPMe → SPM
- 简化选项：`"thermal": "isothermal"`，去掉 SEI/锂沉积
- 使用 JAX 求解器加速：`solver=pybamm.JaxSolver()`

### Q2：如何选择模型复杂度？
- **概念验证/快速迭代** → SPM（秒级）
- **一般工程计算** → SPMe（分钟级）
- **高精度研究/验证** → DFN（小时级）
- **BMS 算法** → Thevenin（毫秒级）

### Q3：参数不确定怎么办？
- 使用 `print_model_parameter_combinations.py` 了解哪些参数是必须的
- 用参数扫描对比不同参数值的影响（参考 4.3 方式 C）

### Q4：如何仿真循环老化？
```python
model = pybamm.lithium_ion.DFN(
    options={"SEI": "reaction limited", "SEI porosity change": "true"}
)
experiment = pybamm.Experiment(
    [("Charge at 1C until 4.2V", "Discharge at 1C until 2.5V")] * 500
)
sim = pybamm.Simulation(model, experiment=experiment)
sol = sim.solve()
```

### Q5：如何仿真半电池？
```python
model = pybamm.lithium_ion.DFN(options={"working electrode": "positive"})  # 正极半电池
# model = pybamm.lithium_ion.DFN(options={"working electrode": "negative"})  # 负极半电池
```

### Q6：报错 "SolverError" 怎么办？
- 尝试降低 C-rate 或缩短仿真时间
- 切换到 `CasadiSolver(mode="safe")` 或 `IDAKLUSolver`
- 检查参数是否合理（如极片厚度、浓度等数量级）

### Q7：没有绘图窗口怎么办？
- 确保安装了 matplotlib：`pip install matplotlib`
- 在 Jupyter Notebook 中已自动内嵌显示
-非交互环境需要设置 `import matplotlib; matplotlib.use('Agg')`

### Q8：想查看可用的模型选项？
```python
model = pybamm.lithium_ion.DFN()
print(model.default_options)
# 或查看所有可能选项
print(pybamm.BatteryModelOptions({}).possible_options)
```

### Q9：想查看某个参数集的所有参数？
```python
import pybamm

# 列出所有可用参数集
print(pybamm.parameter_sets)

# 打印某一参数集的所有参数
param = pybamm.ParameterValues("Chen2020")
print(list(param.keys())[:20])  # 打印前 20 个参数名

# 导出完整参数到文件
parameters = pybamm.LithiumIonParameters()
param.print_parameters(parameters, "chen2020_params.txt")
```
