# PyBaMM 项目使用说明

## 一、项目简介

**PyBaMM** (Python Battery Mathematical Modelling) 是一个用 Python 编写的开源电池仿真包，由 PyBaMM Team 开发并受 NumFOCUS 赞助。其使命是通过提供开源工具来加速电池建模研究，支持多机构、跨学科协作。

PyBaMM 核心包含三大部分：

1. **微分方程求解框架** — 用于编写和求解微分方程系统
2. **电池模型与参数库** — 提供丰富的电池模型和参数集
3. **实验仿真与可视化工具** — 模拟电池特定实验并可视化结果

- **项目主页**: https://pybamm.org
- **文档**: https://docs.pybamm.org
- **源码仓库**: https://github.com/pybamm-team/PyBaMM
- **许可证**: BSD 开源协议
- **Python 版本要求**: >= 3.10, < 3.15
- **版本号格式**: CalVer (YY.MM.N.P)，如 26.5.0 表示 2026 年 5 月首个功能版本

---

## 二、项目文件功能结构

```
pybamm/
├── src/pybamm/                    # ====== 核心源代码 ======
│   ├── __init__.py                # 包入口，导出公共 API
│   ├── batch_study.py             # 批量仿真研究
│   ├── callbacks.py               # 求解回调机制
│   ├── citations.py               # 文献引用管理
│   ├── config.py                  # 配置管理
│   ├── logger.py                  # 日志模块
│   ├── plotting/                  # ====== 可视化模块 ======
│   │   ├── plot.py                # 基础绘图
│   │   ├── quick_plot.py          # 快速绘图
│   │   ├── dynamic_plot.py        # 动态绘图
│   │   ├── nyquist_plot.py        # 奈奎斯特图（EIS）
│   │   ├── plot2D.py              # 二维绘图
│   │   ├── plot_3d_cross_section.py # 三维截面图
│   │   ├── plot_3d_heatmap.py     # 三维热力图
│   │   ├── plot_voltage_components.py  # 电压分量图
│   │   ├── plot_thermal_components.py  # 热分量图
│   │   └── plot_summary_variables.py  # 汇总变量图
│   ├── models/                    # ====== 电池模型 ======
│   │   ├── base_model.py          # 模型基类
│   │   ├── event.py               # 事件（终止条件）
│   │   ├── symbol_processor.py    # 符号处理器
│   │   ├── full_battery_models/   # 完整电池模型
│   │   │   ├── base_battery_model.py  # 电池模型基类
│   │   │   ├── lithium_ion/       # 锂离子电池模型
│   │   │   │   ├── spm.py         # 单粒子模型 (SPM)
│   │   │   │   ├── spme.py        # 单粒子模型+电解液 (SPMe)
│   │   │   │   ├── dfn.py         # Doyle-Fuller-Newman 模型 (DFN)
│   │   │   │   ├── mpm.py         # 多粒子模型 (MPM)
│   │   │   │   ├── msmr.py        # 多状态多反应模型 (MSMR)
│   │   │   │   ├── Yang2017.py    # Yang2017 模型
│   │   │   │   └── ...            # 半电池、复合电极等变体
│   │   │   ├── lead_acid/         # 铅酸电池模型
│   │   │   ├── lithium_metal/     # 金属锂电池模型
│   │   │   ├── sodium_ion/        # 钠离子电池模型
│   │   │   └── equivalent_circuit/ # 等效电路模型 (ECM)
│   │   └── submodels/             # 物理子模型
│   │       ├── electrode/         # 电极动力学
│   │       ├── electrolyte_diffusion/  # 电解液扩散
│   │       ├── electrolyte_conductivity/ # 电解液电导率
│   │       ├── particle/          # 颗粒扩散
│   │       ├── particle_mechanics/ # 颗粒力学（膨胀/收缩）
│   │       ├── thermal/           # 热模型
│   │       ├── porosity/          # 孔隙率
│   │       ├── convection/        # 对流
│   │       ├── interface/         # 界面反应（SEI、锂沉积等）
│   │       ├── external_circuit/   # 外部电路
│   │       └── ...                # 更多子模型
│   ├── simulation/                # ====== 仿真模块 ======
│   │   ├── base_simulation.py     # 仿真基类
│   │   ├── simulation.py          # 主仿真类
│   │   ├── eis_simulation.py      # 电化学阻抗谱仿真
│   │   └── eis_utils.py           # EIS 工具
│   ├── solvers/                   # ====== 求解器 ======
│   │   ├── base_solver.py         # 求解器基类
│   │   ├── casadi_solver.py       # CasADi 求解器
│   │   ├── scipy_solver.py        # SciPy 求解器
│   │   ├── idaklu_solver.py       # IDAKLU 求解器（C++后端）
│   │   ├── jax_solver.py          # JAX 求解器
│   │   ├── jax_bdf_solver.py      # JAX BDF 求解器
│   │   ├── algebraic_solver.py     # 代数求解器
│   │   ├── casadi_algebraic_solver.py # CasADi 代数求解器
│   │   ├── nonlinear_solver.py    # 非线性求解器
│   │   ├── dummy_solver.py        # 虚拟求解器（测试用）
│   │   ├── solution.py            # 解对象
│   │   └── processed_variable.py  # 处理后的变量
│   ├── experiment/                # ====== 实验模块 ======
│   │   ├── experiment.py          # 实验定义（充放电协议）
│   │   └── step/                  # 实验步骤定义
│   ├── expression_tree/           # ====== 表达式树 ======
│   │   ├── symbol.py              # 符号基类
│   │   ├── scalar.py / vector.py  # 标量/向量
│   │   ├── matrix.py / array.py   # 矩阵/数组
│   │   ├── parameter.py           # 参数
│   │   ├── variable.py            # 状态变量
│   │   ├── binary_operators.py    # 二元运算符
│   │   ├── unary_operators.py     # 一元运算符
│   │   ├── functions.py          # 数学函数
│   │   ├── interpolant.py        # 插值
│   │   └── ...                    # 广播、拼接、条件等
│   ├── discretisations/           # ====== 离散化 ======
│   │   └── discretisation.py      # 空间离散化引擎
│   ├── geometry/                  # ====== 几何定义 ======
│   │   ├── geometry.py            # 几何基类
│   │   ├── battery_geometry.py    # 电池几何（层结构）
│   │   └── standard_spatial_vars.py # 标准空间变量
│   ├── meshes/                    # ====== 网格划分 ======
│   │   ├── meshes.py              # 网格类
│   │   ├── one_dimensional_submeshes.py  # 1D 子网格
│   │   ├── two_dimensional_submeshes.py  # 2D 子网格
│   │   ├── zero_dimensional_submesh.py   # 0D 子网格
│   │   ├── scikit_fem_submeshes.py       # FEM 子网格
│   │   └── scikit_fem_submeshes_3d.py    # 3D FEM 子网格
│   ├── spatial_methods/           # ====== 空间离散方法 ======
│   │   ├── spatial_method.py      # 空间方法基类
│   │   ├── finite_volume.py       # 有限体积法 (FVM)
│   │   ├── finite_volume_2d.py    # 二维有限体积法
│   │   ├── spectral_volume.py     # 谱体积法
│   │   ├── scikit_finite_element.py      # 有限元法 (FEM)
│   │   ├── scikit_finite_element_3d.py   # 3D 有限元法
│   │   └── zero_dimensional_method.py    # 零维方法
│   ├── parameters/                # ====== 参数系统 ======
│   │   ├── parameter_values.py   # 参数值管理
│   │   ├── parameter_store.py     # 参数存储
│   │   ├── lithium_ion_parameters.py  # 锂离子参数
│   │   ├── lead_acid_parameters.py    # 铅酸参数
│   │   ├── thermal_parameters.py      # 热参数
│   │   ├── geometric_parameters.py    # 几何参数
│   │   ├── electrical_parameters.py   # 电学参数
│   │   └── ...                    # ECM参数、常数等
│   ├── input/                     # ====== 输入数据 ======
│   │   └── parameters/            # 参数集定义
│   │       ├── lithium_ion/       # 锂离子参数集
│   │       │   ├── Chen2020.py    # Chen2020 参数集
│   │       │   ├── Marquis2019.py # Marquis2019 参数集
│   │       │   ├── Xu2019.py     # Xu2019 参数集
│   │       │   ├── ORegan2022.py # ORegan2022 参数集
│   │       │   └── ...            # 更多参数集
│   │       ├── lead_acid/         # 铅酸参数集
│   │       │   └── Sulzer2019.py  # Sulzer2019 参数集
│   │       ├── sodium_ion/        # 钠离子参数集
│   │       └── ecm/               # ECM 参数集
│   ├── codegen/                   # 代码生成
│   ├── dispatch/                  # 插件/入口点分发
│   ├── telemetry.py               # 遥测统计
│   ├── settings.py                # 全局设置
│   ├── pybamm_data.py             # 数据下载管理
│   ├── util.py                    # 通用工具函数
│   └── type_definitions.py       # 类型定义
│
├── examples/                      # ====== 示例代码 ======
│   ├── scripts/                   # Python 脚本示例
│   │   ├── run_simulation.py      # 最简仿真示例
│   │   ├── DFN.py                 # DFN 模型示例
│   │   ├── SPMe.py                # SPMe 模型示例
│   │   ├── run_eis_simulation.py   # EIS 仿真示例
│   │   ├── run_ecm.py             # 等效电路模型示例
│   │   ├── experiment_drive_cycle.py # 驱动循环实验
│   │   ├── compare_lithium_ion.py # 锂离子模型对比
│   │   ├── compare_dae_solver.py # 求解器对比
│   │   ├── thermal_lithium_ion.py # 热模型示例
│   │   ├── cycling_ageing.py      # 循环老化仿真
│   │   ├── calendar_ageing.py     # 日历老化仿真
│   │   ├── rate_capability.py     # 倍率性能测试
│   │   ├── custom_model.py        # 自定义模型
│   │   ├── print_parameters.py    # 打印参数
│   │   └── ...                    # 更多示例（约50个）
│   └── README.md                  # 示例说明
│
├── docs/                          # ====== 文档 ======
│   ├── source/
│   │   ├── user_guide/            # 用户指南
│   │   ├── api/                   # API 文档
│   │   └── examples/notebooks/   # Jupyter 示例笔记本
│   ├── conf.py                    # Sphinx 配置
│   └── Makefile                   # 文档构建
│
├── tests/                         # ====== 测试 ======
│   ├── unit/                      # 单元测试
│   ├── integration/               # 集成测试
│   ├── memory/                    # 内存泄漏测试
│   └── conftest.py                # 测试配置
│
├── benchmarks/                    # ====== 性能基准测试 ======
│   ├── time_solve_models.py       # 模型求解计时
│   ├── time_sims_experiments.py   # 实验仿真计时
│   ├── unit_benchmarks.py         # 单元基准测试
│   └── memory_sims.py             # 内存基准测试
│
├── scripts/                       # ====== 辅助脚本 ======
│   ├── update_version.py          # 版本更新
│   ├── benchmark_unified_experiment.py # 统一实验基准
│   └── Dockerfile                 # Docker 构建文件
│
├── .github/                       # GitHub CI/CD 配置
├── pyproject.toml                 # 项目构建配置与依赖
├── noxfile.py                     # Nox 测试会话配置
├── asv.conf.json                  # asv 性能测试配置
├── uv.lock                        # uv 包管理器锁文件
├── README.md                      # 项目说明
├── CHANGELOG.md                   # 更新日志
├── CONTRIBUTING.md                # 贡献指南
├── LICENSE.txt                    # BSD 许可证
├── CITATION.cff                   # 引用信息
└── CODE-OF-CONDUCT.md             # 行为准则
```

---

## 三、安装指南

### 3.1 环境要求

- Python >= 3.10, < 3.15
- 建议在虚拟环境中安装

### 3.2 从 pip 安装（推荐用户使用）

```bash
pip install pybamm
```

### 3.3 从 conda 安装

```bash
# 基础安装（无额外依赖）
conda install -c conda-forge pybamm-base

# 完整安装（含绘图、示例等，不含 JAX）
conda install -c conda-forge pybamm
```

### 3.4 从源码安装（开发模式）

```bash
# 克隆项目（本项目已完成此步骤）
git clone https://github.com/pybamm-team/PyBaMM.git
cd PyBaMM

# 使用 pip 以可编辑模式安装
pip install -e .

# 安装所有可选依赖
pip install -e ".[all]"

# 安装开发依赖（测试、lint 等）
pip install -e ".[all,dev]"
```

### 3.5 可选依赖

| 依赖组 | 安装命令 | 说明 |
|--------|---------|------|
| 绘图 | `pip install pybamm[plot]` | matplotlib 绑定 |
| 示例 | `pip install pybamm[examples]` | Jupyter 支持 |
| BPX格式 | `pip install pybamm[bpx]` | Battery Parameter eXchange 格式 |
| 进度条 | `pip install pybamm[tqdm]` | 低开销进度条 |
| JAX求解器 | `pip install pybamm[jax]` | JAX 加速求解器 |
| 引用 | `pip install pybamm[cite]` | BibTeX 引用生成 |
| 全部 | `pip install pybamm[all]` | 除 JAX 和 dev 外全部依赖 |

### 3.6 核心依赖

| 包名 | 用途 |
|------|------|
| pybammsolvers | PyBaMM 专用求解器包 |
| numpy | 数值计算 |
| scipy | 科学计算 |
| xarray | 多维数组 |
| sympy | 符号计算 |
| pandas | 数据处理 |
| anytree | 树形结构 |
| pooch | 数据下载 |
| pyyaml | YAML 配置 |
| black | 代码格式化 |

---

## 四、用户操作指南

### 4.1 快速开始 — 最简仿真

```python
import pybamm

# 1. 选择模型
model = pybamm.lithium_ion.DFN()  # Doyle-Fuller-Newman 模型

# 2. 创建仿真
sim = pybamm.Simulation(model)

# 3. 求解（1C 恒流放电，模拟 3600 秒）
sim.solve([0, 3600])

# 4. 绘图
sim.plot()
```

### 4.2 实验仿真 — CCCV 充放电

```python
import pybamm

# 定义实验协议
experiment = pybamm.Experiment(
    [
        (
            "Discharge at C/10 for 10 hours or until 3.3 V",
            "Rest for 1 hour",
            "Charge at 1 A until 4.1 V",
            "Hold at 4.1 V until 50 mA",
            "Rest for 1 hour",
        )
    ]
    * 3,  # 重复 3 个循环
)

model = pybamm.lithium_ion.DFN()
sim = pybamm.Simulation(model, experiment=experiment)
sim.solve()
sim.plot()
```

### 4.3 可用电池模型

| 模型 | 代码 | 说明 |
|------|------|------|
| SPM | `pybamm.lithium_ion.SPM()` | 单粒子模型，速度最快 |
| SPMe | `pybamm.lithium_ion.SPMe()` | 单粒子+电解液模型，平衡精度与速度 |
| DFN | `pybamm.lithium_ion.DFN()` | Doyle-Fuller-Newman 模型，高精度 |
| MPM | `pybamm.lithium_ion.MPM()` | 多粒子模型 |
| MSMR | `pybamm.lithium_ion.MSMR()` | 多状态多反应模型 |
| NewmanTobias | `pybamm.lithium_ion.NewmanTobias()` | Newman-Tobias 模型 |
| Yang2017 | `pybamm.lithium_ion.Yang2017()` | Yang2017 模型 |

### 4.4 可用参数集

| 参数集 | 代码 | 电池类型 |
|--------|------|---------|
| Chen2020 | `pybamm.input.parameters.lithium_ion.Chen2020` | LG M50 石墨/NMC |
| Marquis2019 | `pybamm.input.parameters.lithium_ion.Marquis2019` | Kokam 石墨/LiCoO2 |
| Xu2019 | `pybamm.input.parameters.lithium_ion.Xu2019` | 锂金属/NMC 半电池 |
| ORegan2022 | `pybamm.input.parameters.lithium_ion.ORegan2022` | LG M50 石墨/NMC811 |
| NCA_Kim2011 | `pybamm.input.parameters.lithium_ion.NCA_Kim2011` | 石墨/NCA |
| Ai2020 | `pybamm.input.parameters.lithium_ion.Ai2020` | Enertech 石墨/LiCoO2 |
| Ramadass2004 | `pybamm.input.parameters.lithium_ion.Ramadass2004` | 石墨/LiCoO2 |
| Sulzer2019 | `pybamm.input.parameters.lead_acid.Sulzer2019` | 铅酸 |

使用参数集：

```python
import pybamm

model = pybamm.lithium_ion.DFN()
param = pybamm.ParameterValues("Chen2020")
sim = pybamm.Simulation(model, parameter_values=param)
sim.solve([0, 3600])
sim.plot()
```

### 4.5 切换求解器

```python
import pybamm

model = pybamm.lithium_ion.DFN()

# CasADi 求解器（默认）
sim = pybamm.Simulation(model, solver=pybamm.CasadiSolver())

# IDAKLU 求解器（当前项目依赖中已包含 pybammsolvers）
sim = pybamm.Simulation(model, solver=pybamm.IDAKLUSolver())

# SciPy 求解器
sim = pybamm.Simulation(model, solver=pybamm.ScipySolver())

# JAX 求解器（需安装 pybamm[jax]）
sim = pybamm.Simulation(model, solver=pybamm.JaxSolver())
```

### 4.6 自定义仿真参数

```python
import pybamm

model = pybamm.lithium_ion.SPM()

# 使用现有参数集并修改部分参数
param = pybamm.ParameterValues("Marquis2019")
param.update(
    {
        "Negative electrode thickness [m]": 80e-6,
        "Positive electrode thickness [m]": 100e-6,
    }
)

sim = pybamm.Simulation(model, parameter_values=param)
sim.solve([0, 3600])
sim.plot()
```

### 4.7 更改模型选项

```python
import pybamm

model = pybamm.lithium_ion.DFN(
    options={
        "thermal": "lumped",          # 热模型选项
        "dimensionality": 0,         # 维度 (0/1/2)
        "surface form": "differential", # 表面形式
        "SEI": "solvent-diffusion limited", # SEI 生长模型
        "lithium plating": "irreversible", # 锂沉积模型
    }
)

sim = pybamm.Simulation(model)
sim.solve([0, 3600])
sim.plot()
```

### 4.8 提取与导出仿真结果

```python
import pybamm

model = pybamm.lithium_ion.SPM()
sim = pybamm.Simulation(model)
sim.solve([0, 3600])

# 获取解对象
solution = sim.solution

# 获取特定变量
voltage = solution["Terminal voltage [V]"]
time = solution["Time [s]"]

# 转换为 numpy 数组
voltage_data = voltage(time)

# 导出为 CSV 或 xarray Dataset
solution.save_data("result.csv", ["Time [s]", "Terminal voltage [V]"])

# 也可以用 xarray
dataset = solution.xarray()
```

### 4.9 批量仿真与参数扫描

```python
import pybamm
import numpy as np

model = pybamm.lithium_ion.SPM()

# 定义不同倍率
C_rates = [0.5, 1, 2, 3]

batch_study = pybamm.BatchStudy(
    {f"{C}C": pybamm.Simulation(model) for C in C_rates},
)

# 批量求解
batch_study.solve([0, 3700 / C for C in C_rates])

# 批量绘图
batch_study.plot()
```

### 4.10 电化学阻抗谱 (EIS) 仿真

```python
import pybamm

model = pybamm.lithium_ion.DFN()
sim = pybamm.Simulation(model)

# 在 SOC=50% 处执行 EIS 仿真
sim.solve([0, 3600])
eis_sim = pybamm.EISSimulation(model, frequency=[1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4])
eis_sim.solve()
eis_sim.plot()
```

### 4.11 引用文献

在脚本末尾添加以下代码获取需要引用的文献：

```python
pybamm.print_citations()          # 打印到终端
pybamm.print_citations("refs.bib") # 保存到 BibTeX 文件
```

---

## 五、运行测试

### 5.1 安装开发依赖

```bash
pip install -e ".[all,dev]"
```

### 5.2 运行测试

```bash
# 运行全部测试（自动并行）
pytest tests/

# 运行特定模块测试
pytest tests/unit/test_expression_tree/
pytest tests/integration/test_solvers/

# 运行单个文件测试
pytest tests/unit/test_expression_tree/test_symbol.py

# 带覆盖率报告
pytest tests/ --cov=src/pybamm
```

### 5.3 运行示例脚本

```bash
python examples/scripts/run_simulation.py
python examples/scripts/DFN.py
python examples/scripts/compare_lithium_ion.py
```

---

## 六、代码质量与格式化

```bash
# 使用 Ruff 进行 lint 和格式化（已配置在 pyproject.toml 中）
pip install ruff
ruff check src/          # lint 检查
ruff format src/         # 自动格式化

# 使用 Black 格式化（核心依赖已包含）
black src/

# 使用 pre-commit 钩子
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

---

## 七、构建文档

```bash
# 安装文档依赖
pip install -e ".[docs]"

# 构建 HTML 文档
cd docs && make html

# 自动构建（文件修改后自动刷新）
sphinx-autobuild docs/source docs/build/html
```

---

## 八、项目架构概览

PyBaMM 的工作流程如下：

```
用户代码
  │
  ├── 选择模型 (models/)
  ├── 设置参数 (input/parameters/ + parameters/)
  ├── 定义实验 (experiment/)
  │
  ▼
Simulation (simulation/)
  │
  ├── 模型构建 → 表达式树 (expression_tree/)
  ├── 几何定义 (geometry/)
  ├── 网格划分 (meshes/)
  ├── 空间离散化 (discretisations/ + spatial_methods/)
  ├── 参数替换 (parameters/)
  │
  ▼
Solver (solvers/)
  │
  ├── CasADi / IDAKLU / SciPy / JAX
  │
  ▼
Solution → 绘图 (plotting/) / 数据导出
```

---

## 九、常用资源链接

| 资源 | 链接 |
|------|------|
| 在线文档 | https://docs.pybamm.org |
| API 文档 | https://docs.pybamm.org/en/latest/source/api/index.html |
| GitHub 仓库 | https://github.com/pybamm-team/PyBaMM |
| 发布页面 | https://github.com/pybamm-team/PyBaMM/releases |
| Google Colab | https://colab.research.google.com/github/pybamm-team/PyBaMM |
| 讨论论坛 | https://pybamm.discourse.group |
| 联系方式 | https://www.pybamm.org/community |
| 辅助材料 | https://github.com/pybamm-team/pybamm-supporting-material |
