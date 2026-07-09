import pybamm

model = pybamm.lithium_ion.DFN()

# 方法1：使用 Experiment 自定义工步
experiment = pybamm.Experiment([
    ("Discharge at 0.5C for 2 hours",),   # 0.5C放电2小时
    ("Rest for 30 minutes",),             # 静置30分钟
    ("Charge at 1C until 4.2V",),         # 1C充电至4.2V
])
sim = pybamm.Simulation(model, experiment=experiment)
sim.solve()
sim.plot()
