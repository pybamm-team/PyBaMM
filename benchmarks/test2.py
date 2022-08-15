import pybamm
import time

def benchmark_experiments(model_,parameter_set,):
    param = pybamm.ParameterValues(parameter_set)
    exp_time = []
    experiments = [[
                "Discharge at C/5 for 10 hours or until 3.3 V",
                "Rest for 1 hour",
                "Charge at 1 A until 4.1 V",
                "Hold at 4.1 V until 10 mA",
                "Rest for 1 hour",
            ],[("Discharge at C/20 for 1 hour", "Rest for 1 hour")] * 20]
    model = model_
    experiments = []
    for exp_ in experiments:
        print(exp_)
        exp = pybamm.Experiment(
                exp_
            )
        t = time.time()
        pybamm.Simulation(model, parameter_values=param, experiment=exp)
        t1 = time.time()
        exp_time.append(t1 - t)
        print(exp_time)
benchmark_experiments(pybamm.lithium_ion.SPM(),"Marquis2019")
