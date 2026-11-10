import numpy as np

import pybamm
import tests


class TestThevenin:
    def test_basic_processing(self):
        model = pybamm.equivalent_circuit.Thevenin()
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_diffusion(self):
        model = pybamm.equivalent_circuit.Thevenin(
            options={"diffusion element": "true"}
        )
        parameter_values = model.default_parameter_values

        parameter_values.update({"Diffusion time constant [s]": 580})
        modeltest = tests.StandardModelTest(model, parameter_values=parameter_values)
        modeltest.test_all()

    def test_diffusion_overpotential_is_a_loss(self):
        model = pybamm.equivalent_circuit.Thevenin(
            options={"diffusion element": "true"}
        )
        parameter_values = model.default_parameter_values
        parameter_values.update({"Diffusion time constant [s]": 580})
        experiment = pybamm.Experiment(
            ["Discharge at 20 A for 30 minutes", "Charge at 20 A for 30 minutes"]
        )
        sim = pybamm.Simulation(
            model, parameter_values=parameter_values, experiment=experiment
        )
        solution = sim.solve()

        # Discharging from rest depletes the surface, which must lower the voltage
        discharge = solution.cycles[0]
        eta = discharge["Diffusion overpotential [V]"](discharge.t)
        assert np.all(eta <= 0)
        assert eta[-1] < 0

        for cycle in solution.cycles:
            t = cycle.t
            current = cycle["Current [A]"](t)
            ocv = cycle["Open-circuit voltage [V]"](t)
            voltage = cycle["Voltage [V]"](t)
            np.testing.assert_allclose(
                cycle["Irreversible heat generation [W]"](t),
                current * (ocv - voltage),
                rtol=1e-10,
                atol=1e-12,
            )
