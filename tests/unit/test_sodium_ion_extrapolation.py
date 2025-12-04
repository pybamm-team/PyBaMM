import pybamm


def test_sodium_ion_interpolants_allow_extrapolation():
    param = pybamm.ParameterValues("Chayambuka2022")
    model = pybamm.sodium_ion.BasicDFN()

    # short nsolve to ensure extrapolation works
    sim = pybamm.Simulation(model, parameter_values=param)

    # This previously crashed with interpolation bounds exceeded
    sol = sim.solve([0, 5])

    assert sol is not None
