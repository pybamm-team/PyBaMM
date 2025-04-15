import pybamm
import numpy as np


def test_heat_transfer_coefficient_scaling():
    param = pybamm.ParameterValues("Chen2020")
    
    # Add required parameters explicitly (with correct names)
    param.update(
        {
            "Negative current collector surface heat transfer coefficient [W.m-2.K-1]": 10,
            "Positive current collector surface heat transfer coefficient [W.m-2.K-1]": 10,
            "Number of electrodes connected in parallel to make a cell": 1,
        },
        check_already_exists=False  # Allow adding new parameters
    )
    
    # Create model to trigger parameter processing
    model = pybamm.lithium_ion.SPM({"thermal": "x-lumped"})
    sim = pybamm.Simulation(model, parameter_values=param)
    sim.build()
    
    # Get processed values
    h_cn_N1 = sim.parameter_values[
        "Negative current collector surface heat transfer coefficient [W.m-2.K-1]"
    ]
    
    # Update N and reprocess
    param.update({"Number of electrodes connected in parallel to make a cell": 2})
    sim.build()  # Rebuild to process new parameters
    h_cn_N2 = sim.parameter_values[
        "Negative current collector surface heat transfer coefficient [W.m-2.K-1]"
    ]
    
    # Verify scaling
    np.testing.assert_allclose(h_cn_N2, h_cn_N1 / 2, rtol=1e-5)

