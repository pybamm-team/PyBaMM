"""Demonstrate adding new variables to a model using CoupledVariable."""

import pybamm

# Example usage
if __name__ == "__main__":
    # Create a simple battery model
    model = pybamm.lithium_ion.SPM()

    # Add variable referencing existing one via CoupledVariable (resolved lazily)
    model.variables["Double voltage [V]"] = 2 * pybamm.CoupledVariable("Voltage [V]")

    print("Added 'Double voltage [V]' variable to model")

    # Solve the model
    sim = pybamm.Simulation(model)
    solution = sim.solve([0, 3600])

    # Access the variables - CoupledVariable is resolved lazily here
    voltage = solution["Voltage [V]"].data
    double_voltage = solution["Double voltage [V]"].data

    print(f"\nFinal voltage: {voltage[-1]:.4f} V")
    print(f"Final double voltage: {double_voltage[-1]:.4f} V")
    print(f"Expected (2x voltage): {2 * voltage[-1]:.4f} V")

    # Verify the relationship
    import numpy as np

    np.testing.assert_allclose(double_voltage, 2 * voltage, rtol=1e-10)
    print("\nVerification passed: Double voltage = 2 * Voltage")
