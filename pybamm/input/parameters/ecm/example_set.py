import pybamm
import os

# An example set of parameters for the equivalent circuit model

path, _ = os.path.split(os.path.abspath(__file__))
ocv = pybamm.parameters.process_1D_data("ecm_example_ocv.csv", path=path)

r0 = pybamm.parameters.process_3D_data_csv("ecm_example_r0.csv", path=path)


def get_rc_parameters(rc_idx):

    r = pybamm.parameters.process_3D_data_csv(f"ecm_example_r{rc_idx}.csv", path=path)
    c = pybamm.parameters.process_3D_data_csv(f"ecm_example_c{rc_idx}.csv", path=path)

    values = {
        f"Element-{rc_idx} initial overpotential": 0,
        f"R{rc_idx} [Ohm]": r,
        f"C{rc_idx} [Ohm]": c,
    }

    return values


dUdT = pybamm.parameters.process_2D_data_csv("ecm_example_dUdT.csv", path=path)


def get_parameters_values(number_of_rc_elements=1):
    """
    Example parameter set for a equivalent circuit model with a
    resistor in series with a single RC element.

    This parameter set is for demonstration purposes only and
    does not reflect the properties of any particular real cell.
    Example functional dependancies have been added for each of
    the parameters to demonstrate the functionality of
    3D look-up tables models.

    The parameter values have been generated in the following
    manner:

        1. Capacity assumed to be 100Ah
        2. 30s DCIR at T25 S50 assumed to be 1mOhm
        3. DCIR assume to be have the following dependencies:
            - quadratic in SoC
            - Arrhenius in temperature
            - linear in current
        4. R0 taken to be 50% of the DCIR
        5. R1 taken to be 50% of the DCIR
        6. C1 is derived from the C1 = tau / R1 where tau=30s
        7. OCV is taken to be a simple linear function of SoC
            starting at 3.0V and ending at 4.2V
        8. dUdT is taken to be 0V/K
    """

    cell_capacity = 100

    values = {
        "Initial SoC": 0.5,
        "Element-1 initial overpotential [V]": 0,
        "Initial cell temperature [degC]": 25,
        "Initial jig temperature [degC]": 25,
        "Cell capacity [A.h]": cell_capacity,
        "Nominal cell capacity [A.h]": cell_capacity,
        "Ambient temperature [degC]": 25,
        "Current function [A]": -100,
        "Upper voltage cut-off [V]": 4.2,
        "Lower voltage cut-off [V]": 3.2,
        "Cell thermal mass [J/K]": 1000,
        "Cell-jig heat transfer coefficient [W/K]": 10,
        "Jig thermal mass [J/K]": 500,
        "Jig-air heat transfer coefficient [W/K]": 10,
        "RCR lookup limit [A]": 340,
    }

    # Add initial overpotentials for RC elements
    for i in range(1, number_of_rc_elements + 1):
        values.update(get_rc_parameters(i))
