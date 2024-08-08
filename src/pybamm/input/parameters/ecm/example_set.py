import pybamm
import os

# An example set of parameters for the equivalent circuit model

path, _ = os.path.split(os.path.abspath(__file__))

ocv_data = pybamm.parameters.process_1D_data("ecm_example_ocv.csv", path=path)

r0_data = pybamm.parameters.process_3D_data_csv("ecm_example_r0.csv", path=path)
r1_data = pybamm.parameters.process_3D_data_csv("ecm_example_r1.csv", path=path)
c1_data = pybamm.parameters.process_3D_data_csv("ecm_example_c1.csv", path=path)

dUdT_data = pybamm.parameters.process_2D_data_csv("ecm_example_dudt.csv", path=path)


def ocv(sto):
    name, (x, y) = ocv_data
    return pybamm.Interpolant(x, y, sto, name)


def r0(T_cell, current, soc):
    name, (x, y) = r0_data
    return pybamm.Interpolant(x, y, [T_cell, current, soc], name)


def r1(T_cell, current, soc):
    name, (x, y) = r1_data
    return pybamm.Interpolant(x, y, [T_cell, current, soc], name)


def c1(T_cell, current, soc):
    name, (x, y) = c1_data
    return pybamm.Interpolant(x, y, [T_cell, current, soc], name)


def dUdT(ocv, T_cell):
    name, (x, y) = dUdT_data
    return pybamm.Interpolant(x, y, [ocv, T_cell], name)


def get_parameter_values():
    """
    Example parameter set for a equivalent circuit model with a
    resistor in series with a single RC element.

    This parameter set is for demonstration purposes only and
    does not reflect the properties of any particular cell.
    Example functional dependancies have been added for each of
    the parameters to demonstrate the functionality of
    3D look-up tables models. Where parameters have been taken
    from the literature, we do not disclose the source
    in order to avoid confusion that these values represent
    any particular cell.

    The parameter values have been generated in the following
    manner:

        1. Capacity assumed to be 100Ah
        2. 100A DCIR at T25 S50 assumed to be 1mOhm
        3. DCIR assume to be have the following dependencies:
            - quadratic in SoC (increasing to 1.2mOhm
              at 0% and 100% SoC)
            - Arrhenius in temperature (with Ea=20000)
            - linear with the magnitude of the current (with
                slope 0.01 mohms per 100 amps)
        4. R0 taken to be 40% of the DCIR
        5. R1 taken to be 60% of the DCIR
        6. C1 is derived from the C1 = tau / R1 where tau=30s
        7. OCV is taken from an undisclosed literature source.
        8. dUdT is taken from an undisclosed literature source.

    """

    # N.B. actual cell capacity and nominal cell capcity
    # can be different hence the two parameters
    cell_capacity = 100

    values = {
        "chemistry": "ecm",
        "Initial SoC": 0.5,
        "Initial temperature [K]": 25 + 273.15,
        "Cell capacity [A.h]": cell_capacity,
        "Nominal cell capacity [A.h]": cell_capacity,
        "Ambient temperature [K]": 25 + 273.15,
        "Current function [A]": 100,
        "Upper voltage cut-off [V]": 4.2,
        "Lower voltage cut-off [V]": 3.2,
        "Cell thermal mass [J/K]": 1000,
        "Cell-jig heat transfer coefficient [W/K]": 10,
        "Jig thermal mass [J/K]": 500,
        "Jig-air heat transfer coefficient [W/K]": 10,
        "Open-circuit voltage [V]": ocv,
        "R0 [Ohm]": r0,
        "Element-1 initial overpotential [V]": 0,
        "R1 [Ohm]": r1,
        "C1 [F]": c1,
        "Entropic change [V/K]": dUdT,
        "RCR lookup limit [A]": 340,
    }

    return values
