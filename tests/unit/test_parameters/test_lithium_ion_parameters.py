#
# Tests lithium-ion parameters load and give expected values
#

import numpy as np

import pybamm


class TestLithiumIonParameterValues:
    def test_print_parameters(self, tmp_path):
        parameters = pybamm.LithiumIonParameters()
        parameter_values = pybamm.lithium_ion.BaseModel().default_parameter_values
        output_file = tmp_path / "lithium_ion_parameters.txt"
        parameter_values.print_parameters(parameters, output_file)

    def test_lithium_ion(self):
        """This test checks that all the parameters are being calculated
        correctly for the specific set of parameters for LCO from dualfoil. The values
        are those converted from those in Scott's transfer which previous versions of
        the DFN work with. A 1C rate corresponds to a 24A/m^2 current density"""
        values = pybamm.lithium_ion.BaseModel().default_parameter_values
        param = pybamm.LithiumIonParameters()

        # particle geometry
        np.testing.assert_almost_equal(
            values.evaluate(param.n.prim.a_typ), 0.18 * 10 ** (6), 2
        )
        np.testing.assert_almost_equal(
            values.evaluate(param.n.prim.R_typ), 1 * 10 ** (-5), 2
        )
        np.testing.assert_almost_equal(
            values.evaluate(param.p.prim.a_typ), 0.15 * 10 ** (6), 2
        )
        np.testing.assert_almost_equal(
            values.evaluate(param.p.prim.R_typ), 1 * 10 ** (-5), 2
        )

        # j0_m
        np.testing.assert_almost_equal(
            values.evaluate(
                param.n.prim.j0(param.c_e_init_av, param.n.prim.c_max / 2, param.T_ref)
            ),
            values.evaluate(
                2 * 10 ** (-5) * param.c_e_init_av**0.5 * param.n.prim.c_max / 2
            ),
            8,
        )

        # j0_p
        np.testing.assert_almost_equal(
            values.evaluate(
                param.p.prim.j0(param.c_e_init_av, param.p.prim.c_max / 2, param.T_ref)
            ),
            values.evaluate(
                6 * 10 ** (-7) * param.c_e_init_av**0.5 * param.p.prim.c_max / 2
            ),
            8,
        )

        # particle dynamics
        # neg diffusion coefficient
        np.testing.assert_almost_equal(
            values.evaluate(
                pybamm.xyz_average(
                    pybamm.r_average(param.n.prim.D(param.n.prim.c_init, param.T_ref))
                )
            ),
            3.9 * 10 ** (-14),
            2,
        )

        # pos diffusion coefficient
        np.testing.assert_almost_equal(
            values.evaluate(
                pybamm.xyz_average(
                    pybamm.r_average(param.p.prim.D(param.p.prim.c_init, param.T_ref))
                )
            ),
            1 * 10 ** (-13),
            2,
        )

        # electrolyte dynamics
        np.testing.assert_almost_equal(
            values.evaluate(param.D_e(param.c_e_init_av, param.T_ref)),
            5.34 * 10 ** (-10) * np.exp(-0.65),
            10,
        )

        # electrolyte conductivity
        np.testing.assert_almost_equal(
            values.evaluate(param.kappa_e(param.c_e_init_av, param.T_ref)), 1.1045, 3
        )

        # electrode conductivities
        # neg
        np.testing.assert_almost_equal(
            values.evaluate(param.n.sigma(param.T_ref)), 100, 3
        )

        # pos
        np.testing.assert_almost_equal(
            values.evaluate(param.p.sigma(param.T_ref)), 10, 3
        )

    def test_thermal_parameters(self):
        values = pybamm.lithium_ion.BaseModel().default_parameter_values
        values.update(
            {
                "Cell heat capacity [J.K-1.m-3]": 2.5e6,
                "Left face heat transfer coefficient [W.m-2.K-1]": 5.0,
                "Right face heat transfer coefficient [W.m-2.K-1]": 5.0,
                "Front face heat transfer coefficient [W.m-2.K-1]": 5.0,
                "Back face heat transfer coefficient [W.m-2.K-1]": 5.0,
                "Bottom face heat transfer coefficient [W.m-2.K-1]": 5.0,
                "Top face heat transfer coefficient [W.m-2.K-1]": 5.0,
                "Inner radius heat transfer coefficient [W.m-2.K-1]": 5.0,
                "Outer radius heat transfer coefficient [W.m-2.K-1]": 5.0,
            },
            check_already_exists=False,
        )
        param = pybamm.LithiumIonParameters()
        T = param.T_ref

        # Density
        np.testing.assert_equal(values.evaluate(param.n.rho_c_p_cc(T)), 8954 * 385)
        np.testing.assert_equal(values.evaluate(param.n.rho_c_p(T)), 1657 * 700)
        np.testing.assert_equal(values.evaluate(param.s.rho_c_p(T)), 397 * 700)
        np.testing.assert_equal(values.evaluate(param.p.rho_c_p(T)), 3262 * 700)
        np.testing.assert_equal(values.evaluate(param.p.rho_c_p_cc(T)), 2707 * 897)

        # Thermal conductivity
        np.testing.assert_equal(values.evaluate(param.n.lambda_cc(T)), 401)
        np.testing.assert_equal(values.evaluate(param.n.lambda_(T)), 1.7)
        np.testing.assert_equal(values.evaluate(param.s.lambda_(T)), 0.16)
        np.testing.assert_equal(values.evaluate(param.p.lambda_(T)), 2.1)
        np.testing.assert_equal(values.evaluate(param.p.lambda_cc(T)), 237)

        # other thermal parameters
        np.testing.assert_equal(values.evaluate(param.T_init), 298.15)
        np.testing.assert_equal(values.evaluate(param.cell_heat_capacity), 2.5e6)
        np.testing.assert_equal(values.evaluate(param.h_edge_x_min), 5.0)
        np.testing.assert_equal(values.evaluate(param.h_edge_x_max), 5.0)
        np.testing.assert_equal(values.evaluate(param.h_edge_y_min), 5.0)
        np.testing.assert_equal(values.evaluate(param.h_edge_y_max), 5.0)
        np.testing.assert_equal(values.evaluate(param.h_edge_z_min), 5.0)
        np.testing.assert_equal(values.evaluate(param.h_edge_z_max), 5.0)
        np.testing.assert_equal(values.evaluate(param.h_edge_radial_min), 5.0)
        np.testing.assert_equal(values.evaluate(param.h_edge_radial_max), 5.0)

    def test_parameter_functions(self):
        values = pybamm.lithium_ion.BaseModel().default_parameter_values
        param = pybamm.LithiumIonParameters()

        T_test = 298.15

        c_e_test = 1000
        values.evaluate(param.D_e(c_e_test, T_test))
        values.evaluate(param.kappa_e(c_e_test, T_test))
