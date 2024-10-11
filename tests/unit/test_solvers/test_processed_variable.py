#
# Tests for the Processed Variable class
#

import casadi
import pybamm
import tests

import numpy as np
import pytest
from scipy.interpolate import CubicHermiteSpline


if pybamm.has_idaklu():
    _hermite_args = [True, False]
else:
    _hermite_args = [False]


def to_casadi(var_pybamm, y, inputs=None):
    t_MX = casadi.MX.sym("t")
    y_MX = casadi.MX.sym("y", y.shape[0])

    inputs_MX_dict = {}
    inputs = inputs or {}
    for key, value in inputs.items():
        inputs_MX_dict[key] = casadi.MX.sym("input", value.shape[0])

    inputs_MX = casadi.vertcat(*[p for p in inputs_MX_dict.values()])

    var_sym = var_pybamm.to_casadi(t_MX, y_MX, inputs=inputs_MX_dict)

    var_casadi = casadi.Function("variable", [t_MX, y_MX, inputs_MX], [var_sym])
    return var_casadi


class TestProcessedVariable:
    @staticmethod
    def _get_yps(y, hermite_interp, values=1):
        if hermite_interp:
            yp_sol = values * np.ones_like(y)
        else:
            yp_sol = None
        return yp_sol

    @staticmethod
    def _sol_default(t_sol, y_sol, yp_sol=None, model=None, inputs=None):
        if inputs is None:
            inputs = {}
        if model is None:
            model = tests.get_base_model_with_battery_geometry()
        return pybamm.Solution(
            t_sol,
            y_sol,
            model,
            inputs,
            all_yps=yp_sol,
        )

    def _process_and_check_2D_variable(
        self,
        var,
        first_spatial_var,
        second_spatial_var,
        disc=None,
        geometry_options=None,
        hermite_interp=False,
    ):
        # first_spatial_var should be on the "smaller" domain, i.e "r" for an "r-x" variable
        if geometry_options is None:
            geometry_options = {}
        if disc is None:
            disc = tests.get_discretisation_for_testing()
        disc.set_variable_slices([var])

        first_sol = disc.process_symbol(first_spatial_var).entries[:, 0]
        second_sol = disc.process_symbol(second_spatial_var).entries[:, 0]

        # Keep only the first iteration of entries
        first_sol = first_sol[: len(first_sol) // len(second_sol)]
        var_sol = disc.process_symbol(var)
        t_sol = np.linspace(0, 1)
        y_sol = 5 * t_sol * np.ones(len(second_sol) * len(first_sol))[:, np.newaxis]
        yp_sol = self._get_yps(y_sol, hermite_interp, values=5)

        var_casadi = to_casadi(var_sol, y_sol)
        model = tests.get_base_model_with_battery_geometry(**geometry_options)
        processed_var = pybamm.process_variable(
            [var_sol],
            [var_casadi],
            self._sol_default(t_sol, y_sol, yp_sol, model),
        )
        np.testing.assert_array_equal(
            processed_var.entries,
            np.reshape(y_sol, [len(first_sol), len(second_sol), len(t_sol)]),
        )

        # check that C++ and Python give the same result
        if pybamm.has_idaklu():
            np.testing.assert_array_equal(
                processed_var._observe_raw_cpp(), processed_var._observe_raw_python()
            )

        return y_sol, first_sol, second_sol, t_sol, yp_sol

    @pytest.mark.parametrize("hermite_interp", _hermite_args)
    def test_processed_variable_0D(self, hermite_interp):
        # without space
        t = pybamm.t
        y = pybamm.StateVector(slice(0, 1))
        var = t * y
        var.mesh = None
        model = pybamm.BaseModel()
        t_sol = np.linspace(0, 1)
        y_sol = np.array([np.linspace(0, 5)])
        yp_sol = self._get_yps(y_sol, hermite_interp)
        var_casadi = to_casadi(var, y_sol)
        processed_var = pybamm.process_variable(
            [var],
            [var_casadi],
            self._sol_default(t_sol, y_sol, yp_sol, model),
        )
        np.testing.assert_array_equal(processed_var.entries, t_sol * y_sol[0])

        # scalar value
        var = y
        var.mesh = None
        t_sol = np.array([0])
        y_sol = np.array([1])[:, np.newaxis]
        yp_sol = np.array([1])[:, np.newaxis]
        sol = self._sol_default(t_sol, y_sol, yp_sol, model)
        var_casadi = to_casadi(var, y_sol)
        processed_var = pybamm.process_variable(
            [var],
            [var_casadi],
            sol,
        )
        np.testing.assert_array_equal(processed_var.entries, y_sol[0])

        # check that repeated calls return the same data
        data1 = processed_var.data

        assert processed_var.entries_raw_initialized

        data2 = processed_var.data

        np.testing.assert_array_equal(data1, data2)

        data_t1 = processed_var(sol.t)

        assert processed_var.xr_array_raw_initialized

        data_t2 = processed_var(sol.t)

        np.testing.assert_array_equal(data_t1, data_t2)

        # check that C++ and Python give the same result
        if pybamm.has_idaklu():
            np.testing.assert_array_equal(
                processed_var._observe_raw_cpp(), processed_var._observe_raw_python()
            )

    @pytest.mark.parametrize("hermite_interp", _hermite_args)
    def test_processed_variable_0D_discrete_data(self, hermite_interp):
        y = pybamm.StateVector(slice(0, 1))
        t_sol = np.linspace(0, 1)
        y_sol = np.array([np.linspace(0, 5)])
        data_const = 3.6
        if hermite_interp:
            yp_sol = 5 * np.ones_like(y_sol)
        else:
            yp_sol = None

        # hermite interpolation can do order 2 interpolation, otherwise make sure result is linear
        order = 2 if hermite_interp else 1

        # data is same timepoints as solution
        data_t = t_sol
        data_v = -data_const * data_t
        data = pybamm.DiscreteTimeData(data_t, data_v, "test_data")
        var = (y - data) ** order
        expected_entries = (y_sol - data_v) ** order
        var.mesh = None
        model = pybamm.BaseModel()
        var_casadi = to_casadi(var, y_sol)
        processed_var = pybamm.process_variable(
            [var],
            [var_casadi],
            self._sol_default(t_sol, y_sol, yp_sol, model),
        )
        np.testing.assert_array_equal(processed_var.entries, expected_entries.flatten())
        np.testing.assert_array_equal(processed_var(data_t), expected_entries.flatten())

        # data is different timepoints as solution
        data_t = np.linspace(0, 1, 7)
        data_v = -data_const * data_t
        y_sol_interp = (np.interp(data_t, t_sol, y_sol[0]),)
        data_v_interp = np.interp(t_sol, data_t, data_v)
        data = pybamm.DiscreteTimeData(data_t, data_v, "test_data")

        # check data interp
        np.testing.assert_array_almost_equal(
            data.evaluate(t=t_sol).flatten(), data_v_interp
        )

        var = (y - data) ** order
        expected = (y_sol_interp - data_v) ** order
        expected_entries = (y_sol - data_v_interp) ** order
        var.mesh = None
        model = pybamm.BaseModel()
        var_casadi = to_casadi(var, y_sol)
        processed_var = pybamm.process_variable(
            [var],
            [var_casadi],
            self._sol_default(t_sol, y_sol, yp_sol, model),
        )
        np.testing.assert_array_almost_equal(
            processed_var.entries, expected_entries.flatten(), decimal=10
        )
        np.testing.assert_array_almost_equal(
            processed_var(t=data_t), expected.flatten(), decimal=10
        )

    @pytest.mark.parametrize("hermite_interp", _hermite_args)
    def test_processed_variable_0D_no_sensitivity(self, hermite_interp):
        # without space
        t = pybamm.t
        y = pybamm.StateVector(slice(0, 1))
        var = t * y
        var.mesh = None
        t_sol = np.linspace(0, 1)
        y_sol = np.array([np.linspace(0, 5)])
        yp_sol = self._get_yps(y_sol, hermite_interp)
        var_casadi = to_casadi(var, y_sol)
        processed_var = pybamm.process_variable(
            [var],
            [var_casadi],
            self._sol_default(t_sol, y_sol, yp_sol, pybamm.BaseModel()),
        )

        # test no inputs (i.e. no sensitivity)
        assert processed_var.sensitivities == {}

        # with parameter
        t = pybamm.t
        y = pybamm.StateVector(slice(0, 1))
        a = pybamm.InputParameter("a")
        var = t * y * a
        var.mesh = None
        t_sol = np.linspace(0, 1)
        y_sol = np.array([np.linspace(0, 5)])
        inputs = {"a": np.array([1.0])}
        var_casadi = to_casadi(var, y_sol, inputs=inputs)
        processed_var = pybamm.process_variable(
            [var],
            [var_casadi],
            pybamm.Solution(t_sol, y_sol, pybamm.BaseModel(), inputs),
        )

        # test no sensitivity raises error
        with pytest.raises(ValueError, match="Cannot compute sensitivities"):
            print(processed_var.sensitivities)

    @pytest.mark.parametrize("hermite_interp", _hermite_args)
    def test_processed_variable_1D(self, hermite_interp):
        t = pybamm.t
        var = pybamm.Variable("var", domain=["negative electrode", "separator"])
        x = pybamm.SpatialVariable("x", domain=["negative electrode", "separator"])
        eqn = t * var + x

        # On nodes
        disc = tests.get_discretisation_for_testing()
        disc.set_variable_slices([var])
        x_sol = disc.process_symbol(x).entries[:, 0]
        var_sol = disc.process_symbol(var)
        eqn_sol = disc.process_symbol(eqn)
        t_sol = np.linspace(0, 1)
        y_sol = np.ones_like(x_sol)[:, np.newaxis] * np.linspace(0, 5)
        yp_sol = self._get_yps(y_sol, hermite_interp, values=5)

        var_casadi = to_casadi(var_sol, y_sol)
        processed_var = pybamm.process_variable(
            [var_sol],
            [var_casadi],
            self._sol_default(t_sol, y_sol, yp_sol),
        )
        np.testing.assert_array_equal(processed_var.entries, y_sol)
        np.testing.assert_array_almost_equal(processed_var(t_sol, x_sol), y_sol)
        eqn_casadi = to_casadi(eqn_sol, y_sol)
        processed_eqn = pybamm.process_variable(
            [eqn_sol],
            [eqn_casadi],
            self._sol_default(t_sol, y_sol, yp_sol),
        )
        np.testing.assert_array_almost_equal(
            processed_eqn(t_sol, x_sol), t_sol * y_sol + x_sol[:, np.newaxis]
        )

        # Test extrapolation
        np.testing.assert_array_equal(processed_var.entries[0], 2 * y_sol[0] - y_sol[1])
        np.testing.assert_array_equal(
            processed_var.entries[1], 2 * y_sol[-1] - y_sol[-2]
        )

        # On edges
        x_s_edge = pybamm.Matrix(disc.mesh["separator"].edges, domain="separator")
        x_s_edge.mesh = disc.mesh["separator"]
        x_s_casadi = to_casadi(x_s_edge, y_sol)
        processed_x_s_edge = pybamm.process_variable(
            [x_s_edge],
            [x_s_casadi],
            self._sol_default(t_sol, y_sol, yp_sol),
        )
        np.testing.assert_array_equal(
            x_s_edge.entries[:, 0], processed_x_s_edge.entries[:, 0]
        )

        # space only
        eqn = var + x
        eqn_sol = disc.process_symbol(eqn)
        t_sol = np.array([0])
        y_sol = np.ones_like(x_sol)[:, np.newaxis]
        yp_sol = self._get_yps(y_sol, hermite_interp, values=0)
        eqn_casadi = to_casadi(eqn_sol, y_sol)
        processed_eqn2 = pybamm.process_variable(
            [eqn_sol],
            [eqn_casadi],
            self._sol_default(t_sol, y_sol, yp_sol),
        )
        np.testing.assert_array_equal(
            processed_eqn2.entries, y_sol + x_sol[:, np.newaxis]
        )

        # check that C++ and Python give the same result
        if pybamm.has_idaklu():
            np.testing.assert_array_equal(
                processed_eqn2._observe_raw_cpp(), processed_eqn2._observe_raw_python()
            )

    @pytest.mark.parametrize("hermite_interp", _hermite_args)
    def test_processed_variable_1D_unknown_domain(self, hermite_interp):
        x = pybamm.SpatialVariable("x", domain="SEI layer", coord_sys="cartesian")
        geometry = pybamm.Geometry(
            {"SEI layer": {x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}}
        )

        submesh_types = {"SEI layer": pybamm.Uniform1DSubMesh}
        var_pts = {x: 100}
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        nt = 100

        y_sol = np.zeros((var_pts[x], nt))
        yp_sol = self._get_yps(y_sol, hermite_interp)
        model = tests.get_base_model_with_battery_geometry()
        model._geometry = geometry
        solution = pybamm.Solution(
            np.linspace(0, 1, nt),
            y_sol,
            model,
            {},
            np.linspace(0, 1, 1),
            np.zeros(var_pts[x]),
            "test",
            all_yps=yp_sol,
        )

        c = pybamm.StateVector(slice(0, var_pts[x]), domain=["SEI layer"])
        c.mesh = mesh["SEI layer"]
        c_casadi = to_casadi(c, y_sol)
        pybamm.process_variable([c], [c_casadi], solution)

    @pytest.mark.parametrize("hermite_interp", _hermite_args)
    def test_processed_variable_2D_x_r(self, hermite_interp):
        var = pybamm.Variable(
            "var",
            domain=["negative particle"],
            auxiliary_domains={"secondary": ["negative electrode"]},
        )
        x = pybamm.SpatialVariable("x", domain=["negative electrode"])
        r = pybamm.SpatialVariable(
            "r",
            domain=["negative particle"],
            auxiliary_domains={"secondary": ["negative electrode"]},
        )

        disc = tests.get_p2d_discretisation_for_testing()
        self._process_and_check_2D_variable(
            var, r, x, disc=disc, hermite_interp=hermite_interp
        )

    @pytest.mark.parametrize("hermite_interp", _hermite_args)
    def test_processed_variable_2D_R_x(self, hermite_interp):
        var = pybamm.Variable(
            "var",
            domain=["negative particle size"],
            auxiliary_domains={"secondary": ["negative electrode"]},
        )
        R = pybamm.SpatialVariable(
            "R",
            domain=["negative particle size"],
            auxiliary_domains={"secondary": ["negative electrode"]},
        )
        x = pybamm.SpatialVariable("x", domain=["negative electrode"])

        disc = tests.get_size_distribution_disc_for_testing()
        self._process_and_check_2D_variable(
            var,
            R,
            x,
            disc=disc,
            geometry_options={"options": {"particle size": "distribution"}},
            hermite_interp=hermite_interp,
        )

    @pytest.mark.parametrize("hermite_interp", _hermite_args)
    def test_processed_variable_2D_R_z(self, hermite_interp):
        var = pybamm.Variable(
            "var",
            domain=["negative particle size"],
            auxiliary_domains={"secondary": ["current collector"]},
        )
        R = pybamm.SpatialVariable(
            "R",
            domain=["negative particle size"],
            auxiliary_domains={"secondary": ["current collector"]},
        )
        z = pybamm.SpatialVariable("z", domain=["current collector"])

        disc = tests.get_size_distribution_disc_for_testing()
        self._process_and_check_2D_variable(
            var,
            R,
            z,
            disc=disc,
            geometry_options={"options": {"particle size": "distribution"}},
            hermite_interp=hermite_interp,
        )

    @pytest.mark.parametrize("hermite_interp", _hermite_args)
    def test_processed_variable_2D_r_R(self, hermite_interp):
        var = pybamm.Variable(
            "var",
            domain=["negative particle"],
            auxiliary_domains={"secondary": ["negative particle size"]},
        )
        r = pybamm.SpatialVariable(
            "r",
            domain=["negative particle"],
            auxiliary_domains={"secondary": ["negative particle size"]},
        )
        R = pybamm.SpatialVariable("R", domain=["negative particle size"])

        disc = tests.get_size_distribution_disc_for_testing()
        self._process_and_check_2D_variable(
            var,
            r,
            R,
            disc=disc,
            geometry_options={"options": {"particle size": "distribution"}},
            hermite_interp=hermite_interp,
        )

    @pytest.mark.parametrize("hermite_interp", _hermite_args)
    def test_processed_variable_2D_x_z(self, hermite_interp):
        var = pybamm.Variable(
            "var",
            domain=["negative electrode", "separator"],
            auxiliary_domains={"secondary": "current collector"},
        )
        x = pybamm.SpatialVariable(
            "x",
            domain=["negative electrode", "separator"],
            auxiliary_domains={"secondary": "current collector"},
        )
        z = pybamm.SpatialVariable("z", domain=["current collector"])

        disc = tests.get_1p1d_discretisation_for_testing()
        y_sol, x_sol, z_sol, t_sol, yp_sol = self._process_and_check_2D_variable(
            var, x, z, disc=disc, hermite_interp=hermite_interp
        )
        del x_sol

        # On edges
        x_s_edge = pybamm.Matrix(
            np.tile(disc.mesh["separator"].edges, len(z_sol)),
            domain="separator",
            auxiliary_domains={"secondary": "current collector"},
        )
        x_s_edge.mesh = disc.mesh["separator"]
        x_s_edge.secondary_mesh = disc.mesh["current collector"]
        x_s_casadi = to_casadi(x_s_edge, y_sol)
        processed_x_s_edge = pybamm.process_variable(
            [x_s_edge],
            [x_s_casadi],
            self._sol_default(t_sol, y_sol, yp_sol),
        )
        np.testing.assert_array_equal(
            x_s_edge.entries.flatten(), processed_x_s_edge.entries[:, :, 0].T.flatten()
        )

    @pytest.mark.parametrize("hermite_interp", _hermite_args)
    def test_processed_variable_2D_space_only(self, hermite_interp):
        var = pybamm.Variable(
            "var",
            domain=["negative particle"],
            auxiliary_domains={"secondary": ["negative electrode"]},
        )
        x = pybamm.SpatialVariable("x", domain=["negative electrode"])
        r = pybamm.SpatialVariable(
            "r",
            domain=["negative particle"],
            auxiliary_domains={"secondary": ["negative electrode"]},
        )

        disc = tests.get_p2d_discretisation_for_testing()
        disc.set_variable_slices([var])
        x_sol = disc.process_symbol(x).entries[:, 0]
        r_sol = disc.process_symbol(r).entries[:, 0]
        # Keep only the first iteration of entries
        r_sol = r_sol[: len(r_sol) // len(x_sol)]
        var_sol = disc.process_symbol(var)
        t_sol = np.array([0])
        y_sol = np.ones(len(x_sol) * len(r_sol))[:, np.newaxis]
        yp_sol = self._get_yps(y_sol, hermite_interp)

        var_casadi = to_casadi(var_sol, y_sol)
        processed_var = pybamm.process_variable(
            [var_sol],
            [var_casadi],
            self._sol_default(t_sol, y_sol, yp_sol),
        )
        np.testing.assert_array_equal(
            processed_var.entries,
            np.reshape(y_sol, [len(r_sol), len(x_sol), len(t_sol)]),
        )

    @pytest.mark.parametrize("hermite_interp", _hermite_args)
    def test_processed_variable_2D_scikit(self, hermite_interp):
        var = pybamm.Variable("var", domain=["current collector"])

        disc = tests.get_2p1d_discretisation_for_testing()
        disc.set_variable_slices([var])
        y = disc.mesh["current collector"].edges["y"]
        z = disc.mesh["current collector"].edges["z"]
        var_sol = disc.process_symbol(var)
        var_sol.mesh = disc.mesh["current collector"]
        t_sol = np.linspace(0, 1)
        u_sol = np.ones(var_sol.shape[0])[:, np.newaxis] * np.linspace(0, 5)
        yp_sol = self._get_yps(u_sol, hermite_interp)

        var_casadi = to_casadi(var_sol, u_sol)
        processed_var = pybamm.process_variable(
            [var_sol],
            [var_casadi],
            self._sol_default(t_sol, u_sol, yp_sol),
        )
        np.testing.assert_array_equal(
            processed_var.entries, np.reshape(u_sol, [len(y), len(z), len(t_sol)])
        )

    @pytest.mark.parametrize("hermite_interp", _hermite_args)
    def test_processed_variable_2D_fixed_t_scikit(self, hermite_interp):
        var = pybamm.Variable("var", domain=["current collector"])

        disc = tests.get_2p1d_discretisation_for_testing()
        disc.set_variable_slices([var])
        y = disc.mesh["current collector"].edges["y"]
        z = disc.mesh["current collector"].edges["z"]
        var_sol = disc.process_symbol(var)
        var_sol.mesh = disc.mesh["current collector"]
        t_sol = np.array([0])
        u_sol = np.ones(var_sol.shape[0])[:, np.newaxis]
        yp_sol = self._get_yps(u_sol, hermite_interp)

        var_casadi = to_casadi(var_sol, u_sol)
        model = tests.get_base_model_with_battery_geometry(
            options={"dimensionality": 2}
        )
        processed_var = pybamm.process_variable(
            [var_sol],
            [var_casadi],
            pybamm.Solution(t_sol, u_sol, model, {}, all_yps=yp_sol),
        )
        np.testing.assert_array_equal(
            processed_var.entries, np.reshape(u_sol, [len(y), len(z), len(t_sol)])
        )

    @pytest.mark.parametrize("hermite_interp", _hermite_args)
    def test_processed_var_0D_interpolation(self, hermite_interp):
        # without spatial dependence
        t = pybamm.t
        y = pybamm.StateVector(slice(0, 1))
        var = y
        eqn = t * y
        var.mesh = None
        eqn.mesh = None

        t_sol = np.linspace(0, 1, 1000)
        y_sol = np.array([5 * t_sol])
        yp_sol = self._get_yps(y_sol, hermite_interp, values=5)
        var_casadi = to_casadi(var, y_sol)
        processed_var = pybamm.process_variable(
            [var],
            [var_casadi],
            self._sol_default(t_sol, y_sol, yp_sol),
        )
        # vector
        np.testing.assert_array_equal(processed_var(t_sol), y_sol[0])
        # scalar
        np.testing.assert_array_almost_equal(processed_var(0.5), 2.5)
        np.testing.assert_array_almost_equal(processed_var(0.7), 3.5)

        eqn_casadi = to_casadi(eqn, y_sol)
        processed_eqn = pybamm.process_variable(
            [eqn],
            [eqn_casadi],
            self._sol_default(t_sol, y_sol, yp_sol),
        )
        np.testing.assert_array_equal(processed_eqn(t_sol), t_sol * y_sol[0])

        assert processed_eqn(0.5).shape == ()

        np.testing.assert_array_almost_equal(processed_eqn(0.5), 0.5 * 2.5)
        np.testing.assert_array_equal(processed_eqn(2, fill_value=100), 100)
        # Suppress warning for this test
        pybamm.set_logging_level("ERROR")
        np.testing.assert_array_equal(processed_eqn(2), np.nan)
        pybamm.set_logging_level("WARNING")

    @pytest.mark.parametrize("hermite_interp", _hermite_args)
    def test_processed_var_0D_fixed_t_interpolation(self, hermite_interp):
        y = pybamm.StateVector(slice(0, 1))
        var = y
        eqn = 2 * y
        var.mesh = None
        eqn.mesh = None

        t_sol = np.array([10])
        y_sol = np.array([[100]])
        yp_sol = self._get_yps(y_sol, hermite_interp)
        eqn_casadi = to_casadi(eqn, y_sol)
        processed_var = pybamm.process_variable(
            [eqn],
            [eqn_casadi],
            self._sol_default(t_sol, y_sol, yp_sol, pybamm.BaseModel()),
        )

        assert processed_var() == 200

    @pytest.mark.parametrize("hermite_interp", _hermite_args)
    def test_processed_var_1D_interpolation(self, hermite_interp):
        t = pybamm.t
        var = pybamm.Variable("var", domain=["negative electrode", "separator"])
        x = pybamm.SpatialVariable("x", domain=["negative electrode", "separator"])
        eqn = t * var + x

        disc = tests.get_discretisation_for_testing()
        disc.set_variable_slices([var])
        x_sol = disc.process_symbol(x).entries[:, 0]
        var_sol = disc.process_symbol(var)
        eqn_sol = disc.process_symbol(eqn)
        t_sol = np.linspace(0, 1)
        y_sol = x_sol[:, np.newaxis] * (5 * t_sol)
        yp_sol = self._get_yps(y_sol, hermite_interp, values=5)

        var_casadi = to_casadi(var_sol, y_sol)
        processed_var = pybamm.process_variable(
            [var_sol],
            [var_casadi],
            self._sol_default(t_sol, y_sol, yp_sol),
        )

        # 2 vectors
        np.testing.assert_array_almost_equal(processed_var(t_sol, x_sol), y_sol)
        # 1 vector, 1 scalar
        np.testing.assert_array_almost_equal(processed_var(0.5, x_sol), 2.5 * x_sol)
        np.testing.assert_array_almost_equal(
            processed_var(t_sol, x_sol[-1]),
            x_sol[-1] * np.linspace(0, 5),
        )
        # 2 scalars
        np.testing.assert_array_almost_equal(
            processed_var(0.5, x_sol[-1]), 2.5 * x_sol[-1]
        )
        eqn_casadi = to_casadi(eqn_sol, y_sol)
        processed_eqn = pybamm.process_variable(
            [eqn_sol],
            [eqn_casadi],
            self._sol_default(t_sol, y_sol, yp_sol),
        )
        # 2 vectors
        np.testing.assert_array_almost_equal(
            processed_eqn(t_sol, x_sol), t_sol * y_sol + x_sol[:, np.newaxis]
        )
        # 1 vector, 1 scalar
        assert processed_eqn(0.5, x_sol[10:30]).shape == (20,)
        assert processed_eqn(t_sol[4:9], x_sol[-1]).shape == (5,)
        # 2 scalars
        assert processed_eqn(0.5, x_sol[-1]).shape == ()

        # test x
        x_disc = disc.process_symbol(x)
        x_casadi = to_casadi(x_disc, y_sol)

        processed_x = pybamm.process_variable(
            [x_disc],
            [x_casadi],
            self._sol_default(t_sol, y_sol, yp_sol),
        )
        np.testing.assert_array_almost_equal(processed_x(t=0, x=x_sol), x_sol)

        # In particles
        r_n = pybamm.Matrix(
            disc.mesh["negative particle"].nodes, domain="negative particle"
        )
        r_n.mesh = disc.mesh["negative particle"]
        r_n_casadi = to_casadi(r_n, y_sol)
        processed_r_n = pybamm.process_variable(
            [r_n],
            [r_n_casadi],
            self._sol_default(t_sol, y_sol, yp_sol),
        )
        np.testing.assert_array_equal(r_n.entries[:, 0], processed_r_n.entries[:, 0])
        r_test = np.linspace(0, 0.5)
        np.testing.assert_array_almost_equal(processed_r_n(0, r=r_test), r_test)

        # On size domain
        R_n = pybamm.Matrix(
            disc.mesh["negative particle size"].nodes, domain="negative particle size"
        )
        R_n.mesh = disc.mesh["negative particle size"]
        R_n_casadi = to_casadi(R_n, y_sol)
        model = tests.get_base_model_with_battery_geometry(
            options={"particle size": "distribution"}
        )
        processed_R_n = pybamm.process_variable(
            [R_n],
            [R_n_casadi],
            pybamm.Solution(t_sol, y_sol, model, {}),
        )
        np.testing.assert_array_equal(R_n.entries[:, 0], processed_R_n.entries[:, 0])
        R_test = np.linspace(0, 1)
        np.testing.assert_array_almost_equal(processed_R_n(0, R=R_test), R_test)

    @pytest.mark.parametrize("hermite_interp", _hermite_args)
    def test_processed_var_1D_fixed_t_interpolation(self, hermite_interp):
        var = pybamm.Variable("var", domain=["negative electrode", "separator"])
        x = pybamm.SpatialVariable("x", domain=["negative electrode", "separator"])
        eqn = var + x

        disc = tests.get_discretisation_for_testing()
        disc.set_variable_slices([var])
        x_sol = disc.process_symbol(x).entries[:, 0]
        eqn_sol = disc.process_symbol(eqn)
        t_sol = np.array([1])
        y_sol = x_sol[:, np.newaxis]
        yp_sol = self._get_yps(y_sol, hermite_interp)

        eqn_casadi = to_casadi(eqn_sol, y_sol)
        processed_var = pybamm.process_variable(
            [eqn_sol],
            [eqn_casadi],
            self._sol_default(t_sol, y_sol, yp_sol),
        )

        # vector
        np.testing.assert_array_almost_equal(
            processed_var(x=x_sol), 2 * x_sol[:, np.newaxis]
        )
        # scalar
        np.testing.assert_array_almost_equal(processed_var(x=0.5), 1)

    @pytest.mark.parametrize("hermite_interp", _hermite_args)
    def test_processed_var_wrong_spatial_variable_names(self, hermite_interp):
        var = pybamm.Variable(
            "var",
            domain=["domain A", "domain B"],
        )
        a = pybamm.SpatialVariable("a", domain=["domain A"])
        b = pybamm.SpatialVariable("b", domain=["domain B"])
        geometry = {
            "domain A": {a: {"min": 0, "max": 1}},
            "domain B": {b: {"min": 1, "max": 2}},
        }
        submesh_types = {
            "domain A": pybamm.Uniform1DSubMesh,
            "domain B": pybamm.Uniform1DSubMesh,
        }
        var_pts = {a: 10, b: 20}
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        spatial_methods = {
            "domain A": pybamm.FiniteVolume(),
            "domain B": pybamm.FiniteVolume(),
        }

        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.set_variable_slices([var])
        a_sol = disc.process_symbol(a).entries[:, 0]
        b_sol = disc.process_symbol(b).entries[:, 0]
        var_sol = disc.process_symbol(var)
        t_sol = np.linspace(0, 1)
        y_sol = np.ones(len(a_sol) * len(b_sol))[:, np.newaxis] * np.linspace(0, 5)
        yp_sol = self._get_yps(y_sol, hermite_interp)

        var_casadi = to_casadi(var_sol, y_sol)
        model = pybamm.BaseModel()
        model._geometry = pybamm.Geometry(
            {
                "domain A": {a: {"min": 0, "max": 1}},
                "domain B": {b: {"min": 0, "max": 1}},
            }
        )
        with pytest.raises(NotImplementedError, match="Spatial variable name"):
            pybamm.process_variable(
                [var_sol],
                [var_casadi],
                pybamm.Solution(t_sol, y_sol, model, {}, all_yps=yp_sol),
            ).initialise()

    @pytest.mark.parametrize("hermite_interp", _hermite_args)
    def test_processed_var_2D_interpolation(self, hermite_interp):
        var = pybamm.Variable(
            "var",
            domain=["negative particle"],
            auxiliary_domains={"secondary": ["negative electrode"]},
        )
        x = pybamm.SpatialVariable("x", domain=["negative electrode"])
        r = pybamm.SpatialVariable(
            "r",
            domain=["negative particle"],
            auxiliary_domains={"secondary": ["negative electrode"]},
        )

        disc = tests.get_p2d_discretisation_for_testing()
        disc.set_variable_slices([var])
        x_sol = disc.process_symbol(x).entries[:, 0]
        r_sol = disc.process_symbol(r).entries[:, 0]
        # Keep only the first iteration of entries
        r_sol = r_sol[: len(r_sol) // len(x_sol)]
        var_sol = disc.process_symbol(var)
        t_sol = np.linspace(0, 1)
        y_sol = np.ones(len(x_sol) * len(r_sol))[:, np.newaxis] * np.linspace(0, 5)
        yp_sol = self._get_yps(y_sol, hermite_interp)

        var_casadi = to_casadi(var_sol, y_sol)
        processed_var = pybamm.process_variable(
            [var_sol],
            [var_casadi],
            self._sol_default(t_sol, y_sol, yp_sol),
        )
        # 3 vectors
        np.testing.assert_array_equal(
            processed_var(t_sol, x_sol, r_sol).shape, (10, 40, 50)
        )
        np.testing.assert_array_almost_equal(
            processed_var(t_sol, x_sol, r_sol),
            np.reshape(y_sol, [len(r_sol), len(x_sol), len(t_sol)]),
        )
        # 2 vectors, 1 scalar
        np.testing.assert_array_equal(processed_var(0.5, x_sol, r_sol).shape, (10, 40))
        np.testing.assert_array_equal(processed_var(t_sol, 0.2, r_sol).shape, (10, 50))
        np.testing.assert_array_equal(processed_var(t_sol, x_sol, 0.5).shape, (40, 50))
        # 1 vectors, 2 scalar
        np.testing.assert_array_equal(processed_var(0.5, 0.2, r_sol).shape, (10,))
        np.testing.assert_array_equal(processed_var(0.5, x_sol, 0.5).shape, (40,))
        np.testing.assert_array_equal(processed_var(t_sol, 0.2, 0.5).shape, (50,))
        # 3 scalars
        np.testing.assert_array_equal(processed_var(0.2, 0.2, 0.2).shape, ())

        # positive particle
        var = pybamm.Variable(
            "var",
            domain=["positive particle"],
            auxiliary_domains={"secondary": ["positive electrode"]},
        )
        x = pybamm.SpatialVariable("x", domain=["positive electrode"])
        r = pybamm.SpatialVariable(
            "r",
            domain=["positive particle"],
            auxiliary_domains={"secondary": ["positive electrode"]},
        )

        disc.set_variable_slices([var])
        x_sol = disc.process_symbol(x).entries[:, 0]
        r_sol = disc.process_symbol(r).entries[:, 0]
        # Keep only the first iteration of entries
        r_sol = r_sol[: len(r_sol) // len(x_sol)]
        var_sol = disc.process_symbol(var)
        t_sol = np.linspace(0, 1)
        y_sol = np.ones(len(x_sol) * len(r_sol))[:, np.newaxis] * np.linspace(0, 5)

        var_casadi = to_casadi(var_sol, y_sol)
        processed_var = pybamm.process_variable(
            [var_sol],
            [var_casadi],
            self._sol_default(t_sol, y_sol, yp_sol),
        )
        # 3 vectors
        np.testing.assert_array_equal(
            processed_var(t_sol, x_sol, r_sol).shape, (10, 35, 50)
        )

    @pytest.mark.parametrize("hermite_interp", _hermite_args)
    def test_processed_var_2D_fixed_t_interpolation(self, hermite_interp):
        var = pybamm.Variable(
            "var",
            domain=["negative particle"],
            auxiliary_domains={"secondary": ["negative electrode"]},
        )
        x = pybamm.SpatialVariable("x", domain=["negative electrode"])
        r = pybamm.SpatialVariable(
            "r",
            domain=["negative particle"],
            auxiliary_domains={"secondary": ["negative electrode"]},
        )

        disc = tests.get_p2d_discretisation_for_testing()
        disc.set_variable_slices([var])
        x_sol = disc.process_symbol(x).entries[:, 0]
        r_sol = disc.process_symbol(r).entries[:, 0]
        # Keep only the first iteration of entries
        r_sol = r_sol[: len(r_sol) // len(x_sol)]
        var_sol = disc.process_symbol(var)
        t_sol = np.array([0])
        y_sol = np.ones(len(x_sol) * len(r_sol))[:, np.newaxis]
        yp_sol = self._get_yps(y_sol, hermite_interp)

        var_casadi = to_casadi(var_sol, y_sol)
        processed_var = pybamm.process_variable(
            [var_sol],
            [var_casadi],
            self._sol_default(t_sol, y_sol, yp_sol),
        )
        # 2 vectors
        np.testing.assert_array_equal(
            processed_var(t=0, x=x_sol, r=r_sol).shape, (10, 40)
        )
        # 1 vector, 1 scalar
        np.testing.assert_array_equal(processed_var(t=0, x=0.2, r=r_sol).shape, (10,))
        np.testing.assert_array_equal(processed_var(t=0, x=x_sol, r=0.5).shape, (40,))
        # 2 scalars
        np.testing.assert_array_equal(processed_var(t=0, x=0.2, r=0.2).shape, ())

    @pytest.mark.parametrize("hermite_interp", _hermite_args)
    def test_processed_var_2D_secondary_broadcast(self, hermite_interp):
        var = pybamm.Variable("var", domain=["negative particle"])
        broad_var = pybamm.SecondaryBroadcast(var, "negative electrode")
        x = pybamm.SpatialVariable("x", domain=["negative electrode"])
        r = pybamm.SpatialVariable("r", domain=["negative particle"])

        disc = tests.get_discretisation_for_testing()
        disc.set_variable_slices([var])
        x_sol = disc.process_symbol(x).entries[:, 0]
        r_sol = disc.process_symbol(r).entries[:, 0]
        var_sol = disc.process_symbol(broad_var)
        t_sol = np.linspace(0, 1)
        y_sol = np.ones(len(x_sol) * len(r_sol))[:, np.newaxis] * np.linspace(0, 5)
        yp_sol = self._get_yps(y_sol, hermite_interp)

        var_casadi = to_casadi(var_sol, y_sol)
        processed_var = pybamm.process_variable(
            [var_sol],
            [var_casadi],
            self._sol_default(t_sol, y_sol, yp_sol),
        )
        # 3 vectors
        np.testing.assert_array_equal(
            processed_var(t_sol, x_sol, r_sol).shape, (10, 40, 50)
        )
        np.testing.assert_array_almost_equal(
            processed_var(t_sol, x_sol, r_sol),
            np.reshape(y_sol, [len(r_sol), len(x_sol), len(t_sol)]),
        )
        # 2 vectors, 1 scalar
        np.testing.assert_array_equal(processed_var(0.5, x_sol, r_sol).shape, (10, 40))
        np.testing.assert_array_equal(processed_var(t_sol, 0.2, r_sol).shape, (10, 50))
        np.testing.assert_array_equal(processed_var(t_sol, x_sol, 0.5).shape, (40, 50))
        # 1 vectors, 2 scalar
        np.testing.assert_array_equal(processed_var(0.5, 0.2, r_sol).shape, (10,))
        np.testing.assert_array_equal(processed_var(0.5, x_sol, 0.5).shape, (40,))
        np.testing.assert_array_equal(processed_var(t_sol, 0.2, 0.5).shape, (50,))
        # 3 scalars
        np.testing.assert_array_equal(processed_var(0.2, 0.2, 0.2).shape, ())

        # positive particle
        var = pybamm.Variable("var", domain=["positive particle"])
        broad_var = pybamm.SecondaryBroadcast(var, "positive electrode")
        x = pybamm.SpatialVariable("x", domain=["positive electrode"])
        r = pybamm.SpatialVariable("r", domain=["positive particle"])

        disc.set_variable_slices([var])
        x_sol = disc.process_symbol(x).entries[:, 0]
        r_sol = disc.process_symbol(r).entries[:, 0]
        var_sol = disc.process_symbol(broad_var)
        t_sol = np.linspace(0, 1)
        y_sol = np.ones(len(x_sol) * len(r_sol))[:, np.newaxis] * np.linspace(0, 5)
        yp_sol = self._get_yps(y_sol, hermite_interp, values=5)

        var_casadi = to_casadi(var_sol, y_sol)
        processed_var = pybamm.process_variable(
            [var_sol],
            [var_casadi],
            self._sol_default(t_sol, y_sol, yp_sol),
        )
        # 3 vectors
        np.testing.assert_array_equal(
            processed_var(t_sol, x_sol, r_sol).shape, (10, 35, 50)
        )

    @pytest.mark.parametrize("hermite_interp", _hermite_args)
    def test_processed_var_2_d_scikit_interpolation(self, hermite_interp):
        var = pybamm.Variable("var", domain=["current collector"])

        disc = tests.get_2p1d_discretisation_for_testing()
        disc.set_variable_slices([var])
        y_sol = disc.mesh["current collector"].edges["y"]
        z_sol = disc.mesh["current collector"].edges["z"]
        var_sol = disc.process_symbol(var)
        var_sol.mesh = disc.mesh["current collector"]
        t_sol = np.linspace(0, 1)
        u_sol = np.ones(var_sol.shape[0])[:, np.newaxis] * np.linspace(0, 5)
        yp_sol = self._get_yps(u_sol, hermite_interp)

        var_casadi = to_casadi(var_sol, u_sol)
        processed_var = pybamm.process_variable(
            [var_sol],
            [var_casadi],
            self._sol_default(t_sol, u_sol, yp_sol),
        )
        # 3 vectors
        np.testing.assert_array_equal(
            processed_var(t_sol, y=y_sol, z=z_sol).shape, (15, 15, 50)
        )
        np.testing.assert_array_almost_equal(
            processed_var(t_sol, y=y_sol, z=z_sol),
            np.reshape(u_sol, [len(y_sol), len(z_sol), len(t_sol)]),
        )
        # 2 vectors, 1 scalar
        np.testing.assert_array_equal(
            processed_var(0.5, y=y_sol, z=z_sol).shape, (15, 15)
        )
        np.testing.assert_array_equal(
            processed_var(t_sol, y=0.2, z=z_sol).shape, (15, 50)
        )
        np.testing.assert_array_equal(
            processed_var(t_sol, y=y_sol, z=0.5).shape, (15, 50)
        )
        # 1 vectors, 2 scalar
        np.testing.assert_array_equal(processed_var(0.5, y=0.2, z=z_sol).shape, (15,))
        np.testing.assert_array_equal(processed_var(0.5, y=y_sol, z=0.5).shape, (15,))
        np.testing.assert_array_equal(processed_var(t_sol, y=0.2, z=0.5).shape, (50,))
        # 3 scalars
        np.testing.assert_array_equal(processed_var(0.2, y=0.2, z=0.2).shape, ())

    @pytest.mark.parametrize("hermite_interp", _hermite_args)
    def test_processed_var_2D_fixed_t_scikit_interpolation(self, hermite_interp):
        var = pybamm.Variable("var", domain=["current collector"])

        disc = tests.get_2p1d_discretisation_for_testing()
        disc.set_variable_slices([var])
        y_sol = disc.mesh["current collector"].edges["y"]
        z_sol = disc.mesh["current collector"].edges["z"]
        var_sol = disc.process_symbol(var)
        var_sol.mesh = disc.mesh["current collector"]
        t_sol = np.array([0])
        u_sol = np.ones(var_sol.shape[0])[:, np.newaxis]
        yp_sol = self._get_yps(u_sol, hermite_interp)

        var_casadi = to_casadi(var_sol, u_sol)
        processed_var = pybamm.process_variable(
            [var_sol],
            [var_casadi],
            self._sol_default(t_sol, u_sol, yp_sol),
        )
        # 2 vectors
        np.testing.assert_array_equal(
            processed_var(t=0, y=y_sol, z=z_sol).shape, (15, 15)
        )
        # 1 vector, 1 scalar
        np.testing.assert_array_equal(processed_var(t=0, y=0.2, z=z_sol).shape, (15,))
        np.testing.assert_array_equal(processed_var(t=0, y=y_sol, z=0.5).shape, (15,))
        # 2 scalars
        np.testing.assert_array_equal(processed_var(t=0, y=0.2, z=0.2).shape, ())

    @pytest.mark.parametrize("hermite_interp", _hermite_args)
    def test_processed_var_2D_unknown_domain(self, hermite_interp):
        var = pybamm.Variable(
            "var",
            domain=["domain B"],
            auxiliary_domains={"secondary": ["domain A"]},
        )
        x = pybamm.SpatialVariable("x", domain=["domain A"])
        z = pybamm.SpatialVariable(
            "z",
            domain=["domain B"],
            auxiliary_domains={"secondary": ["domain A"]},
        )

        geometry = {
            "domain A": {x: {"min": 0, "max": 1}},
            "domain B": {z: {"min": 0, "max": 1}},
        }
        submesh_types = {
            "domain A": pybamm.Uniform1DSubMesh,
            "domain B": pybamm.Uniform1DSubMesh,
        }
        var_pts = {x: 10, z: 20}
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        spatial_methods = {
            "domain A": pybamm.FiniteVolume(),
            "domain B": pybamm.FiniteVolume(),
        }

        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.set_variable_slices([var])
        x_sol = disc.process_symbol(x).entries[:, 0]
        z_sol = disc.process_symbol(z).entries[:, 0]
        # Keep only the first iteration of entries
        z_sol = z_sol[: len(z_sol) // len(x_sol)]
        var_sol = disc.process_symbol(var)
        t_sol = np.linspace(0, 1)
        y_sol = np.ones(len(x_sol) * len(z_sol))[:, np.newaxis] * np.linspace(0, 5)
        yp_sol = self._get_yps(y_sol, hermite_interp, values=5)

        var_casadi = to_casadi(var_sol, y_sol)
        model = pybamm.BaseModel()
        model._geometry = pybamm.Geometry(
            {
                "domain A": {x: {"min": 0, "max": 1}},
                "domain B": {z: {"min": 0, "max": 1}},
            }
        )
        processed_var = pybamm.process_variable(
            [var_sol],
            [var_casadi],
            pybamm.Solution(t_sol, y_sol, model, {}, all_yps=yp_sol),
        )
        # 3 vectors
        np.testing.assert_array_equal(
            processed_var(t=t_sol, x=x_sol, z=z_sol).shape, (20, 10, 50)
        )
        np.testing.assert_array_almost_equal(
            processed_var(t_sol, x=x_sol, z=z_sol),
            np.reshape(y_sol, [len(z_sol), len(x_sol), len(t_sol)]),
        )
        # 2 vectors, 1 scalar
        np.testing.assert_array_equal(
            processed_var(t=0.5, x=x_sol, z=z_sol).shape, (20, 10)
        )
        np.testing.assert_array_equal(
            processed_var(t=t_sol, x=0.2, z=z_sol).shape, (20, 50)
        )
        np.testing.assert_array_equal(
            processed_var(t=t_sol, x=x_sol, z=0.5).shape, (10, 50)
        )
        # 1 vectors, 2 scalar
        np.testing.assert_array_equal(processed_var(t=0.5, x=0.2, z=z_sol).shape, (20,))
        np.testing.assert_array_equal(processed_var(t=0.5, x=x_sol, z=0.5).shape, (10,))
        np.testing.assert_array_equal(processed_var(t=t_sol, x=0.2, z=0.5).shape, (50,))
        # 3 scalars
        np.testing.assert_array_equal(processed_var(t=0.2, x=0.2, z=0.2).shape, ())

    @pytest.mark.parametrize("hermite_interp", _hermite_args)
    def test_3D_raises_error(self, hermite_interp):
        var = pybamm.Variable(
            "var",
            domain=["negative electrode"],
            auxiliary_domains={"secondary": ["current collector"]},
        )

        disc = tests.get_2p1d_discretisation_for_testing()
        disc.set_variable_slices([var])
        var_sol = disc.process_symbol(var)
        t_sol = np.array([0, 1, 2])
        u_sol = np.ones(var_sol.shape[0] * 3)[:, np.newaxis]
        yp_sol = self._get_yps(u_sol, hermite_interp, values=0)
        var_casadi = to_casadi(var_sol, u_sol)

        with pytest.raises(NotImplementedError, match="Shape not recognized"):
            pybamm.process_variable(
                [var_sol],
                [var_casadi],
                self._sol_default(t_sol, u_sol, yp_sol),
            )

    def test_process_spatial_variable_names(self):
        # initialise dummy variable to access method
        t = pybamm.t
        y = pybamm.StateVector(slice(0, 1))
        var = t * y
        var.mesh = None
        t_sol = np.linspace(0, 1)
        y_sol = np.array([np.linspace(0, 5)])
        var_casadi = to_casadi(var, y_sol)
        processed_var = pybamm.process_variable(
            [var],
            [var_casadi],
            pybamm.Solution(t_sol, y_sol, pybamm.BaseModel(), {}),
        )

        # Test empty list returns None
        assert processed_var._process_spatial_variable_names([]) is None

        # Test tabs is ignored
        assert processed_var._process_spatial_variable_names(["tabs", "var"]) == "var"

        # Test strings stay strings
        assert processed_var._process_spatial_variable_names(["y"]) == "y"

        # Test spatial variables are converted to strings
        x = pybamm.SpatialVariable("x", domain=["domain"])
        assert processed_var._process_spatial_variable_names([x]) == "x"

        # Test renaming for PyBaMM convention
        assert processed_var._process_spatial_variable_names(["x_a", "x_b"]) == "x"
        assert processed_var._process_spatial_variable_names(["r_a", "r_b"]) == "r"
        assert processed_var._process_spatial_variable_names(["R_a", "R_b"]) == "R"

        # Test error raised if spatial variable name not recognised
        with pytest.raises(NotImplementedError, match="Spatial variable name"):
            processed_var._process_spatial_variable_names(["var1", "var2"])

    def test_hermite_interpolator(self):
        if not pybamm.has_idaklu():
            pytest.skip("Cannot test Hermite interpolation without IDAKLU")

        # initialise dummy solution to access method
        def solution_setup(t_sol, sign):
            y_sol = np.array([sign * np.sin(t_sol)])
            yp_sol = np.array([sign * np.cos(t_sol)])
            sol = self._sol_default(t_sol, y_sol, yp_sol)
            return sol

        # without spatial dependence
        t = pybamm.t
        y = pybamm.StateVector(slice(0, 1))
        var = y
        eqn = t * y
        var.mesh = None
        eqn.mesh = None

        sign1 = +1
        t_sol1 = np.linspace(0, 1, 100)
        sol1 = solution_setup(t_sol1, sign1)

        # Discontinuity in the solution
        sign2 = -1
        t_sol2 = np.linspace(np.nextafter(t_sol1[-1], np.inf), t_sol1[-1] + 3, 99)
        sol2 = solution_setup(t_sol2, sign2)

        sol = sol1 + sol2
        var_casadi = to_casadi(var, sol.all_ys[0])
        processed_var = pybamm.process_variable(
            [var] * len(sol.all_ts),
            [var_casadi] * len(sol.all_ts),
            sol,
        )

        # Ground truth spline interpolants from scipy
        spls = [
            CubicHermiteSpline(t, y, yp, axis=1)
            for t, y, yp in zip(sol.all_ts, sol.all_ys, sol.all_yps)
        ]

        def spl(t):
            t = np.array(t)
            out = np.zeros(len(t))
            for i, spl in enumerate(spls):
                t0 = sol.all_ts[i][0]
                tf = sol.all_ts[i][-1]

                mask = t >= t0
                # Extrapolation is allowed for the final solution
                if i < len(spls) - 1:
                    mask &= t <= tf

                out[mask] = spl(t[mask]).flatten()
            return out

        t0 = sol.t[0]
        tf = sol.t[-1]

        # Test extrapolation before the first solution time
        t_left_extrap = t0 - 1
        with pytest.raises(
            ValueError, match="interpolation points must be greater than"
        ):
            processed_var(t_left_extrap)

        # Test extrapolation after the last solution time
        t_right_extrap = [tf + 1]
        np.testing.assert_almost_equal(
            spl(t_right_extrap),
            processed_var(t_right_extrap, fill_value="extrapolate"),
            decimal=8,
        )

        t_dense = np.linspace(t0, tf + 1, 1000)
        np.testing.assert_almost_equal(
            spl(t_dense),
            processed_var(t_dense, fill_value="extrapolate"),
            decimal=8,
        )

        t_extended = np.union1d(sol.t, sol.t[-1] + 1)
        np.testing.assert_almost_equal(
            spl(t_extended),
            processed_var(t_extended, fill_value="extrapolate"),
            decimal=8,
        )

        ## Unsorted arrays
        t_unsorted = np.array([0.5, 0.4, 0.6, 0, 1])
        idxs_sort = np.argsort(t_unsorted)
        t_sorted = np.sort(t_unsorted)

        y_sorted = processed_var(t_sorted)

        idxs_unsort = np.zeros_like(idxs_sort)
        idxs_unsort[idxs_sort] = np.arange(len(t_unsorted))

        # Check that the unsorted and sorted arrays are the same
        assert np.all(t_sorted == t_unsorted[idxs_sort])

        y_unsorted = processed_var(t_unsorted)

        # Check that the unsorted and sorted arrays are the same
        assert np.all(y_unsorted == y_sorted[idxs_unsort])
