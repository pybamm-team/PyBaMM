#
# Tests for DiffSLExport class
#
import numpy as np
import pytest
import scipy.sparse

import pybamm


class TestDiffSLExport:
    # fixture of a model with all the characteristics we want to test
    @pytest.fixture
    def model(self):
        model = pybamm.BaseModel()

        x = pybamm.StateVector(slice(0, 2), domain="negative electrode")
        y = pybamm.StateVector(slice(2, 4), domain="positive electrode")
        z = pybamm.StateVector(slice(4, 5))
        z2 = pybamm.StateVector(slice(5, 7))
        A = pybamm.Matrix(
            scipy.sparse.csr_matrix(([4.12345, 4.12345], [0, 1], [0, 1, 2]))
        )
        B = pybamm.Matrix(np.array([[4, -2], [3, -1]]))
        C = pybamm.Matrix(scipy.sparse.csr_matrix(([1], [0], [0, 1, 1]), shape=(2, 2)))
        b = pybamm.Matrix(scipy.sparse.csr_matrix(([2, 2], [0, 1], [0, 2])))
        c = pybamm.Vector(scipy.sparse.csr_matrix(([1], [0], [0, 1, 1])))
        d = pybamm.Matrix(np.array([2, 3]).reshape((1, 2)))
        u = pybamm.StateVector(slice(0, 4))
        p = pybamm.InputParameter("p")
        dup = pybamm.maximum(x * x, 0)
        dxdt = A @ dup + c + b @ dup + C @ dup + d @ x + A @ pybamm.minimum(x, 0)
        dydt = A @ y + y @ z + pybamm.cos(p**2) + pybamm.t

        x_n = pybamm.SpatialVariable("x_n", domain="negative electrode")
        x_p = pybamm.SpatialVariable("x_p", domain="positive electrode")
        geometry = pybamm.Geometry(
            {
                "negative electrode": {
                    x_n: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}
                },
                "positive electrode": {
                    x_p: {"min": pybamm.Scalar(1), "max": pybamm.Scalar(2)}
                },
            }
        )
        submesh_types = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
        }
        var_pts = {x_n: 2, x_p: 2}
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
        dudt = pybamm.DomainConcatenation([dxdt, dydt], mesh)

        model.rhs = {u: u * dudt + u / dudt}
        model.algebraic = {z: z, z2: z2}
        model.initial_conditions = {
            u: pybamm.Vector(np.array([1, 2, 1, 2])),
            z: pybamm.Scalar(0),
            z2: pybamm.Vector(np.array([0, 0])),
        }
        model.variables = {"x": x, "y": y, "z": z}
        model.events = [pybamm.Event("event1", x - B @ x)]
        return model

    def test_model(self, model, snapshot):
        export = pybamm.DiffSLExport(model, float_precision=6).to_diffeq(outputs=["x"])
        assert "u_i" in export
        assert "event" in export
        assert "constant" in export
        assert "varying" in export
        assert "dudt_i" in export
        assert "F_i" in export
        assert "out_i" in export
        assert "stop_i" in export
        assert "in_i" in export
        assert "cos" in export
        assert "max" in export
        assert "min" in export
        snapshot.assert_match(export, "diffsl_export.snapshot")

    def test_float_precision(self, model):
        export = pybamm.DiffSLExport(model, float_precision=6).to_diffeq(outputs=["x"])
        assert "4.12345" in export
        export = pybamm.DiffSLExport(model, float_precision=2).to_diffeq(outputs=["x"])
        assert "4.12345" not in export
        assert "4.1" in export
        with pytest.raises(ValueError):
            model = pybamm.DiffSLExport(model, float_precision=-1)

    def test_inputs(self, model):
        with pytest.raises(TypeError):
            pybamm.DiffSLExport(model).to_diffeq(outputs="not a list")
        with pytest.raises(ValueError):
            pybamm.DiffSLExport(model).to_diffeq(outputs=["not in model"])
        with pytest.raises(ValueError):
            pybamm.DiffSLExport(model).to_diffeq(outputs=[])

    def test_ode(self, snapshot):
        model = pybamm.BaseModel()

        x = pybamm.Variable("x")
        y = pybamm.Variable("y")

        dxdt = 4 * x - 2 * y
        dydt = 3 * x - y

        model.rhs = {x: dxdt, y: dydt}
        model.initial_conditions = {x: pybamm.Scalar(1), y: pybamm.Scalar(2)}
        model.variables = {"x": x, "y": y, "z": x + 4 * y}

        disc = pybamm.Discretisation()
        model = disc.process_model(model)

        model = pybamm.DiffSLExport(model)
        snapshot.assert_match(
            model.to_diffeq(outputs=["x", "y", "z"]), "diffsl_export_ode.snapshot"
        )

    def test_heat_equation(self, snapshot):
        model = pybamm.BaseModel()

        x = pybamm.SpatialVariable("x", domain="rod", coord_sys="cartesian")
        T = pybamm.Variable("Temperature", domain="rod")
        k = pybamm.Parameter("Thermal diffusivity")

        N = -k * pybamm.grad(T)
        dTdt = -pybamm.div(N)
        model.rhs = {T: dTdt}

        model.boundary_conditions = {
            T: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(0), "Dirichlet"),
            }
        }

        model.initial_conditions = {T: x}
        model.variables = {"Temperature": T, "Heat flux": N}
        geometry = {"rod": {x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(10)}}}
        param = pybamm.ParameterValues({"Thermal diffusivity": 1.0})
        param.process_model(model)
        param.process_geometry(geometry)

        submesh_types = {"rod": pybamm.Uniform1DSubMesh}
        var_pts = {x: 10}
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
        spatial_methods = {"rod": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        disc.process_model(model)

        model = pybamm.DiffSLExport(model)
        snapshot.assert_match(
            model.to_diffeq(outputs=["Heat flux", "Temperature"]),
            "diffsl_export_heat_equation.snapshot",
        )

    def test_reg_power_and_arcsinh2_export(self):
        model = pybamm.BaseModel()

        x = pybamm.Variable("x")
        y = pybamm.Variable("y")
        special = pybamm.reg_power(x, 2, scale=3) + pybamm.arcsinh2(x, y, eps=1e-4)

        model.rhs = {x: special, y: x - y}
        model.initial_conditions = {x: pybamm.Scalar(1), y: pybamm.Scalar(2)}
        model.variables = {"special": special}

        disc = pybamm.Discretisation()
        model = disc.process_model(model)

        export = pybamm.DiffSLExport(model).to_diffeq(outputs=["special"])

        # reg_power branch in diffsl export includes the scale^a factor
        assert "* pow(3, 2)" in export
        # arcsinh2 branch in diffsl export uses regularised denominator form
        assert "copysign(sqrt(pow(" in export

    def test_simulation_auto_build_export_matches_built_model(self):
        model = pybamm.BaseModel()

        x = pybamm.Variable("x")
        y = pybamm.Variable("y")

        model.rhs = {x: 4 * x - 2 * y, y: 3 * x - y}
        model.initial_conditions = {x: pybamm.Scalar(1), y: pybamm.Scalar(2)}
        model.variables = {"x": x, "y": y}

        sim = pybamm.Simulation(model)

        export_from_sim = pybamm.DiffSLExport(sim).to_diffeq(outputs=["x"])

        assert sim.built_model is not None

        export_from_model = pybamm.DiffSLExport(sim.built_model).to_diffeq(
            outputs=["x"]
        )

        assert export_from_sim == export_from_model

    def test_simulation_with_legacy_experiment_errors(self):
        experiment = pybamm.Experiment(
            [
                "Discharge at C/20 for 1 hour",
                "Rest for 10 minutes",
            ]
        )
        sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            experiment=experiment,
            solver=pybamm.CasadiSolver(),
            experiment_model_mode="legacy",
        )

        with pytest.raises(ValueError, match="experiment_model_mode='unified'"):
            pybamm.DiffSLExport(sim).to_diffeq(outputs=["Voltage [V]"])

    def test_simulation_with_unified_experiment_uses_model_index(self):
        experiment = pybamm.Experiment(
            [
                "Discharge at C/20 for 1 hour",
                "Rest for 10 minutes",
            ]
        )
        sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            experiment=experiment,
            solver=pybamm.CasadiSolver(),
            experiment_model_mode="unified",
        )
        sim.build_for_experiment()

        output_name = next(iter(sim._built_experiment_model.variables))
        export = pybamm.DiffSLExport(sim).to_diffeq(outputs=[output_name])

        assert sim._built_experiment_model is not None
        assert "experimentstepindex" not in export
        assert "[N]" in export

    def test_conditional_scalar_export_uses_branch_vector(self):
        model = pybamm.BaseModel()

        x = pybamm.Variable("x")
        selector = pybamm.InputParameter(pybamm.Simulation._STEP_INDEX_INPUT)
        special = pybamm.Conditional(selector, x + 1, x + 2)

        model.rhs = {x: special}
        model.initial_conditions = {x: pybamm.Scalar(1)}
        model.variables = {"special": special}

        disc = pybamm.Discretisation()
        model = disc.process_model(model)

        export = pybamm.DiffSLExport(model).to_diffeq(outputs=["special"])

        assert "0.0," in export
        assert "[N]" in export

    def test_conditional_vector_export_not_implemented(self):
        model = pybamm.BaseModel()

        u = pybamm.StateVector(slice(0, 2))
        selector = pybamm.InputParameter(pybamm.Simulation._STEP_INDEX_INPUT)
        special = pybamm.Conditional(
            selector,
            pybamm.Vector(np.array([1, 2])),
            pybamm.Vector(np.array([3, 4])),
        )

        model.rhs = {u: special}
        model.initial_conditions = {u: pybamm.Vector(np.array([0, 0]))}
        model.variables = {"special": special}

        with pytest.raises(
            NotImplementedError, match="only supports scalar Conditionals"
        ):
            pybamm.DiffSLExport(model).to_diffeq(outputs=["special"])
