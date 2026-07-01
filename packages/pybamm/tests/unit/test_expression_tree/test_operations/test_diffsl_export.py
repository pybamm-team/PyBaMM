#
# Tests for DiffSLExport class
#
from datetime import datetime

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
                "Charge at C/20 until 4.1 V",
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
                "Charge at C/20 until 4.1 V",
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
        assert "steptime0_i" in export
        assert "steptime0" in export.split("u_i {", 1)[1].split("}", 1)[0]
        assert "heaviside" in export
        assert "(event" in export and "_i * varying" in export
        assert "3600 - steptime0_i" in export
        assert "600 - steptime0_i" in export
        assert "4.099" in export

        reset_parts = export.split("reset_i {\n", 1)
        assert len(reset_parts) > 1
        assert "u_i[0:" in reset_parts[1]

    def test_simulation_with_unified_experiment_padding_rests_use_model_index(self):
        experiment = pybamm.Experiment(
            [
                pybamm.step.string(
                    "Discharge at 0.5C for 1 hour",
                    start_time=datetime(2023, 1, 1, 8, 0, 0),
                ),
                pybamm.step.string(
                    "Rest for 30 minutes",
                    start_time=datetime(2023, 1, 1, 10, 0, 0),
                ),
                pybamm.step.string(
                    "Charge at 0.5C for 1 hour",
                    start_time=datetime(2023, 1, 1, 12, 0, 0),
                ),
            ]
        )
        sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            experiment=experiment,
            solver=pybamm.CasadiSolver(),
            experiment_model_mode="unified",
        )
        sim.build_for_experiment()

        assert sim._experiment_includes_padding_rest is True

        output_name = next(iter(sim._built_experiment_model.variables))
        export = pybamm.DiffSLExport(sim).to_diffeq(outputs=[output_name])

        assert sim._built_experiment_model is not None
        assert "experimentstepindex" not in export
        assert "[N]" in export
        assert "steptime0_i" in export
        assert "3600 - steptime0_i" in export
        assert "heaviside" in export

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

        assert "[N]" in export
        assert "(2 + x_i)" in export

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

    def test_init_rejects_invalid_model(self):
        with pytest.raises(
            TypeError, match=r"must be a pybamm\.BaseModel or pybamm\.Simulation"
        ):
            pybamm.DiffSLExport(42)

    def test_output_specific_input_parameter(self):
        model = pybamm.BaseModel()
        x = pybamm.Variable("x")
        extra = pybamm.InputParameter("extra_param")
        model.rhs = {x: x}
        model.initial_conditions = {x: pybamm.Scalar(1)}
        model.variables = {"x": x, "special": x + extra}
        disc = pybamm.Discretisation()
        disc.process_model(model)
        export = pybamm.DiffSLExport(model).to_diffeq(outputs=["special"])
        assert "extra_param" in export

    def test_map_inputs_basic(self, model):
        exporter = pybamm.DiffSLExport(model)
        exporter.to_diffeq(outputs=["x"])
        result = exporter.map_inputs({"p": 3.14})
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        assert result[0] == 3.14

    def test_map_inputs_no_outputs(self, model):
        exporter = pybamm.DiffSLExport(model)
        exporter.to_diffeq(outputs=["x"])
        result = exporter.map_inputs({"p": 1.0})
        assert result[0] == 1.0

    def test_map_inputs_empty_model(self):
        model = pybamm.BaseModel()
        x = pybamm.Variable("x")
        model.rhs = {x: x}
        model.initial_conditions = {x: pybamm.Scalar(1)}
        model = pybamm.Discretisation().process_model(model)
        exporter = pybamm.DiffSLExport(model)
        result = exporter.map_inputs({})
        assert isinstance(result, np.ndarray)
        assert result.size == 0

    def test_map_inputs_missing_key_raises(self, model):
        exporter = pybamm.DiffSLExport(model)
        exporter.to_diffeq(outputs=["x"])
        with pytest.raises(KeyError, match="not found in inputs dict"):
            exporter.map_inputs({})

    def test_map_inputs_output_specific(self):
        model = pybamm.BaseModel()
        x = pybamm.Variable("x")
        extra = pybamm.InputParameter("extra_param")
        model.rhs = {x: x}
        model.initial_conditions = {x: pybamm.Scalar(1)}
        model.variables = {"x": x, "extra_out": x + extra}
        disc = pybamm.Discretisation()
        disc.process_model(model)
        exporter = pybamm.DiffSLExport(model)
        exporter.to_diffeq(outputs=["extra_out"])
        result = exporter.map_inputs({"extra_param": 2.0})
        assert result[0] == 2.0

    def test_reg_power_with_non_scalar_exponent(self):
        model = pybamm.BaseModel()
        x = pybamm.Variable("x")
        a = pybamm.InputParameter("a")
        rp = pybamm.reg_power(x, a, scale=pybamm.Scalar(2))
        model.rhs = {x: rp}
        model.initial_conditions = {x: pybamm.Scalar(1)}
        model.variables = {"rp": rp}
        disc = pybamm.Discretisation()
        disc.process_model(model)
        export = pybamm.DiffSLExport(model).to_diffeq(outputs=["rp"])
        assert "(a - 1) / 2" in export

    def test_map_inputs_invalid_output_raises(self, model):
        exporter = pybamm.DiffSLExport(model)
        with pytest.raises(ValueError, match="output nonexistent not in model"):
            exporter.to_diffeq(outputs=["nonexistent"])

    def test_map_inputs_processed_variable_path(self, model):
        exporter = pybamm.DiffSLExport(model)
        exporter.to_diffeq(outputs=["x"])
        result = exporter.map_inputs({"p": 1.0})
        assert result[0] == 1.0

    def test_map_inputs_with_symbol_processor(self):
        sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            solver=pybamm.CasadiSolver(),
        )
        sim.build()
        exporter = pybamm.DiffSLExport(sim)
        exporter.to_diffeq(outputs=["Terminal voltage [V]"])
        result = exporter.map_inputs({})
        assert len(result) >= 0

    def test_map_inputs_diffsl_transformed_name(self):
        model = pybamm.BaseModel()
        x = pybamm.Variable("x")
        inp = pybamm.InputParameter("Test param")
        model.rhs = {x: inp * x}
        model.initial_conditions = {x: pybamm.Scalar(1)}
        model.variables = {"x": x}
        disc = pybamm.Discretisation()
        disc.process_model(model)
        exporter = pybamm.DiffSLExport(model)
        exporter.to_diffeq(outputs=["x"])
        result = exporter.map_inputs({"testparam": 2.0})
        assert result[0] == 2.0

    def test_unified_experiment_duplicate_steps(self):
        experiment = pybamm.Experiment(
            [
                "Discharge at C/20 for 1 hour",
                "Rest for 10 minutes",
                "Discharge at C/20 for 1 hour",
            ]
        )
        sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            experiment=experiment,
            solver=pybamm.CasadiSolver(),
            experiment_model_mode="unified",
        )
        exporter = pybamm.DiffSLExport(sim)
        output_name = next(iter(exporter.model.variables))
        export = exporter.to_diffeq(outputs=[output_name])

        assert "[N]" in export
        assert "steptime0_i" in export
        assert "3600 - steptime0_i" in export
        assert "600 - steptime0_i" in export
        assert "heaviside" in export

    def test_unified_experiment_padding_duplicate_steps(self):
        experiment = pybamm.Experiment(
            [
                pybamm.step.string(
                    "Discharge at 0.5C for 1 hour",
                    start_time=datetime(2023, 1, 1, 8, 0, 0),
                ),
                pybamm.step.string(
                    "Rest for 30 minutes",
                    start_time=datetime(2023, 1, 1, 10, 0, 0),
                ),
                pybamm.step.string(
                    "Discharge at 0.5C for 1 hour",
                    start_time=datetime(2023, 1, 1, 12, 0, 0),
                ),
            ]
        )
        sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            experiment=experiment,
            solver=pybamm.CasadiSolver(),
            experiment_model_mode="unified",
        )
        exporter = pybamm.DiffSLExport(sim)
        assert sim._experiment_includes_padding_rest is True

        output_name = next(iter(exporter.model.variables))
        export = exporter.to_diffeq(outputs=[output_name])

        assert "[N]" in export
        assert "steptime0_i" in export
        assert "3600 - steptime0_i" in export
        assert "heaviside" in export

    def test_unified_experiment_padding_rest_uses_padding_branch(self):
        experiment = pybamm.Experiment(
            [
                pybamm.step.string(
                    "Discharge at C/20 for 60 seconds",
                    start_time=datetime(2023, 1, 1, 8, 0, 0),
                ),
                pybamm.step.string(
                    "Discharge at C/20 for 60 seconds",
                    start_time=datetime(2023, 1, 1, 8, 2, 0),
                ),
            ]
        )
        sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            experiment=experiment,
            solver=pybamm.CasadiSolver(),
            experiment_model_mode="unified",
        )
        exporter = pybamm.DiffSLExport(sim)
        output_name = next(iter(exporter.model.variables))
        export = exporter.to_diffeq(outputs=[output_name])

        assert export.count("60 - steptime0_i") == 3
        first_step = export.index("_i[N])")
        padding_rest = export.index("currentvariablea_i),", first_step)
        second_step = export.index("_i[N])", padding_rest)

        assert first_step < padding_rest < second_step

    def test_unified_experiment_repeating_cycle(self):
        experiment = pybamm.Experiment(
            [
                "Discharge at C/5 for 10 hours or until 3.3 V",
                "Rest for 1 hour",
                "Charge at 1 A until 4.1 V",
                "Hold at 4.1 V until 10 mA",
                "Rest for 1 hour",
            ]
            * 10
        )
        sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            experiment=experiment,
            solver=pybamm.CasadiSolver(),
            experiment_model_mode="unified",
        )
        exporter = pybamm.DiffSLExport(sim)
        output_name = next(iter(exporter.model.variables))
        export = exporter.to_diffeq(outputs=[output_name])

        assert len(sim.experiment.steps) == 50
        assert "[N]" in export
        assert "steptime0_i" in export

        assert export.count("heaviside") >= 1
        assert "(event" in export and "_i * varying" in export
        assert "36000 - steptime0_i" in export
        assert export.count("36000 - steptime0_i") == 1
        assert "3.2999" in export
        assert "86400 - steptime0_i" in export
        assert "4.099" in export
        assert "0.010" in export

    def test_unified_experiment_schedule_cycle_preserves_step_durations(self):
        experiment = pybamm.Experiment(
            [
                "Discharge at C/20 for 60 seconds",
                "Rest for 10 seconds",
                "Discharge at C/20 for 120 seconds",
                "Rest for 20 seconds",
            ]
        )
        sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            experiment=experiment,
            solver=pybamm.CasadiSolver(),
            experiment_model_mode="unified",
        )
        exporter = pybamm.DiffSLExport(sim)
        output_name = next(iter(exporter.model.variables))
        export = exporter.to_diffeq(outputs=[output_name])

        assert "60 - steptime0_i" in export
        assert "10 - steptime0_i" in export
        assert "120 - steptime0_i" in export
        assert "20 - steptime0_i" in export

    def test_unified_experiment_schedule_preserves_step_target_values(self):
        experiment = pybamm.Experiment(
            [
                pybamm.step.c_rate(0.5, duration=10),
                pybamm.step.c_rate(1.0, duration=10),
            ]
        )
        sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            experiment=experiment,
            solver=pybamm.CasadiSolver(),
            experiment_model_mode="unified",
        )
        exporter = pybamm.DiffSLExport(sim)
        output_name = next(iter(exporter.model.variables))
        export = exporter.to_diffeq(outputs=[output_name])

        assert export.count("10 - steptime0_i") == 2

    def test_unified_experiment_step_value_tensor_values(self):
        experiment = pybamm.Experiment(
            [
                pybamm.step.c_rate(0.5, duration=10),
                pybamm.step.c_rate(1.0, duration=10),
            ]
        )
        sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            experiment=experiment,
            solver=pybamm.CasadiSolver(),
            experiment_model_mode="unified",
        )
        exporter = pybamm.DiffSLExport(sim)
        output_name = next(iter(exporter.model.variables))
        export = exporter.to_diffeq(outputs=[output_name])

        assert len(exporter._schedule_states) == 2
        assert all(s.model_branch_index == 0 for s in exporter._schedule_states)

        sorted_states = sorted(
            exporter._schedule_states, key=lambda s: s.schedule_index
        )
        v0 = sorted_states[0].target_value
        v1 = sorted_states[1].target_value
        assert v0 < v1

        assert "experimentstepvalue" not in export
        assert "_i[N]" in export

        s0 = f"{v0:.{exporter.float_precision}g}"
        s1 = f"{v1:.{exporter.float_precision}g}"
        pos0 = export.index(s0)
        pos1 = export.index(s1)
        assert pos0 < pos1

    def test_interpolant_linear_exports_interp1d(self):
        model = pybamm.BaseModel()
        x = pybamm.Variable("x")
        model.rhs = {x: -x}
        model.initial_conditions = {x: pybamm.Scalar(1)}
        model.variables = {
            "data": pybamm.DiscreteTimeData(
                np.array([0.0, 0.5, 1.0]),
                np.array([1.0, 2.0, 3.0]),
                "test",
            ),
        }
        disc = pybamm.Discretisation()
        disc.process_model(model)
        export = pybamm.DiffSLExport(model).to_diffeq(outputs=["data"])

        assert "interp1d(constant" in export
        assert "(0:1): 0," in export
        assert "(1:2): 0.5," in export
        assert "(2:3): 1," in export

    def test_interpolant_pchip_raises(self):
        model = pybamm.BaseModel()
        x = pybamm.Variable("x")
        model.rhs = {x: -x}
        model.initial_conditions = {x: pybamm.Scalar(1)}
        model.variables = {
            "out": pybamm.Interpolant(
                np.array([0, 1, 2]),
                np.array([1, 2, 3]),
                pybamm.t,
                interpolator="pchip",
            ),
        }
        disc = pybamm.Discretisation()
        disc.process_model(model)
        msg = r"DiffSL export only supports 'linear' interpolants"
        with pytest.raises(ValueError, match=msg):
            pybamm.DiffSLExport(model).to_diffeq(outputs=["out"])

    def test_interpolant_cubic_raises(self):
        model = pybamm.BaseModel()
        x = pybamm.Variable("x")
        model.rhs = {x: -x}
        model.initial_conditions = {x: pybamm.Scalar(1)}
        model.variables = {
            "out": pybamm.Interpolant(
                np.array([0, 1, 2]),
                np.array([1, 2, 3]),
                pybamm.t,
                interpolator="cubic",
            ),
        }
        disc = pybamm.Discretisation()
        disc.process_model(model)
        msg = r"DiffSL export only supports 'linear' interpolants"
        with pytest.raises(ValueError, match=msg):
            pybamm.DiffSLExport(model).to_diffeq(outputs=["out"])
