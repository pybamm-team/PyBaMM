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

    def test_model(self, model):
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

    def test_ode(self):
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
        correct_export = "constant0_i {\n  (0:1): 1,\n}\nconstant1_i {\n  (0:1): 2,\n}\nu_i {\n  x = constant0_i,\n  y = constant1_i,\n}\nF_i {\n  ((4 * x_i) - (2 * y_i)),\n  ((3 * x_i) - y_i),\n}\nout_i {\n  x_i,\n  y_i,\n  (x_i + (4 * y_i)),\n}"
        assert correct_export == model.to_diffeq(outputs=["x", "y", "z"])

    def test_heat_equation(self):
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

        # The order of the matrix entries is not deterministic, so we need to check both possible options
        correct_export_option_1 = "constant0_ij {\n  (0,0): 2,\n  (1,0): -1,\n  (1,1): 1,\n  (2,1): -1,\n  (2,2): 1,\n  (3,2): -1,\n  (3,3): 1,\n  (4,3): -1,\n  (4,4): 1,\n  (5,4): -1,\n  (5,5): 1,\n  (6,5): -1,\n  (6,6): 1,\n  (7,6): -1,\n  (7,7): 1,\n  (8,7): -1,\n  (8,8): 1,\n  (9,8): -1,\n  (9,9): 1,\n  (10,9): -2,\n}\nconstant1_ij {\n  (0,0): -3,\n  (0,1): 1,\n  (1,1): -2,\n  (1,0): 1,\n  (1,2): 1,\n  (2,2): -2,\n  (2,1): 1,\n  (2,3): 1,\n  (3,3): -2,\n  (3,2): 1,\n  (3,4): 1,\n  (4,4): -2,\n  (4,3): 1,\n  (4,5): 1,\n  (5,5): -2,\n  (5,4): 1,\n  (5,6): 1,\n  (6,6): -2,\n  (6,5): 1,\n  (6,7): 1,\n  (7,7): -2,\n  (7,6): 1,\n  (7,8): 1,\n  (8,8): -2,\n  (8,7): 1,\n  (8,9): 1,\n  (9,9): -3,\n  (9,8): 1,\n}\nconstant2_i {\n  (0:1): 0.5,\n  (1:2): 1.5,\n  (2:3): 2.5,\n  (3:4): 3.5,\n  (4:5): 4.5,\n  (5:6): 5.5,\n  (6:7): 6.5,\n  (7:8): 7.5,\n  (8:9): 8.5,\n  (9:10): 9.5,\n}\nu_i {\n  temperature = constant2_i,\n}\nF_i {\n  (constant1_ij * temperature_j),\n}\nvariable3_i {\n  (constant0_ij * temperature_j),\n}\nout_i {\n  -(variable3_i),\n  temperature_i,\n}"
        correct_export_option_2 = "constant0_ij {\n  (0,0): -3,\n  (0,1): 1,\n  (1,1): -2,\n  (1,0): 1,\n  (1,2): 1,\n  (2,2): -2,\n  (2,1): 1,\n  (2,3): 1,\n  (3,3): -2,\n  (3,2): 1,\n  (3,4): 1,\n  (4,4): -2,\n  (4,3): 1,\n  (4,5): 1,\n  (5,5): -2,\n  (5,4): 1,\n  (5,6): 1,\n  (6,6): -2,\n  (6,5): 1,\n  (6,7): 1,\n  (7,7): -2,\n  (7,6): 1,\n  (7,8): 1,\n  (8,8): -2,\n  (8,7): 1,\n  (8,9): 1,\n  (9,9): -3,\n  (9,8): 1,\n}\nconstant1_ij {\n  (0,0): 2,\n  (1,0): -1,\n  (1,1): 1,\n  (2,1): -1,\n  (2,2): 1,\n  (3,2): -1,\n  (3,3): 1,\n  (4,3): -1,\n  (4,4): 1,\n  (5,4): -1,\n  (5,5): 1,\n  (6,5): -1,\n  (6,6): 1,\n  (7,6): -1,\n  (7,7): 1,\n  (8,7): -1,\n  (8,8): 1,\n  (9,8): -1,\n  (9,9): 1,\n  (10,9): -2,\n}\nconstant2_i {\n  (0:1): 0.5,\n  (1:2): 1.5,\n  (2:3): 2.5,\n  (3:4): 3.5,\n  (4:5): 4.5,\n  (5:6): 5.5,\n  (6:7): 6.5,\n  (7:8): 7.5,\n  (8:9): 8.5,\n  (9:10): 9.5,\n}\nu_i {\n  temperature = constant2_i,\n}\nF_i {\n  (constant0_ij * temperature_j),\n}\nvariable3_i {\n  (constant1_ij * temperature_j),\n}\nout_i {\n  -(variable3_i),\n  temperature_i,\n}"
        model_output = model.to_diffeq(outputs=["Heat flux", "Temperature"])
        assert correct_export_option_1 == str(
            model_output
        ) or correct_export_option_2 == str(model_output)
