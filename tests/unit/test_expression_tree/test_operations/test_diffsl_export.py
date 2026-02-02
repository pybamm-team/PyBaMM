#
# Tests for DiffSLExport class
#
import pybamm


class TestDiffSLExport:
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
        assert correct_export == model.to_diffeq(inputs=[], outputs=["x", "y", "z"])

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
        model_output = model.to_diffeq(inputs=[], outputs=["Heat flux", "Temperature"])
        assert correct_export_option_1 == str(
            model_output
        ) or correct_export_option_2 == str(model_output)
