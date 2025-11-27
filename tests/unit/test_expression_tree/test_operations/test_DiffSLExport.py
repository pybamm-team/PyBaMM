#
# Tests for DiffslExport class
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

        model = pybamm.DiffslExport(model)
        correct_export = "in = []\nxinput_i { \n  1.\n}\n\nyinput_i { \n  2.\n}\nu_i {\n  x = xinput_i,\n  y = yinput_i,\n}\nF_i {\n  ((4.0 * x_i) - (2.0 * y_i)),\n  ((3.0 * x_i) - y_i),\n}\nout_i {\n  x_i,\n  y_i,\n  (x_i + (4.0 * y_i)),\n}"
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

        model = pybamm.DiffslExport(model)
        correct_export = """in = []\nconstant0_ij {\n  (0,0): 2.0,\n  (1,1): 1.0,\n  (1,0): -1.0,\n  (2,2): 1.0,\n  (2,1): -1.0,\n  (3,3): 1.0,\n  (3,2): -1.0,\n  (4,4): 1.0,\n  (4,3): -1.0,\n  (5,5): 1.0,\n  (5,4): -1.0,\n  (6,6): 1.0,\n  (6,5): -1.0,\n  (7,7): 1.0,\n  (7,6): -1.0,\n  (8,8): 1.0,\n  (8,7): -1.0,\n  (9,9): 1.0,\n  (9,8): -1.0,\n  (10,9): -2.0,\n}\nconstant1_ij {\n  (0,1): 1.0,\n  (0,0): -3.0,\n  (1,2): 1.0,\n  (1,0): 1.0,\n  (1,1): -2.0,\n  (2,3): 1.0,\n  (2,1): 1.0,\n  (2,2): -2.0,\n  (3,4): 1.0,\n  (3,2): 1.0,\n  (3,3): -2.0,\n  (4,5): 1.0,\n  (4,3): 1.0,\n  (4,4): -2.0,\n  (5,6): 1.0,\n  (5,4): 1.0,\n  (5,5): -2.0,\n  (6,7): 1.0,\n  (6,5): 1.0,\n  (6,6): -2.0,\n  (7,8): 1.0,\n  (7,6): 1.0,\n  (7,7): -2.0,\n  (8,9): 1.0,\n  (8,7): 1.0,\n  (8,8): -2.0,\n  (9,8): 1.0,\n  (9,9): -3.0,\n}\ntemperatureinput_i { \n  0.5,\n  1.5,\n  2.5,\n  3.5,\n  4.5,\n  5.5,\n  6.5,\n  7.5,\n  8.5,\n  9.5\n}\nu_i {\n  temperature = temperatureinput_i,\n}\nvarying0_i {\n  (constant0_ij * temperature_j),\n}\nF_i {\n  (constant1_ij * temperature_j),\n}\nout_i {\n  -(varying0_i),\n  temperature_i,\n}"""
        assert correct_export == model.to_diffeq(
            inputs=[], outputs=["Heat flux", "Temperature"]
        )
