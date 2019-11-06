import pybamm
import numpy as np
import matplotlib.pyplot as plt

# Set up model ----------------------------------------------------------------
model = pybamm.BaseModel()
a = 3
b = 4
c = 5
var = pybamm.standard_spatial_vars
u = pybamm.Variable("variable", domain="current collector")
y = var.y
z = var.z
source = (np.pi**2 * (a*z**2 + b*z + c) - 2*a) * pybamm.cos(np.pi*y)
model.algebraic = {u: pybamm.laplacian(u) + pybamm.source(source, u)}
# set boundary conditions ("negative tab" = bottom of unit square,
# "positive tab" = top of unit square, elsewhere normal derivative is zero)
model.boundary_conditions = {
            u: {
                "negative tab": (pybamm.Scalar(c) * pybamm.cos(np.pi*y), "Dirichlet"),
                "positive tab": (pybamm.Scalar(a+b+c) * pybamm.cos(np.pi*y), "Dirichlet"),
            }
        }
# bad initial guess
model.initial_conditions = {u: pybamm.Scalar(1)}
model.variables = {"u": u}

# Set up params, geometry and mesh---------------------------------------------
param = pybamm.ParameterValues(
    values={
        "Electrode width [m]": 1,
        "Electrode height [m]": 1,
        "Negative tab width [m]": 1,
        "Negative tab centre y-coordinate [m]": 0.5,
        "Negative tab centre z-coordinate [m]": 0,
        "Positive tab width [m]": 1,
        "Positive tab centre y-coordinate [m]": 0.5,
        "Positive tab centre z-coordinate [m]": 1,
        "Negative electrode thickness [m]": 0.3,
        "Separator thickness [m]": 0.3,
        "Positive electrode thickness [m]": 0.3,
     }
)

geometry = pybamm.Geometryxp1DMacro(cc_dimension=2)
param.process_geometry(geometry)

var_pts = {var.x_n: 3, var.x_s: 3, var.x_p: 3, var.y: 32, var.z: 32}

submesh_types = {
  "negative electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
  "separator": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
  "positive electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
  "current collector": pybamm.MeshGenerator(pybamm.ScikitUniform2DSubMesh),
}

mesh =  pybamm.Mesh(geometry, submesh_types, var_pts)
spatial_methods = {
    "macroscale": pybamm.FiniteVolume,
    "current collector": pybamm.ScikitFiniteElement,
}
disc = pybamm.Discretisation(mesh, spatial_methods)
disc.process_model(model)

# Solve -----------------------------------------------------------------------
solver = pybamm.AlgebraicSolver()
solution = solver.solve(model)

# Compare solution ------------------------------------------------------------
y = mesh["current collector"][0].coordinates[0, :][:, np.newaxis]
z = mesh["current collector"][0].coordinates[1, :][:, np.newaxis]
u_exact = (a*z**2 + b*z + c)* np.cos(np.pi*y)
print(np.linalg.norm(u_exact - solution.y))
plt.plot(u_exact, 'o')
plt.plot(solution.y, '-')
plt.show()
