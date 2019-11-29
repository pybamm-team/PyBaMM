import vtk
from vtk.util import numpy_support
import numpy as np

import pybamm
import models

from pyevtk.hl import gridToVTK


t_eval = np.linspace(0, 0.17, 100)
C_rate = 1
var_pts = {
    pybamm.standard_spatial_vars.x_n: 5,
    pybamm.standard_spatial_vars.x_s: 5,
    pybamm.standard_spatial_vars.x_p: 5,
    pybamm.standard_spatial_vars.y: 5,
    pybamm.standard_spatial_vars.z: 5,
    pybamm.standard_spatial_vars.r_n: 5,
    pybamm.standard_spatial_vars.r_p: 5,
}

spmecc = models.solve_spmecc(t_eval=t_eval, var_pts=var_pts, C_rate=C_rate)

x = np.linspace(0, 1, 10)
y = np.linspace(0, 1.5, 100)
z = np.linspace(0, 1, 100)
NumPy_data = spmecc["Negative current collector potential [V]"](t=0, y=y, z=z)

juliaStacked = np.dstack(NumPy_data)

gridToVTK("./julia", x, y, z, cellData={"julia": juliaStacked})

# writer = vtk.vtkUnstructuredGridWriter()
# writer.SetInputData(VTK_data)
# writer.SetFileName("test.vtk")
# writer.Write()
