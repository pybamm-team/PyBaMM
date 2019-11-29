from vtk.util import numpy_support
import vtk
import numpy as np

data = np.zeros((3, 3, 3))

x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)
xx, yy, zz = np.meshgrid(x, y, x, sparse=True)
data = np.sin(xx ** 2 + yy ** 2 + zz) / (xx ** 2 + yy ** 2)


# vtkImageData is the vtk image volume type
imdata = vtk.vtkImageData()
# this is where the conversion happens
depthArray = numpy_support.numpy_to_vtk(
    data.ravel(), deep=True, array_type=vtk.VTK_DOUBLE
)

# fill the vtk image data object
imdata.SetDimensions(data.shape)
imdata.SetSpacing([1, 1, 1])
imdata.SetOrigin([0, 0, 0])
imdata.GetPointData().SetScalars(depthArray)

# f.ex. save it as mhd file
writer = vtk.vtkMetaImageWriter()
writer.SetFileName("test.mhd")
writer.SetInputData(imdata)
writer.Write()
