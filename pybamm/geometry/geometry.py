#
# Geometry class for storing the geometry of the model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import pybamm


class Geometry(dict):
    """A geometry class to store the details features of the cell geometry

     **Extends**: :class:`dict`

     Parameters
     ----------

     custom_geometry : dict containing any extra user defined geometry
     """

    # left empty for now but I think we should have methods to
    # add Geometries together (i.e. choose 3D macro with 1D micro etc)


class Geometry1DMacro(Geometry):
    def __init__(self, custom_geometry={}):
        super().__init__()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        x = pybamm.IndependentVariable("x", whole_cell)
        ln = pybamm.standard_parameters.ln
        ls = pybamm.standard_parameters.ls

        self["negative electrode"] = {x: {"min": pybamm.Scalar(0), "max": ln}}
        self["separator"] = {x: {"min": ln, "max": ln + ls}}
        self["positive electrode"] = {x: {"min": ln + ls, "max": pybamm.Scalar(1)}}

        # update with custom geometry if non empty
        self.update(custom_geometry)


class Geometry1DMicro(Geometry):
    def __init__(self, custom_geometry={}):
        super().__init__()

        r = pybamm.IndependentVariable("r", ["negative particle", "positive particle"])

        self["negative particle"] = {
            r: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}
        }
        self["positive particle"] = {
            r: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}
        }
        # update with custom geometry if non empty
        self.update(custom_geometry)


class Geometry3DMacro(Geometry1DMacro):
    """A geometry class to store the details features of the cell geometry

     **Extends**: :class:`dict`

     Parameters
     ----------

     custom_geometry : dict containing any extra user defined geometry
     """

    def __init__(self, custom_geometry={}):
        super().__init__()

        whole_cell = ["negative electrode", "separator", "positive electrode"]
        y = pybamm.IndependentVariable("y", whole_cell)
        z = pybamm.IndependentVariable("z", whole_cell)

        y_lim = {"min": pybamm.Scalar(0), "max": pybamm.standard_parameters.ly}

        z_lim = {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}

        MACRO_DOMAINS = ["negative electrode", "separator", "positive electrode"]
        for domain in MACRO_DOMAINS:
            self[domain][y] = y_lim
            self[domain][z] = z_lim
        self.update(custom_geometry)
