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

    def __init__(self, *geometries, custom_geometry={}):
        for geometry in geometries:
            if geometry == "1D macro":
                geometry = Geometry1DMacro()
            elif geometry == "3D macro":
                geometry = Geometry3DMacro()
            elif geometry == "1D micro":
                geometry = Geometry1DMicro()
            elif geometry == "1+1D micro":
                geometry = Geometry1p1DMicro()
            # avoid combining geometries that clash
            if any([k in self.keys() for k in geometry.keys()]):
                raise ValueError("trying to overwrite existing geometry")
            self.update(geometry)
        # Allow overwriting with a custom geometry
        self.update(custom_geometry)


class Geometry1DMacro(Geometry):
    def __init__(self, custom_geometry={}):
        super().__init__()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        x = pybamm.SpatialVariable("x", whole_cell)
        l_n = pybamm.standard_parameters.l_n
        l_s = pybamm.standard_parameters.l_s

        self["negative electrode"] = {
            "primary": {x: {"min": pybamm.Scalar(0), "max": l_n}}
        }
        self["separator"] = {"primary": {x: {"min": l_n, "max": l_n + l_s}}}
        self["positive electrode"] = {
            "primary": {x: {"min": l_n + l_s, "max": pybamm.Scalar(1)}}
        }

        # update with custom geometry if non empty
        self.update(custom_geometry)


class Geometry1DMicro(Geometry):
    def __init__(self, custom_geometry={}):
        super().__init__()

        r_n = pybamm.SpatialVariable("r", ["negative particle"])
        r_p = pybamm.SpatialVariable("r", ["positive particle"])

        self["negative particle"] = {
            "primary": {r_n: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
        }
        self["positive particle"] = {
            "primary": {r_p: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
        }
        # update with custom geometry if non empty
        self.update(custom_geometry)


class Geometry1p1DMicro(Geometry):
    def __init__(self, custom_geometry={}):
        super().__init__()

        whole_cell = ["negative electrode", "separator", "positive electrode"]
        x = pybamm.SpatialVariable("x", whole_cell)
        r_n = pybamm.SpatialVariable("r", ["negative particle"])
        r_p = pybamm.SpatialVariable("r", ["positive particle"])
        l_n = pybamm.standard_parameters.l_n
        l_s = pybamm.standard_parameters.l_s

        self["negative particle"] = {
            "primary": {r_n: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}},
            "secondary": {x: {"min": pybamm.Scalar(0), "max": l_n}},
        }
        self["positive particle"] = {
            "primary": {r_p: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}},
            "secondary": {x: {"min": l_n + l_s, "max": pybamm.Scalar(1)}},
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
        y = pybamm.SpatialVariable("y", whole_cell)
        z = pybamm.SpatialVariable("z", whole_cell)

        y_lim = {"min": pybamm.Scalar(0), "max": pybamm.standard_parameters.l_y}

        z_lim = {"min": pybamm.Scalar(0), "max": pybamm.standard_parameters.l_z}

        MACRO_DOMAINS = ["negative electrode", "separator", "positive electrode"]
        for domain in MACRO_DOMAINS:
            self[domain]["primary"][y] = y_lim
            self[domain]["primary"][z] = z_lim
        self.update(custom_geometry)
