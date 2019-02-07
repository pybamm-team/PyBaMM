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

    def __init__(self, custom_geometry={}):
        super().__init__()
        ln = pybamm.standard_parameters.ln
        ls = pybamm.standard_parameters.ls

        self["negative electrode"] = {"x": {"min": pybamm.Scalar(0), "max": ln}}
        self["separator"] = {"x": {"min": ln, "max": ln + ls}}
        self["positive electrode"] = {"x": {"min": ln + ls, "max": pybamm.Scalar(1)}}
        self["negative particle"] = {
            "r": {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}
        }
        self["positive particle"] = {
            "r": {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}
        }
        # update with custom geometry if non empty
        self.update(custom_geometry)


class Geometry3D(Geometry):
    """A geometry class to store the details features of the cell geometry

     **Extends**: :class:`dict`

     Parameters
     ----------

     custom_geometry : dict containing any extra user defined geometry
     """

    def __init__(self, custom_geometry={}):
        super().__init__()

        y = {"min": pybamm.Scalar(0), "max": pybamm.standard_parameters.ly}

        z = {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}

        for domain in self["macro"]:
            self["macro"][domain]["y"] = y
            self["macro"][domain]["z"] = z
        self.update(custom_geometry)
