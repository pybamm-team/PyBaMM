#
# Geometry class for storing the geometry of the model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import pybamm


class Geometry(dict):
    """A geometry class to store the details features of the cell geometry.
        Geometry extends the class dictionary and uses the key words:
        "negative electrode", "positive electrode", etc to indicate the subdomain.
        Within each subdomain, there are "primary" and "secondary" dimensions.
        "primary" dimensions correspond to dimensiones on which spatial
        operators will be applied (e.g. the gradient and divergence). In contrast,
        spatial operators do not act along "secondary" dimensions. This allows for
        multiple independent particles to be included into a model.

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
        var = pybamm.standard_spatial_vars
        l_n = pybamm.geometric_parameters.l_n
        l_s = pybamm.geometric_parameters.l_s

        self["negative electrode"] = {
            "primary": {var.x_n: {"min": pybamm.Scalar(0), "max": l_n}}
        }
        self["separator"] = {"primary": {var.x_s: {"min": l_n, "max": l_n + l_s}}}
        self["positive electrode"] = {
            "primary": {var.x_p: {"min": l_n + l_s, "max": pybamm.Scalar(1)}}
        }

        # update with custom geometry if non empty
        self.update(custom_geometry)


class Geometry1DMicro(Geometry):
    def __init__(self, custom_geometry={}):
        super().__init__()

        var = pybamm.standard_spatial_vars

        self["negative particle"] = {
            "primary": {var.r_n: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
        }
        self["positive particle"] = {
            "primary": {var.r_p: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
        }
        # update with custom geometry if non empty
        self.update(custom_geometry)


class Geometry1p1DMicro(Geometry):
    def __init__(self, custom_geometry={}):
        super().__init__()

        var = pybamm.standard_spatial_vars
        l_n = pybamm.geometric_parameters.l_n
        l_s = pybamm.geometric_parameters.l_s

        self["negative particle"] = {
            "primary": {var.r_n: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}},
            "secondary": {var.x_n: {"min": pybamm.Scalar(0), "max": l_n}},
        }
        self["positive particle"] = {
            "primary": {var.r_p: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}},
            "secondary": {var.x_p: {"min": l_n + l_s, "max": pybamm.Scalar(1)}},
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

        var = pybamm.standard_spatial_vars

        y_lim = {"min": pybamm.Scalar(0), "max": pybamm.geometric_parameters.l_y}

        z_lim = {"min": pybamm.Scalar(0), "max": pybamm.geometric_parameters.l_z}

        MACRO_DOMAINS = ["negative electrode", "separator", "positive electrode"]
        for domain in MACRO_DOMAINS:
            self[domain]["primary"][var.y] = y_lim
            self[domain]["primary"][var.z] = z_lim
        self.update(custom_geometry)
