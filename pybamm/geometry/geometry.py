#
# Geometry class for storing the geometry of the model
#
import pybamm


class Geometry(dict):

    """
    A geometry class to store the details features of the cell geometry.

    Geometry extends the class dictionary and uses the key words: "negative electrode",
    "positive electrode", etc to indicate the subdomain.  Within each subdomain, there
    are "primary", "secondary" or "tabs" dimensions.  "primary" dimensions correspond to
    dimensions on which spatial operators will be applied (e.g. the gradient and
    divergence). In contrast, spatial operators do not act along "secondary" dimensions.
    This allows for multiple independent particles to be included into a model.

    The values assigned to each domain are dictionaries containing the spatial variables
    in that domain, along with expression trees giving their min and maximum extents.
    For example, the following dictionary structure would represent a Geometry with a
    single domain "negative electrode", defined using the variable `x_n` which has a
    range from 0 to the pre-defined parameter `l_n`.

    .. code-block:: python

       {
           "negative electrode": {
               "primary": {x_n: {"min": pybamm.Scalar(0), "max": l_n}}
           }
       }

    A user can create a new Geometry by combining one or more of the pre-defined
    geometries defined with the names given below.

    - "1D macro": macroscopic 1D cell geometry (i.e. electrodes)
    - "3D macro": macroscopic 3D cell geometry
    - "1+1D macro": 1D macroscopic cell geometry with a 1D current collector
    - "1+2D macro": 1D macroscopic cell geometry with a 2D current collector
    - "1D micro": 1D microscopic cell geometry (i.e. particles)
    - "1+1D micro": This is the geometry used in the standard DFN or P2D model
    - "(1+0)+1D micro": 0D macroscopic cell geometry with 1D current collector,
                        along with the microscopic 1D particle geometry.
    - "(2+0)+1D micro": 0D macroscopic cell geometry with 1D current collector,
                        along with the microscopic 1D particle geometry.
    - "(1+1)+1D micro": 1D macroscopic cell geometry, with 1D current collector model,
                        along with the microscopic 1D particle geometry.
    - "(2+1)+1D micro": 1D macroscopic cell geometry, with 2D current collector model,
                        along with the microscopic 1D particle geometry.
    - "2D current collector": macroscopic 2D current collector geometry

    **Extends**: :class:`dict`

    Parameters
    ----------

    geometries: one or more strings or Geometry objects. A string will be assumed to be
                one of the predefined Geometries given above

    custom_geometry : dict containing any extra user defined geometry
    """

    def __init__(self, *geometries, custom_geometry={}):
        for geometry in geometries:
            if geometry == "1D macro":
                geometry = Geometry1DMacro()
            elif geometry == "3D macro":
                geometry = Geometry3DMacro()
            elif geometry == "1+1D macro":
                geometry = Geometryxp1DMacro(cc_dimension=1)
            elif geometry == "2+1D macro":
                geometry = Geometryxp1DMacro(cc_dimension=2)
            elif geometry == "1D micro":
                geometry = Geometry1DMicro()
            elif geometry == "1+1D micro":
                geometry = Geometry1p1DMicro()
            elif geometry == "(1+0)+1D micro":
                geometry = Geometryxp0p1DMicro(cc_dimension=1)
            elif geometry == "(2+0)+1D micro":
                geometry = Geometryxp0p1DMicro(cc_dimension=2)
            elif geometry == "(1+1)+1D micro":
                geometry = Geometryxp1p1DMicro(cc_dimension=1)
            elif geometry == "(2+1)+1D micro":
                geometry = Geometryxp1p1DMicro(cc_dimension=2)
            elif geometry == "2D current collector":
                geometry = Geometry2DCurrentCollector()
            # avoid combining geometries that clash
            if any([k in self.keys() for k in geometry.keys()]):
                raise ValueError("trying to overwrite existing geometry")

            for k, v in geometry.items():
                self.add_domain(k, v)

        # Allow overwriting with a custom geometry
        for k, v in custom_geometry.items():
            self.add_domain(k, v)

    def add_domain(self, name, geometry):
        """
        Add a new domain to the geometry

        Parameters
        ----------

        name: string giving the name of the domain

        geometry: dict of variables in the domain, along with the minimum and maximum
                extents (e.g. {"primary": {x_n: {"min": pybamm.Scalar(0), "max": l_n}}}
        """
        if not isinstance(name, str):
            raise ValueError("name must be a string")

        for k, v in geometry.items():
            if k not in ["primary", "secondary", "tabs"]:
                raise ValueError(
                    'keys of geometry must be either "primary", "secondary" or '
                    '"tabs"'
                )
            if k != "tabs":
                for variable, rnge in v.items():
                    if not isinstance(variable, pybamm.SpatialVariable):
                        raise ValueError(
                            "inner dict of geometry must have pybamm.SpatialVariable "
                            "as keys"
                        )
                    if list(rnge.keys()) != ["position"]:
                        if "min" not in rnge.keys():
                            raise ValueError(
                                "no minimum extents for variable {}".format(variable)
                            )
                        if "max" not in rnge.keys():
                            raise ValueError(
                                "no maximum extents for variable {}".format(variable)
                            )
            else:
                for region, params in v.items():
                    if region not in ["negative", "positive"]:
                        raise ValueError('tabs region must be "negative" or "positive"')
                    for pname in params.keys():
                        if pname not in ["y_centre", "z_centre", "width"]:
                            raise ValueError(
                                'tabs region params must be "y_centre", '
                                '"z_centre" or "width"'
                            )

        self.update({name: geometry})


class Geometry1DMacro(Geometry):
    """
    A geometry class to store the details features of the macroscopic 1D cell geometry.

    **Extends**: :class:`Geometry`

    Parameters
    ----------

    custom_geometry : dict containing any extra user defined geometry
    """

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
        self["current collector"] = {"primary": {var.z: {"position": pybamm.Scalar(1)}}}

        # update with custom geometry if non empty
        self.update(custom_geometry)


class Geometry3DMacro(Geometry1DMacro):
    """
    A geometry class to store the details features of the macroscopic 3D cell geometry.

    **Extends**: :class:`Geometry1DMacro`

    Parameters
    ----------

    custom_geometry : dict containing any extra user defined geometry
    """

    def __init__(self, custom_geometry={}):
        super().__init__()

        var = pybamm.standard_spatial_vars

        y_lim = {"min": pybamm.Scalar(0), "max": pybamm.geometric_parameters.l_y}

        z_lim = {"min": pybamm.Scalar(0), "max": pybamm.geometric_parameters.l_z}

        for domain in self.keys():
            self[domain]["primary"][var.y] = y_lim
            self[domain]["primary"][var.z] = z_lim
        self.update(custom_geometry)


class Geometry1DMicro(Geometry):
    """
    A geometry class to store the details features of the microscopic 1D particle
    geometry.

    **Extends**: :class:`Geometry`

    Parameters
    ----------

    custom_geometry : dict containing any extra user defined geometry
    """

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
    """
    A geometry class to store the details features of the 1+1D cell geometry.
    This is the geometry used in the standard DFN or P2D model.

    **Extends**: :class:`Geometry`

    Parameters
    ----------

    custom_geometry : dict containing any extra user defined geometry
    """

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


class Geometryxp1DMacro(Geometry1DMacro):
    """
    A geometry class to store the details features of x+1D macroscopic cell
    geometry, where x is the dimension of the current collector model.

    **Extends**: :class:`Geometry1DMacro`

    Parameters
    ----------

    cc_dimension : int, optional
        the dimension of the current collector model
    custom_geometry : dict, optional
        dictionary containing any extra user defined geometry
    """

    def __init__(self, cc_dimension=1, custom_geometry={}):
        super().__init__()

        var = pybamm.standard_spatial_vars

        if cc_dimension == 1:
            # Add secondary domains to x-domains
            for geom in self.values():
                geom["secondary"] = {
                    var.z: {
                        "min": pybamm.Scalar(0),
                        "max": pybamm.geometric_parameters.l_z,
                    }
                }
            # Add primary current collector domain
            self["current collector"] = {
                "primary": {
                    var.z: {
                        "min": pybamm.Scalar(0),
                        "max": pybamm.geometric_parameters.l_z,
                    }
                },
                "tabs": {
                    "negative": {
                        "z_centre": pybamm.geometric_parameters.centre_z_tab_n
                    },
                    "positive": {
                        "z_centre": pybamm.geometric_parameters.centre_z_tab_p
                    },
                },
            }
        elif cc_dimension == 2:
            # Add secondary domains to x-domains
            for geom in self.values():
                geom["secondary"] = {
                    var.y: {
                        "min": pybamm.Scalar(0),
                        "max": pybamm.geometric_parameters.l_y,
                    },
                    var.z: {
                        "min": pybamm.Scalar(0),
                        "max": pybamm.geometric_parameters.l_z,
                    },
                }
            # Add primary current collector domain
            self["current collector"] = {
                "primary": {
                    var.y: {
                        "min": pybamm.Scalar(0),
                        "max": pybamm.geometric_parameters.l_y,
                    },
                    var.z: {
                        "min": pybamm.Scalar(0),
                        "max": pybamm.geometric_parameters.l_z,
                    },
                },
                "tabs": {
                    "negative": {
                        "y_centre": pybamm.geometric_parameters.centre_y_tab_n,
                        "z_centre": pybamm.geometric_parameters.centre_z_tab_n,
                        "width": pybamm.geometric_parameters.l_tab_n,
                    },
                    "positive": {
                        "y_centre": pybamm.geometric_parameters.centre_y_tab_p,
                        "z_centre": pybamm.geometric_parameters.centre_z_tab_p,
                        "width": pybamm.geometric_parameters.l_tab_p,
                    },
                },
            }
        else:
            raise pybamm.GeometryError(
                "current collector dimension must be 1 or 2, not {}".format(
                    cc_dimension
                )
            )

        # update with custom geometry if non empty
        self.update(custom_geometry)


class Geometryxp0p1DMicro(Geometry1DMicro):
    """
    A geometry class to store the details features of x+0D macroscopic cell
    geometry, where x is the dimension of the current collector model, along with
    the microscopic 1D particle geometry.

    **Extends**: :class:`Geometry1DMicro`

    Parameters
    ----------

    cc_dimension : int, optional
        the dimension of the current collector model
    custom_geometry : dict, optional
        dictionary containing any extra user defined geometry
    """

    def __init__(self, cc_dimension=1, custom_geometry={}):
        super().__init__()

        var = pybamm.standard_spatial_vars

        # Add secondary domains to x-domains
        if cc_dimension == 1:
            for domain in self.keys():
                self[domain]["secondary"] = {
                    var.z: {
                        "min": pybamm.Scalar(0),
                        "max": pybamm.geometric_parameters.l_z,
                    }
                }
        elif cc_dimension == 2:
            for domain in self.keys():
                self[domain]["secondary"] = {
                    var.y: {
                        "min": pybamm.Scalar(0),
                        "max": pybamm.geometric_parameters.l_y,
                    },
                    var.z: {
                        "min": pybamm.Scalar(0),
                        "max": pybamm.geometric_parameters.l_z,
                    },
                }
        else:
            raise pybamm.GeometryError(
                "current collector dimension must be 1 or 2, not {}".format(
                    cc_dimension
                )
            )

        # update with custom geometry if non empty
        self.update(custom_geometry)


class Geometryxp1p1DMicro(Geometry1DMicro):
    """
    A geometry class to store the details features of x+1D macroscopic cell
    geometry, where x is the dimension of the current collector model, along with
    the microscopic 1D particle geometry.

    **Extends**: :class:`Geometry1DMicro`

    Parameters
    ----------

    cc_dimension : int, optional
        the dimension of the current collector model
    custom_geometry : dict, optional
        dictionary containing any extra user defined geometry
    """

    def __init__(self, cc_dimension=1, custom_geometry={}):
        super().__init__()

        var = pybamm.standard_spatial_vars
        l_n = pybamm.geometric_parameters.l_n
        l_s = pybamm.geometric_parameters.l_s

        # Add secondary domains to x-domains
        if cc_dimension == 1:
            self["negative particle"]["secondary"] = {
                var.x_n: {"min": pybamm.Scalar(0), "max": l_n},
                var.z: {
                    "min": pybamm.Scalar(0),
                    "max": pybamm.geometric_parameters.l_z,
                },
            }
            self["positive particle"]["secondary"] = {
                var.x_p: {"min": l_n + l_s, "max": pybamm.Scalar(1)},
                var.z: {
                    "min": pybamm.Scalar(0),
                    "max": pybamm.geometric_parameters.l_z,
                },
            }
        elif cc_dimension == 2:
            self["negative particle"]["secondary"] = {
                var.x_n: {"min": pybamm.Scalar(0), "max": l_n},
                var.y: {
                    "min": pybamm.Scalar(0),
                    "max": pybamm.geometric_parameters.l_y,
                },
                var.z: {
                    "min": pybamm.Scalar(0),
                    "max": pybamm.geometric_parameters.l_z,
                },
            }
            self["positive particle"]["secondary"] = {
                var.x_p: {"min": l_n + l_s, "max": pybamm.Scalar(1)},
                var.y: {
                    "min": pybamm.Scalar(0),
                    "max": pybamm.geometric_parameters.l_y,
                },
                var.z: {
                    "min": pybamm.Scalar(0),
                    "max": pybamm.geometric_parameters.l_z,
                },
            }
        else:
            raise pybamm.GeometryError(
                "current collector dimension must be 1 or 2, not {}".format(
                    cc_dimension
                )
            )

        # update with custom geometry if non empty
        self.update(custom_geometry)


class Geometry2DCurrentCollector(Geometry):
    """
    A geometry class to store the details features of the macroscopic 2D
    current collector geometry.

    **Extends**: :class:`Geometry`

    Parameters
    ----------

    custom_geometry : dict containing any extra user defined geometry
    """

    def __init__(self, custom_geometry={}):
        super().__init__()
        var = pybamm.standard_spatial_vars

        self["current collector"] = {
            "primary": {
                var.y: {
                    "min": pybamm.Scalar(0),
                    "max": pybamm.geometric_parameters.l_y,
                },
                var.z: {
                    "min": pybamm.Scalar(0),
                    "max": pybamm.geometric_parameters.l_z,
                },
            },
            "tabs": {
                "negative": {
                    "y_centre": pybamm.geometric_parameters.centre_y_tab_n,
                    "z_centre": pybamm.geometric_parameters.centre_z_tab_n,
                    "width": pybamm.geometric_parameters.l_tab_n,
                },
                "positive": {
                    "y_centre": pybamm.geometric_parameters.centre_y_tab_p,
                    "z_centre": pybamm.geometric_parameters.centre_z_tab_p,
                    "width": pybamm.geometric_parameters.l_tab_p,
                },
            },
        }

        # update with custom geometry if non empty
        self.update(custom_geometry)
