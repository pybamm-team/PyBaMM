#
# Function to create battery geometries
#
import pybamm


def battery_geometry(include_particles=True, current_collector_dimension=0):
    """
    A convenience function to create battery geometries.

    Parameters
    ----------
    include_particles : bool
        Whether to include particle domains
    current_collector_dimensions : int, default
        The dimensions of the current collector. Should be 0 (default), 1 or 2

    Returns
    -------
    :class:`pybamm.Geometry`
        A geometry class for the battery

    """
    var = pybamm.standard_spatial_vars
    l_n = pybamm.geometric_parameters.l_n
    l_s = pybamm.geometric_parameters.l_s

    geometry = {
        "negative electrode": {var.x_n: {"min": 0, "max": l_n}},
        "separator": {var.x_s: {"min": l_n, "max": l_n + l_s}},
        "positive electrode": {var.x_p: {"min": l_n + l_s, "max": 1}},
    }
    # Add particle domains
    if include_particles is True:
        geometry.update(
            {
                "negative particle": {var.r_n: {"min": 0, "max": 1}},
                "positive particle": {var.r_p: {"min": 0, "max": 1}},
            }
        )

    if current_collector_dimension == 0:
        geometry["current collector"] = {var.z: {"position": 1}}
    elif current_collector_dimension == 1:
        geometry["current collector"] = {
            var.z: {"min": 0, "max": 1},
            "tabs": {
                "negative": {"z_centre": pybamm.geometric_parameters.centre_z_tab_n},
                "positive": {"z_centre": pybamm.geometric_parameters.centre_z_tab_p},
            },
        }
    elif current_collector_dimension == 2:
        geometry["current collector"] = {
            var.y: {"min": 0, "max": pybamm.geometric_parameters.l_y},
            var.z: {"min": 0, "max": pybamm.geometric_parameters.l_z},
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
            "Invalid current collector dimension '{}' (should be 0, 1 or 2)".format(
                current_collector_dimension
            )
        )

    return pybamm.Geometry(geometry)
