#
# Function to create battery geometries
#
import pybamm
from pybamm.geometry import half_cell_spatial_vars


def half_cell_geometry(
    include_particles=True, current_collector_dimension=0, working_electrode="positive"
):
    """
    A convenience function to create battery geometries.

    Parameters
    ----------
    include_particles : bool
        Whether to include particle domains
    current_collector_dimensions : int, default
        The dimensions of the current collector. Should be 0 (default), 1 or 2
    current_collector_dimensions : string
        The electrode taking as working electrode. Should be "positive" or "negative"

    Returns
    -------
    :class:`pybamm.Geometry`
        A geometry class for the battery

    """
    var = half_cell_spatial_vars
    geo = pybamm.geometric_parameters
    if working_electrode == "positive":
        l_w = geo.l_p
    elif working_electrode == "negative":
        l_w = geo.l_n
    else:
        raise ValueError(
            "The option 'working_electrode' should be either 'positive'"
            " or 'negative'"
        )
    l_Li = geo.l_Li
    l_s = geo.l_s

    geometry = {
        "lithium counter electrode": {var.x_Li: {"min": 0, "max": l_Li}},
        "separator": {var.x_s: {"min": l_Li, "max": l_Li + l_s}},
        "working electrode": {var.x_w: {"min": l_Li + l_s, "max": l_Li + l_s + l_w}},
    }
    # Add particle domains
    if include_particles is True:
        geometry.update({"working particle": {var.r_w: {"min": 0, "max": 1}}})

    if current_collector_dimension == 0:
        geometry["current collector"] = {var.z: {"position": 1}}
    elif current_collector_dimension == 1:
        geometry["current collector"] = {
            var.z: {"min": 0, "max": 1},
            "tabs": {
                "negative": {"z_centre": geo.centre_z_tab_n},
                "positive": {"z_centre": geo.centre_z_tab_p},
            },
        }
    elif current_collector_dimension == 2:
        geometry["current collector"] = {
            var.y: {"min": 0, "max": geo.l_y},
            var.z: {"min": 0, "max": geo.l_z},
            "tabs": {
                "negative": {
                    "y_centre": geo.centre_y_tab_n,
                    "z_centre": geo.centre_z_tab_n,
                    "width": geo.l_tab_n,
                },
                "positive": {
                    "y_centre": geo.centre_y_tab_p,
                    "z_centre": geo.centre_z_tab_p,
                    "width": geo.l_tab_p,
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
