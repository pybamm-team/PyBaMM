#
# Function to create battery geometries
#
import pybamm


def battery_geometry(
    include_particles=True,
    options=None,
    current_collector_dimension=0,
    form_factor="pouch",
):
    """
    A convenience function to create battery geometries.

    Parameters
    ----------
    include_particles : bool, optional
        Whether to include particle domains. Can be True (default) or False.
    options : dict, optional
        Dictionary of model options. Necessary for "particle-size geometry",
        relevant for lithium-ion chemistries.
    current_collector_dimensions : int, optional
        The dimensions of the current collector. Can be 0 (default), 1 or 2. For
        a "cylindrical" form factor the current collector dimension must be 0 or 1.
    form_factor : str, optional
        The form factor of the cell. Can be "pouch" (default) or "cylindrical".

    Returns
    -------
    :class:`pybamm.Geometry`
        A geometry class for the battery

    """
    geo = pybamm.geometric_parameters
    l_n = geo.l_n
    l_s = geo.l_s
    l_n_l_s = l_n + l_s
    # Override print_name
    l_n_l_s.print_name = "l_n + l_s"

    # Set up electrode/separator/electrode geometry
    geometry = {
        "negative electrode": {"x_n": {"min": 0, "max": l_n}},
        "separator": {"x_s": {"min": l_n, "max": l_n_l_s}},
        "positive electrode": {"x_p": {"min": l_n_l_s, "max": 1}},
    }
    # Add particle domains
    if include_particles is True:
        geometry.update(
            {
                "negative particle": {"r_n": {"min": 0, "max": 1}},
                "positive particle": {"r_p": {"min": 0, "max": 1}},
            }
        )
    # Add particle size domains
    if options is not None and options["particle size"] == "distribution":
        R_min_n = geo.R_min_n
        R_min_p = geo.R_min_p
        R_max_n = geo.R_max_n
        R_max_p = geo.R_max_p
        geometry.update(
            {
                "negative particle size": {"R_n": {"min": R_min_n, "max": R_max_n}},
                "positive particle size": {"R_p": {"min": R_min_p, "max": R_max_p}},
            }
        )
    # Add current collector domains
    if form_factor == "pouch":
        if current_collector_dimension == 0:
            geometry["current collector"] = {"z": {"position": 1}}
        elif current_collector_dimension == 1:
            geometry["current collector"] = {
                "z": {"min": 0, "max": 1},
                "tabs": {
                    "negative": {"z_centre": geo.centre_z_tab_n},
                    "positive": {"z_centre": geo.centre_z_tab_p},
                },
            }
        elif current_collector_dimension == 2:
            geometry["current collector"] = {
                "y": {"min": 0, "max": geo.l_y},
                "z": {"min": 0, "max": geo.l_z},
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
    elif form_factor == "cylindrical":
        if current_collector_dimension == 0:
            geometry["current collector"] = {"r_macro": {"position": 1}}
        elif current_collector_dimension == 1:
            geometry["current collector"] = {
                "r_macro": {"min": geo.r_inner, "max": 1},
            }
        else:
            raise pybamm.GeometryError(
                "Invalid current collector dimension '{}' (should be 0 or 1 for "
                "a 'cylindrical' battery geometry)".format(current_collector_dimension)
            )
    else:
        raise pybamm.GeometryError(
            "Invalid form factor '{}' (should be 'pouch' or 'cylindrical'".format(
                form_factor
            )
        )

    return pybamm.Geometry(geometry)
