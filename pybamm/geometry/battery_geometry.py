#
# Function to create battery geometries
#
import pybamm


def battery_geometry(
    include_particles=True,
    options=None,
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
    form_factor : str, optional
        The form factor of the cell. Can be "pouch" (default) or "cylindrical".

    Returns
    -------
    :class:`pybamm.Geometry`
        A geometry class for the battery

    """
    if options is None or type(options) == dict:
        options = pybamm.BatteryModelOptions(options)

    geo = pybamm.geometric_parameters
    l_n = geo.n.l
    l_s = geo.s.l
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
        zero_one = {"min": 0, "max": 1}
        geometry.update(
            {
                "negative particle": {"r_n": zero_one},
                "positive particle": {"r_p": zero_one},
            }
        )
        for domain in ["negative", "positive"]:
            phases = int(getattr(options, domain)["particle phases"])
            if phases >= 2:
                geometry.update(
                    {
                        f"{domain} primary particle": {"r_n_prim": zero_one},
                        f"{domain} secondary particle": {"r_n_sec": zero_one},
                    }
                )
    # Add particle size domains
    if options is not None and options["particle size"] == "distribution":
        R_min_n = geo.n.prim.R_min
        R_min_p = geo.p.prim.R_min
        R_max_n = geo.n.prim.R_max
        R_max_p = geo.p.prim.R_max
        geometry.update(
            {
                "negative particle size": {"R_n": {"min": R_min_n, "max": R_max_n}},
                "positive particle size": {"R_p": {"min": R_min_p, "max": R_max_p}},
            }
        )
    # Add current collector domains
    current_collector_dimension = options["dimensionality"]
    if form_factor == "pouch":
        if current_collector_dimension == 0:
            geometry["current collector"] = {"z": {"position": 1}}
        elif current_collector_dimension == 1:
            geometry["current collector"] = {
                "z": {"min": 0, "max": 1},
                "tabs": {
                    "negative": {"z_centre": geo.n.centre_z_tab},
                    "positive": {"z_centre": geo.p.centre_z_tab},
                },
            }
        elif current_collector_dimension == 2:
            geometry["current collector"] = {
                "y": {"min": 0, "max": geo.l_y},
                "z": {"min": 0, "max": geo.l_z},
                "tabs": {
                    "negative": {
                        "y_centre": geo.n.centre_y_tab,
                        "z_centre": geo.n.centre_z_tab,
                        "width": geo.n.l_tab,
                    },
                    "positive": {
                        "y_centre": geo.p.centre_y_tab,
                        "z_centre": geo.p.centre_z_tab,
                        "width": geo.p.l_tab,
                    },
                },
            }
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
