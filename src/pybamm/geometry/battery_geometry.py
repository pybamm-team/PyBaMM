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
    if options is None or type(options) == dict:  # noqa: E721
        options = pybamm.BatteryModelOptions(options)
    geo = pybamm.GeometricParameters(options)
    L_n = geo.n.L
    L_s = geo.s.L
    L_n_L_s = L_n + L_s
    # Override print_name
    L_n_L_s.print_name = "L_n + L_s"

    # Set up electrode/separator/electrode geometry
    geometry = {
        "negative electrode": {"x_n": {"min": 0, "max": L_n}, "coord_sys": "cartesian"},
        "separator": {"x_s": {"min": L_n, "max": L_n_L_s}, "coord_sys": "cartesian"},
        "positive electrode": {
            "x_p": {"min": L_n_L_s, "max": geo.L_x},
            "coord_sys": "cartesian",
        },
    }

    # Add particle domains
    if include_particles is True:
        for domain in ["negative", "positive"]:
            domain_options = getattr(options, domain)
            if options.electrode_types[domain] == "porous":
                particle_coord_sys = domain_options["particle shape"] + " polar"
                geo_domain = geo.domain_params[domain]
                d = domain[0]
                geometry.update(
                    {
                        f"{domain} particle": {
                            f"r_{d}": {"min": 0, "max": geo_domain.prim.R_typ},
                            "coord_sys": particle_coord_sys,
                        },
                    }
                )
                phases = int(domain_options["particle phases"])
                if phases >= 2:
                    geometry.update(
                        {
                            f"{domain} primary particle": {
                                f"r_{d}_prim": {"min": 0, "max": geo_domain.prim.R_typ},
                                "coord_sys": particle_coord_sys,
                            },
                            f"{domain} secondary particle": {
                                f"r_{d}_sec": {"min": 0, "max": geo_domain.sec.R_typ},
                                "coord_sys": particle_coord_sys,
                            },
                        }
                    )

                if domain_options["particle size"] == "distribution":
                    if phases == 1:
                        geometry.update(
                            {
                                f"{domain} particle size": {
                                    f"R_{d}": {
                                        "min": geo_domain.prim.R_min,
                                        "max": geo_domain.prim.R_max,
                                    },
                                    "coord_sys": "cartesian",
                                },
                            }
                        )
                    elif phases == 2:
                        geometry.update(
                            {
                                f"{domain} primary particle size": {
                                    f"R_{d}_prim": {
                                        "min": geo_domain.prim.R_min,
                                        "max": geo_domain.prim.R_max,
                                    },
                                    "coord_sys": "cartesian",
                                },
                                f"{domain} secondary particle size": {
                                    f"R_{d}_sec": {
                                        "min": geo_domain.sec.R_min,
                                        "max": geo_domain.sec.R_max,
                                    },
                                    "coord_sys": "cartesian",
                                },
                            }
                        )

    # Add current collector domains
    current_collector_dimension = options["dimensionality"]
    if form_factor == "pouch":
        if current_collector_dimension == 0:
            geometry["current collector"] = {"z": {"position": 1}}
        elif current_collector_dimension == 1:
            geometry["current collector"] = {
                "z": {"min": 0, "max": geo.L_z},
                "coord_sys": "cartesian",
                "tabs": {
                    "negative": {"z_centre": geo.n.centre_z_tab},
                    "positive": {"z_centre": geo.p.centre_z_tab},
                },
            }
        elif current_collector_dimension == 2:
            geometry["current collector"] = {
                "y": {"min": 0, "max": geo.L_y},
                "z": {"min": 0, "max": geo.L_z},
                "coord_sys": "cartesian",
                "tabs": {
                    "negative": {
                        "y_centre": geo.n.centre_y_tab,
                        "z_centre": geo.n.centre_z_tab,
                        "width": geo.n.L_tab,
                    },
                    "positive": {
                        "y_centre": geo.p.centre_y_tab,
                        "z_centre": geo.p.centre_z_tab,
                        "width": geo.p.L_tab,
                    },
                },
            }
        elif current_collector_dimension == 3:
            geometry["current collector"] = {"z": {"position": 1}}
            geometry["cell"] = {
                "x": {"min": 0, "max": geo.L_x},
                "y": {"min": 0, "max": geo.L_y},
                "z": {"min": 0, "max": geo.L_z},
                "coord_sys": "cartesian",
            }

    elif form_factor == "cylindrical":
        if current_collector_dimension == 0:
            geometry["current collector"] = {"r_macro": {"position": 1}}
        elif current_collector_dimension == 1:
            geometry["current collector"] = {
                "r_macro": {"min": geo.r_inner, "max": 1},
                "coord_sys": "cylindrical polar",
            }
        elif current_collector_dimension == 3:
            geometry["current collector"] = {"z": {"position": 1}}
            geometry["cell"] = {
                "r_macro": {"min": geo.r_inner, "max": geo.r_outer},
                "z": {"min": 0, "max": geo.L_z},
                "coord_sys": "cylindrical polar",
            }
        else:
            raise pybamm.GeometryError(
                f"Invalid current collector dimension '{current_collector_dimension}' (should be 0 or 1 for "
                "a 'cylindrical' battery geometry)"
            )
    else:
        raise pybamm.GeometryError(
            f"Invalid form factor '{form_factor}' (should be 'pouch' or 'cylindrical')"
        )

    return pybamm.Geometry(geometry)
