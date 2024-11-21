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
        "negative electrode": pybamm.Domain1D((0, L_n)),
        "separator": pybamm.Domain1D((L_n, L_n_L_s)),
        "positive electrode": pybamm.Domain1D((L_n_L_s, geo.L_x)),
    }

    # Add particle domains
    for domain in ["negative", "positive"]:
        if include_particles is True:
            if options.electrode_types[domain] == "porous":
                geo_domain = geo.domain_params[domain]
                geometry.update(
                    {
                        f"{domain} particle": pybamm.Domain1D(
                            (0, geo_domain.prim.R_typ), coord_sys="spherical polar"
                        )
                    }
                )
                phases = int(getattr(options, domain)["particle phases"])
                if phases >= 2:
                    geometry.update(
                        {
                            f"{domain} primary particle": pybamm.Domain1D(
                                (0, geo_domain.prim.R_typ), coord_sys="spherical polar"
                            ),
                            f"{domain} secondary particle": pybamm.Domain1D(
                                (0, geo_domain.sec.R_typ), coord_sys="spherical polar"
                            ),
                        }
                    )

                # Add particle size domains
                if getattr(options, domain)["particle size"] == "distribution":
                    R_min = getattr(geo_domain, f"{domain} particle").R_min
                    R_max = getattr(geo_domain, f"{domain} particle").R_max
                    geometry.update(
                        {f"{domain} particle size": pybamm.Domain1D((R_min, R_max))}
                    )

    # Add current collector domains
    # current_collector_dimension = options["dimensionality"]
    # if form_factor == "pouch":
    #     if current_collector_dimension == 0:
    #         geometry["current collector"] = {"z": {"position": 1}}
    #     elif current_collector_dimension == 1:
    #         geometry["current collector"] = {
    #             "z": {"min": 0, "max": geo.L_z},
    #             "tabs": {
    #                 "negative": {"z_centre": geo.n.centre_z_tab},
    #                 "positive": {"z_centre": geo.p.centre_z_tab},
    #             },
    #         }
    #     elif current_collector_dimension == 2:
    #         geometry["current collector"] = {
    #             "y": {"min": 0, "max": geo.L_y},
    #             "z": {"min": 0, "max": geo.L_z},
    #             "tabs": {
    #                 "negative": {
    #                     "y_centre": geo.n.centre_y_tab,
    #                     "z_centre": geo.n.centre_z_tab,
    #                     "width": geo.n.L_tab,
    #                 },
    #                 "positive": {
    #                     "y_centre": geo.p.centre_y_tab,
    #                     "z_centre": geo.p.centre_z_tab,
    #                     "width": geo.p.L_tab,
    #                 },
    #             },
    #         }
    # elif form_factor == "cylindrical":
    #     if current_collector_dimension == 0:
    #         geometry["current collector"] = {"r_macro": {"position": 1}}
    #     elif current_collector_dimension == 1:
    #         geometry["current collector"] = {
    #             "r_macro": {"min": geo.r_inner, "max": 1},
    #         }
    #     else:
    #         raise pybamm.GeometryError(
    #             f"Invalid current collector dimension '{current_collector_dimension}' (should be 0 or 1 for "
    #             "a 'cylindrical' battery geometry)"
    #         )
    # else:
    #     raise pybamm.GeometryError(
    #         f"Invalid form factor '{form_factor}' (should be 'pouch' or 'cylindrical'"
    #     )

    return pybamm.Geometry(geometry)
