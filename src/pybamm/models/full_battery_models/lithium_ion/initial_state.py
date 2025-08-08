import pybamm


def set_initial_state(
    initial_value,
    parameter_values,
    param=None,
    known_value="cyclable lithium capacity",
    inplace=True,
    options=None,
    inputs=None,
    tol=1e-6,
):
    """
    Set the values of the parameters representing the initial model states.

    Parameters
    ----------
    initial_value : float
        Target initial value.
        If float, interpreted as SOC, must be between 0 and 1.
        If string e.g. "4 V", interpreted as voltage, must be between V_min and V_max.
    parameter_values : :class:`pybamm.ParameterValues`
        Parameters and their corresponding values.
    param : :class:`pybamm.LithiumIonParameters`, optional
        The symbolic parameter set to use for the simulation.
        If not provided, the default parameter set will be used.
    known_value : str, optional
        The known value needed to complete the electrode SOH model.
        Can be "cyclable lithium capacity" (default) or "cell capacity".
    inplace: bool, optional
        If True, replace the parameters values in place. Otherwise, return a new set of
        parameter values. Default is True.
    options : dict-like, optional
        A dictionary of options to be passed to the model, see
        :class:`pybamm.BatteryModelOptions`.
    inputs : dict, optional
        A dictionary of input parameters to pass to the model when solving.
    tol : float, optional
        The tolerance for the solver used to compute the initial stoichiometries.
        A lower value results in higher precision but may increase computation time.
        Default is 1e-6.
    """
    parameter_values = parameter_values if inplace else parameter_values.copy()
    param = param or pybamm.LithiumIonParameters(options)

    if options is not None and options.get("open-circuit potential", None) == "MSMR":
        """
        Set the initial OCP of each electrode, based on the initial SOC or voltage.
        """
        Un, Up = pybamm.lithium_ion.get_initial_ocps(
            initial_value,
            parameter_values,
            param=param,
            known_value=known_value,
            options=options,
            inputs=inputs,
        )
        parameter_values.update(
            {
                "Initial voltage in negative electrode [V]": Un,
                "Initial voltage in positive electrode [V]": Up,
            }
        )
    elif options is not None and options.get("working electrode", None) == "positive":
        """
        Set the initial stoichiometry of the working electrode, based on the initial
        SOC or voltage.
        """
        x = pybamm.lithium_ion.get_initial_stoichiometry_half_cell(
            initial_value,
            parameter_values,
            param=param,
            known_value=known_value,
            options=options,
            inputs=inputs,
        )
        c_max = parameter_values.evaluate(param.p.prim.c_max, inputs=inputs)
        parameter_values.update(
            {"Initial concentration in positive electrode [mol.m-3]": x * c_max}
        )
    else:
        """
        Set the initial stoichiometry of each electrode, based on the initial SOC or
        voltage.
        """
        x, y = pybamm.lithium_ion.get_initial_stoichiometries(
            initial_value,
            parameter_values,
            param=param,
            known_value=known_value,
            options=options,
            tol=tol,
            inputs=inputs,
        )
        c_n_max = parameter_values.evaluate(param.n.prim.c_max, inputs=inputs)
        c_p_max = parameter_values.evaluate(param.p.prim.c_max, inputs=inputs)
        parameter_values.update(
            {
                "Initial concentration in negative electrode [mol.m-3]": x * c_n_max,
                "Initial concentration in positive electrode [mol.m-3]": y * c_p_max,
            }
        )

    return parameter_values
