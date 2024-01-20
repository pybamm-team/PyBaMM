def raise_error():
    raise NotImplementedError(
        "parameters cli has been deprecated. "
        "Parameters should now be defined via python files (see "
        "https://github.com/pybamm-team/PyBaMM/tree/develop/pybamm/input/parameters/lithium_ion/Ai2020.py"
        " for example)"
    )


def add_parameter(arguments=None):
    raise_error()


def remove_parameter(arguments=None):
    raise_error()


def edit_parameter(arguments=None):
    raise_error()
