#
# Standard variables for the models
#
import pybamm


class StandardVariables:
    def __init__(self):
        # Potential difference
        self.delta_phi_n = pybamm.Variable(
            "Negative electrode surface potential difference [V]",
            domain="negative electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.delta_phi_p = pybamm.Variable(
            "Positive electrode surface potential difference [V]",
            domain="positive electrode",
            auxiliary_domains={"secondary": "current collector"},
        )

        self.delta_phi_n_av = pybamm.Variable(
            "X-averaged negative electrode surface potential difference [V]",
            domain="current collector",
        )
        self.delta_phi_p_av = pybamm.Variable(
            "X-averaged positive electrode surface potential difference [V]",
            domain="current collector",
        )

    def __setattr__(self, name, value):
        value.print_name = name
        super().__setattr__(name, value)


standard_variables = StandardVariables()
