#
# Standard variables for the models
#
import pybamm


class StandardVariables:
    def __init__(self):
        # Discharge capacity and energy
        self.Q_Ah = pybamm.Variable("Discharge capacity [A.h]")
        self.Q_Wh = pybamm.Variable("Discharge energy [W.h]")

        # Throughput capacity and energy (cumulative)
        self.Qt_Ah = pybamm.Variable("Throughput capacity [A.h]")
        self.Qt_Wh = pybamm.Variable("Throughput energy [W.h]")

        # Electrode potential
        self.phi_s_n = pybamm.Variable(
            "Negative electrode potential",
            domain="negative electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.phi_s_p = pybamm.Variable(
            "Positive electrode potential",
            domain="positive electrode",
            auxiliary_domains={"secondary": "current collector"},
        )

        # Potential difference
        self.delta_phi_n = pybamm.Variable(
            "Negative electrode surface potential difference",
            domain="negative electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.delta_phi_p = pybamm.Variable(
            "Positive electrode surface potential difference",
            domain="positive electrode",
            auxiliary_domains={"secondary": "current collector"},
        )

        self.delta_phi_n_av = pybamm.Variable(
            "X-averaged negative electrode surface potential difference",
            domain="current collector",
        )
        self.delta_phi_p_av = pybamm.Variable(
            "X-averaged positive electrode surface potential difference",
            domain="current collector",
        )

        # current collector variables
        self.phi_s_cn = pybamm.Variable(
            "Negative current collector potential", domain="current collector"
        )
        self.phi_s_cp = pybamm.Variable(
            "Positive current collector potential", domain="current collector"
        )
        self.i_boundary_cc = pybamm.Variable(
            "Current collector current density", domain="current collector"
        )
        self.phi_s_cn_composite = pybamm.Variable(
            "Composite negative current collector potential", domain="current collector"
        )
        self.phi_s_cp_composite = pybamm.Variable(
            "Composite positive current collector potential", domain="current collector"
        )
        self.i_boundary_cc_composite = pybamm.Variable(
            "Composite current collector current density", domain="current collector"
        )

    def __setattr__(self, name, value):
        value.print_name = name
        super().__setattr__(name, value)


standard_variables = StandardVariables()
