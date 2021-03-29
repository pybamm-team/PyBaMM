import pybamm

a_n = pybamm.FullBroadcast(
    pybamm.Scalar(0), ["negative electrode"], "current collector"
)
a_s = pybamm.FullBroadcast(pybamm.Scalar(0), ["separator"], "current collector")
a_p = pybamm.FullBroadcast(
    pybamm.Scalar(0), ["positive electrode"], "current collector"
)
b = pybamm.PrimaryBroadcast(pybamm.Scalar(1), "current collector")

coupled_variables = {
    "Negative electrode surface area to volume ratio": a_n,
    "Positive electrode surface area to volume ratio": a_p,
    "Negative electrode interfacial current density": a_n,
    "Positive electrode interfacial current density": a_p,
    "Negative electrode reaction overpotential": a_n,
    "Positive electrode reaction overpotential": a_p,
    "Negative electrode entropic change": a_n,
    "Positive electrode entropic change": a_p,
    "Negative electrolyte potential": a_n,
    "Separator electrolyte potential": a_s,
    "Positive electrolyte potential": a_p,
    "Negative electrolyte current density": a_n,
    "Separator current density": a_s,
    "Positive electrolyte current density": a_p,
    "Electrolyte potential": pybamm.concatenation(a_n, a_s, a_p),
    "Electrolyte current density": pybamm.concatenation(a_n, a_s, a_p),
    "Negative electrode potential": a_n,
    "Negative electrode current density": a_n,
    "Positive electrode potential": a_p,
    "Positive electrode current density": a_p,
    "Current collector current density": b,
    "Negative current collector potential": b,
    "Positive current collector potential": b,
}
