# 3D NMC electrode of Paten et al. (2026)

Data for an NMC cathode half cell reconstructed from a tomographic image,
without carbon-binder domain, used for the direct numerical simulation (DNS)
comparisons of Paten et al. (2026). The cell parameters are
`parameter_sets.paten2026_3d_case()`, and `parameter_sets.load_3d_case_data()`
reads these files.

- `closure_transient.csv`: surface-averaged transient closure variable
  `<s^t>_A(t)` [mol.m-1.A-1] solved on the microstructure, with an
  active-material diffusivity of 4e-14 m2.s-1. Its steady value is
  -416.7084594772896 mol.m-1.A-1. The first values (-51.6 at 1.2 ms) are far
  beyond the short-time limit, a sign that the closure solve does not resolve
  the earliest times; `fit_transient_closure` warns about this.
- `dns_discharge_relaxation.csv`: DNS of a discharge at 50.3357753 A.m-2
  (negative current density) down to 3.0 V, reached at 1742 s, followed by a
  900 s rest: time [s], voltage [V] and current density [A.m-2].
