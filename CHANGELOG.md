# [Unreleased](https://github.com/pybamm-team/PyBaMM)

## Features

-   Add interface to CasADi solver ([#687](https://github.com/pybamm-team/PyBaMM/pull/687))
-   Add option to use CasADi's Algorithmic Differentiation framework to calculate Jacobians ([#687](https://github.com/pybamm-team/PyBaMM/pull/687))
-   Add method to evaluate parameters more easily ([#669](https://github.com/pybamm-team/PyBaMM/pull/669))
-   Add `Jacobian` class to reuse known Jacobians of expressions ([#665](https://github.com/pybamm-team/PyBaMM/pull/670))
-   Add `Interpolant` class to interpolate experimental data (e.g. OCP curves) ([#661](https://github.com/pybamm-team/PyBaMM/pull/661))
-   Added interface (via pybind11) to sundials with the IDA KLU sparse linear solver ([#657](https://github.com/pybamm-team/PyBaMM/pull/657))
-   Allow parameters to be set by material or by specifying a particular paper ([#647](https://github.com/pybamm-team/PyBaMM/pull/647))
-   Set relative and absolute tolerances independently in solvers ([#645](https://github.com/pybamm-team/PyBaMM/pull/645))
-   Add some non-uniform meshes in 1D and 2D ([#617](https://github.com/pybamm-team/PyBaMM/pull/617))

## Optimizations

-   Avoid re-checking size when making a copy of an `Index` object ([#656](https://github.com/pybamm-team/PyBaMM/pull/656))
-   Avoid recalculating `_evaluation_array` when making a copy of a `StateVector` object ([#653](https://github.com/pybamm-team/PyBaMM/pull/653))

## Bug fixes

-   Adds missing temperature dependence in electrolyte and interface submodels ([#698](https://github.com/pybamm-team/PyBaMM/pull/698))
-   Fix differentiation of functions that have more than one argument ([#687](https://github.com/pybamm-team/PyBaMM/pull/687))
-   Add warning if `ProcessedVariable` is called outisde its interpolation range ([#681](https://github.com/pybamm-team/PyBaMM/pull/681))
-   Improve the way `ProcessedVariable` objects are created in higher dimensions ([#581](https://github.com/pybamm-team/PyBaMM/pull/581))

# [v0.1.0](https://github.com/pybamm-team/PyBaMM/tree/v0.1.0) - 2019-10-08

This is the first official version of PyBaMM.
Please note that PyBaMM in still under active development, and so the API may change in the future.

## Features

### Models

#### Lithium-ion

- Single Particle Model (SPM)
- Single Particle Model with electrolyte (SPMe)
- Doyle-Fuller-Newman (DFN) model

with the following optional physics:

- Thermal effects
- Fast diffusion in particles
- 2+1D (pouch cell)

#### Lead-acid

- Leading-Order Quasi-Static model
- First-Order Quasi-Static model
- Composite model
- Full model

with the following optional physics:

- Hydrolysis side reaction
- Capacitance effects
- 2+1D


### Spatial discretisations

- Finite Volume (1D only)
- Finite Element (scikit, 2D only)

### Solvers

- Scipy
- Scikits ODE
- Scikits DAE
- IDA KLU sparse linear solver (Sundials)
- Algebraic (root-finding)
