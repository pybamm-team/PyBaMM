# [Unreleased](https://github.com/pybamm-team/PyBaMM)

## Features

- Allow parameters to be set by material or by specifying a particular paper (#647)
- Set relative and absolute tolerances independently in solvers (#645)

## Optimizations

- Avoid re-checking size when making a copy of an `Index` object (#656)
- Avoid recalculating `_evaluation_array` when making a copy of a `StateVector` object (#653)

## Bug fixes

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
- Algebraic (root-finding)
