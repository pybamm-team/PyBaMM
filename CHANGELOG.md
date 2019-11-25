# [Unreleased](https://github.com/pybamm-team/PyBaMM)

## Features

-   Separated active and inactive material volume fractions ([#726](https://github.com/pybamm-team/PyBaMM/pull/726))
-   Added submodels for tortuosity ([#726](https://github.com/pybamm-team/PyBaMM/pull/726))
-   Simplified the interface for setting current functions ([#723](https://github.com/pybamm-team/PyBaMM/pull/723))
-   Added Heaviside operator ([#723](https://github.com/pybamm-team/PyBaMM/pull/723))
-   Added some "Getting Started" documentation ([#703](https://github.com/pybamm-team/PyBaMM/pull/703))
-   Added Simulation class ([#693](https://github.com/pybamm-team/PyBaMM/pull/693)) with load/save functionality ([#732](https://github.com/pybamm-team/PyBaMM/pull/732))
-   Added interface to CasADi solver ([#687](https://github.com/pybamm-team/PyBaMM/pull/687), [#691](https://github.com/pybamm-team/PyBaMM/pull/691), [#714](https://github.com/pybamm-team/PyBaMM/pull/714)). This makes the SUNDIALS DAE solvers (Scikits and KLU) truly optional (though IDA KLU is recommended for solving the DFN).
-   Added option to use CasADi's Algorithmic Differentiation framework to calculate Jacobians ([#687](https://github.com/pybamm-team/PyBaMM/pull/687))
-   Added method to evaluate parameters more easily ([#669](https://github.com/pybamm-team/PyBaMM/pull/669))
-   Added `Jacobian` class to reuse known Jacobians of expressions ([#665](https://github.com/pybamm-team/PyBaMM/pull/670))
-   Added `Interpolant` class to interpolate experimental data (e.g. OCP curves) ([#661](https://github.com/pybamm-team/PyBaMM/pull/661))
-   Added interface (via pybind11) to sundials with the IDA KLU sparse linear solver ([#657](https://github.com/pybamm-team/PyBaMM/pull/657))
-   Allowed parameters to be set by material or by specifying a particular paper ([#647](https://github.com/pybamm-team/PyBaMM/pull/647))
-   Set relative and absolute tolerances independently in solvers ([#645](https://github.com/pybamm-team/PyBaMM/pull/645))
-   Added basic method to allow (a part of) the State Vector to be updated with results obtained from another solution or package ([#624](https://github.com/pybamm-team/PyBaMM/pull/624))
-   Added some non-uniform meshes in 1D and 2D ([#617](https://github.com/pybamm-team/PyBaMM/pull/617))

## Optimizations

-   Added an option to skip model checks during discretisation, which could be slow for large models ([#739](https://github.com/pybamm-team/PyBaMM/pull/739))
-   Use CasADi's automatic differentation algorithms by default when solving a model ([#714](https://github.com/pybamm-team/PyBaMM/pull/714))
-   Avoid re-checking size when making a copy of an `Index` object ([#656](https://github.com/pybamm-team/PyBaMM/pull/656))
-   Avoid recalculating `_evaluation_array` when making a copy of a `StateVector` object ([#653](https://github.com/pybamm-team/PyBaMM/pull/653))

## Bug fixes

-   Fixed bug with wrong temperature in initial conditions ([#737](https://github.com/pybamm-team/PyBaMM/pull/737))
-   Improved flexibility of parameter values so that parameters (such as diffusivity or current) can be set as functions or scalars ([#723](https://github.com/pybamm-team/PyBaMM/pull/723))
-   Fixed a bug where boundary conditions were sometimes handled incorrectly in 1+1D models ([#713](https://github.com/pybamm-team/PyBaMM/pull/713))
-   Corrected a sign error in Dirichlet boundary conditions in the Finite Element Method ([#706](https://github.com/pybamm-team/PyBaMM/pull/706))
-   Passed the correct dimensional temperature to open circuit potential ([#702](https://github.com/pybamm-team/PyBaMM/pull/702))
-   Added missing temperature dependence in electrolyte and interface submodels ([#698](https://github.com/pybamm-team/PyBaMM/pull/698))
-   Fixed differentiation of functions that have more than one argument ([#687](https://github.com/pybamm-team/PyBaMM/pull/687))
-   Added warning if `ProcessedVariable` is called outside its interpolation range ([#681](https://github.com/pybamm-team/PyBaMM/pull/681))
-   Improved the way `ProcessedVariable` objects are created in higher dimensions ([#581](https://github.com/pybamm-team/PyBaMM/pull/581))

## Breaking changes

-   The parameters "Bruggeman coefficient" must now be specified separately as "Bruggeman coefficient (electrolyte)" and "Bruggeman coefficient (electrode)"
-   The current classes (`GetConstantCurrent`, `GetUserCurrent` and `GetUserData`) have now been removed. Please refer to the [`change-input-current` notebook](https://github.com/pybamm-team/PyBaMM/blob/master/examples/notebooks/change-input-current.ipynb) for information on how to specify an input current
-   Parameter functions must now use pybamm functions instead of numpy functions (e.g. `pybamm.exp` instead of `numpy.exp`), as these are then used to construct the expression tree directly. Generally, pybamm syntax follows numpy syntax; please get in touch if a function you need is missing.


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
