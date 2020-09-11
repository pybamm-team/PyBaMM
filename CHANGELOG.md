# [v0.2.4](https://github.com/pybamm-team/PyBaMM/tree/v0.2.4) - 2020-09-07

This release adds new operators for more complex models, some basic sensitivity analysis, and a spectral volumes spatial method, as well as some small bug fixes.

## Features

-   Added variables which track the total amount of lithium in the system ([#1136](https://github.com/pybamm-team/PyBaMM/pull/1136))
-   Added `Upwind` and `Downwind` operators for convection ([#1134](https://github.com/pybamm-team/PyBaMM/pull/1134))
-   Added Getting Started notebook on solver options and changing the mesh. Also added a notebook detailing the different thermal options, and a notebook explaining the steps that occur behind the scenes in the `Simulation` class ([#1131](https://github.com/pybamm-team/PyBaMM/pull/1131))
-   Added particle submodel that use a polynomial approximation to the concentration within the electrode particles ([#1130](https://github.com/pybamm-team/PyBaMM/pull/1130))
-   Added `Modulo`, `Floor` and `Ceiling` operators ([#1121](https://github.com/pybamm-team/PyBaMM/pull/1121))
-   Added DFN model for a half cell ([#1121](https://github.com/pybamm-team/PyBaMM/pull/1121))
-   Automatically compute surface area per unit volume based on particle shape for li-ion models ([#1120](https://github.com/pybamm-team/PyBaMM/pull/1120))
-   Added "R-averaged particle concentration" variables ([#1118](https://github.com/pybamm-team/PyBaMM/pull/1118))
-   Added support for sensitivity calculations to the casadi solver ([#1109](https://github.com/pybamm-team/PyBaMM/pull/1109))
-   Added support for index 1 semi-explicit dae equations and sensitivity calculations to JAX BDF solver ([#1107](https://github.com/pybamm-team/PyBaMM/pull/1107))
-   Allowed keyword arguments to be passed to `Simulation.plot()` ([#1099](https://github.com/pybamm-team/PyBaMM/pull/1099))
-   Added the Spectral Volumes spatial method and the submesh that it works with ([#900](https://github.com/pybamm-team/PyBaMM/pull/900))

## Bug fixes

-   Fixed bug where some parameters were not being set by the `EcReactionLimited` SEI model ([#1136](https://github.com/pybamm-team/PyBaMM/pull/1136))
-   Fixed bug on electrolyte potential for `BasicDFNHalfCell` ([#1133](https://github.com/pybamm-team/PyBaMM/pull/1133))
-   Fixed `r_average` to work with `SecondaryBroadcast` ([#1118](https://github.com/pybamm-team/PyBaMM/pull/1118))
-   Fixed finite volume discretisation of spherical integrals ([#1118](https://github.com/pybamm-team/PyBaMM/pull/1118))
-   `t_eval` now gets changed to a `linspace` if a list of length 2 is passed ([#1113](https://github.com/pybamm-team/PyBaMM/pull/1113))
-   Fixed bug when setting a function with an `InputParameter` ([#1111](https://github.com/pybamm-team/PyBaMM/pull/1111))

## Breaking changes

-   The "fast diffusion" particle option has been renamed "uniform profile" ([#1130](https://github.com/pybamm-team/PyBaMM/pull/1130))
-   The modules containing standard parameters are now classes so they can take options
(e.g. `standard_parameters_lithium_ion` is now `LithiumIonParameters`) ([#1120](https://github.com/pybamm-team/PyBaMM/pull/1120))
-   Renamed `quick_plot_vars` to `output_variables` in `Simulation` to be consistent with `QuickPlot`. Passing `quick_plot_vars` to `Simulation.plot()` has been deprecated and `output_variables` should be passed instead ([#1099](https://github.com/pybamm-team/PyBaMM/pull/1099))


# [v0.2.3](https://github.com/pybamm-team/PyBaMM/tree/v0.2.3) - 2020-07-01

This release enables the use of [Google Colab](https://colab.research.google.com/github/pybamm-team/PyBaMM/blob/master/) for running example notebooks, and adds some small new features and bug fixes.

## Features

-   Added JAX evaluator, and ODE solver ([#1038](https://github.com/pybamm-team/PyBaMM/pull/1038))
-   Reformatted Getting Started notebooks ([#1083](https://github.com/pybamm-team/PyBaMM/pull/1083))
-   Reformatted Landesfeind electrolytes ([#1064](https://github.com/pybamm-team/PyBaMM/pull/1064))
-   Adapted examples to be run in Google Colab ([#1061](https://github.com/pybamm-team/PyBaMM/pull/1061))
-   Added some new solvers for algebraic models ([#1059](https://github.com/pybamm-team/PyBaMM/pull/1059))
-   Added `length_scales` attribute to models ([#1058](https://github.com/pybamm-team/PyBaMM/pull/1058))
-   Added averaging in secondary dimensions ([#1057](https://github.com/pybamm-team/PyBaMM/pull/1057))
-   Added SEI reaction based on Yang et. al. 2017 and reduction in porosity ([#1009](https://github.com/pybamm-team/PyBaMM/issues/1009))

## Optimizations

-   Reformatted CasADi "safe" mode to deal with events better ([#1089](https://github.com/pybamm-team/PyBaMM/pull/1089))

## Bug fixes

-   Fixed a bug in `InterstitialDiffusionLimited` ([#1097](https://github.com/pybamm-team/PyBaMM/pull/1097))
-   Fixed `Simulation` to keep different copies of the model so that parameters can be changed between simulations ([#1090](https://github.com/pybamm-team/PyBaMM/pull/1090))
-   Fixed `model.new_copy()` to keep custom submodels ([#1090](https://github.com/pybamm-team/PyBaMM/pull/1090))
-   2D processed variables can now be evaluated at the domain boundaries ([#1088](https://github.com/pybamm-team/PyBaMM/pull/1088))
-   Update the default variable points to better capture behaviour in the solid particles in li-ion models ([#1081](https://github.com/pybamm-team/PyBaMM/pull/1081))
-   Fix `QuickPlot` to display variables discretised by FEM (in y-z) properly ([#1078](https://github.com/pybamm-team/PyBaMM/pull/1078))
-   Add length scales to `EffectiveResistance` models ([#1071](https://github.com/pybamm-team/PyBaMM/pull/1071))
-   Allowed for pybamm functions exp, sin, cos, sqrt to be used in expression trees that
    are converted to casadi format ([#1067](https://github.com/pybamm-team/PyBaMM/pull/1067))
-   Fix a bug where variables that depend on y and z were transposed in `QuickPlot` ([#1055](https://github.com/pybamm-team/PyBaMM/pull/1055))

## Breaking changes

-   `Simulation.specs` and `Simulation.set_defaults` have been deprecated. Users should create a new `Simulation` object for each different case instead ([#1090](https://github.com/pybamm-team/PyBaMM/pull/1090))
-   The solution times `t_eval` must now be provided to `Simulation.solve()` when not using an experiment or prescribing the current using drive cycle data ([#1086](https://github.com/pybamm-team/PyBaMM/pull/1086))

# [v0.2.2](https://github.com/pybamm-team/PyBaMM/tree/v0.2.2) - 2020-06-01

New SEI models, simplification of submodel structure, as well as optimisations and general bug fixes.

## Features

-   Reformatted `Geometry` and `Mesh` classes ([#1032](https://github.com/pybamm-team/PyBaMM/pull/1032))
-   Added arbitrary geometry to the lumped thermal model ([#718](https://github.com/pybamm-team/PyBaMM/issues/718))
-   Allowed `ProcessedVariable` to handle cases where `len(solution.t)=1` ([#1020](https://github.com/pybamm-team/PyBaMM/pull/1020))
-   Added `BackwardIndefiniteIntegral` symbol ([#1014](https://github.com/pybamm-team/PyBaMM/pull/1014))
-   Added `plot` and `plot2D` to enable easy plotting of `pybamm.Array` objects ([#1008](https://github.com/pybamm-team/PyBaMM/pull/1008))
-   Updated effective current collector models and added example notebook ([#1007](https://github.com/pybamm-team/PyBaMM/pull/1007))
-   Added SEI film resistance as an option ([#994](https://github.com/pybamm-team/PyBaMM/pull/994))
-   Added `parameters` attribute to `pybamm.BaseModel` and `pybamm.Geometry` that lists all of the required parameters ([#993](https://github.com/pybamm-team/PyBaMM/pull/993))
-   Added tab, edge, and surface cooling ([#965](https://github.com/pybamm-team/PyBaMM/pull/965))
-   Added functionality to solver to automatically discretise a 0D model ([#947](https://github.com/pybamm-team/PyBaMM/pull/947))
-   Added sensitivity to `CasadiAlgebraicSolver` ([#940](https://github.com/pybamm-team/PyBaMM/pull/940))
-   Added `ProcessedSymbolicVariable` class, which can handle symbolic variables (i.e. variables for which the inputs are symbolic) ([#940](https://github.com/pybamm-team/PyBaMM/pull/940))
-   Made `QuickPlot` compatible with Google Colab ([#935](https://github.com/pybamm-team/PyBaMM/pull/935))
-   Added `BasicFull` model for lead-acid ([#932](https://github.com/pybamm-team/PyBaMM/pull/932))
-   Added 'arctan' function  ([#973](https://github.com/pybamm-team/PyBaMM/pull/973))

## Optimizations

-   Implementing the use of GitHub Actions for CI ([#855](https://github.com/pybamm-team/PyBaMM/pull/855))
-   Changed default solver for DAE models to `CasadiSolver` ([#978](https://github.com/pybamm-team/PyBaMM/pull/978))
-   Added some extra simplifications to the expression tree ([#971](https://github.com/pybamm-team/PyBaMM/pull/971))
-   Changed the behaviour of "safe" mode in `CasadiSolver` ([#956](https://github.com/pybamm-team/PyBaMM/pull/956))
-   Sped up model building ([#927](https://github.com/pybamm-team/PyBaMM/pull/927))
-   Changed default solver for lead-acid to `CasadiSolver` ([#927](https://github.com/pybamm-team/PyBaMM/pull/927))

## Bug fixes

-   Fix a bug where slider plots do not update properly in notebooks ([#1041](https://github.com/pybamm-team/PyBaMM/pull/1041))
-   Fix storing and plotting external variables in the solution ([#1026](https://github.com/pybamm-team/PyBaMM/pull/1026))
-   Fix running a simulation with a model that is already discretized ([#1025](https://github.com/pybamm-team/PyBaMM/pull/1025))
-   Fix CI not triggering for PR. ([#1013](https://github.com/pybamm-team/PyBaMM/pull/1013))
-   Fix schedule testing running too often. ([#1010](https://github.com/pybamm-team/PyBaMM/pull/1010))
-   Fix doctests failing due to mismatch in unsorted output.([#990](https://github.com/pybamm-team/PyBaMM/pull/990))
-   Added extra checks when creating a model, for clearer errors ([#971](https://github.com/pybamm-team/PyBaMM/pull/971))
-   Fixed `Interpolant` ids to allow processing ([#962](https://github.com/pybamm-team/PyBaMM/pull/962))
-   Fixed a bug in the initial conditions of the potential pair model ([#954](https://github.com/pybamm-team/PyBaMM/pull/954))
-   Changed simulation attributes to assign copies rather than the objects themselves ([#952](https://github.com/pybamm-team/PyBaMM/pull/952))
-   Added default values to base model so that it works with the `Simulation` class ([#952](https://github.com/pybamm-team/PyBaMM/pull/952))
-   Fixed solver to recompute initial conditions when inputs are changed ([#951](https://github.com/pybamm-team/PyBaMM/pull/951))
-   Reformatted thermal submodels ([#938](https://github.com/pybamm-team/PyBaMM/pull/938))
-   Reformatted electrolyte submodels ([#927](https://github.com/pybamm-team/PyBaMM/pull/927))
-   Reformatted convection submodels ([#635](https://github.com/pybamm-team/PyBaMM/pull/635))

## Breaking changes

-   Geometry should no longer be given keys 'primary' or 'secondary' ([#1032](https://github.com/pybamm-team/PyBaMM/pull/1032))
-   Calls to `ProcessedVariable` objects are now made using dimensional time and space ([#1028](https://github.com/pybamm-team/PyBaMM/pull/1028))
-   For variables discretised using finite elements the result returned by calling `ProcessedVariable` is now transposed ([#1020](https://github.com/pybamm-team/PyBaMM/pull/1020))
-   Renamed "surface area density" to "surface area to volume ratio" ([#975](https://github.com/pybamm-team/PyBaMM/pull/975))
-   Replaced "reaction rate" with "exchange-current density" ([#975](https://github.com/pybamm-team/PyBaMM/pull/975))
-   Changed the implementation of reactions in submodels ([#948](https://github.com/pybamm-team/PyBaMM/pull/948))
-   Removed some inputs like `T_inf`, `R_g` and activation energies to some of the standard function parameters. This is because each of those inputs is specific to a particular function (e.g. the reference temperature at which the function was measured). To change a property such as the activation energy, users should create a new function, specifying the relevant property as a `Parameter` or `InputParameter` ([#942](https://github.com/pybamm-team/PyBaMM/pull/942))
-   The thermal option 'xyz-lumped' has been removed. The option 'thermal current collector' has also been removed ([#938](https://github.com/pybamm-team/PyBaMM/pull/938))
-   The 'C-rate' parameter has been deprecated. Use 'Current function [A]' instead. The cell capacity can be accessed as 'Cell capacity [A.h]', and used to calculate current from C-rate ([#952](https://github.com/pybamm-team/PyBaMM/pull/952))

# [v0.2.1](https://github.com/pybamm-team/PyBaMM/tree/v0.2.1) - 2020-03-31

New expression tree node types, models, parameter sets and solvers, as well as general bug fixes and new examples.

## Features

-   Store variable slices in model for inspection ([#925](https://github.com/pybamm-team/PyBaMM/pull/925))
-   Added LiNiCoO2 parameter set from Ecker et. al. ([#922](https://github.com/pybamm-team/PyBaMM/pull/922))
-   Made t_plus (optionally) a function of electrolyte concentration, and added (1 + dlnf/dlnc) to models ([#921](https://github.com/pybamm-team/PyBaMM/pull/921))
-   Added `DummySolver` for empty models ([#915](https://github.com/pybamm-team/PyBaMM/pull/915))
-   Added functionality to broadcast to edges ([#891](https://github.com/pybamm-team/PyBaMM/pull/891))
-   Reformatted and cleaned up `QuickPlot` ([#886](https://github.com/pybamm-team/PyBaMM/pull/886))
-   Added thermal effects to lead-acid models ([#885](https://github.com/pybamm-team/PyBaMM/pull/885))
-   Added a helper function for info on function parameters ([#881](https://github.com/pybamm-team/PyBaMM/pull/881))
-   Added additional notebooks showing how to create and compare models ([#877](https://github.com/pybamm-team/PyBaMM/pull/877))
-   Added `Minimum`, `Maximum` and `Sign` operators
    ([#876](https://github.com/pybamm-team/PyBaMM/pull/876))
-   Added a search feature to `FuzzyDict` ([#875](https://github.com/pybamm-team/PyBaMM/pull/875))
-   Add ambient temperature as a function of time ([#872](https://github.com/pybamm-team/PyBaMM/pull/872))
-   Added `CasadiAlgebraicSolver` for solving algebraic systems with CasADi ([#868](https://github.com/pybamm-team/PyBaMM/pull/868))
-   Added electrolyte functions from Landesfeind ([#860](https://github.com/pybamm-team/PyBaMM/pull/860))
-   Add new symbols `VariableDot`, representing the derivative of a variable wrt time,
    and `StateVectorDot`, representing the derivative of a state vector wrt time
    ([#858](https://github.com/pybamm-team/PyBaMM/issues/858))

## Bug fixes

-   Filter out discontinuities that occur after solve times
    ([#941](https://github.com/pybamm-team/PyBaMM/pull/945))
-   Fixed tight layout for QuickPlot in jupyter notebooks ([#930](https://github.com/pybamm-team/PyBaMM/pull/930))
-   Fixed bug raised if function returns a scalar ([#919](https://github.com/pybamm-team/PyBaMM/pull/919))
-   Fixed event handling in `ScipySolver` ([#905](https://github.com/pybamm-team/PyBaMM/pull/905))
-   Made input handling clearer in solvers ([#905](https://github.com/pybamm-team/PyBaMM/pull/905))
-   Updated Getting started notebook 2 ([#903](https://github.com/pybamm-team/PyBaMM/pull/903))
-   Reformatted external circuit submodels ([#879](https://github.com/pybamm-team/PyBaMM/pull/879))
-   Some bug fixes to generalize specifying models that aren't battery models, see [#846](https://github.com/pybamm-team/PyBaMM/issues/846)
-   Reformatted interface submodels to be more readable ([#866](https://github.com/pybamm-team/PyBaMM/pull/866))
-   Removed double-counted "number of electrodes connected in parallel" from simulation ([#864](https://github.com/pybamm-team/PyBaMM/pull/864))

## Breaking changes

-   Changed keyword argument `u` for inputs (when evaluating an object) to `inputs` ([#905](https://github.com/pybamm-team/PyBaMM/pull/905))
-   Removed "set external temperature" and "set external potential" options. Use "external submodels" option instead ([#862](https://github.com/pybamm-team/PyBaMM/pull/862))

# [v0.2.0](https://github.com/pybamm-team/PyBaMM/tree/v0.2.0) - 2020-02-26

This release introduces many new features and optimizations. All models can now be solved using the pip installation - in particular, the DFN can be solved in around 0.1s. Other highlights include an improved user interface, simulations of experimental protocols (GITT, CCCV, etc), new parameter sets for NCA and LGM50, drive cycles, "input parameters" and "external variables" for quickly solving models with different parameter values and coupling with external software, and general bug fixes and optimizations.

## Features

-   Added LG M50 parameter set ([#854](https://github.com/pybamm-team/PyBaMM/pull/854))
-   Changed rootfinding algorithm to CasADi, scipy.optimize.root still accessible as an option ([#844](https://github.com/pybamm-team/PyBaMM/pull/844))
-   Added capacitance effects to lithium-ion models ([#842](https://github.com/pybamm-team/PyBaMM/pull/842))
-   Added NCA parameter set ([#824](https://github.com/pybamm-team/PyBaMM/pull/824))
-   Added functionality to `Solution` that automatically gets `t_eval` from the data when simulating drive cycles and performs checks to ensure the output has the required resolution to accurately capture the input current ([#819](https://github.com/pybamm-team/PyBaMM/pull/819))
-   Added `Citations` object to print references when specific functionality is used ([#818](https://github.com/pybamm-team/PyBaMM/pull/818))
-   Updated `Solution` to allow exporting to matlab and csv formats ([#811](https://github.com/pybamm-team/PyBaMM/pull/811))
-   Allow porosity to vary in space ([#809](https://github.com/pybamm-team/PyBaMM/pull/809))
-   Added functionality to solve DAE models with non-smooth current inputs ([#808](https://github.com/pybamm-team/PyBaMM/pull/808))
-   Added functionality to simulate experiments and testing protocols ([#807](https://github.com/pybamm-team/PyBaMM/pull/807))
-   Added fuzzy string matching for parameters and variables ([#796](https://github.com/pybamm-team/PyBaMM/pull/796))
-   Changed ParameterValues to raise an error when a parameter that wasn't previously defined is updated ([#796](https://github.com/pybamm-team/PyBaMM/pull/796))
-   Added some basic models (BasicSPM and BasicDFN) in order to clearly demonstrate the PyBaMM model structure for battery models ([#795](https://github.com/pybamm-team/PyBaMM/pull/795))
-   Allow initial conditions in the particle to depend on x ([#786](https://github.com/pybamm-team/PyBaMM/pull/786))
-   Added the harmonic mean to the Finite Volume method, which is now used when computing fluxes ([#783](https://github.com/pybamm-team/PyBaMM/pull/783))
-   Refactored `Solution` to make it a dictionary that contains all of the solution variables. This automatically creates `ProcessedVariable` objects when required, so that the solution can be obtained much more easily. ([#781](https://github.com/pybamm-team/PyBaMM/pull/781))
-   Added notebook to explain broadcasts ([#776](https://github.com/pybamm-team/PyBaMM/pull/776))
-   Added a step to discretisation that automatically compute the inverse of the mass matrix of the differential part of the problem so that the underlying DAEs can be provided in semi-explicit form, as required by the CasADi solver ([#769](https://github.com/pybamm-team/PyBaMM/pull/769))
-   Added the gradient operation for the Finite Element Method ([#767](https://github.com/pybamm-team/PyBaMM/pull/767))
-   Added `InputParameter` node for quickly changing parameter values ([#752](https://github.com/pybamm-team/PyBaMM/pull/752))
-   Added submodels for operating modes other than current-controlled ([#751](https://github.com/pybamm-team/PyBaMM/pull/751))
-   Changed finite volume discretisation to use exact values provided by Neumann boundary conditions when computing the gradient instead of adding ghost nodes([#748](https://github.com/pybamm-team/PyBaMM/pull/748))
-   Added optional R(x) distribution in particle models ([#745](https://github.com/pybamm-team/PyBaMM/pull/745))
-   Generalized importing of external variables ([#728](https://github.com/pybamm-team/PyBaMM/pull/728))
-   Separated active and inactive material volume fractions ([#726](https://github.com/pybamm-team/PyBaMM/pull/726))
-   Added submodels for tortuosity ([#726](https://github.com/pybamm-team/PyBaMM/pull/726))
-   Simplified the interface for setting current functions ([#723](https://github.com/pybamm-team/PyBaMM/pull/723))
-   Added Heaviside operator ([#723](https://github.com/pybamm-team/PyBaMM/pull/723))
-   New extrapolation methods ([#707](https://github.com/pybamm-team/PyBaMM/pull/707))
-   Added some "Getting Started" documentation ([#703](https://github.com/pybamm-team/PyBaMM/pull/703))
-   Allow abs tolerance to be set by variable for IDA KLU solver ([#700](https://github.com/pybamm-team/PyBaMM/pull/700))
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

-   Now simplifying objects that are constant as soon as they are created ([#801](https://github.com/pybamm-team/PyBaMM/pull/801))
-   Simplified solver interface ([#800](https://github.com/pybamm-team/PyBaMM/pull/800))
-   Added caching for shape evaluation, used during discretisation ([#780](https://github.com/pybamm-team/PyBaMM/pull/780))
-   Added an option to skip model checks during discretisation, which could be slow for large models ([#739](https://github.com/pybamm-team/PyBaMM/pull/739))
-   Use CasADi's automatic differentation algorithms by default when solving a model ([#714](https://github.com/pybamm-team/PyBaMM/pull/714))
-   Avoid re-checking size when making a copy of an `Index` object ([#656](https://github.com/pybamm-team/PyBaMM/pull/656))
-   Avoid recalculating `_evaluation_array` when making a copy of a `StateVector` object ([#653](https://github.com/pybamm-team/PyBaMM/pull/653))

## Bug fixes

-   Fixed a bug where current loaded from data was incorrectly scaled with the cell capacity ([#852](https://github.com/pybamm-team/PyBaMM/pull/852))
-   Moved evaluation of initial conditions to solver ([#839](https://github.com/pybamm-team/PyBaMM/pull/839))
-   Fixed a bug where the first line of the data wasn't loaded when parameters are loaded from data ([#819](https://github.com/pybamm-team/PyBaMM/pull/819))
-   Made `graphviz` an optional dependency ([#810](https://github.com/pybamm-team/PyBaMM/pull/810))
-   Fixed examples to run with basic pip installation ([#800](https://github.com/pybamm-team/PyBaMM/pull/800))
-   Added events for CasADi solver when stepping ([#800](https://github.com/pybamm-team/PyBaMM/pull/800))
-   Improved implementation of broadcasts ([#776](https://github.com/pybamm-team/PyBaMM/pull/776))
-   Fixed a bug which meant that the Ohmic heating in the current collectors was incorrect if using the Finite Element Method ([#767](https://github.com/pybamm-team/PyBaMM/pull/767))
-   Improved automatic broadcasting ([#747](https://github.com/pybamm-team/PyBaMM/pull/747))
-   Fixed bug with wrong temperature in initial conditions ([#737](https://github.com/pybamm-team/PyBaMM/pull/737))
-   Improved flexibility of parameter values so that parameters (such as diffusivity or current) can be set as functions or scalars ([#723](https://github.com/pybamm-team/PyBaMM/pull/723))
-   Fixed a bug where boundary conditions were sometimes handled incorrectly in 1+1D models ([#713](https://github.com/pybamm-team/PyBaMM/pull/713))
-   Corrected a sign error in Dirichlet boundary conditions in the Finite Element Method ([#706](https://github.com/pybamm-team/PyBaMM/pull/706))
-   Passed the correct dimensional temperature to open circuit potential ([#702](https://github.com/pybamm-team/PyBaMM/pull/702))
-   Added missing temperature dependence in electrolyte and interface submodels ([#698](https://github.com/pybamm-team/PyBaMM/pull/698))
-   Fixed differentiation of functions that have more than one argument ([#687](https://github.com/pybamm-team/PyBaMM/pull/687))
-   Added warning if `ProcessedVariable` is called outside its interpolation range ([#681](https://github.com/pybamm-team/PyBaMM/pull/681))
-   Updated installation instructions for Mac OS ([#680](https://github.com/pybamm-team/PyBaMM/pull/680))
-   Improved the way `ProcessedVariable` objects are created in higher dimensions ([#581](https://github.com/pybamm-team/PyBaMM/pull/581))

## Breaking changes

-   Time for solver should now be given in seconds ([#832](https://github.com/pybamm-team/PyBaMM/pull/832))
-   Model events are now represented as a list of `pybamm.Event` ([#759](https://github.com/pybamm-team/PyBaMM/issues/759)
-   Removed `ParameterValues.update_model`, whose functionality is now replaced by `InputParameter` ([#801](https://github.com/pybamm-team/PyBaMM/pull/801))
-   Removed `Outer` and `Kron` nodes as no longer used ([#777](https://github.com/pybamm-team/PyBaMM/pull/777))
-   Moved `results` to separate repositories ([#761](https://github.com/pybamm-team/PyBaMM/pull/761))
-   The parameters "Bruggeman coefficient" must now be specified separately as "Bruggeman coefficient (electrolyte)" and "Bruggeman coefficient (electrode)"
-   The current classes (`GetConstantCurrent`, `GetUserCurrent` and `GetUserData`) have now been removed. Please refer to the [`change-input-current` notebook](https://github.com/pybamm-team/PyBaMM/blob/master/examples/notebooks/change-input-current.ipynb) for information on how to specify an input current
-   Parameter functions must now use pybamm functions instead of numpy functions (e.g. `pybamm.exp` instead of `numpy.exp`), as these are then used to construct the expression tree directly. Generally, pybamm syntax follows numpy syntax; please get in touch if a function you need is missing.
-   The current must now be updated by changing "Current function [A]" or "C-rate" instead of "Typical current [A]"


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
