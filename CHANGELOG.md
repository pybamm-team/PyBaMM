# [Unreleased](https://github.com/pybamm-team/PyBaMM/)

## Features

- Renamed "Terminal voltage [V]" to just "Voltage [V]". "Terminal voltage [V]" can still be used and will return the same value as "Voltage [V]".
- Added "Negative electrode surface potential difference at separator interface [V]", which is the value of the surface potential difference (`phi_s - phi_e`) at the anode/separator interface, commonly controlled in fast-charging algorithms to avoid plating. Also added "Positive electrode surface potential difference at separator interface [V]". ([#2740](https://github.com/pybamm-team/PyBaMM/pull/2740))
- Added "Open-circuit voltage [V]", which is the open-circuit voltage as calculated from the bulk particle concentrations. The old variable "Measured open circuit voltage [V]", which referred to the open-circuit potential as calculated from the surface particle concentrations, has been renamed to "Surface open-circuit voltage [V]". ([#2740](https://github.com/pybamm-team/PyBaMM/pull/2740))
- Added an example for `plot_voltage_components`, explaining what the different voltage components are. ([#2740](https://github.com/pybamm-team/PyBaMM/pull/2740))

## Bug fixes

- Fixed `plot_voltage_components` so that the sum of overpotentials is now equal to the voltage ([#2740](https://github.com/pybamm-team/PyBaMM/pull/2740))

## Optimizations

- Migrated to [Lychee](https://github.com/lycheeverse/lychee-action) workflow for checking URLs [#2734](https://github.com/pybamm-team/PyBaMM/pull/2734)

## Breaking changes

- Renamed "Measured open circuit voltage [V]" to "Surface open-circuit voltage [V]". This variable was calculated from surface particle concentrations, and hence "hid" the overpotential from particle gradients. The new variable "Open-circuit voltage [V]" is calculated from bulk particle concentrations instead. ([#2740](https://github.com/pybamm-team/PyBaMM/pull/2740))
- Renamed all references to "open circuit" to be "open-circuit" instead. ([#2740](https://github.com/pybamm-team/PyBaMM/pull/2740))
- All PyBaMM models are now dimensional. This has been benchmarked against dimensionless models and found to give around the same solve time. Implementing dimensional models greatly reduces the barrier to entry for adding new models. However, this comes with several breaking changes: (i) the `timescale` and `length_scales` attributes of a model have been removed (they are no longer needed) (ii) several dimensionless variables are no longer defined, but the corresponding dimensional variables can still be accessed by adding the units to the name (iii) some parameters used only for non-dimensionalization, such as "Typical current [A]", have been removed ([#2419](https://github.com/pybamm-team/PyBaMM/pull/2419))

# [v23.2](https://github.com/pybamm-team/PyBaMM/tree/v23.2) - 2023-02-28

## Features

- Added an option for using a banded jacobian and sundials banded solvers for the IDAKLU solve [#2677](https://github.com/pybamm-team/PyBaMM/pull/2677)
- The "particle size" option can now be a tuple to allow different behaviour in each electrode([#2672](https://github.com/pybamm-team/PyBaMM/pull/2672)).
- Added temperature control to experiment class. [#2518](https://github.com/pybamm-team/PyBaMM/pull/2518)

## Bug fixes

- Fixed current_sigmoid_ocp to be valid for both electrodes ([#2719](https://github.com/pybamm-team/PyBaMM/pull/2719)).
- Fixed the length scaling for the first dimension of r-R plots ([#2663](https://github.com/pybamm-team/PyBaMM/pull/2663)).

# [v23.1](https://github.com/pybamm-team/PyBaMM/tree/v23.1) - 2023-01-31

## Features

- Changed linting from `flake8` to `ruff` ([#2630](https://github.com/pybamm-team/PyBaMM/pull/2630)).
- Changed docs theme to pydata theme and start to improve docs in general ([#2618](https://github.com/pybamm-team/PyBaMM/pull/2618)).
- New `contact resistance` option, new parameter `Contact resistance [Ohm]` and new variable `Contact overpotential [V]` ([#2598](https://github.com/pybamm-team/PyBaMM/pull/2598)).
- Steps in `Experiment` can now be tagged and cycle numbers be searched based on those tags ([#2593](https://github.com/pybamm-team/PyBaMM/pull/2593)).

## Bug fixes

- Fixed a bug where the solid phase conductivity was double-corrected for tortuosity when loading parameters from a BPX file ([#2638](https://github.com/pybamm-team/PyBaMM/pull/2638)).
- Changed termination from "success" to "final time" for algebraic solvers to match ODE/DAE solvers ([#2613](https://github.com/pybamm-team/PyBaMM/pull/2613)).

# [v22.12](https://github.com/pybamm-team/PyBaMM/tree/v22.12) - 2022-12-31

## Features

- Added functionality to create `pybamm.ParameterValues` from a [BPX standard](https://github.com/pybamm-team/BPX) JSON file ([#2555](https://github.com/pybamm-team/PyBaMM/pull/2555)).
- Allow the option "surface form" to be "differential" in the `MPM` ([#2533](https://github.com/pybamm-team/PyBaMM/pull/2533))
- Added variables "Loss of lithium due to loss of active material in negative/positive electrode [mol]". These should be included in the calculation of "total lithium in system" to make sure that lithium is truly conserved. ([#2529](https://github.com/pybamm-team/PyBaMM/pull/2529))
- `initial_soc` can now be a string "x V", in which case the simulation is initialized to start from that voltage ([#2508](https://github.com/pybamm-team/PyBaMM/pull/2508))
- The `ElectrodeSOH` solver can now calculate electrode balance based on a target "cell capacity" (requires cell capacity "Q" as input), as well as the default "cyclable cell capacity" (requires cyclable lithium capacity "Q_Li" as input). Use the keyword argument `known_value` to control which is used. ([#2508](https://github.com/pybamm-team/PyBaMM/pull/2508))

## Bug fixes

- Allow models that subclass `BaseBatteryModel` to use custom options classes ([#2571](https://github.com/pybamm-team/PyBaMM/pull/2571))
- Fixed bug with `EntryPoints` in Spyder IDE ([#2584](https://github.com/pybamm-team/PyBaMM/pull/2584))
- Fixed electrolyte conservation when options {"surface form": "algebraic"} are used
- Fixed "constant concentration" electrolyte model so that "porosity times concentration" is conserved when porosity changes ([#2529](https://github.com/pybamm-team/PyBaMM/pull/2529))
- Fix installation on `Google Colab` (`pybtex` and `Colab` issue) ([#2526](https://github.com/pybamm-team/PyBaMM/pull/2526))

## Breaking changes

- Renamed "Negative/Positive electrode SOC" to "Negative/Positive electrode stoichiometry" to avoid confusion with cell SOC ([#2529](https://github.com/pybamm-team/PyBaMM/pull/2529))
- Removed external variables and submodels. InputParameter should now be used in all cases ([#2502](https://github.com/pybamm-team/PyBaMM/pull/2502))
- Trying to use a solver to solve multiple models results in a RuntimeError exception ([#2481](https://github.com/pybamm-team/PyBaMM/pull/2481))
- Inputs for the `ElectrodeSOH` solver are now (i) "Q_Li", the total cyclable capacity of lithium in the electrodes (previously "n_Li", the total number of moles, n_Li = 3600/F \* Q_Li) (ii) "Q_n", the capacity of the negative electrode (previously "C_n"), and "Q_p", the capacity of the positive electrode (previously "C_p") ([#2508](https://github.com/pybamm-team/PyBaMM/pull/2508))

# [v22.11.1](https://github.com/pybamm-team/PyBaMM/tree/v22.11.1) - 2022-12-13

## Bug fixes

- Fixed installation on Google Colab (`pybtex` issues) ([#2547](https://github.com/pybamm-team/PyBaMM/pull/2547/files))

# [v22.11](https://github.com/pybamm-team/PyBaMM/tree/v22.11) - 2022-11-30

## Features

- Updated parameter sets so that interpolants are created explicitly in the parameter set python file. This does not change functionality but allows finer control, e.g. specifying a "cubic" interpolator instead of the default "linear" ([#2510](https://github.com/pybamm-team/PyBaMM/pull/2510))
- Equivalent circuit models ([#2478](https://github.com/pybamm-team/PyBaMM/pull/2478))
- New Idaklu solver options for jacobian type and linear solver, support Sundials v6 ([#2444](https://github.com/pybamm-team/PyBaMM/pull/2444))
- Added `scale` and `reference` attributes to `Variable` objects, which can be use to make the ODE/DAE solver better conditioned ([#2440](https://github.com/pybamm-team/PyBaMM/pull/2440))
- SEI reactions can now be asymmetric ([#2425](https://github.com/pybamm-team/PyBaMM/pull/2425))

## Bug fixes

- Switched from `pkg_resources` to `importlib_metadata` for handling entry points ([#2500](https://github.com/pybamm-team/PyBaMM/pull/2500))
- Fixed some bugs related to processing `FunctionParameter` to `Interpolant` ([#2494](https://github.com/pybamm-team/PyBaMM/pull/2494))

## Optimizations

- `ParameterValues` now avoids trying to process children if a function parameter is an object that doesn't depend on its children ([#2477](https://github.com/pybamm-team/PyBaMM/pull/2477))
- Implemented memoization via `cache` and `cached_property` from functools ([#2465](https://github.com/pybamm-team/PyBaMM/pull/2465))
- Added more rules for simplifying expressions, especially around Concatenations. Also, meshes constructed from multiple domains are now cached ([#2443](https://github.com/pybamm-team/PyBaMM/pull/2443))
- Added more rules for simplifying expressions. Constants in binary operators are now moved to the left by default (e.g. `x*2` returns `2*x`) ([#2424](https://github.com/pybamm-team/PyBaMM/pull/2424))

## Breaking changes

- Interpolants created from parameter data are now "linear" by default (was "cubic") ([#2494](https://github.com/pybamm-team/PyBaMM/pull/2494))
- Renamed entry point for parameter sets to `pybamm_parameter_sets` ([#2475](https://github.com/pybamm-team/PyBaMM/pull/2475))
- Removed code for generating `ModelingToolkit` problems ([#2432](https://github.com/pybamm-team/PyBaMM/pull/2432))
- Removed `FirstOrder` and `Composite` lead-acid models, and some submodels specific to those models ([#2431](https://github.com/pybamm-team/PyBaMM/pull/2431))

# [v22.10.post1](https://github.com/pybamm-team/PyBaMM/tree/v22.10.post1) - 2022-10-31

## Breaking changes

- Removed all julia generation code ([#2453](https://github.com/pybamm-team/PyBaMM/pull/2453)). Julia code will be hosted at [PyBaMM.jl](https://github.com/tinosulzer/PyBaMM.jl) from now on.

# [v22.10](https://github.com/pybamm-team/PyBaMM/tree/v22.10) - 2022-10-31

## Features

- Third-party parameter sets can be added by registering entry points to ~~`pybamm_parameter_set`~~`pybamm_parameter_sets` ([#2396](https://github.com/pybamm-team/PyBaMM/pull/2396), changed in [#2475](https://github.com/pybamm-team/PyBaMM/pull/2475))
- Added three-dimensional interpolation ([#2380](https://github.com/pybamm-team/PyBaMM/pull/2380))

## Bug fixes

- `pybamm.have_julia()` now checks that julia is properly configured ([#2402](https://github.com/pybamm-team/PyBaMM/pull/2402))
- For simulations with events that cause the simulation to stop early, the sensitivities could be evaluated incorrectly to zero ([#2337](https://github.com/pybamm-team/PyBaMM/pull/2337))

## Optimizations

- Reformatted how simulations with experiments are built ([#2395](https://github.com/pybamm-team/PyBaMM/pull/2395))
- Added small perturbation to initial conditions for casadi solver. This seems to help the solver converge better in some cases ([#2356](https://github.com/pybamm-team/PyBaMM/pull/2356))
- Added `ExplicitTimeIntegral` functionality to move variables which do not appear anywhere on the rhs to a new location, and to integrate those variables explicitly when `get` is called by the solution object. ([#2348](https://github.com/pybamm-team/PyBaMM/pull/2348))
- Added more rules for simplifying expressions ([#2211](https://github.com/pybamm-team/PyBaMM/pull/2211))
- Sped up calculations of Electrode SOH variables for summary variables ([#2210](https://github.com/pybamm-team/PyBaMM/pull/2210))

## Breaking change

- Removed `pybamm.SymbolReplacer` as it is no longer needed to set up simulations with experiments, which is the only place where it was being used ([#2395](https://github.com/pybamm-team/PyBaMM/pull/2395))
- Removed `get_infinite_nested_dict`, `BaseModel.check_default_variables_dictionaries`, and `Discretisation.create_jacobian` methods, which were not used by any other functionality in the repository ([#2384](https://github.com/pybamm-team/PyBaMM/pull/2384))
- Dropped support for Python 3.7 after the release of Numpy v1.22.0 ([#2379](https://github.com/pybamm-team/PyBaMM/pull/2379))
- Removed parameter cli tools (add/edit/remove parameters). Parameter sets can now more easily be added via python scripts. ([#2342](https://github.com/pybamm-team/PyBaMM/pull/2342))
- Parameter sets should now be provided as single python files containing all parameters and functions. Parameters provided as "data" (e.g. OCP vs SOC) can still be csv files, but must be either in the same folder as the parameter file or in a subfolder called "data/". See for example [Ai2020](https://github.com/pybamm-team/PyBaMM/tree/develop/pybamm/input/parameters/lithium_ion/Ai2020.py) ([#2342](https://github.com/pybamm-team/PyBaMM/pull/2342))

# [v22.9](https://github.com/pybamm-team/PyBaMM/tree/v22.9) - 2022-09-30

## Features

- Added function `pybamm.get_git_commit_info()`, which returns information about the last git commit, useful for reproducibility ([#2293](https://github.com/pybamm-team/PyBaMM/pull/2293))
- Added SEI model for composite electrodes ([#2290](https://github.com/pybamm-team/PyBaMM/pull/2290))
- For experiments, the simulation now automatically checks and skips steps that cannot be performed (e.g. "Charge at 1C until 4.2V" from 100% SOC) ([#2212](https://github.com/pybamm-team/PyBaMM/pull/2212))

## Bug fixes

- Arrhenius function for `nmc_OKane2022` positive electrode actually gets used now ([#2309](https://github.com/pybamm-team/PyBaMM/pull/2309))
- Added `SEI on cracks` to loop over all interfacial reactions ([#2262](https://github.com/pybamm-team/PyBaMM/pull/2262))
- Fixed `X-averaged SEI on cracks concentration` so it's an average over x only, not y and z ([#2262](https://github.com/pybamm-team/PyBaMM/pull/2262))
- Corrected initial state for SEI on cracks ([#2262](https://github.com/pybamm-team/PyBaMM/pull/2262))

## Optimizations

- Default options for `particle mechanics` now dealt with differently in each electrode ([#2262](https://github.com/pybamm-team/PyBaMM/pull/2262))
- Sped up calculations of Electrode SOH variables for summary variables ([#2210](https://github.com/pybamm-team/PyBaMM/pull/2210))

## Breaking changes

- When creating a `pybamm.Interpolant` the default interpolator is now "linear". Passing data directly to `ParameterValues` using the `[data]` tag will be still used to create a cubic spline interpolant, as before ([#2258](https://github.com/pybamm-team/PyBaMM/pull/2258))
- Events must now be defined in such a way that they are positive at the initial conditions (events will be triggered when they become negative, instead of when they change sign in either direction) ([#2212](https://github.com/pybamm-team/PyBaMM/pull/2212))

# [v22.8](https://github.com/pybamm-team/PyBaMM/tree/v22.8) - 2022-08-31

## Features

- Added `CurrentSigmoidOpenCircuitPotential` model to model voltage hysteresis for charge/discharge ([#2256](https://github.com/pybamm-team/PyBaMM/pull/2256))
- Added "Chen2020_composite" parameter set for a composite graphite/silicon electrode. ([#2256](https://github.com/pybamm-team/PyBaMM/pull/2256))
- Added new cumulative variables `Throughput capacity [A.h]` and `Throughput energy [W.h]` to standard variables and summary variables, to assist with degradation studies. Throughput variables are only calculated if `calculate discharge energy` is set to `true`. `Time [s]` and `Time [h]` also added to summary variables. ([#2249](https://github.com/pybamm-team/PyBaMM/pull/2249))
- Added `lipf6_OKane2022` electrolyte to `OKane2022` parameter set ([#2249](https://github.com/pybamm-team/PyBaMM/pull/2249))
- Reformated submodel structure to allow composite electrodes. Composite positive electrode is now also possible. With current implementation, electrodes can have at most two phases. ([#2248](https://github.com/pybamm-team/PyBaMM/pull/2248))

## Bug fixes

- Added new parameter `Ratio of lithium moles to SEI moles` (short name z_sei) to fix a bug where this number was incorrectly hardcoded to 1. ([#2222](https://github.com/pybamm-team/PyBaMM/pull/2222))
- Changed short name of parameter `Inner SEI reaction proportion` from alpha_SEI to inner_sei_proportion, to avoid confusion with transfer coefficients. ([#2222](https://github.com/pybamm-team/PyBaMM/pull/2222))
- Deleted legacy parameters with short names beta_sei and beta_plating. ([#2222](https://github.com/pybamm-team/PyBaMM/pull/2222))
- Corrected initial SEI thickness for OKane2022 parameter set. ([#2218](https://github.com/pybamm-team/PyBaMM/pull/2218))

## Optimizations

- Simplified scaling for the exchange-current density. The dimensionless parameter `C_r` is kept, but no longer used anywhere ([#2238](https://github.com/pybamm-team/PyBaMM/pull/2238))
- Added limits for variables in some functions to avoid division by zero, sqrt(negative number), etc ([#2213](https://github.com/pybamm-team/PyBaMM/pull/2213))

## Breaking changes

- Parameters specific to a (primary/secondary) phase in a domain are doubly nested. e.g. `param.c_n_max` is now `param.n.prim.c_max` ([#2248](https://github.com/pybamm-team/PyBaMM/pull/2248))

# [v22.7](https://github.com/pybamm-team/PyBaMM/tree/v22.7) - 2022-07-31

## Features

- Moved general code about submodels to `BaseModel` instead of `BaseBatteryModel`, making it easier to build custom models from submodels. ([#2169](https://github.com/pybamm-team/PyBaMM/pull/2169))
- Events can now be plotted as a regular variable (under the name "Event: event_name", e.g. "Event: Minimum voltage [V]") ([#2158](https://github.com/pybamm-team/PyBaMM/pull/2158))
- Added example showing how to print whether a model is compatible with a parameter set ([#2112](https://github.com/pybamm-team/PyBaMM/pull/2112))
- Added SEI growth on cracks ([#2104](https://github.com/pybamm-team/PyBaMM/pull/2104))
- Added Arrhenius temperature dependence of SEI growth ([#2104](https://github.com/pybamm-team/PyBaMM/pull/2104))
- The "Inner SEI reaction proportion" parameter actually gets used now ([#2104](https://github.com/pybamm-team/PyBaMM/pull/2104))
- New OKane2022 parameter set replaces Chen2020_plating ([#2104](https://github.com/pybamm-team/PyBaMM/pull/2104))
- SEI growth, lithium plating and porosity change can now be set to distributed in `SPMe`. There is an additional option called `x-average side reactions` which allows to set this (note that for `SPM` it is always x-averaged). ([#2099](https://github.com/pybamm-team/PyBaMM/pull/2099))

## Optimizations

- Improved eSOH calculations to be more robust ([#2192](https://github.com/pybamm-team/PyBaMM/pull/2192),[#2199](https://github.com/pybamm-team/PyBaMM/pull/2199))
- The (2x2x2=8) particle diffusion submodels have been consolidated into just three submodels (Fickian diffusion, polynomial profile, and x-averaged polynomial profile) with optional x-averaging and size distribution. Polynomial profile and x-averaged polynomial profile are still two separate submodels, since they deal with surface concentration differently.
- Added error for when solution vector gets too large, to help debug solver errors ([#2138](https://github.com/pybamm-team/PyBaMM/pull/2138))

## Bug fixes

- Fixed error reporting for simulation with experiment ([#2213](https://github.com/pybamm-team/PyBaMM/pull/2213))
- Fixed a bug in `Simulation` that caused initial conditions to change when solving an experiment multiple times ([#2204](https://github.com/pybamm-team/PyBaMM/pull/2204))
- Fixed labels and ylims in `plot_voltage_components`([#2183](https://github.com/pybamm-team/PyBaMM/pull/2183))
- Fixed 2D interpolant ([#2180](https://github.com/pybamm-team/PyBaMM/pull/2180))
- Fixes a bug where the SPMe always builds even when `build=False` ([#2169](https://github.com/pybamm-team/PyBaMM/pull/2169))
- Some events have been removed in the case where they are constant, i.e. can never be reached ([#2158](https://github.com/pybamm-team/PyBaMM/pull/2158))
- Raise explicit `NotImplementedError` if trying to call `bool()` on a pybamm Symbol (e.g. in an if statement condition) ([#2141](https://github.com/pybamm-team/PyBaMM/pull/2141))
- Fixed bug causing cut-off voltage to change after setting up a simulation with a model ([#2138](https://github.com/pybamm-team/PyBaMM/pull/2138))
- A single solution cycle can now be used as a starting solution for a simulation ([#2138](https://github.com/pybamm-team/PyBaMM/pull/2138))

## Breaking changes

- Exchange-current density functions (and some other functions) now take an additional argument, the maximum particle concentration for that phase ([#2134](https://github.com/pybamm-team/PyBaMM/pull/2134))
- Loss of lithium to SEI on cracks is now a degradation variable, so setting a particle mechanics submodel is now compulsory (NoMechanics will suffice) ([#2104](https://github.com/pybamm-team/PyBaMM/pull/2104))

# [v22.6](https://github.com/pybamm-team/PyBaMM/tree/v22.6) - 2022-06-30

## Features

- Added open-circuit potential as a separate submodel ([#2094](https://github.com/pybamm-team/PyBaMM/pull/2094))
- Added partially reversible lithium plating model and new `OKane2022` parameter set to go with it ([#2043](https://github.com/pybamm-team/PyBaMM/pull/2043))
- Added `__eq__` and `__hash__` methods for `Symbol` objects, using `.id` ([#1978](https://github.com/pybamm-team/PyBaMM/pull/1978))

## Optimizations

- Stoichiometry inputs to OCP functions are now bounded between 1e-10 and 1-1e-10, with singularities at 0 and 1 so that OCP goes to +- infinity ([#2095](https://github.com/pybamm-team/PyBaMM/pull/2095))

## Breaking changes

- Changed some dictionary keys to `Symbol` instead of `Symbol.id` (internal change only, should not affect external facing functions) ([#1978](https://github.com/pybamm-team/PyBaMM/pull/1978))

# [v22.5](https://github.com/pybamm-team/PyBaMM/tree/v22.5) - 2022-05-31

## Features

- Added a casadi version of the IDKLU solver, which is used for `model.convert_to_format = "casadi"` ([#2002](https://github.com/pybamm-team/PyBaMM/pull/2002))
- Added functionality to generate Julia expressions from a model. See [PyBaMM.jl](https://github.com/tinosulzer/PyBaMM.jl) for how to use these ([#1942](https://github.com/pybamm-team/PyBaMM/pull/1942)))
- Added basic callbacks to the Simulation class, and a LoggingCallback ([#1880](https://github.com/pybamm-team/PyBaMM/pull/1880)))

## Bug fixes

- Corrected legend order in "plot_voltage_components.py", so each entry refers to the correct overpotential. ([#2061](https://github.com/pybamm-team/PyBaMM/pull/2061))

## Breaking changes

- Changed domain-specific parameter names to a nested attribute. `param.n.l_n` is now `param.n.l` ([#2063](https://github.com/pybamm-team/PyBaMM/pull/2063))

# [v22.4](https://github.com/pybamm-team/PyBaMM/tree/v22.4) - 2022-04-30

## Features

- Added a casadi version of the IDKLU solver, which is used for `model.convert_to_format = "casadi"` ([#2002](https://github.com/pybamm-team/PyBaMM/pull/2002))

## Bug fixes

- Remove old deprecation errors, including those in `parameter_values.py` that caused the simulation if, for example, the reaction rate is re-introduced manually ([#2022](https://github.com/pybamm-team/PyBaMM/pull/2022))

# [v22.3](https://github.com/pybamm-team/PyBaMM/tree/v22.3) - 2022-03-31

## Features

- Added "Discharge energy [W.h]", which is the integral of the power in Watts, as an optional output. Set the option "calculate discharge energy" to "true" to get this output ("false" by default, since it can slow down some of the simple models) ([#1969](https://github.com/pybamm-team/PyBaMM/pull/1969)))
- Added an option "calculate heat source for isothermal models" to choose whether or not the heat generation terms are computed when running models with the option `thermal="isothermal"` ([#1958](https://github.com/pybamm-team/PyBaMM/pull/1958))

## Optimizations

- Simplified `model.new_copy()` ([#1977](https://github.com/pybamm-team/PyBaMM/pull/1977))

## Bug fixes

- Fix bug where sensitivity calculation failed if len of `calculate_sensitivities` was less than `inputs` ([#1897](https://github.com/pybamm-team/PyBaMM/pull/1897))
- Fixed a bug in the eSOH variable calculation when OCV is given as data ([#1975](https://github.com/pybamm-team/PyBaMM/pull/1975))
- Fixed a bug where isothermal models did not compute any heat source terms ([#1958](https://github.com/pybamm-team/PyBaMM/pull/1958))

## Breaking changes

- Removed `model.new_empty_copy()` (use `model.new_copy()` instead) ([#1977](https://github.com/pybamm-team/PyBaMM/pull/1977))
- Dropped support for Windows 32-bit architecture ([#1964](https://github.com/pybamm-team/PyBaMM/pull/1964))

# [v22.2](https://github.com/pybamm-team/PyBaMM/tree/v22.2) - 2022-02-28

## Features

- Isothermal models now calculate heat source terms (but the temperature remains constant). The models now also account for current collector heating when `dimensionality=0` ([#1929](https://github.com/pybamm-team/PyBaMM/pull/1929))
- Added new models for power control and resistance control ([#1917](https://github.com/pybamm-team/PyBaMM/pull/1917))
- Initial concentrations can now be provided as a function of `r` as well as `x` ([#1866](https://github.com/pybamm-team/PyBaMM/pull/1866))

## Bug fixes

- Fixed a bug where thermal submodels could not be used with half-cells ([#1929](https://github.com/pybamm-team/PyBaMM/pull/1929))
- Parameters can now be imported from a directory having "pybamm" in its name ([#1919](https://github.com/pybamm-team/PyBaMM/pull/1919))
- `scikit.odes` and `SUNDIALS` can now be installed using `pybamm_install_odes` ([#1916](https://github.com/pybamm-team/PyBaMM/pull/1916))

## Breaking changes

- The `domain` setter and `auxiliary_domains` getter have been deprecated, `domains` setter/getter should be used instead. The `domain` getter is still active. We now recommend creating symbols with `domains={...}` instead of `domain=..., auxiliary_domains={...}`, but the latter is not yet deprecated ([#1866](https://github.com/pybamm-team/PyBaMM/pull/1866))

# [v22.1](https://github.com/pybamm-team/PyBaMM/tree/v22.1) - 2022-01-31

## Features

- Half-cell models can now be run with "surface form" ([#1913](https://github.com/pybamm-team/PyBaMM/pull/1913))
- Added option for different kinetics on anode and cathode ([#1913](https://github.com/pybamm-team/PyBaMM/pull/1913))
- Allow `pybamm.Solution.save_data()` to return a string if filename is None, and added json to_format option ([#1909](https://github.com/pybamm-team/PyBaMM/pull/1909))
- Added an option to force install compatible versions of jax and jaxlib if already installed using CLI ([#1881](https://github.com/pybamm-team/PyBaMM/pull/1881))

## Optimizations

- The `Symbol` nodes no longer subclasses `anytree.NodeMixIn`. This removes some checks that were not really needed ([#1912](https://github.com/pybamm-team/PyBaMM/pull/1912))

## Bug fixes

- Parameters can now be imported from any given path in `Windows` ([#1900](https://github.com/pybamm-team/PyBaMM/pull/1900))
- Fixed initial conditions for the EC SEI model ([#1895](https://github.com/pybamm-team/PyBaMM/pull/1895))
- Fixed issue in extraction of sensitivites ([#1894](https://github.com/pybamm-team/PyBaMM/pull/1894))

# [v21.12](https://github.com/pybamm-team/PyBaMM/tree/v21.11) - 2021-12-29

## Features

- Added new kinetics models for asymmetric Butler-Volmer, linear kinetics, and Marcus-Hush-Chidsey ([#1858](https://github.com/pybamm-team/PyBaMM/pull/1858))
- Experiments can be set to terminate when a voltage is reached (across all steps) ([#1832](https://github.com/pybamm-team/PyBaMM/pull/1832))
- Added cylindrical geometry and finite volume method ([#1824](https://github.com/pybamm-team/PyBaMM/pull/1824))

## Bug fixes

- `PyBaMM` is now importable in `Linux` systems where `jax` is already installed ([#1874](https://github.com/pybamm-team/PyBaMM/pull/1874))
- Simulations with drive cycles now support `initial_soc` ([#1842](https://github.com/pybamm-team/PyBaMM/pull/1842))
- Fixed bug in expression tree simplification ([#1831](https://github.com/pybamm-team/PyBaMM/pull/1831))
- Solid tortuosity is now correctly calculated with Bruggeman coefficient of the respective electrode ([#1773](https://github.com/pybamm-team/PyBaMM/pull/1773))

# [v21.11](https://github.com/pybamm-team/PyBaMM/tree/v21.11) - 2021-11-30

## Features

- The name of a parameter set can be passed to `ParameterValues` as a string, e.g. `ParameterValues("Chen2020")` ([#1822](https://github.com/pybamm-team/PyBaMM/pull/1822))
- Added submodels for interface utilisation ([#1821](https://github.com/pybamm-team/PyBaMM/pull/1821))
- Reformatted SEI growth models into a single submodel with conditionals ([#1808](https://github.com/pybamm-team/PyBaMM/pull/1808))
- Stress-induced diffusion is now a separate model option instead of being automatically included when using the particle mechanics submodels ([#1797](https://github.com/pybamm-team/PyBaMM/pull/1797))
- `Experiment`s with drive cycles can be solved ([#1793](https://github.com/pybamm-team/PyBaMM/pull/1793))
- Added surface area to volume ratio as a factor to the SEI equations ([#1790](https://github.com/pybamm-team/PyBaMM/pull/1790))
- Half-cell SPM and SPMe have been implemented ([#1731](https://github.com/pybamm-team/PyBaMM/pull/1731))

## Bug fixes

- Fixed `sympy` operators for `Arctan` and `Exponential` ([#1786](https://github.com/pybamm-team/PyBaMM/pull/1786))
- Fixed finite volume discretization in spherical polar coordinates ([#1782](https://github.com/pybamm-team/PyBaMM/pull/1782))
- Fixed bug when using `Experiment` with a pouch cell model ([#1707](https://github.com/pybamm-team/PyBaMM/pull/1707))
- Fixed bug when using `Experiment` with a plating model ([#1707](https://github.com/pybamm-team/PyBaMM/pull/1707))
- Fixed hack for potentials in the SPMe model ([#1707](https://github.com/pybamm-team/PyBaMM/pull/1707))

## Breaking changes

- The `chemistry` keyword argument in `ParameterValues` has been deprecated. Use `ParameterValues(chem)` instead of `ParameterValues(chemistry=chem)` ([#1822](https://github.com/pybamm-team/PyBaMM/pull/1822))
- Raise error when trying to convert an `Interpolant` with the "pchip" interpolator to CasADI ([#1791](https://github.com/pybamm-team/PyBaMM/pull/1791))
- Raise error if `Concatenation` is used directly with `Variable` objects (`concatenation` should be used instead) ([#1789](https://github.com/pybamm-team/PyBaMM/pull/1789))
- Made jax, jaxlib and the PyBaMM JaxSolver optional ([#1767](https://github.com/pybamm-team/PyBaMM/pull/1767), [#1803](https://github.com/pybamm-team/PyBaMM/pull/1803))

# [v21.10](https://github.com/pybamm-team/PyBaMM/tree/v21.10) - 2021-10-31

## Features

- Summary variables can now be user-determined ([#1760](https://github.com/pybamm-team/PyBaMM/pull/1760))
- Added `all_first_states` to the `Solution` object for a simulation with experiment ([#1759](https://github.com/pybamm-team/PyBaMM/pull/1759))
- Added a new method (`create_gif`) in `QuickPlot`, `Simulation` and `BatchStudy` to create a GIF of a simulation ([#1754](https://github.com/pybamm-team/PyBaMM/pull/1754))
- Added more examples for the `BatchStudy` class ([#1747](https://github.com/pybamm-team/PyBaMM/pull/1747))
- SEI models can now be included in the half-cell model ([#1705](https://github.com/pybamm-team/PyBaMM/pull/1705))

## Bug fixes

- Half-cell model and lead-acid models can now be simulated with `Experiment`s ([#1759](https://github.com/pybamm-team/PyBaMM/pull/1759))
- Removed in-place modification of the solution objects by `QuickPlot` ([#1747](https://github.com/pybamm-team/PyBaMM/pull/1747))
- Fixed vector-vector multiplication bug that was causing errors in the SPM with constant voltage or power ([#1735](https://github.com/pybamm-team/PyBaMM/pull/1735))

# [v21.9](https://github.com/pybamm-team/PyBaMM/tree/v21.9) - 2021-09-30

## Features

- Added thermal parameters (thermal conductivity, specific heat, etc.) to the `Ecker2015` parameter set from Zhao et al. (2018) and Hales et al. (2019) ([#1683](https://github.com/pybamm-team/PyBaMM/pull/1683))
- Added `plot_summary_variables` to plot and compare summary variables ([#1678](https://github.com/pybamm-team/PyBaMM/pull/1678))
- The DFN model can now be used directly (instead of `BasicDFNHalfCell`) to simulate a half-cell ([#1600](https://github.com/pybamm-team/PyBaMM/pull/1600))

## Breaking changes

- Dropped support for Python 3.6 ([#1696](https://github.com/pybamm-team/PyBaMM/pull/1696))
- The substring 'negative electrode' has been removed from variables related to SEI and lithium plating (e.g. 'Total negative electrode SEI thickness [m]' replaced by 'Total SEI thickness [m]') ([#1654](https://github.com/pybamm-team/PyBaMM/pull/1654))

# [v21.08](https://github.com/pybamm-team/PyBaMM/tree/v21.08) - 2021-08-26

This release introduces:

- the switch to calendar versioning: from now on we will use year.month version number
- sensitivity analysis of solutions with respect to input parameters
- several new models, including many-particle and state-of-health models
- improvement on how CasADI solver's handle events, including a new "fast with events" mode
- several other new features, optimizations, and bug fixes, summarized below

## Features

- Added submodels and functionality for particle-size distributions in the DFN model, including an
  example notebook ([#1602](https://github.com/pybamm-team/PyBaMM/pull/1602))
- Added UDDS and WLTC drive cycles ([#1601](https://github.com/pybamm-team/PyBaMM/pull/1601))
- Added LG M50 (NMC811 and graphite + SiOx) parameter set from O'Regan 2022 ([#1594](https://github.com/pybamm-team/PyBaMM/pull/1594))
- `pybamm.base_solver.solve` function can take a list of input parameters to calculate the sensitivities of the solution with respect to. Alternatively, it can be set to `True` to calculate the sensitivities for all input parameters ([#1552](https://github.com/pybamm-team/PyBaMM/pull/1552))
- Added capability for `quaternary` domains (in addition to `primary`, `secondary` and `tertiary`), increasing the maximum number of domains that a `Symbol` can have to 4. ([#1580](https://github.com/pybamm-team/PyBaMM/pull/1580))
- Tabs can now be placed at the bottom of the cell in 1+1D thermal models ([#1581](https://github.com/pybamm-team/PyBaMM/pull/1581))
- Added temperature dependence on electrode electronic conductivity ([#1570](https://github.com/pybamm-team/PyBaMM/pull/1570))
- `pybamm.base_solver.solve` function can take a list of input parameters to calculate the sensitivities of the solution with respect to. Alternatively, it can be set to `True` to calculate the sensitivities for all input parameters ([#1552](https://github.com/pybamm-team/PyBaMM/pull/1552))
- Added a new lithium-ion model `MPM` or Many-Particle Model, with a distribution of particle sizes in each electrode. ([#1529](https://github.com/pybamm-team/PyBaMM/pull/1529))
- Added 2 new submodels for lithium transport in a size distribution of electrode particles: Fickian diffusion (`FickianSingleSizeDistribution`) and uniform concentration profile (`FastSingleSizeDistribution`). ([#1529](https://github.com/pybamm-team/PyBaMM/pull/1529))
- Added a "particle size" domain to the default lithium-ion geometry, including plotting capabilities (`QuickPlot`) and processing of variables (`ProcessedVariable`). ([#1529](https://github.com/pybamm-team/PyBaMM/pull/1529))
- Added fitted expressions for OCPs for the Chen2020 parameter set ([#1526](https://github.com/pybamm-team/PyBaMM/pull/1497))
- Added `initial_soc` argument to `Simualtion.solve` for specifying the initial SOC when solving a model ([#1512](https://github.com/pybamm-team/PyBaMM/pull/1512))
- Added `print_name` to some symbols ([#1495](https://github.com/pybamm-team/PyBaMM/pull/1495), [#1497](https://github.com/pybamm-team/PyBaMM/pull/1497))
- Added Base Parameters class and SymPy in dependencies ([#1495](https://github.com/pybamm-team/PyBaMM/pull/1495))
- Added a new "reaction-driven" model for LAM from Reniers et al (2019) ([#1490](https://github.com/pybamm-team/PyBaMM/pull/1490))
- Some features ("loss of active material" and "particle mechanics") can now be specified separately for the negative electrode and positive electrode by passing a 2-tuple ([#1490](https://github.com/pybamm-team/PyBaMM/pull/1490))
- `plot` and `plot2D` now take and return a matplotlib Axis to allow for easier customization ([#1472](https://github.com/pybamm-team/PyBaMM/pull/1472))
- `ParameterValues.evaluate` can now return arrays to allow function parameters to be easily evaluated ([#1472](https://github.com/pybamm-team/PyBaMM/pull/1472))
- Added option to save only specific cycle numbers when simulating an `Experiment` ([#1459](https://github.com/pybamm-team/PyBaMM/pull/1459))
- Added capacity-based termination conditions when simulating an `Experiment` ([#1459](https://github.com/pybamm-team/PyBaMM/pull/1459))
- Added "summary variables" to track degradation over several cycles ([#1459](https://github.com/pybamm-team/PyBaMM/pull/1459))
- Added `ElectrodeSOH` model for calculating capacities and stoichiometric limits ([#1459](https://github.com/pybamm-team/PyBaMM/pull/1459))
- Added Batch Study class ([#1455](https://github.com/pybamm-team/PyBaMM/pull/1455))
- Added `ConcatenationVariable`, which is automatically created when variables are concatenated ([#1453](https://github.com/pybamm-team/PyBaMM/pull/1453))
- Added "fast with events" mode for the CasADi solver, which solves a model and finds events more efficiently than "safe" mode. As of PR #1450 this feature is still being tested and "safe" mode remains the default ([#1450](https://github.com/pybamm-team/PyBaMM/pull/1450))

## Optimizations

- Models that mostly use x-averaged quantities (SPM and SPMe) now use x-averaged degradation models ([#1490](https://github.com/pybamm-team/PyBaMM/pull/1490))
- Improved how the CasADi solver's "safe" mode finds events ([#1450](https://github.com/pybamm-team/PyBaMM/pull/1450))
- Perform more automatic simplifications of the expression tree ([#1449](https://github.com/pybamm-team/PyBaMM/pull/1449))
- Reduce time taken to hash a sparse `Matrix` object ([#1449](https://github.com/pybamm-team/PyBaMM/pull/1449))

## Bug fixes

- Fixed bug with `load_function` ([#1675](https://github.com/pybamm-team/PyBaMM/pull/1675))
- Updated documentation to include some previously missing functions, such as `erf` and `tanh` ([#1628](https://github.com/pybamm-team/PyBaMM/pull/1628))
- Fixed reading citation file without closing ([#1620](https://github.com/pybamm-team/PyBaMM/pull/1620))
- Porosity variation for SEI and plating models is calculated from the film thickness rather than from a separate ODE ([#1617](https://github.com/pybamm-team/PyBaMM/pull/1617))
- Fixed a bug where the order of the indexing for the entries of variables discretised using FEM was incorrect ([#1556](https://github.com/pybamm-team/PyBaMM/pull/1556))
- Fix broken module import for spyder when running a script twice ([#1555](https://github.com/pybamm-team/PyBaMM/pull/1555))
- Fixed ElectrodeSOH model for multi-dimensional simulations ([#1548](https://github.com/pybamm-team/PyBaMM/pull/1548))
- Removed the overly-restrictive check "each variable in the algebraic eqn keys must appear in the eqn" ([#1510](https://github.com/pybamm-team/PyBaMM/pull/1510))
- Made parameters importable through pybamm ([#1475](https://github.com/pybamm-team/PyBaMM/pull/1475))

## Breaking changes

- Refactored the `particle` submodel module, with the models having no size distribution now found in `particle.no_distribution`, and those with a size distribution in `particle.size_distribution`. Renamed submodels to indicate the transport model (Fickian diffusion, polynomial profile) and if they are "x-averaged". E.g., `FickianManyParticles` and `FickianSingleParticle` are now `no_distribution.FickianDiffusion` and `no_distribution.XAveragedFickianDiffusion` ([#1602](https://github.com/pybamm-team/PyBaMM/pull/1602))
- Changed sensitivity API. Removed `ProcessedSymbolicVariable`, all sensitivity now handled within the solvers and `ProcessedVariable` ([#1552](https://github.com/pybamm-team/PyBaMM/pull/1552),[#2276](https://github.com/pybamm-team/PyBaMM/pull/2276))
- The `Yang2017` parameter set has been removed as the complete parameter set is not publicly available in the literature ([#1577](https://github.com/pybamm-team/PyBaMM/pull/1577))
- Changed how options are specified for the "loss of active material" and "particle cracking" submodels. "loss of active material" can now be one of "none", "stress-driven", or "reaction-driven", or a 2-tuple for different options in negative and positive electrode. Similarly "particle cracking" (now called "particle mechanics") can now be "none", "swelling only", "swelling and cracking", or a 2-tuple ([#1490](https://github.com/pybamm-team/PyBaMM/pull/1490))
- Changed the variable in the full diffusion model from "Electrolyte concentration" to "Porosity times concentration" ([#1476](https://github.com/pybamm-team/PyBaMM/pull/1476))
- Renamed `lithium-ion` folder to `lithium_ion` and `lead-acid` folder to `lead_acid` in parameters ([#1464](https://github.com/pybamm-team/PyBaMM/pull/1464))

# [v0.4.0](https://github.com/pybamm-team/PyBaMM/tree/v0.4.0) - 2021-03-28

This release introduces:

- several new models, including reversible and irreversible plating submodels, submodels for loss of active material, Yang et al.'s (2017) coupled SEI/plating/pore clogging model, and the Newman-Tobias model
- internal optimizations for solving models, particularly for simulating experiments, with more accurate event detection and more efficient numerical methods and post-processing
- parallel solutions of a model with different inputs
- a cleaner installation process for Mac when installing from PyPI, no longer requiring a Homebrew installation of Sundials
- improved plotting functionality, including adding a new 'voltage component' plot
- several other new features, optimizations, and bug fixes, summarized below

## Features

- Added `NewmanTobias` li-ion battery model ([#1423](https://github.com/pybamm-team/PyBaMM/pull/1423))
- Added `plot_voltage_components` to easily plot the component overpotentials that make up the voltage ([#1419](https://github.com/pybamm-team/PyBaMM/pull/1419))
- Made `QuickPlot` more customizable and added an example ([#1419](https://github.com/pybamm-team/PyBaMM/pull/1419))
- `Solution` objects can now be created by stepping _different_ models ([#1408](https://github.com/pybamm-team/PyBaMM/pull/1408))
- Added Yang et al 2017 model that couples irreversible lithium plating, SEI growth and change in porosity which produces a transition from linear to nonlinear degradation pattern of lithium-ion battery over extended cycles([#1398](https://github.com/pybamm-team/PyBaMM/pull/1398))
- Added support for Python 3.9 and dropped support for Python 3.6. Python 3.6 may still work but is now untested ([#1370](https://github.com/pybamm-team/PyBaMM/pull/1370))
- Added the electrolyte overpotential and Ohmic losses for full conductivity, including surface form ([#1350](https://github.com/pybamm-team/PyBaMM/pull/1350))
- Added functionality to `Citations` to print formatted citations ([#1340](https://github.com/pybamm-team/PyBaMM/pull/1340))
- Updated the way events are handled in `CasadiSolver` for more accurate event location ([#1328](https://github.com/pybamm-team/PyBaMM/pull/1328))
- Added error message if initial conditions are outside the bounds of a variable ([#1326](https://github.com/pybamm-team/PyBaMM/pull/1326))
- Added temperature dependence to density, heat capacity and thermal conductivity ([#1323](https://github.com/pybamm-team/PyBaMM/pull/1323))
- Added temperature dependence to the transference number (`t_plus`) ([#1317](https://github.com/pybamm-team/PyBaMM/pull/1317))
- Added new functionality for `Interpolant` ([#1312](https://github.com/pybamm-team/PyBaMM/pull/1312))
- Added option to express experiments (and extract solutions) in terms of cycles of operating condition ([#1309](https://github.com/pybamm-team/PyBaMM/pull/1309))
- The event time and state are now returned as part of `Solution.t` and `Solution.y` so that the event is accurately captured in the returned solution ([#1300](https://github.com/pybamm-team/PyBaMM/pull/1300))
- Added reversible and irreversible lithium plating models ([#1287](https://github.com/pybamm-team/PyBaMM/pull/1287))
- Reformatted the `BasicDFNHalfCell` to be consistent with the other models ([#1282](https://github.com/pybamm-team/PyBaMM/pull/1282))
- Added option to make the total interfacial current density a state ([#1280](https://github.com/pybamm-team/PyBaMM/pull/1280))
- Added functionality to initialize a model using the solution from another model ([#1278](https://github.com/pybamm-team/PyBaMM/pull/1278))
- Added submodels for active material ([#1262](https://github.com/pybamm-team/PyBaMM/pull/1262))
- Updated solvers' method `solve()` so it can take a list of inputs dictionaries as the `inputs` keyword argument. In this case the model is solved for each input set in the list, and a list of solutions mapping the set of inputs to the solutions is returned. Note that `solve()` can still take a single dictionary as the `inputs` keyword argument. In this case the behaviour is unchanged compared to previous versions.([#1261](https://github.com/pybamm-team/PyBaMM/pull/1261))
- Added composite surface form electrolyte models: `CompositeDifferential` and `CompositeAlgebraic` ([#1207](https://github.com/pybamm-team/PyBaMM/issues/1207))

## Optimizations

- Improved the way an `Experiment` is simulated to reduce solve time (at the cost of slightly higher set-up time) ([#1408](https://github.com/pybamm-team/PyBaMM/pull/1408))
- Add script and workflow to automatically update parameter_sets.py docstrings ([#1371](https://github.com/pybamm-team/PyBaMM/pull/1371))
- Add URLs checker in workflows ([#1347](https://github.com/pybamm-team/PyBaMM/pull/1347))
- The `Solution` class now only creates the concatenated `y` when the user asks for it. This is an optimization step as the concatenation can be slow, especially with larger experiments ([#1331](https://github.com/pybamm-team/PyBaMM/pull/1331))
- If solver method `solve()` is passed a list of inputs as the `inputs` keyword argument, the resolution of the model for each input set is spread across several Python processes, usually running in parallel on different processors. The default number of processes is the number of processors available. `solve()` takes a new keyword argument `nproc` which can be used to set this number a manually.
- Variables are now post-processed using CasADi ([#1316](https://github.com/pybamm-team/PyBaMM/pull/1316))
- Operations such as `1*x` and `0+x` now directly return `x` ([#1252](https://github.com/pybamm-team/PyBaMM/pull/1252))

## Bug fixes

- Fixed a bug on the boundary conditions of `FickianSingleParticle` and `FickianManyParticles` to ensure mass is conserved ([#1421](https://github.com/pybamm-team/PyBaMM/pull/1421))
- Fixed a bug where the `PolynomialSingleParticle` submodel gave incorrect results with "dimensionality" equal to 2 ([#1411](https://github.com/pybamm-team/PyBaMM/pull/1411))
- Fixed a bug where volume averaging in 0D gave the wrong result ([#1411](https://github.com/pybamm-team/PyBaMM/pull/1411))
- Fixed a sign error in the positive electrode ohmic losses ([#1407](https://github.com/pybamm-team/PyBaMM/pull/1407))
- Fixed the formulation of the EC reaction SEI model ([#1397](https://github.com/pybamm-team/PyBaMM/pull/1397))
- Simulations now stop when an experiment becomes infeasible ([#1395](https://github.com/pybamm-team/PyBaMM/pull/1395))
- Added a check for domains in `Concatenation` ([#1368](https://github.com/pybamm-team/PyBaMM/pull/1368))
- Differentiation now works even when the differentiation variable is a constant ([#1294](https://github.com/pybamm-team/PyBaMM/pull/1294))
- Fixed a bug where the event time and state were no longer returned as part of the solution ([#1344](https://github.com/pybamm-team/PyBaMM/pull/1344))
- Fixed a bug in `CasadiSolver` safe mode which crashed when there were extrapolation events but no termination events ([#1321](https://github.com/pybamm-team/PyBaMM/pull/1321))
- When an `Interpolant` is extrapolated an error is raised for `CasadiSolver` (and a warning is raised for the other solvers) ([#1315](https://github.com/pybamm-team/PyBaMM/pull/1315))
- Fixed `Simulation` and `model.new_copy` to fix a bug where changes to the model were overwritten ([#1278](https://github.com/pybamm-team/PyBaMM/pull/1278))

## Breaking changes

- Removed `Simplification` class and `.simplify()` function ([#1369](https://github.com/pybamm-team/PyBaMM/pull/1369))
- All example notebooks in PyBaMM's GitHub repository must now include the command `pybamm.print_citations()`, otherwise the tests will fail. This is to encourage people to use this command to cite the relevant papers ([#1340](https://github.com/pybamm-team/PyBaMM/pull/1340))
- Notation has been homogenised to use positive and negative electrode (instead of cathode and anode). This applies to the parameter folders (now called `'positive_electrodes'` and `'negative_electrodes'`) and the options of `active_material` and `particle_cracking` submodels (now called `'positive'` and `'negative'`) ([#1337](https://github.com/pybamm-team/PyBaMM/pull/1337))
- `Interpolant` now takes `x` and `y` instead of a single `data` entry ([#1312](https://github.com/pybamm-team/PyBaMM/pull/1312))
- Boolean model options ('sei porosity change', 'convection') must now be given in string format ('true' or 'false' instead of True or False) ([#1280](https://github.com/pybamm-team/PyBaMM/pull/1280))
- Operations such as `1*x` and `0+x` now directly return `x`. This can be bypassed by explicitly creating the binary operators, e.g. `pybamm.Multiplication(1, x)` ([#1252](https://github.com/pybamm-team/PyBaMM/pull/1252))
- `'Cell capacity [A.h]'` has been renamed to `'Nominal cell capacity [A.h]'`. `'Cell capacity [A.h]'` will be deprecated in the next release. ([#1352](https://github.com/pybamm-team/PyBaMM/pull/1352))

# [v0.3.0](https://github.com/pybamm-team/PyBaMM/tree/v0.3.0) - 2020-12-01

This release introduces a new aging model for particle mechanics, a new reduced-order model (TSPMe), and a parameter set for A123 LFP cells. Additionally, there have been several backend optimizations to speed up model creation and solving, and other minor features and bug fixes.

## Features

- Added a submodel for particle mechanics ([#1232](https://github.com/pybamm-team/PyBaMM/pull/1232))
- Added a notebook on how to speed up the solver and handle instabilities ([#1223](https://github.com/pybamm-team/PyBaMM/pull/1223))
- Improve string printing of `BinaryOperator`, `Function`, and `Concatenation` objects ([#1223](https://github.com/pybamm-team/PyBaMM/pull/1223))
- Added `Solution.integration_time`, which is the time taken just by the integration subroutine, without extra setups ([#1223](https://github.com/pybamm-team/PyBaMM/pull/1223))
- Added parameter set for an A123 LFP cell ([#1209](https://github.com/pybamm-team/PyBaMM/pull/1209))
- Added variables related to equivalent circuit models ([#1204](https://github.com/pybamm-team/PyBaMM/pull/1204))
- Added the `Integrated` electrolyte conductivity submodel ([#1188](https://github.com/pybamm-team/PyBaMM/pull/1188))
- Added an example script to check conservation of lithium ([#1186](https://github.com/pybamm-team/PyBaMM/pull/1186))
- Added `erf` and `erfc` functions ([#1184](https://github.com/pybamm-team/PyBaMM/pull/1184))

## Optimizations

- Add (optional) smooth approximations for the `Minimum`, `Maximum`, `Heaviside`, and `AbsoluteValue` operators ([#1223](https://github.com/pybamm-team/PyBaMM/pull/1223))
- Avoid unnecessary repeated computations in the solvers ([#1222](https://github.com/pybamm-team/PyBaMM/pull/1222))
- Rewrite `Symbol.is_constant` to be more efficient ([#1222](https://github.com/pybamm-team/PyBaMM/pull/1222))
- Cache shape and size calculations ([#1222](https://github.com/pybamm-team/PyBaMM/pull/1222))
- Only instantiate the geometric, electrical and thermal parameter classes once ([#1222](https://github.com/pybamm-team/PyBaMM/pull/1222))

## Bug fixes

- Quickplot now works when timescale or lengthscale is a function of an input parameter ([#1234](https://github.com/pybamm-team/PyBaMM/pull/1234))
- Fix bug that was slowing down creation of the EC reaction SEI submodel ([#1227](https://github.com/pybamm-team/PyBaMM/pull/1227))
- Add missing separator thermal parameters for the Ecker parameter set ([#1226](https://github.com/pybamm-team/PyBaMM/pull/1226))
- Make sure simulation solves when evaluated timescale is a function of an input parameter ([#1218](https://github.com/pybamm-team/PyBaMM/pull/1218))
- Raise error if saving to MATLAB with variable names that MATLAB can't read, and give option of providing alternative variable names ([#1206](https://github.com/pybamm-team/PyBaMM/pull/1206))
- Raise error if the boundary condition at the origin in a spherical domain is other than no-flux ([#1175](https://github.com/pybamm-team/PyBaMM/pull/1175))
- Fix boundary conditions at r = 0 for Creating Models notebooks ([#1173](https://github.com/pybamm-team/PyBaMM/pull/1173))

## Breaking changes

- The parameters "Positive/Negative particle distribution in x" and "Positive/Negative surface area to volume ratio distribution in x" have been deprecated. Instead, users can provide "Positive/Negative particle radius [m]" and "Positive/Negative surface area to volume ratio [m-1]" directly as functions of through-cell position (x [m]) ([#1237](https://github.com/pybamm-team/PyBaMM/pull/1237))

# [v0.2.4](https://github.com/pybamm-team/PyBaMM/tree/v0.2.4) - 2020-09-07

This release adds new operators for more complex models, some basic sensitivity analysis, and a spectral volumes spatial method, as well as some small bug fixes.

## Features

- Added variables which track the total amount of lithium in the system ([#1136](https://github.com/pybamm-team/PyBaMM/pull/1136))
- Added `Upwind` and `Downwind` operators for convection ([#1134](https://github.com/pybamm-team/PyBaMM/pull/1134))
- Added Getting Started notebook on solver options and changing the mesh. Also added a notebook detailing the different thermal options, and a notebook explaining the steps that occur behind the scenes in the `Simulation` class ([#1131](https://github.com/pybamm-team/PyBaMM/pull/1131))
- Added particle submodel that use a polynomial approximation to the concentration within the electrode particles ([#1130](https://github.com/pybamm-team/PyBaMM/pull/1130))
- Added `Modulo`, `Floor` and `Ceiling` operators ([#1121](https://github.com/pybamm-team/PyBaMM/pull/1121))
- Added DFN model for a half cell ([#1121](https://github.com/pybamm-team/PyBaMM/pull/1121))
- Automatically compute surface area to volume ratio based on particle shape for li-ion models ([#1120](https://github.com/pybamm-team/PyBaMM/pull/1120))
- Added "R-averaged particle concentration" variables ([#1118](https://github.com/pybamm-team/PyBaMM/pull/1118))
- Added support for sensitivity calculations to the casadi solver ([#1109](https://github.com/pybamm-team/PyBaMM/pull/1109))
- Added support for index 1 semi-explicit dae equations and sensitivity calculations to JAX BDF solver ([#1107](https://github.com/pybamm-team/PyBaMM/pull/1107))
- Allowed keyword arguments to be passed to `Simulation.plot()` ([#1099](https://github.com/pybamm-team/PyBaMM/pull/1099))
- Added the Spectral Volumes spatial method and the submesh that it works with ([#900](https://github.com/pybamm-team/PyBaMM/pull/900))

## Bug fixes

- Fixed bug where some parameters were not being set by the `EcReactionLimited` SEI model ([#1136](https://github.com/pybamm-team/PyBaMM/pull/1136))
- Fixed bug on electrolyte potential for `BasicDFNHalfCell` ([#1133](https://github.com/pybamm-team/PyBaMM/pull/1133))
- Fixed `r_average` to work with `SecondaryBroadcast` ([#1118](https://github.com/pybamm-team/PyBaMM/pull/1118))
- Fixed finite volume discretisation of spherical integrals ([#1118](https://github.com/pybamm-team/PyBaMM/pull/1118))
- `t_eval` now gets changed to a `linspace` if a list of length 2 is passed ([#1113](https://github.com/pybamm-team/PyBaMM/pull/1113))
- Fixed bug when setting a function with an `InputParameter` ([#1111](https://github.com/pybamm-team/PyBaMM/pull/1111))

## Breaking changes

- The "fast diffusion" particle option has been renamed "uniform profile" ([#1130](https://github.com/pybamm-team/PyBaMM/pull/1130))
- The modules containing standard parameters are now classes so they can take options
  (e.g. `standard_parameters_lithium_ion` is now `LithiumIonParameters`) ([#1120](https://github.com/pybamm-team/PyBaMM/pull/1120))
- Renamed `quick_plot_vars` to `output_variables` in `Simulation` to be consistent with `QuickPlot`. Passing `quick_plot_vars` to `Simulation.plot()` has been deprecated and `output_variables` should be passed instead ([#1099](https://github.com/pybamm-team/PyBaMM/pull/1099))

# [v0.2.3](https://github.com/pybamm-team/PyBaMM/tree/v0.2.3) - 2020-07-01

This release enables the use of [Google Colab](https://colab.research.google.com/github/pybamm-team/PyBaMM/blob/main/) for running example notebooks, and adds some small new features and bug fixes.

## Features

- Added JAX evaluator, and ODE solver ([#1038](https://github.com/pybamm-team/PyBaMM/pull/1038))
- Reformatted Getting Started notebooks ([#1083](https://github.com/pybamm-team/PyBaMM/pull/1083))
- Reformatted Landesfeind electrolytes ([#1064](https://github.com/pybamm-team/PyBaMM/pull/1064))
- Adapted examples to be run in Google Colab ([#1061](https://github.com/pybamm-team/PyBaMM/pull/1061))
- Added some new solvers for algebraic models ([#1059](https://github.com/pybamm-team/PyBaMM/pull/1059))
- Added `length_scales` attribute to models ([#1058](https://github.com/pybamm-team/PyBaMM/pull/1058))
- Added averaging in secondary dimensions ([#1057](https://github.com/pybamm-team/PyBaMM/pull/1057))
- Added SEI reaction based on Yang et. al. 2017 and reduction in porosity ([#1009](https://github.com/pybamm-team/PyBaMM/issues/1009))

## Optimizations

- Reformatted CasADi "safe" mode to deal with events better ([#1089](https://github.com/pybamm-team/PyBaMM/pull/1089))

## Bug fixes

- Fixed a bug in `InterstitialDiffusionLimited` ([#1097](https://github.com/pybamm-team/PyBaMM/pull/1097))
- Fixed `Simulation` to keep different copies of the model so that parameters can be changed between simulations ([#1090](https://github.com/pybamm-team/PyBaMM/pull/1090))
- Fixed `model.new_copy()` to keep custom submodels ([#1090](https://github.com/pybamm-team/PyBaMM/pull/1090))
- 2D processed variables can now be evaluated at the domain boundaries ([#1088](https://github.com/pybamm-team/PyBaMM/pull/1088))
- Update the default variable points to better capture behaviour in the solid particles in li-ion models ([#1081](https://github.com/pybamm-team/PyBaMM/pull/1081))
- Fix `QuickPlot` to display variables discretised by FEM (in y-z) properly ([#1078](https://github.com/pybamm-team/PyBaMM/pull/1078))
- Add length scales to `EffectiveResistance` models ([#1071](https://github.com/pybamm-team/PyBaMM/pull/1071))
- Allowed for pybamm functions exp, sin, cos, sqrt to be used in expression trees that
  are converted to casadi format ([#1067](https://github.com/pybamm-team/PyBaMM/pull/1067))
- Fix a bug where variables that depend on y and z were transposed in `QuickPlot` ([#1055](https://github.com/pybamm-team/PyBaMM/pull/1055))

## Breaking changes

- `Simulation.specs` and `Simulation.set_defaults` have been deprecated. Users should create a new `Simulation` object for each different case instead ([#1090](https://github.com/pybamm-team/PyBaMM/pull/1090))
- The solution times `t_eval` must now be provided to `Simulation.solve()` when not using an experiment or prescribing the current using drive cycle data ([#1086](https://github.com/pybamm-team/PyBaMM/pull/1086))

# [v0.2.2](https://github.com/pybamm-team/PyBaMM/tree/v0.2.2) - 2020-06-01

New SEI models, simplification of submodel structure, as well as optimisations and general bug fixes.

## Features

- Reformatted `Geometry` and `Mesh` classes ([#1032](https://github.com/pybamm-team/PyBaMM/pull/1032))
- Added arbitrary geometry to the lumped thermal model ([#718](https://github.com/pybamm-team/PyBaMM/issues/718))
- Allowed `ProcessedVariable` to handle cases where `len(solution.t)=1` ([#1020](https://github.com/pybamm-team/PyBaMM/pull/1020))
- Added `BackwardIndefiniteIntegral` symbol ([#1014](https://github.com/pybamm-team/PyBaMM/pull/1014))
- Added `plot` and `plot2D` to enable easy plotting of `pybamm.Array` objects ([#1008](https://github.com/pybamm-team/PyBaMM/pull/1008))
- Updated effective current collector models and added example notebook ([#1007](https://github.com/pybamm-team/PyBaMM/pull/1007))
- Added SEI film resistance as an option ([#994](https://github.com/pybamm-team/PyBaMM/pull/994))
- Added `parameters` attribute to `pybamm.BaseModel` and `pybamm.Geometry` that lists all of the required parameters ([#993](https://github.com/pybamm-team/PyBaMM/pull/993))
- Added tab, edge, and surface cooling ([#965](https://github.com/pybamm-team/PyBaMM/pull/965))
- Added functionality to solver to automatically discretise a 0D model ([#947](https://github.com/pybamm-team/PyBaMM/pull/947))
- Added sensitivity to `CasadiAlgebraicSolver` ([#940](https://github.com/pybamm-team/PyBaMM/pull/940))
- Added `ProcessedSymbolicVariable` class, which can handle symbolic variables (i.e. variables for which the inputs are symbolic) ([#940](https://github.com/pybamm-team/PyBaMM/pull/940))
- Made `QuickPlot` compatible with Google Colab ([#935](https://github.com/pybamm-team/PyBaMM/pull/935))
- Added `BasicFull` model for lead-acid ([#932](https://github.com/pybamm-team/PyBaMM/pull/932))
- Added 'arctan' function ([#973](https://github.com/pybamm-team/PyBaMM/pull/973))

## Optimizations

- Implementing the use of GitHub Actions for CI ([#855](https://github.com/pybamm-team/PyBaMM/pull/855))
- Changed default solver for DAE models to `CasadiSolver` ([#978](https://github.com/pybamm-team/PyBaMM/pull/978))
- Added some extra simplifications to the expression tree ([#971](https://github.com/pybamm-team/PyBaMM/pull/971))
- Changed the behaviour of "safe" mode in `CasadiSolver` ([#956](https://github.com/pybamm-team/PyBaMM/pull/956))
- Sped up model building ([#927](https://github.com/pybamm-team/PyBaMM/pull/927))
- Changed default solver for lead-acid to `CasadiSolver` ([#927](https://github.com/pybamm-team/PyBaMM/pull/927))

## Bug fixes

- Fix a bug where slider plots do not update properly in notebooks ([#1041](https://github.com/pybamm-team/PyBaMM/pull/1041))
- Fix storing and plotting external variables in the solution ([#1026](https://github.com/pybamm-team/PyBaMM/pull/1026))
- Fix running a simulation with a model that is already discretized ([#1025](https://github.com/pybamm-team/PyBaMM/pull/1025))
- Fix CI not triggering for PR. ([#1013](https://github.com/pybamm-team/PyBaMM/pull/1013))
- Fix schedule testing running too often. ([#1010](https://github.com/pybamm-team/PyBaMM/pull/1010))
- Fix doctests failing due to mismatch in unsorted output.([#990](https://github.com/pybamm-team/PyBaMM/pull/990))
- Added extra checks when creating a model, for clearer errors ([#971](https://github.com/pybamm-team/PyBaMM/pull/971))
- Fixed `Interpolant` ids to allow processing ([#962](https://github.com/pybamm-team/PyBaMM/pull/962))
- Fixed a bug in the initial conditions of the potential pair model ([#954](https://github.com/pybamm-team/PyBaMM/pull/954))
- Changed simulation attributes to assign copies rather than the objects themselves ([#952](https://github.com/pybamm-team/PyBaMM/pull/952))
- Added default values to base model so that it works with the `Simulation` class ([#952](https://github.com/pybamm-team/PyBaMM/pull/952))
- Fixed solver to recompute initial conditions when inputs are changed ([#951](https://github.com/pybamm-team/PyBaMM/pull/951))
- Reformatted thermal submodels ([#938](https://github.com/pybamm-team/PyBaMM/pull/938))
- Reformatted electrolyte submodels ([#927](https://github.com/pybamm-team/PyBaMM/pull/927))
- Reformatted convection submodels ([#635](https://github.com/pybamm-team/PyBaMM/pull/635))

## Breaking changes

- Geometry should no longer be given keys 'primary' or 'secondary' ([#1032](https://github.com/pybamm-team/PyBaMM/pull/1032))
- Calls to `ProcessedVariable` objects are now made using dimensional time and space ([#1028](https://github.com/pybamm-team/PyBaMM/pull/1028))
- For variables discretised using finite elements the result returned by calling `ProcessedVariable` is now transposed ([#1020](https://github.com/pybamm-team/PyBaMM/pull/1020))
- Renamed "surface area density" to "surface area to volume ratio" ([#975](https://github.com/pybamm-team/PyBaMM/pull/975))
- Replaced "reaction rate" with "exchange-current density" ([#975](https://github.com/pybamm-team/PyBaMM/pull/975))
- Changed the implementation of reactions in submodels ([#948](https://github.com/pybamm-team/PyBaMM/pull/948))
- Removed some inputs like `T_inf`, `R_g` and activation energies to some of the standard function parameters. This is because each of those inputs is specific to a particular function (e.g. the reference temperature at which the function was measured). To change a property such as the activation energy, users should create a new function, specifying the relevant property as a `Parameter` or `InputParameter` ([#942](https://github.com/pybamm-team/PyBaMM/pull/942))
- The thermal option 'xyz-lumped' has been removed. The option 'thermal current collector' has also been removed ([#938](https://github.com/pybamm-team/PyBaMM/pull/938))
- The 'C-rate' parameter has been deprecated. Use 'Current function [A]' instead. The cell capacity can be accessed as 'Cell capacity [A.h]', and used to calculate current from C-rate ([#952](https://github.com/pybamm-team/PyBaMM/pull/952))

# [v0.2.1](https://github.com/pybamm-team/PyBaMM/tree/v0.2.1) - 2020-03-31

New expression tree node types, models, parameter sets and solvers, as well as general bug fixes and new examples.

## Features

- Store variable slices in model for inspection ([#925](https://github.com/pybamm-team/PyBaMM/pull/925))
- Added LiNiCoO2 parameter set from Ecker et. al. ([#922](https://github.com/pybamm-team/PyBaMM/pull/922))
- Made t_plus (optionally) a function of electrolyte concentration, and added (1 + dlnf/dlnc) to models ([#921](https://github.com/pybamm-team/PyBaMM/pull/921))
- Added `DummySolver` for empty models ([#915](https://github.com/pybamm-team/PyBaMM/pull/915))
- Added functionality to broadcast to edges ([#891](https://github.com/pybamm-team/PyBaMM/pull/891))
- Reformatted and cleaned up `QuickPlot` ([#886](https://github.com/pybamm-team/PyBaMM/pull/886))
- Added thermal effects to lead-acid models ([#885](https://github.com/pybamm-team/PyBaMM/pull/885))
- Added a helper function for info on function parameters ([#881](https://github.com/pybamm-team/PyBaMM/pull/881))
- Added additional notebooks showing how to create and compare models ([#877](https://github.com/pybamm-team/PyBaMM/pull/877))
- Added `Minimum`, `Maximum` and `Sign` operators
  ([#876](https://github.com/pybamm-team/PyBaMM/pull/876))
- Added a search feature to `FuzzyDict` ([#875](https://github.com/pybamm-team/PyBaMM/pull/875))
- Add ambient temperature as a function of time ([#872](https://github.com/pybamm-team/PyBaMM/pull/872))
- Added `CasadiAlgebraicSolver` for solving algebraic systems with CasADi ([#868](https://github.com/pybamm-team/PyBaMM/pull/868))
- Added electrolyte functions from Landesfeind ([#860](https://github.com/pybamm-team/PyBaMM/pull/860))
- Add new symbols `VariableDot`, representing the derivative of a variable wrt time,
  and `StateVectorDot`, representing the derivative of a state vector wrt time
  ([#858](https://github.com/pybamm-team/PyBaMM/issues/858))

## Bug fixes

- Filter out discontinuities that occur after solve times
  ([#941](https://github.com/pybamm-team/PyBaMM/pull/945))
- Fixed tight layout for QuickPlot in jupyter notebooks ([#930](https://github.com/pybamm-team/PyBaMM/pull/930))
- Fixed bug raised if function returns a scalar ([#919](https://github.com/pybamm-team/PyBaMM/pull/919))
- Fixed event handling in `ScipySolver` ([#905](https://github.com/pybamm-team/PyBaMM/pull/905))
- Made input handling clearer in solvers ([#905](https://github.com/pybamm-team/PyBaMM/pull/905))
- Updated Getting started notebook 2 ([#903](https://github.com/pybamm-team/PyBaMM/pull/903))
- Reformatted external circuit submodels ([#879](https://github.com/pybamm-team/PyBaMM/pull/879))
- Some bug fixes to generalize specifying models that aren't battery models, see [#846](https://github.com/pybamm-team/PyBaMM/issues/846)
- Reformatted interface submodels to be more readable ([#866](https://github.com/pybamm-team/PyBaMM/pull/866))
- Removed double-counted "number of electrodes connected in parallel" from simulation ([#864](https://github.com/pybamm-team/PyBaMM/pull/864))

## Breaking changes

- Changed keyword argument `u` for inputs (when evaluating an object) to `inputs` ([#905](https://github.com/pybamm-team/PyBaMM/pull/905))
- Removed "set external temperature" and "set external potential" options. Use "external submodels" option instead ([#862](https://github.com/pybamm-team/PyBaMM/pull/862))

# [v0.2.0](https://github.com/pybamm-team/PyBaMM/tree/v0.2.0) - 2020-02-26

This release introduces many new features and optimizations. All models can now be solved using the pip installation - in particular, the DFN can be solved in around 0.1s. Other highlights include an improved user interface, simulations of experimental protocols (GITT, CCCV, etc), new parameter sets for NCA and LGM50, drive cycles, "input parameters" and "external variables" for quickly solving models with different parameter values and coupling with external software, and general bug fixes and optimizations.

## Features

- Added LG M50 parameter set from Chen 2020 ([#854](https://github.com/pybamm-team/PyBaMM/pull/854))
- Changed rootfinding algorithm to CasADi, scipy.optimize.root still accessible as an option ([#844](https://github.com/pybamm-team/PyBaMM/pull/844))
- Added capacitance effects to lithium-ion models ([#842](https://github.com/pybamm-team/PyBaMM/pull/842))
- Added NCA parameter set ([#824](https://github.com/pybamm-team/PyBaMM/pull/824))
- Added functionality to `Solution` that automatically gets `t_eval` from the data when simulating drive cycles and performs checks to ensure the output has the required resolution to accurately capture the input current ([#819](https://github.com/pybamm-team/PyBaMM/pull/819))
- Added `Citations` object to print references when specific functionality is used ([#818](https://github.com/pybamm-team/PyBaMM/pull/818))
- Updated `Solution` to allow exporting to matlab and csv formats ([#811](https://github.com/pybamm-team/PyBaMM/pull/811))
- Allow porosity to vary in space ([#809](https://github.com/pybamm-team/PyBaMM/pull/809))
- Added functionality to solve DAE models with non-smooth current inputs ([#808](https://github.com/pybamm-team/PyBaMM/pull/808))
- Added functionality to simulate experiments and testing protocols ([#807](https://github.com/pybamm-team/PyBaMM/pull/807))
- Added fuzzy string matching for parameters and variables ([#796](https://github.com/pybamm-team/PyBaMM/pull/796))
- Changed ParameterValues to raise an error when a parameter that wasn't previously defined is updated ([#796](https://github.com/pybamm-team/PyBaMM/pull/796))
- Added some basic models (BasicSPM and BasicDFN) in order to clearly demonstrate the PyBaMM model structure for battery models ([#795](https://github.com/pybamm-team/PyBaMM/pull/795))
- Allow initial conditions in the particle to depend on x ([#786](https://github.com/pybamm-team/PyBaMM/pull/786))
- Added the harmonic mean to the Finite Volume method, which is now used when computing fluxes ([#783](https://github.com/pybamm-team/PyBaMM/pull/783))
- Refactored `Solution` to make it a dictionary that contains all of the solution variables. This automatically creates `ProcessedVariable` objects when required, so that the solution can be obtained much more easily. ([#781](https://github.com/pybamm-team/PyBaMM/pull/781))
- Added notebook to explain broadcasts ([#776](https://github.com/pybamm-team/PyBaMM/pull/776))
- Added a step to discretisation that automatically compute the inverse of the mass matrix of the differential part of the problem so that the underlying DAEs can be provided in semi-explicit form, as required by the CasADi solver ([#769](https://github.com/pybamm-team/PyBaMM/pull/769))
- Added the gradient operation for the Finite Element Method ([#767](https://github.com/pybamm-team/PyBaMM/pull/767))
- Added `InputParameter` node for quickly changing parameter values ([#752](https://github.com/pybamm-team/PyBaMM/pull/752))
- Added submodels for operating modes other than current-controlled ([#751](https://github.com/pybamm-team/PyBaMM/pull/751))
- Changed finite volume discretisation to use exact values provided by Neumann boundary conditions when computing the gradient instead of adding ghost nodes([#748](https://github.com/pybamm-team/PyBaMM/pull/748))
- Added optional R(x) distribution in particle models ([#745](https://github.com/pybamm-team/PyBaMM/pull/745))
- Generalized importing of external variables ([#728](https://github.com/pybamm-team/PyBaMM/pull/728))
- Separated active and inactive material volume fractions ([#726](https://github.com/pybamm-team/PyBaMM/pull/726))
- Added submodels for tortuosity ([#726](https://github.com/pybamm-team/PyBaMM/pull/726))
- Simplified the interface for setting current functions ([#723](https://github.com/pybamm-team/PyBaMM/pull/723))
- Added Heaviside operator ([#723](https://github.com/pybamm-team/PyBaMM/pull/723))
- New extrapolation methods ([#707](https://github.com/pybamm-team/PyBaMM/pull/707))
- Added some "Getting Started" documentation ([#703](https://github.com/pybamm-team/PyBaMM/pull/703))
- Allow abs tolerance to be set by variable for IDA KLU solver ([#700](https://github.com/pybamm-team/PyBaMM/pull/700))
- Added Simulation class ([#693](https://github.com/pybamm-team/PyBaMM/pull/693)) with load/save functionality ([#732](https://github.com/pybamm-team/PyBaMM/pull/732))
- Added interface to CasADi solver ([#687](https://github.com/pybamm-team/PyBaMM/pull/687), [#691](https://github.com/pybamm-team/PyBaMM/pull/691), [#714](https://github.com/pybamm-team/PyBaMM/pull/714)). This makes the SUNDIALS DAE solvers (Scikits and KLU) truly optional (though IDA KLU is recommended for solving the DFN).
- Added option to use CasADi's Algorithmic Differentiation framework to calculate Jacobians ([#687](https://github.com/pybamm-team/PyBaMM/pull/687))
- Added method to evaluate parameters more easily ([#669](https://github.com/pybamm-team/PyBaMM/pull/669))
- Added `Jacobian` class to reuse known Jacobians of expressions ([#665](https://github.com/pybamm-team/PyBaMM/pull/670))
- Added `Interpolant` class to interpolate experimental data (e.g. OCP curves) ([#661](https://github.com/pybamm-team/PyBaMM/pull/661))
- Added interface (via pybind11) to sundials with the IDA KLU sparse linear solver ([#657](https://github.com/pybamm-team/PyBaMM/pull/657))
- Allowed parameters to be set by material or by specifying a particular paper ([#647](https://github.com/pybamm-team/PyBaMM/pull/647))
- Set relative and absolute tolerances independently in solvers ([#645](https://github.com/pybamm-team/PyBaMM/pull/645))
- Added basic method to allow (a part of) the State Vector to be updated with results obtained from another solution or package ([#624](https://github.com/pybamm-team/PyBaMM/pull/624))
- Added some non-uniform meshes in 1D and 2D ([#617](https://github.com/pybamm-team/PyBaMM/pull/617))

## Optimizations

- Now simplifying objects that are constant as soon as they are created ([#801](https://github.com/pybamm-team/PyBaMM/pull/801))
- Simplified solver interface ([#800](https://github.com/pybamm-team/PyBaMM/pull/800))
- Added caching for shape evaluation, used during discretisation ([#780](https://github.com/pybamm-team/PyBaMM/pull/780))
- Added an option to skip model checks during discretisation, which could be slow for large models ([#739](https://github.com/pybamm-team/PyBaMM/pull/739))
- Use CasADi's automatic differentation algorithms by default when solving a model ([#714](https://github.com/pybamm-team/PyBaMM/pull/714))
- Avoid re-checking size when making a copy of an `Index` object ([#656](https://github.com/pybamm-team/PyBaMM/pull/656))
- Avoid recalculating `_evaluation_array` when making a copy of a `StateVector` object ([#653](https://github.com/pybamm-team/PyBaMM/pull/653))

## Bug fixes

- Fixed a bug where current loaded from data was incorrectly scaled with the cell capacity ([#852](https://github.com/pybamm-team/PyBaMM/pull/852))
- Moved evaluation of initial conditions to solver ([#839](https://github.com/pybamm-team/PyBaMM/pull/839))
- Fixed a bug where the first line of the data wasn't loaded when parameters are loaded from data ([#819](https://github.com/pybamm-team/PyBaMM/pull/819))
- Made `graphviz` an optional dependency ([#810](https://github.com/pybamm-team/PyBaMM/pull/810))
- Fixed examples to run with basic pip installation ([#800](https://github.com/pybamm-team/PyBaMM/pull/800))
- Added events for CasADi solver when stepping ([#800](https://github.com/pybamm-team/PyBaMM/pull/800))
- Improved implementation of broadcasts ([#776](https://github.com/pybamm-team/PyBaMM/pull/776))
- Fixed a bug which meant that the Ohmic heating in the current collectors was incorrect if using the Finite Element Method ([#767](https://github.com/pybamm-team/PyBaMM/pull/767))
- Improved automatic broadcasting ([#747](https://github.com/pybamm-team/PyBaMM/pull/747))
- Fixed bug with wrong temperature in initial conditions ([#737](https://github.com/pybamm-team/PyBaMM/pull/737))
- Improved flexibility of parameter values so that parameters (such as diffusivity or current) can be set as functions or scalars ([#723](https://github.com/pybamm-team/PyBaMM/pull/723))
- Fixed a bug where boundary conditions were sometimes handled incorrectly in 1+1D models ([#713](https://github.com/pybamm-team/PyBaMM/pull/713))
- Corrected a sign error in Dirichlet boundary conditions in the Finite Element Method ([#706](https://github.com/pybamm-team/PyBaMM/pull/706))
- Passed the correct dimensional temperature to open circuit potential ([#702](https://github.com/pybamm-team/PyBaMM/pull/702))
- Added missing temperature dependence in electrolyte and interface submodels ([#698](https://github.com/pybamm-team/PyBaMM/pull/698))
- Fixed differentiation of functions that have more than one argument ([#687](https://github.com/pybamm-team/PyBaMM/pull/687))
- Added warning if `ProcessedVariable` is called outside its interpolation range ([#681](https://github.com/pybamm-team/PyBaMM/pull/681))
- Updated installation instructions for Mac OS ([#680](https://github.com/pybamm-team/PyBaMM/pull/680))
- Improved the way `ProcessedVariable` objects are created in higher dimensions ([#581](https://github.com/pybamm-team/PyBaMM/pull/581))

## Breaking changes

- Time for solver should now be given in seconds ([#832](https://github.com/pybamm-team/PyBaMM/pull/832))
- Model events are now represented as a list of `pybamm.Event` ([#759](https://github.com/pybamm-team/PyBaMM/issues/759)
- Removed `ParameterValues.update_model`, whose functionality is now replaced by `InputParameter` ([#801](https://github.com/pybamm-team/PyBaMM/pull/801))
- Removed `Outer` and `Kron` nodes as no longer used ([#777](https://github.com/pybamm-team/PyBaMM/pull/777))
- Moved `results` to separate repositories ([#761](https://github.com/pybamm-team/PyBaMM/pull/761))
- The parameters "Bruggeman coefficient" must now be specified separately as "Bruggeman coefficient (electrolyte)" and "Bruggeman coefficient (electrode)"
- The current classes (`GetConstantCurrent`, `GetUserCurrent` and `GetUserData`) have now been removed. Please refer to the [`change-input-current` notebook](https://github.com/pybamm-team/PyBaMM/blob/main/examples/notebooks/change-input-current.ipynb) for information on how to specify an input current
- Parameter functions must now use pybamm functions instead of numpy functions (e.g. `pybamm.exp` instead of `numpy.exp`), as these are then used to construct the expression tree directly. Generally, pybamm syntax follows numpy syntax; please get in touch if a function you need is missing.
- The current must now be updated by changing "Current function [A]" or "C-rate" instead of "Typical current [A]"

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
