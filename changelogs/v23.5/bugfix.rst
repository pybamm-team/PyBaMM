Bug Fixes 
=========

- Realign 'count' increment in CasadiSolver.\_integrate() ([#2986](https://github.com/pybamm-team/PyBaMM/pull/2986))
- Fix `pybamm_install_odes` and update the required SUNDIALS version ([#2958](https://github.com/pybamm-team/PyBaMM/pull/2958))
- Fixed a bug where all data included in a BPX was incorrectly assumed to be given as a function of time.([#2957](https://github.com/pybamm-team/PyBaMM/pull/2957))
- Remove brew install for Mac from the recommended developer installation options for SUNDIALS ([#2925](https://github.com/pybamm-team/PyBaMM/pull/2925))
- Fix `bpx.py` to correctly generate parameters for "lumped" thermal model ([#2860](https://github.com/pybamm-team/PyBaMM/issues/2860))