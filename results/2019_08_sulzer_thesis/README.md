# Results: Valentin Sulzer's Thesis

This folder contains the scripts used to generate results for Valentin Sulzer's thesis:

V Sulzer. *Mathematical modelling of lead-acid batteries*. PhD thesis, University of Oxford, 2019.

The plots were formatted using a formatting file `matplotlibrc` identical to [this one](_matplotlibrc) (but not included in the GitHub repo to avoid clashes with different formatting files).
Each file can be run individually to generate the results described below.
To generate all results, run the [main file](main.py).


## Chapter 2 - Model

- (Dimensionless and dimensional) [parameters](print_lead_acid_parameters.py)
    - Function to print out all standard parameters to a text file, including dependence on C-rate for dimensionless parameters

## Chapter 3 - Simplified models for slow discharge

- [Effect of capacitance](effect_of_capacitance.py): comparison of the one-dimensional model with and without capacitance terms included
    - Voltages
    - Time taken for full solution
- [Effect of convection](effect_of_convection.py):
    - Voltage at various C-rates with and without convection
    - Concentration and velocity profiles
    - Increasing the volume changes to see more of an effect
- [Asymptotics results for discharge](lead_acid_discharge.py):
    - Comparison of voltage curves for the full porous electrode model and a hierarchy of simplified models (Leading-Order Quasi-Static, First-Order Quasi-Static and Composite)
    - Comparison of variable profiles at various times for the full and reduced-order models
        - Electrolyte concentration
        - Electrolyte potential
        - Interfacial current density
    - Decomposition of voltage into constituent overpotentials
- [Times and errors](discharge_times_and_errors.py): Errors compared to full model and time taken to solve each model

## Chapter 4 - Small aspect ratio cells

- Comparison of voltage with COMSOL ([1D](compare_comsol/compare_comsol_lead_acid_1D.py) and [2D](compare_comsol/compare_comsol_lead_acid_2D.py))
- Further asymptotics for small discharge rate with different conductivities:
    - [Poorly conducting](2D/2D_lead_acid_dicharge_poorly_conducting.py):
        - Voltages
        - Concentration snapshots at a fixed time (as functions of x and z)
        - X-averaged concentration at various times (as functions of z)
    - [Quite conductive](2D/2D_lead_acid_dicharge_quite_conducting.py):
        - Voltages
        - Decomposition of voltage into constituent overpotentials
- [Times and errors](2D/2D_all_times_errors.py): Errors compared to full model and time taken to solve each model
- Further asymptotics

## Chapter 5 - Preliminary model for recharge

- [Effect of side reactions](effect_of_side_reactions.py): comparison of solving the model for constant-current recharge with and without side reactions included
    - Voltages
    - Average interfacial current densities
- [Asymptotics results for charge](lead_acid_charge.py):
    - Comparison of voltage curves for the full porous electrode model and a hierarchy of simplified models (Leading-Order Quasi-Static and Composite)
    - Comparison of variable profiles at various times for the full and reduced-order models
        - Electrolyte concentration
        - Oxygen concentration
    - Comparison of average interfacial current densities for each of the side reactions for the full porous electrode model and a hierarchy of simplified models (Leading-Order Quasi-Static and Composite)
    - Decomposition of voltage into constituent overpotentials
