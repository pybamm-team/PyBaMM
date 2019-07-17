# Results: Valentin Sulzer's Thesis

This folder contains the scripts used to generate results for Valentin Sulzers thesis

## Chapter 2 - Model

- (Dimensionless and dimensional) [parameters](print_lead_acid_parameters.py)
    - Function to print out all standard parameters to a text file, including dependence on C-rate for dimensionless parameters

## Chapter 3 - Simplified models for slow discharge

- [Effect of capacitance](effect_of_capacitance.py): comparison of the one-dimensional model with and without capacitance terms included
    - Voltages
    - Time taken for full solution
- [Discharge results](lead_acid_discharge.py):
    - Comparison of voltage curves for the full porous electrode model and a hierarchy of simplified models (Leading-Order Quasi-Static, First-Order Quasi-Static and Composite)
    - Comparison of variable profiles at various times for the full and reduced-order models
      - Electrolyte concentration
      - Electrolyte potential
      - Interfacial current density
    - Errors compared to full model and time taken to solve each model
    - Breakdown of voltage into constituent overpotentials
