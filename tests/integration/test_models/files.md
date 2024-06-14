.
├── __init__.py
├── standard_model_tests.py
├── standard_output_comparison.py
├── standard_output_tests.py
├── test_full_battery_models
│   ├── __init__.py
│   ├── test_equivalent_circuit
│   │   ├── __init__.py
│   │   └── test_thevenin.py
│   ├── test_lead_acid
│   │   ├── __init__.py
│   │   ├── test_asymptotics_convergence.py
│   │   ├── test_compare_basic_models.py
│   │   ├── test_compare_outputs.py
│   │   ├── test_full.py
│   │   ├── test_loqs.py
│   │   ├── test_loqs_surface_form.py
│   │   └── test_side_reactions
│   │       ├── test_full_side_reactions.py
│   │       └── test_loqs_side_reactions.py
│   └── test_lithium_ion
│       ├── __init__.py
│       ├── base_lithium_ion_half_cell_tests.py
│       ├── base_lithium_ion_tests.py
│       ├── test_basic_models.py
│       ├── test_compare_basic_models.py
│       ├── test_compare_outputs.py
│       ├── test_compare_outputs_two_phase.py
│       ├── test_dfn.py
│       ├── test_dfn_half_cell.py
│       ├── test_external
│       │   ├── __init__.py
│       │   └── test_external_temperature.py
│       ├── test_initial_soc.py
│       ├── test_mpm.py
│       ├── test_newman_tobias.py
│       ├── test_spm.py
│       ├── test_spm_half_cell.py
│       ├── test_spme.py
│       ├── test_spme_half_cell.py
│       ├── test_thermal_models.py
│       └── test_yang2017.py
└── test_submodels
    ├── __init__.py
    ├── test_external_circuit
    │   ├── __init__.py
    │   └── test_function_control.py
    └── test_interface
        ├── __init__.py
        ├── test_butler_volmer.py
        ├── test_lead_acid.py
        └── test_lithium_ion.py

20 directories, 43 files
