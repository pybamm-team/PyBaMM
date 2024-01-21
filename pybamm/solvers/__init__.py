import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'algebraic_solver',
        'base_solver',
        'c_solvers',
        'casadi_algebraic_solver',
        'casadi_solver',
        'dummy_solver',
        'idaklu_solver',
        'jax_bdf_solver',
        'jax_solver',
        'lrudict',
        'processed_variable',
        'processed_variable_computed',
        'scikits_dae_solver',
        'scikits_ode_solver',
        'scipy_solver',
        'solution',
    },
    submod_attrs={
        'algebraic_solver': [
            'AlgebraicSolver',
        ],
        'base_solver': [
            'BaseSolver',
            'process',
        ],
        'casadi_algebraic_solver': [
            'CasadiAlgebraicSolver',
        ],
        'casadi_solver': [
            'CasadiSolver',
        ],
        'dummy_solver': [
            'DummySolver',
        ],
        'idaklu_solver': [
            'IDAKLUSolver',
            'have_idaklu',
            'idaklu_spec',
        ],
        'jax_bdf_solver': [
            'jax_bdf_integrate',
        ],
        'jax_solver': [
            'JaxSolver',
        ],
        'lrudict': [
            'LRUDict',
        ],
        'processed_variable': [
            'ProcessedVariable',
        ],
        'processed_variable_computed': [
            'ProcessedVariableComputed',
        ],
        'scikits_dae_solver': [
            'ScikitsDaeSolver',
            'scikits_odes_spec',
        ],
        'scikits_ode_solver': [
            'ScikitsOdeSolver',
            'have_scikits_odes',
            'scikits_odes_spec',
        ],
        'scipy_solver': [
            'ScipySolver',
        ],
        'solution': [
            'EmptySolution',
            'NumpyEncoder',
            'Solution',
            'make_cycle_solution',
        ],
    },
)

__all__ = ['AlgebraicSolver', 'BaseSolver', 'CasadiAlgebraicSolver',
           'CasadiSolver', 'DummySolver', 'EmptySolution', 'IDAKLUSolver',
           'JaxSolver', 'LRUDict', 'NumpyEncoder', 'ProcessedVariable',
           'ProcessedVariableComputed', 'ScikitsDaeSolver', 'ScikitsOdeSolver',
           'ScipySolver', 'Solution', 'algebraic_solver', 'base_solver',
           'c_solvers', 'casadi_algebraic_solver', 'casadi_solver',
           'dummy_solver', 'have_idaklu', 'have_scikits_odes', 'idaklu_solver',
           'idaklu_spec', 'jax_bdf_integrate', 'jax_bdf_solver', 'jax_solver',
           'lrudict', 'make_cycle_solution', 'process', 'processed_variable',
           'processed_variable_computed', 'scikits_dae_solver',
           'scikits_ode_solver', 'scikits_odes_spec', 'scipy_solver',
           'solution']
