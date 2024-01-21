#
# Root of the lithium-ion models module.
#
import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'Yang2017',
        'base_lithium_ion_model',
        'basic_dfn',
        'basic_dfn_composite',
        'basic_dfn_half_cell',
        'basic_spm',
        'dfn',
        'electrode_soh',
        'electrode_soh_half_cell',
        'mpm',
        'msmr',
        'newman_tobias',
        'spm',
        'spme',
    },
    submod_attrs={
        'Yang2017': [
            'Yang2017',
        ],
        'base_lithium_ion_model': [
            'BaseModel',
        ],
        'basic_dfn': [
            'BasicDFN',
        ],
        'basic_dfn_composite': [
            'BasicDFNComposite',
        ],
        'basic_dfn_half_cell': [
            'BasicDFNHalfCell',
        ],
        'basic_spm': [
            'BasicSPM',
        ],
        'dfn': [
            'DFN',
        ],
        'electrode_soh': [
            'ElectrodeSOHSolver',
            'calculate_theoretical_energy',
            'get_initial_ocps',
            'get_initial_stoichiometries',
            'get_min_max_ocps',
            'get_min_max_stoichiometries',
            'theoretical_energy_integral',
        ],
        'electrode_soh_half_cell': [
            'ElectrodeSOHHalfCell',
            'get_initial_stoichiometry_half_cell',
            'get_min_max_stoichiometries',
        ],
        'mpm': [
            'MPM',
        ],
        'msmr': [
            'MSMR',
        ],
        'newman_tobias': [
            'NewmanTobias',
        ],
        'spm': [
            'SPM',
        ],
        'spme': [
            'SPMe',
        ],
    },
)

__all__ = ['BaseModel', 'BasicDFN', 'BasicDFNComposite', 'BasicDFNHalfCell',
           'BasicSPM', 'DFN', 'ElectrodeSOHHalfCell', 'ElectrodeSOHSolver',
           'MPM', 'MSMR', 'NewmanTobias', 'SPM', 'SPMe', 'Yang2017',
           'base_lithium_ion_model', 'basic_dfn', 'basic_dfn_composite',
           'basic_dfn_half_cell', 'basic_spm', 'calculate_theoretical_energy',
           'dfn', 'electrode_soh', 'electrode_soh_half_cell',
           'get_initial_ocps', 'get_initial_stoichiometries',
           'get_initial_stoichiometry_half_cell', 'get_min_max_ocps',
           'get_min_max_stoichiometries', 'mpm', 'msmr', 'newman_tobias',
           'spm', 'spme', 'theoretical_energy_integral']
