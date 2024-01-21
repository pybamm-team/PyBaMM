import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'base_parameters',
        'bpx',
        'constants',
        'ecm_parameters',
        'electrical_parameters',
        'geometric_parameters',
        'lead_acid_parameters',
        'lithium_ion_parameters',
        'parameter_sets',
        'parameter_values',
        'process_parameter_data',
        'size_distribution_parameters',
        'thermal_parameters',
    },
    submod_attrs={
        'base_parameters': [
            'BaseParameters',
            'NullParameters',
        ],
        'bpx': [
            'Domain',
            'cell',
            'electrolyte',
            'experiment',
            'negative_current_collector',
            'negative_electrode',
            'positive_current_collector',
            'positive_electrode',
            'preamble',
            'separator',
        ],
        'constants': [
            'F',
            'R',
            'k_b',
            'q_e',
        ],
        'ecm_parameters': [
            'EcmParameters',
        ],
        'electrical_parameters': [
            'ElectricalParameters',
            'electrical_parameters',
        ],
        'geometric_parameters': [
            'DomainGeometricParameters',
            'GeometricParameters',
            'ParticleGeometricParameters',
            'geometric_parameters',
        ],
        'lead_acid_parameters': [
            'DomainLeadAcidParameters',
            'LeadAcidParameters',
            'PhaseLeadAcidParameters',
        ],
        'lithium_ion_parameters': [
            'DomainLithiumIonParameters',
            'LithiumIonParameters',
            'ParticleLithiumIonParameters',
        ],
        'parameter_sets': [
            'ParameterSets',
            'parameter_sets',
        ],
        'parameter_values': [
            'ParameterValues',
        ],
        'process_parameter_data': [
            'process_1D_data',
            'process_2D_data',
            'process_2D_data_csv',
            'process_3D_data_csv',
        ],
        'size_distribution_parameters': [
            'get_size_distribution_parameters',
            'lognormal',
        ],
        'thermal_parameters': [
            'DomainThermalParameters',
            'ThermalParameters',
            'thermal_parameters',
        ],
    },
)

__all__ = ['BaseParameters', 'Domain', 'DomainGeometricParameters',
           'DomainLeadAcidParameters', 'DomainLithiumIonParameters',
           'DomainThermalParameters', 'EcmParameters', 'ElectricalParameters',
           'F', 'GeometricParameters', 'LeadAcidParameters',
           'LithiumIonParameters', 'NullParameters', 'ParameterSets',
           'ParameterValues', 'ParticleGeometricParameters',
           'ParticleLithiumIonParameters', 'PhaseLeadAcidParameters', 'R',
           'ThermalParameters', 'base_parameters', 'bpx', 'cell', 'constants',
           'ecm_parameters', 'electrical_parameters', 'electrolyte',
           'experiment', 'geometric_parameters',
           'get_size_distribution_parameters', 'k_b', 'lead_acid_parameters',
           'lithium_ion_parameters', 'lognormal', 'negative_current_collector',
           'negative_electrode', 'parameter_sets', 'parameter_values',
           'positive_current_collector', 'positive_electrode', 'preamble',
           'process_1D_data', 'process_2D_data', 'process_2D_data_csv',
           'process_3D_data_csv', 'process_parameter_data', 'q_e', 'separator',
           'size_distribution_parameters', 'thermal_parameters']
