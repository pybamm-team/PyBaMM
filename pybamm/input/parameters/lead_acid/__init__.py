import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'Sulzer2019',
    },
    submod_attrs={
        'Sulzer2019': [
            'conductivity_Gu1997',
            'darken_thermodynamic_factor_Chapman1968',
            'diffusivity_Gu1997',
            'get_parameter_values',
            'lead_dioxide_exchange_current_density_Sulzer2019',
            'lead_dioxide_ocp_Bode1977',
            'lead_exchange_current_density_Sulzer2019',
            'lead_ocp_Bode1977',
            'oxygen_exchange_current_density_Sulzer2019',
        ],
    },
)

__all__ = ['Sulzer2019', 'conductivity_Gu1997',
           'darken_thermodynamic_factor_Chapman1968', 'diffusivity_Gu1997',
           'get_parameter_values',
           'lead_dioxide_exchange_current_density_Sulzer2019',
           'lead_dioxide_ocp_Bode1977',
           'lead_exchange_current_density_Sulzer2019', 'lead_ocp_Bode1977',
           'oxygen_exchange_current_density_Sulzer2019']
