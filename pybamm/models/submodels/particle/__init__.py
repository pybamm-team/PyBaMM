import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'base_particle',
        'fickian_diffusion',
        'msmr_diffusion',
        'polynomial_profile',
        'total_particle_concentration',
        'x_averaged_polynomial_profile',
    },
    submod_attrs={
        'base_particle': [
            'BaseParticle',
        ],
        'fickian_diffusion': [
            'FickianDiffusion',
        ],
        'msmr_diffusion': [
            'MSMRDiffusion',
        ],
        'polynomial_profile': [
            'PolynomialProfile',
        ],
        'total_particle_concentration': [
            'TotalConcentration',
        ],
        'x_averaged_polynomial_profile': [
            'XAveragedPolynomialProfile',
        ],
    },
)

__all__ = ['BaseParticle', 'FickianDiffusion', 'MSMRDiffusion',
           'PolynomialProfile', 'TotalConcentration',
           'XAveragedPolynomialProfile', 'base_particle', 'fickian_diffusion',
           'msmr_diffusion', 'polynomial_profile',
           'total_particle_concentration', 'x_averaged_polynomial_profile']
