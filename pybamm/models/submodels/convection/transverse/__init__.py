import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'base_transverse_convection',
        'full_convection',
        'no_convection',
        'uniform_convection',
    },
    submod_attrs={
        'base_transverse_convection': [
            'BaseTransverseModel',
        ],
        'full_convection': [
            'Full',
        ],
        'no_convection': [
            'NoConvection',
        ],
        'uniform_convection': [
            'Uniform',
        ],
    },
)

__all__ = ['BaseTransverseModel', 'Full', 'NoConvection', 'Uniform',
           'base_transverse_convection', 'full_convection', 'no_convection',
           'uniform_convection']
