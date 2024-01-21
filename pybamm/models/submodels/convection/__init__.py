import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'base_convection',
        'through_cell',
        'transverse',
    },
    submod_attrs={
        'base_convection': [
            'BaseModel',
        ],
        'through_cell': [
            'BaseThroughCellModel',
            'Explicit',
            'Full',
            'NoConvection',
            'base_through_cell_convection',
            'explicit_convection',
            'full_convection',
            'no_convection',
        ],
        'transverse': [
            'BaseTransverseModel',
            'Full',
            'NoConvection',
            'Uniform',
            'base_transverse_convection',
            'full_convection',
            'no_convection',
            'uniform_convection',
        ],
    },
)

__all__ = ['BaseModel', 'BaseThroughCellModel', 'BaseTransverseModel',
           'Explicit', 'Full', 'NoConvection', 'Uniform', 'base_convection',
           'base_through_cell_convection', 'base_transverse_convection',
           'explicit_convection', 'full_convection', 'no_convection',
           'through_cell', 'transverse', 'uniform_convection']
