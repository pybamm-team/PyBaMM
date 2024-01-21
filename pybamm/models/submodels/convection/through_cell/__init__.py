import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'base_through_cell_convection',
        'explicit_convection',
        'full_convection',
        'no_convection',
    },
    submod_attrs={
        'base_through_cell_convection': [
            'BaseThroughCellModel',
        ],
        'explicit_convection': [
            'Explicit',
        ],
        'full_convection': [
            'Full',
        ],
        'no_convection': [
            'NoConvection',
        ],
    },
)

__all__ = ['BaseThroughCellModel', 'Explicit', 'Full', 'NoConvection',
           'base_through_cell_convection', 'explicit_convection',
           'full_convection', 'no_convection']
