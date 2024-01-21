import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'discretisation',
    },
    submod_attrs={
        'discretisation': [
            'Discretisation',
            'has_bc_of_form',
        ],
    },
)

__all__ = ['Discretisation', 'discretisation', 'has_bc_of_form']
