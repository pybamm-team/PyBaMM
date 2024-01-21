import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'inverse_butler_volmer',
    },
    submod_attrs={
        'inverse_butler_volmer': [
            'CurrentForInverseButlerVolmer',
            'CurrentForInverseButlerVolmerLithiumMetal',
            'InverseButlerVolmer',
        ],
    },
)

__all__ = ['CurrentForInverseButlerVolmer',
           'CurrentForInverseButlerVolmerLithiumMetal', 'InverseButlerVolmer',
           'inverse_butler_volmer']
