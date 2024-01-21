import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'base_plating',
        'no_plating',
        'plating',
    },
    submod_attrs={
        'base_plating': [
            'BasePlating',
        ],
        'no_plating': [
            'NoPlating',
        ],
        'plating': [
            'Plating',
        ],
    },
)

__all__ = ['BasePlating', 'NoPlating', 'Plating', 'base_plating', 'no_plating',
           'plating']
