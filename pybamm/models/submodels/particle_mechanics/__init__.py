import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'base_mechanics',
        'crack_propagation',
        'no_mechanics',
        'swelling_only',
    },
    submod_attrs={
        'base_mechanics': [
            'BaseMechanics',
        ],
        'crack_propagation': [
            'CrackPropagation',
        ],
        'no_mechanics': [
            'NoMechanics',
        ],
        'swelling_only': [
            'SwellingOnly',
        ],
    },
)

__all__ = ['BaseMechanics', 'CrackPropagation', 'NoMechanics', 'SwellingOnly',
           'base_mechanics', 'crack_propagation', 'no_mechanics',
           'swelling_only']
