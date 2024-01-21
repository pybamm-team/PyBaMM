import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'base_porosity',
        'constant_porosity',
        'reaction_driven_porosity',
        'reaction_driven_porosity_ode',
    },
    submod_attrs={
        'base_porosity': [
            'BaseModel',
        ],
        'constant_porosity': [
            'Constant',
        ],
        'reaction_driven_porosity': [
            'ReactionDriven',
        ],
        'reaction_driven_porosity_ode': [
            'ReactionDrivenODE',
        ],
    },
)

__all__ = ['BaseModel', 'Constant', 'ReactionDriven', 'ReactionDrivenODE',
           'base_porosity', 'constant_porosity', 'reaction_driven_porosity',
           'reaction_driven_porosity_ode']
