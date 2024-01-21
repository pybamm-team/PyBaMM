import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'base_kinetics',
        'butler_volmer',
        'diffusion_limited',
        'inverse_kinetics',
        'linear',
        'marcus',
        'msmr_butler_volmer',
        'no_reaction',
        'tafel',
        'total_main_kinetics',
    },
    submod_attrs={
        'base_kinetics': [
            'BaseKinetics',
        ],
        'butler_volmer': [
            'AsymmetricButlerVolmer',
            'SymmetricButlerVolmer',
        ],
        'diffusion_limited': [
            'DiffusionLimited',
        ],
        'inverse_kinetics': [
            'CurrentForInverseButlerVolmer',
            'CurrentForInverseButlerVolmerLithiumMetal',
            'InverseButlerVolmer',
            'inverse_butler_volmer',
        ],
        'linear': [
            'Linear',
        ],
        'marcus': [
            'Marcus',
            'MarcusHushChidsey',
        ],
        'msmr_butler_volmer': [
            'MSMRButlerVolmer',
        ],
        'no_reaction': [
            'NoReaction',
        ],
        'tafel': [
            'ForwardTafel',
        ],
        'total_main_kinetics': [
            'TotalMainKinetics',
        ],
    },
)

__all__ = ['AsymmetricButlerVolmer', 'BaseKinetics',
           'CurrentForInverseButlerVolmer',
           'CurrentForInverseButlerVolmerLithiumMetal', 'DiffusionLimited',
           'ForwardTafel', 'InverseButlerVolmer', 'Linear', 'MSMRButlerVolmer',
           'Marcus', 'MarcusHushChidsey', 'NoReaction',
           'SymmetricButlerVolmer', 'TotalMainKinetics', 'base_kinetics',
           'butler_volmer', 'diffusion_limited', 'inverse_butler_volmer',
           'inverse_kinetics', 'linear', 'marcus', 'msmr_butler_volmer',
           'no_reaction', 'tafel', 'total_main_kinetics']
