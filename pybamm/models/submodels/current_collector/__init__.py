import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'base_current_collector',
        'effective_resistance_current_collector',
        'homogeneous_current_collector',
        'potential_pair',
    },
    submod_attrs={
        'base_current_collector': [
            'BaseModel',
        ],
        'effective_resistance_current_collector': [
            'AlternativeEffectiveResistance2D',
            'BaseEffectiveResistance',
            'EffectiveResistance',
        ],
        'homogeneous_current_collector': [
            'Uniform',
        ],
        'potential_pair': [
            'BasePotentialPair',
            'PotentialPair1plus1D',
            'PotentialPair2plus1D',
        ],
    },
)

__all__ = ['AlternativeEffectiveResistance2D', 'BaseEffectiveResistance',
           'BaseModel', 'BasePotentialPair', 'EffectiveResistance',
           'PotentialPair1plus1D', 'PotentialPair2plus1D', 'Uniform',
           'base_current_collector', 'effective_resistance_current_collector',
           'homogeneous_current_collector', 'potential_pair']
