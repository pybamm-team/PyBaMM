import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'base_sei',
        'constant_sei',
        'no_sei',
        'sei_growth',
        'total_sei',
    },
    submod_attrs={
        'base_sei': [
            'BaseModel',
        ],
        'constant_sei': [
            'ConstantSEI',
        ],
        'no_sei': [
            'NoSEI',
        ],
        'sei_growth': [
            'SEIGrowth',
        ],
        'total_sei': [
            'TotalSEI',
        ],
    },
)

__all__ = ['BaseModel', 'ConstantSEI', 'NoSEI', 'SEIGrowth', 'TotalSEI',
           'base_sei', 'constant_sei', 'no_sei', 'sei_growth', 'total_sei']
