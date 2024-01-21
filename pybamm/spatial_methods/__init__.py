import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'finite_volume',
        'scikit_finite_element',
        'spatial_method',
        'spectral_volume',
        'zero_dimensional_method',
    },
    submod_attrs={
        'finite_volume': [
            'FiniteVolume',
        ],
        'scikit_finite_element': [
            'ScikitFiniteElement',
        ],
        'spatial_method': [
            'SpatialMethod',
        ],
        'spectral_volume': [
            'SpectralVolume',
        ],
        'zero_dimensional_method': [
            'ZeroDimensionalSpatialMethod',
        ],
    },
)

__all__ = ['FiniteVolume', 'ScikitFiniteElement', 'SpatialMethod',
           'SpectralVolume', 'ZeroDimensionalSpatialMethod', 'finite_volume',
           'scikit_finite_element', 'spatial_method', 'spectral_volume',
           'zero_dimensional_method']
