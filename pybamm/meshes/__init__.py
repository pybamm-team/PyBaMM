import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'meshes',
        'one_dimensional_submeshes',
        'scikit_fem_submeshes',
        'zero_dimensional_submesh',
    },
    submod_attrs={
        'meshes': [
            'Mesh',
            'MeshGenerator',
            'SubMesh',
        ],
        'one_dimensional_submeshes': [
            'Chebyshev1DSubMesh',
            'Exponential1DSubMesh',
            'SpectralVolume1DSubMesh',
            'SubMesh1D',
            'Uniform1DSubMesh',
            'UserSupplied1DSubMesh',
        ],
        'scikit_fem_submeshes': [
            'ScikitChebyshev2DSubMesh',
            'ScikitExponential2DSubMesh',
            'ScikitSubMesh2D',
            'ScikitUniform2DSubMesh',
            'UserSupplied2DSubMesh',
        ],
        'zero_dimensional_submesh': [
            'SubMesh0D',
        ],
    },
)

__all__ = ['Chebyshev1DSubMesh', 'Exponential1DSubMesh', 'Mesh',
           'MeshGenerator', 'ScikitChebyshev2DSubMesh',
           'ScikitExponential2DSubMesh', 'ScikitSubMesh2D',
           'ScikitUniform2DSubMesh', 'SpectralVolume1DSubMesh', 'SubMesh',
           'SubMesh0D', 'SubMesh1D', 'Uniform1DSubMesh',
           'UserSupplied1DSubMesh', 'UserSupplied2DSubMesh', 'meshes',
           'one_dimensional_submeshes', 'scikit_fem_submeshes',
           'zero_dimensional_submesh']
