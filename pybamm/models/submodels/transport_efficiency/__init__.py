import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'base_transport_efficiency',
        'bruggeman_transport_efficiency',
    },
    submod_attrs={
        'base_transport_efficiency': [
            'BaseModel',
        ],
        'bruggeman_transport_efficiency': [
            'Bruggeman',
        ],
    },
)

__all__ = ['BaseModel', 'Bruggeman', 'base_transport_efficiency',
           'bruggeman_transport_efficiency']
