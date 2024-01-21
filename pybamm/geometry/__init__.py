import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'battery_geometry',
        'geometry',
        'standard_spatial_vars',
    },
    submod_attrs={
        'battery_geometry': [
            'battery_geometry',
        ],
        'geometry': [
            'Geometry',
        ],
        'standard_spatial_vars': [
            'R_n',
            'R_n_edge',
            'R_p',
            'R_p_edge',
            'r_macro',
            'r_macro_edge',
            'r_n',
            'r_n_edge',
            'r_n_prim',
            'r_n_sec',
            'r_p',
            'r_p_edge',
            'r_p_prim',
            'r_p_sec',
            'whole_cell',
            'x',
            'x_edge',
            'x_n',
            'x_n_edge',
            'x_p',
            'x_p_edge',
            'x_s',
            'x_s_edge',
            'y',
            'y_edge',
            'z',
            'z_edge',
        ],
    },
)

__all__ = ['Geometry', 'R_n', 'R_n_edge', 'R_p', 'R_p_edge',
           'battery_geometry', 'geometry', 'r_macro', 'r_macro_edge', 'r_n',
           'r_n_edge', 'r_n_prim', 'r_n_sec', 'r_p', 'r_p_edge', 'r_p_prim',
           'r_p_sec', 'standard_spatial_vars', 'whole_cell', 'x', 'x_edge',
           'x_n', 'x_n_edge', 'x_p', 'x_p_edge', 'x_s', 'x_s_edge', 'y',
           'y_edge', 'z', 'z_edge']
