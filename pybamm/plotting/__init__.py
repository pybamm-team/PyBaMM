import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'dynamic_plot',
        'plot',
        'plot2D',
        'plot_summary_variables',
        'plot_voltage_components',
        'quick_plot',
    },
    submod_attrs={
        'dynamic_plot': [
            'dynamic_plot',
        ],
        'plot': [
            'plot',
        ],
        'plot2D': [
            'plot2D',
        ],
        'plot_summary_variables': [
            'plot_summary_variables',
        ],
        'plot_voltage_components': [
            'plot_voltage_components',
        ],
        'quick_plot': [
            'LoopList',
            'QuickPlot',
            'QuickPlotAxes',
            'ax_max',
            'ax_min',
            'close_plots',
            'split_long_string',
        ],
    },
)

__all__ = ['LoopList', 'QuickPlot', 'QuickPlotAxes', 'ax_max', 'ax_min',
           'close_plots', 'dynamic_plot', 'plot', 'plot2D',
           'plot_summary_variables', 'plot_voltage_components', 'quick_plot',
           'split_long_string']
