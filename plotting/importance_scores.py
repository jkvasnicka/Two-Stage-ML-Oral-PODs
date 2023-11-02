'''
'''

import matplotlib.pyplot as plt 
from matplotlib.ticker import MaxNLocator

from . import utilities

#region: importances_boxplots
def importances_boxplots(results_analyzer, plot_settings):
    '''
    '''
    _feature_importances_subplot(
        results_analyzer, 
        'importances',
        plot_settings,
        importances_boxplots
    )
#endregion

#region: importances_replicates_boxplots
def importances_replicates_boxplots(results_analyzer, plot_settings):
    '''
    '''
    _feature_importances_subplot(
        results_analyzer, 
        'importances_replicates',
        plot_settings,
        importances_replicates_boxplots
    )
#endregion

#region: _feature_importances_subplot
def _feature_importances_subplot(
        results_analyzer, 
        result_type,
        plot_settings,
        function,
        figsize=(8, 10)
        ):
    '''
    '''
    model_keys = results_analyzer.read_model_keys(exclusion_string='false')

    for model_key in model_keys:

        df_wide = results_analyzer.read_result(model_key, result_type)
        
        # FIXME: Pass the subset directly into the helper function?
        xlim = utilities.compute_global_x_limits(
            [df_wide[list(plot_settings.label_for_scoring)]]
            )

        fig, axs = plt.subplots(
            ncols=len(plot_settings.label_for_scoring),
            figsize=figsize
        )

        utilities.vertical_boxplots_subplot(
            axs,
            df_wide,
            'feature',
            plot_settings.label_for_scoring, 
            'scoring',
            xlim=xlim,
            ylabel=plot_settings.feature_names_label
        )

        # Set tick marks dynamically
        for ax in axs:
            ax.xaxis.set_major_locator(MaxNLocator(5))

        fig.tight_layout()

        utilities.save_figure(
            fig, 
            function, 
            model_key
            )
#endregion