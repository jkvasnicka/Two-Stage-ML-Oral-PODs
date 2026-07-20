'''
Plotting module for visualizing feature importance scores. 

See Also
--------
plot.py
    The main plotting module where this sub-module is implemented as part of 
    the main package.
'''

import matplotlib.pyplot as plt 
from matplotlib.ticker import MaxNLocator

from . import utilities

#region: importances_boxplots
def importances_boxplots(
        results_analyzer, 
        plot_settings,
        output_dir=None
        ):
    '''
    Generate boxplots for feature importances across all models.

    Parameters
    ----------
    results_analyzer : ResultsAnalyzer
        An object to analyze and retrieve data for plotting.
    plot_settings : SimpleNamespace
        Configuration settings for plotting.
    '''
    _feature_importances_subplot(
        results_analyzer, 
        'importances',
        plot_settings,
        importances_boxplots,
        output_dir=output_dir
    )
#endregion

#region: importances_replicates_boxplots
def importances_replicates_boxplots(
        results_analyzer, 
        plot_settings,
        output_dir=None
        ):
    '''
    Generate boxplots for feature importances across replicates within each model.

    Parameters
    ----------
    results_analyzer : ResultsAnalyzer
        An object to analyze and retrieve data for plotting.
    plot_settings : SimpleNamespace
        Configuration settings for plotting.
    '''
    _feature_importances_subplot(
        results_analyzer, 
        'importances_replicates',
        plot_settings,
        importances_replicates_boxplots,
        output_dir=output_dir
    )
#endregion

#region: _feature_importances_subplot
def _feature_importances_subplot(
        results_analyzer, 
        result_type,
        plot_settings,
        function,
        figsize=(8, 10),
        output_dir=None
        ):
    '''
    Helper function to generate a subplot for feature importances or replicates.

    This function is not meant to be called directly but through the
    `importances_boxplots` and `importances_replicates_boxplots` functions.

    Parameters
    ----------
    results_analyzer : ResultsAnalyzer
        An object to analyze and retrieve data for plotting.
    result_type : str
        The type of result to plot ('importances' or 'importances_replicates').
    plot_settings : SimpleNamespace
        Configuration settings for plotting.
    function : function
        The calling function, used to generate appropriate file names for saving.
    figsize : tuple, optional
        The size of the figure to be created. Default is (8, 10).

    Returns
    -------
    None
        The figures are saved to a dedicated directory derived from the 
        function name.
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
            model_key,
            output_dir=output_dir
            )
#endregion