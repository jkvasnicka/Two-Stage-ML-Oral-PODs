'''
'''

import matplotlib.pyplot as plt 
import pandas as pd

from . import utilities

#region: sensitivity_analysis_boxplots
def sensitivity_analysis_boxplots(
        results_manager, 
        data_manager,
        plot_settings, 
        xlim=(0., 1.),
        figsize=(7, 5)
        ):
    '''
    Generate boxplots for sensitivity analysis from performance data.

    Parameters
    ----------
    results_manager : instance of ResultsManager
        Object responsible for managing and combining results from models.
    data_manager : instance of DataManager
        Data manager object for loading dataset features and targets.
    plot_settings : SimpleNamespace
        Contains settings for plotting, such as model labels, metrics, and effects.
    xlim : tuple, optional
        Limits for the x-axis of the boxplots. Default is (0., 1.).
    figsize : tuple, optional
        Figure size for the plots. Default is (7, 5).

    Returns
    -------
    None
    '''
    ## Get the data
    performances = results_manager.combine_results('performances')
    without_selection = results_manager.read_model_keys(
        inclusion_string='false'
        )
    performances = results_manager.combine_results(
        'performances', 
        model_keys=without_selection
    )

    effects = performances.columns.unique(level='target_effect')

    n_evaluation_labels = len(plot_settings.label_for_metric)
    n_effects = len(effects)

    fig, axs = plt.subplots(
        nrows=n_effects,
        ncols=n_evaluation_labels,
        figsize=figsize
    )

    model_key_names = results_manager.read_model_key_names()
    for j, effect in enumerate(effects):
        
        df_wide = prepare_data_for_plotting(
            performances, 
            effect, 
            data_manager, 
            model_key_names, 
            plot_settings
            )

        utilities.vertical_boxplots_subplot(
            axs[j, :],
            df_wide,
            'model_name',
            plot_settings.label_for_metric,
            'metric',
            do_sort=False,
            ascending=True,
            xlim=xlim,
            palette='vlag',
        )
                  
        # Set a title only in the left-most Axes
        title = plot_settings.label_for_effect[effect]
        axs[j, 0].set_title(title, loc='left', fontsize=10)

    for ax in axs.flatten():
        ax.set_xticklabels(
            [_format_tick_label(label.get_text()) 
             for label in ax.get_xticklabels()]
             )  # avoids overlapping ticklabels
        
    fig.tight_layout()

    utilities.save_figure(
        fig, 
        sensitivity_analysis_boxplots, 
        'performances-without-selection',
        bbox_inches='tight'
        )
#endregion

#region: prepare_data_for_plotting
def prepare_data_for_plotting(
        performances, 
        effect, 
        data_manager, 
        model_key_names, 
        plot_settings
        ):
    '''
    Prepare the dataframe for plotting.
     
    Rename and order the columns based on the model configuration.

    Parameters
    ----------
    performances : DataFrame
        DataFrame containing performance metrics.
    effect : str
        The target effect category being analyzed.
    data_manager : instance of DataManager
        Data manager object for loading features and targets.
    model_key_names : list of str
        List of model key names.
    plot_settings : SimpleNamespace
        Settings for plotting, including model labels and configurations.

    Returns
    -------
    DataFrame
        DataFrame ready for plotting with renamed and ordered columns.
    '''
    df_wide = performances.xs(effect, axis=1, level='target_effect')
    df_wide_new = pd.DataFrame(index=df_wide.index)
    n_samples_for = {}

    # Map from model configuration (tuple) to label
    label_for_model = {
        tuple(model): label for label, model 
        in plot_settings.model_for_label.items()
        }

    for col in df_wide.columns:
        model_key = (effect, *col[:-1])
        if model_key not in n_samples_for:
            # Compute the sample size
            _, y_true = data_manager.load_features_and_target(
                **dict(zip(model_key_names, model_key))
                )
            n_samples_for[model_key] = utilities.comma_separated(len(y_true))

        # Use label_for_model to get the label for the current column
        model_label = label_for_model.get(col[:-1], None)
        if model_label:
            n_samples = n_samples_for[model_key]
            new_label = f'{model_label} ({n_samples})'
            metric = col[-1]
            df_wide_new[(new_label, metric)] = df_wide[col]

    # Order columns to match the order in plot_settings.model_for_label
    ordered_columns = []
    for label in plot_settings.model_for_label.keys():
        for col in df_wide_new.columns:
            if label in col[0]:  # Matching label part of the column
                ordered_columns.append(col)
    df_wide_new = df_wide_new.loc[:, ordered_columns]

    names = ['model_name', 'metric']
    df_wide_new.columns = pd.MultiIndex.from_tuples(
        df_wide_new.columns, 
        names=names
        )

    return df_wide_new
#endregion

#region: _format_tick_label
def _format_tick_label(label_text):
    '''
    Format a tick label to have one significant digit of precision.

    Parameters
    ----------
    label_text : str
        The label text to be formatted.

    Returns
    -------
    str
        The formatted label text with one significant digit. If the input 
        cannot be converted to a float (e.g., an empty string), it returns 
        the input unchanged.
    '''
    try:
        return "{:.1g}".format(float(label_text))
    except ValueError:
        return label_text
#endregion
