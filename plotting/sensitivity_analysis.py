'''
'''

import matplotlib.pyplot as plt 
import pandas as pd

from . import utilities

# FIXME: Split into helper functions
#region: sensitivity_analysis_boxplots
def sensitivity_analysis_boxplots(
        results_manager, 
        data_manager,
        plot_settings, 
        xlim=(0., 1.),
        figsize=(7, 5)
        ):
    '''
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

    # Convert JSON format to model key
    label_for_model = {
        tuple(model) : label 
        for label, model in plot_settings.model_for_label.items()
    }

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
        
        df_wide = performances.xs(effect, axis=1, level='target_effect')
        
        ## Reformat the data 
        
        df_wide_new = pd.DataFrame(index=df_wide.index)

        ## Rename the columns based on the mapping

        n_samples_for = {}  # initialize

        for col in df_wide.columns:

            model_key = (effect, *col[:-1])

            if model_key not in n_samples_for:
                # Compute the sample size from the labeled data
                _, y_true = data_manager.load_features_and_target(
                    **dict(zip(model_key_names, model_key))
                    )
                n_samples_for[model_key] = utilities.comma_separated(len(y_true))

            model_label = label_for_model.get(col[:-1], None)
            n_samples = n_samples_for[model_key]
            new_label = f'{model_label} ({n_samples})'

            if new_label:
                # Rename the column
                metric = col[-1]
                df_wide_new[(new_label, metric)] = df_wide[col]

        ## Re-order the labels to match the order in plot_settings
        # FIXME: This is a quick and rough fix during manuscript revision
        ordered_model_labels = [
            label for label in plot_settings.model_for_label.keys()
            ]
        ordered_columns = []
        for label in ordered_model_labels:
            for col in df_wide_new.columns:
            # Check if the label part of the column matches the ordered label
                if label in col[0]:  
                    ordered_columns.append(col)
        df_wide_new = df_wide_new[ordered_columns]

        names = ['model_name', df_wide.columns.names[-1]]
        df_wide_new.columns = pd.MultiIndex.from_tuples(
            df_wide_new.columns, names=names
            )
        
        df_wide = df_wide_new

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
