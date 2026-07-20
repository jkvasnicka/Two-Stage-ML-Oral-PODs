'''
Plotting module for visualizing the predictions stratified by missing features.

See Also
--------
plot.py
    The main plotting module where this sub-module is implemented as part of 
    the main package.
'''

import matplotlib.pyplot as plt 
import seaborn as sns

from . import utilities

#region: predictions_by_missing_feature
def predictions_by_missing_feature(
        results_analyzer, 
        plot_settings,
        output_dir=None
    ):
    '''
    Visualize the impact of missing features on model predictions.

    This function generates a series of boxplots to compare the distributions 
    of model predictions stratified by missing features.

    Parameters
    ----------
    results_analyzer : ResultsAnalyzer
        An object to analyze and retrieve data for plotting. 
    plot_settings : SimpleNamespace
        Configuration settings for plotting.

    Returns
    -------
    None
        The figures are saved to a dedicated directory derived from the 
        function name.
    '''
    with sns.axes_style('whitegrid'):

        all_samples_color = '#00008b'
        remaining_color = '#ffff99'

        model_key_names = results_analyzer.read_model_key_names()
        grouped_keys = results_analyzer.group_model_keys(
            'target_effect',
            model_keys=plot_settings.final_model_keys
            )
        
        for grouping_key, model_keys in grouped_keys:

            ## Define the global x-limits based on the interquartile ranges
            series_list = []
            for model_key in model_keys:
                series_list.append(
                    results_analyzer.predict(
                        model_key, 
                        exclude_training=True
                        )[0]
                    )
                series_list.append(
                    results_analyzer.get_in_sample_prediction(model_key)[0]
                    )
            x_limits = utilities.compute_global_x_limits(series_list)

            n_effects = len(model_keys)
            fig, axs = plt.subplots(
                nrows=n_effects, 
                ncols=2, 
                figsize=(8, 4 * n_effects)
                )

            for i, model_key in enumerate(model_keys):
                key_for = dict(zip(model_key_names, model_key))
                y_pred_out, X_out, *_ = (
                    results_analyzer.predict(
                        model_key, 
                        exclude_training=True
                        )
                )
                y_pred_in, X_in, *_ = (
                    results_analyzer.get_in_sample_prediction(model_key)
                )

                dfs_out = _boxplot_by_missing_feature(
                    axs[i, 0], 
                    y_pred_out, 
                    X_out, 
                    all_samples_color, 
                    remaining_color, 
                    plot_settings.prediction_label
                )
                sort_order = list(dfs_out.keys())
                dfs_in = _boxplot_by_missing_feature(
                    axs[i, 1], 
                    y_pred_in, 
                    X_in, 
                    all_samples_color, 
                    remaining_color, 
                    plot_settings.prediction_label, 
                    sort_order
                )

                _set_ytick_labels(axs[i, 0], dfs_out, True)
                _set_ytick_labels(axs[i, 1], dfs_in, False)
                
                ## Set the Axes titles
                effect = plot_settings.label_for_effect[key_for['target_effect']]
                left_title = f"{effect}\n{plot_settings.label_for_sample_type['out']}"
                right_title = plot_settings.label_for_sample_type['in']

                axs[i, 0].set_title(left_title, size='medium', loc='left')
                axs[i, 1].set_title(right_title, size='medium', loc='left')
                
                axs[i, 0].set_xlim(x_limits)
                axs[i, 1].set_xlim(x_limits)

            fig.tight_layout()
            utilities.save_figure(
                fig, 
                predictions_by_missing_feature, 
                grouping_key,
                output_dir=output_dir
                )
#endregion

#region: _boxplot_by_missing_feature
def _boxplot_by_missing_feature(
        ax, 
        df, 
        X, 
        all_samples_color, 
        remaining_color, 
        prediction_label, 
        sort_order=None
        ):
    '''
    Generate boxplot by missing feature on the provided axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to draw the boxplot on.
    df : pd.DataFrame
        The input data.
    X : pd.DataFrame
        DataFrame containing feature values, used to identify missing samples.
    all_samples_color : str
        Color for the 'All Samples' box in the boxplot.
    remaining_color : str
        Color for the remaining boxes in the boxplot.
    sort_order : list of str, optional
        Order to sort the features. If None, sort by the sample size.

    Returns
    -------
    df_for_name : dict
        A dictionary with keys as feature names and values as dataframes 
        representing the distributions.
    '''
    df_for_name = {}
    df_for_name['All Samples'] = df
    for feature_name in X.columns:
        missing_samples = X[X[feature_name].isna()].index
        df_for_name[feature_name] = df.loc[missing_samples]

    # If no sort order is provided, sort by the sample size.
    if sort_order is None:
        sort_order = sorted(
            df_for_name.keys(), 
            key=lambda name: df_for_name[name].size, 
            reverse=False
        )

    df_for_name = {name: df_for_name[name] for name in sort_order}

    _create_boxplot(
        ax, 
        df_for_name, 
        all_samples_color, 
        remaining_color, 
        prediction_label
        )

    return df_for_name
#endregion

#region: _create_boxplot
def _create_boxplot(
        ax, 
        df_for_name, 
        all_samples_color, 
        remaining_color, 
        prediction_label
        ):
    '''
    Draw a boxplot on the given axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to draw the boxplot on.
    df_for_name : dict of pd.DataFrame
        Dictionary with keys as feature names and values as dataframes 
        representing the distributions.
    all_samples_color : str
        Color for the 'All Samples' box in the boxplot.
    remaining_color : str
        Color for the remaining boxes in the boxplot.
    '''
    boxplot = ax.boxplot(
        list(df_for_name.values()),
        vert=False,
        labels=[None]*len(df_for_name),  # will be updated later
        widths=0.6,
        patch_artist=True,
        medianprops={'color': 'black'},
        showfliers=False
    )

    for patch in boxplot['boxes']:
        patch.set(facecolor=remaining_color)

    patch.set(facecolor=all_samples_color)

    for key in ['whiskers', 'caps', 'medians']:
        for element in boxplot[key]:
            element.set(color='black')

    median_value = boxplot['medians'][-1].get_xdata()[0]
    ax.axvline(
        x=median_value,
        color=all_samples_color,
        linestyle='--',
        alpha=0.5
    )

    ax.set_xlabel(f'Predicted {prediction_label}')
    ax.tick_params(axis='y', which='major', labelsize=8)
#endregion

#region: _set_ytick_labels
def _set_ytick_labels(ax, df_for_name, do_label_features):
    '''
    Sets the ytick labels for a given axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to draw the boxplot on.
    df_for_name : dict of pd.DataFrame
        Dictionary with keys as feature names and values as dataframes 
        representing the distributions.
    do_label_features : bool
        If True, feature labels are set for the y-axis ticks. If False, 
        feature labels are not set.
    '''
    labels = [
        _get_box_tick_label(name, series, do_label_features) 
        for name, series in df_for_name.items()
        ]
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_ylabel(None)
#endregion

#region: _get_box_tick_label
def _get_box_tick_label(name, series, do_label_features=True):
    '''
    Return the box label components.

    Parameters
    ----------
    name : str
        Feature name.
    series : pd.Series
        Data series.

    Returns
    -------
    feature_label : str
        Updated box label for the feature.
    sample_size_label : str
        Sample size of the data series in a string format.
    '''
    if name != 'All Samples':
        prefix = 'Missing'
        suffix = f'"{name}"'
    else:
        prefix = ''
        suffix = name
    feature_label = f'{prefix} {suffix}'

    sample_size_label = utilities.comma_separated(len(series))

    if do_label_features:
        label = f'{feature_label} ({sample_size_label})'
    else:
        label = f'({sample_size_label})'
    return label 
#endregion