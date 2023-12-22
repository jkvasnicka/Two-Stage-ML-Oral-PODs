'''
Plotting module for Margin of Exposure (MOE) data. 

This module contains functions for plotting cumulative distribution functions
(CDFs) of POD and MOE data across various models and datasets.

Notes
-----
This module is part of a larger plotting sub-package focused on the 
visualization of results from the main package. It relies on external classes 
such as ResultsAnalyzer and PlotSetting for data processing and configuration.
'''

import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np

from . import utilities

POD_XLABEL = '$\log_{10}POD$ [mg∙(kg∙d)$^{-1}$]'
MOE_XLABEL = '$\log_{10}MOE$'

# Define the limits of the vertical spans in log10 units of MOE.
# log10(0) is undefined and will be handled dynamically
MOE_CATEGORIES = {
    'Low Concern' : (2., np.inf),
    'Moderate Concern' : (0., 2.),  # 1, 100
    'High Concern' : (-np.inf, 0.)  # 0, 1
}
MOE_CATEGORY_KWARGS = {
    'Low Concern' : {'color' : 'white', 'alpha' : 0.},  # transparent
    'Moderate Concern' : {'color' : '#a6cee3', 'alpha' : 0.2},
    'High Concern' : {'color' : '#1f78b4', 'alpha' : 0.2},
}

#region: margins_of_exposure_cumulative
def margins_of_exposure_cumulative(results_analyzer, plot_settings):
    '''
    Plot distributions of margin of exposure (MOE), with uncertainty, across
    chemicals.

    This function creates one Axes for each target effect category. Each Axes
    shows the cumulative count of chemicals in relation to their MOE. 
    Exposure uncertainty is represented by separate cumulative distributions 
    for different exposure percentiles. Hazard (POD) uncertainty is 
    represented by prediction intervals for each exposure percentile. MOE
    concern categories are illustrated by vertical spans.

    Parameters
    ----------
    results_analyzer : ResultsAnalyzer
        An instance of ResultsAnalyzer used to retrieve and process MOE data.
    plot_settings : PlotSettings
        An instance of PlotSettings containing label and color configurations.

    Returns
    -------
    None
        The function creates and saves the subplots based on the provided
        results_analyzer and plot_settings.
    '''
    # Get x-axis truncation limit if present
    right_truncation = plot_settings.__dict__.get('moe_right_truncation', None)

    # TODO: Create a method of ResultsAnalyzer and reuse?
    model_key_names, grouped_keys = group_model_keys(results_analyzer)

    for grouping_key, model_keys in grouped_keys:

        nrows = 2  # POD & MOE
        ncols = len(model_keys)  # N effect categories
        fig, axs = plt.subplots(
            nrows,
            ncols,
            figsize=(5*ncols, 5*nrows),
        )

        global_pod_xlim = utilities.initialize_global_limits()
        global_moe_xlim = utilities.initialize_global_limits()

        for i, model_key in enumerate(model_keys):

            ylabel = 'Cumulative Count of Chemicals' if i == 0 else None 

            # TODO: Create a method and reuse in other modules?
            title = get_effect_label(
                model_key, 
                model_key_names, 
                plot_settings.label_for_effect
            )

            plot_model_pod_data(  # first row, column i
                axs[0, i],
                model_key, 
                results_analyzer,
                global_xlim=global_pod_xlim
            )
            format_axes(
                axs[0, i], 
                POD_XLABEL,
                title=title,
                ylabel=ylabel,
                global_xlim=global_pod_xlim
            )

            plot_model_moe_data(  # second row, column i
                axs[1, i], 
                model_key, 
                results_analyzer, 
                plot_settings,
                global_xlim=global_moe_xlim
            )
            format_axes(
                axs[1, i], 
                MOE_XLABEL,
                ylabel=ylabel,
                global_xlim=global_moe_xlim,
                right_truncation=right_truncation
            )
            annotate_vertical_spans(
                axs[1, i], 
                MOE_CATEGORIES, 
                MOE_CATEGORY_KWARGS
                )
        utilities.set_centralized_legend(
            fig, 
            axs[1, -1],
            bottom=0.08,
            bbox_to_anchor=(0.5, -0.01)        
        )

        utilities.save_figure(
            fig,
            margins_of_exposure_cumulative,
            grouping_key
        )
#endregion

#region: single_model_moes
def single_model_moes(
        model_key, 
        results_analyzer, 
        plot_settings, 
        title=None
        ):
    '''
    Plot the distribution of margin of exposure (MOE) for a single model.

    Parameters
    ----------
    model_key : tuple of str
        The key representing the model to plot.
    results_analyzer : ResultsAnalyzer
        An instance of ResultsAnalyzer used to retrieve and process MOE data.
    plot_settings : PlotSettings
        An instance of PlotSettings containing label and color configurations.
    title : str, optional
        Title for the plot. If None, no title is set.

    Returns
    -------
    2-tuple
        matplotlib.pyplot Figure and Axes
    '''
    # Get x-axis truncation limit if present
    right_truncation = plot_settings.__dict__.get('moe_right_truncation', None)

    fig, ax = plt.subplots()

    plot_model_moe_data(
        ax, 
        model_key, 
        results_analyzer, 
        plot_settings
    )

    format_axes(
        ax, 
        MOE_XLABEL,
        title=title,
        ylabel='Cumulative Count of Chemicals',
        right_truncation=right_truncation
    )

    annotate_vertical_spans(
        ax, 
        MOE_CATEGORIES, 
        MOE_CATEGORY_KWARGS
        )

    utilities.set_centralized_legend(
        fig, 
        ax,
        bottom=0.17,
        bbox_to_anchor=(0.5, -0.01)        
    )

    return fig, ax
#endregion

#region: plot_model_pod_data
def plot_model_pod_data(
        ax, 
        model_key, 
        results_analyzer,
        global_xlim=None
        ):
    '''
    Plot Point of Departure (POD) data for a single model.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object to plot on.
    model_key : str
        The key representing the model to plot.
    results_analyzer : ResultsAnalyzer
        An instance of ResultsAnalyzer used to retrieve MOE data.
    global_xlim : list, optional
        The current global limits as a list [min, max].

    Returns
    -------
    None
    '''
    results = results_analyzer.pod_and_prediction_interval(model_key)

    plot_with_prediction_interval(
        ax,
        results['pod'],
        results['cum_count'],
        results['lb'],
        results['ub'],
        color='black'
    )

    if global_xlim:
        utilities.update_global_limits(global_xlim, ax.get_xlim())
#endregion

#region: plot_model_moe_data
def plot_model_moe_data(
        ax, 
        model_key, 
        results_analyzer, 
        plot_settings,
        global_xlim=None
        ):
    '''
    Plot Margin of Exposure (MOE) data for a single model.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object to plot on.
    model_key : str
        The key representing the model to plot.
    results_analyzer : ResultsAnalyzer
        An instance of ResultsAnalyzer used to retrieve MOE data.
    plot_settings : PlotSettings
        An instance of PlotSettings containing label and color configurations.
    global_xlim : list, optional
        The current global limits as a list [min, max].

    Returns
    -------
    None
    '''
    results_for_percentile = results_analyzer.moe_and_prediction_intervals(model_key)
    percentile_colors = sns.color_palette('Set2', len(results_for_percentile))

    for j, (percentile, results) in enumerate(results_for_percentile.items()):
        plot_with_prediction_interval(
            ax,
            results['moe'],
            results['cum_count'],
            results['lb'],
            results['ub'],
            color=percentile_colors[j],
            label=plot_settings.label_for_exposure_column[percentile]
        )

    if global_xlim:
        utilities.update_global_limits(global_xlim, ax.get_xlim())
#endregion

#region: plot_with_prediction_interval
def plot_with_prediction_interval(
        ax, 
        sorted_values, 
        cumulative_counts, 
        lower_bound, 
        upper_bound, 
        color=None, 
        label=None
    ):
    '''
    Plot values with prediction interval.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    sorted_values : pd.Series
        Sorted values in log10 scale.
    cumulative_counts : np.ndarray
        Cumulative counts for the sorted values.
    lower_bound : pd.Series
        Lower bound of the prediction interval.
    upper_bound : pd.Series
        Upper bound of the prediction interval.
    color : str
        Color of the plot.
    label : str, optional
        Label for the line plot.

    Returns
    -------
    None
    '''
    # Plot the prediction interval
    ax.fill_betweenx(
        cumulative_counts, 
        lower_bound, 
        upper_bound, 
        color=color, 
        alpha=0.2
        )

    # Plot the main line
    ax.plot(
        sorted_values, 
        cumulative_counts, 
        color=color, 
        label=label
        )
#endregion

#region: format_axes
def format_axes(
        ax, 
        xlabel,
        title=None,
        ylabel=None,
        global_xlim=None,
        right_truncation=None
        ):
    '''
    Format the Axes of a subplot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object to format.
    title : str, optional
        Title for the subplot. If None, no title is set.
    ylabel : str, optional
        Y-axis label for the subplot. If None, no ylabel is set.
    global_xlim : list
        The current global limits as a list [min, max].
    right_truncation : float, optional
        If provided, sets the right truncation limit for x-axis.

    Returns
    -------
    None
    '''
    ## Update the limits.
    set_even_ticks(ax, axis_type='x', data_type='fill')
    set_even_log_ticks(ax, axis_type='y', data_type='fill')
    if global_xlim:
        ax.set_xlim(global_xlim)
    if right_truncation:
        ax.set_xlim(ax.get_xlim()[0], right_truncation)

    # Remove the padding; the data limits will define the axis limits.
    ax.margins(x=0, y=0)

    ## Set labels, etc. 
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    if ylabel: 
        ax.set_ylabel(ylabel)
#endregion

#region: get_effect_label
def get_effect_label(model_key, model_key_names, label_for_effect):
    '''
    Retrieve the label for the target effect for the specified model.

    Parameters
    ----------
    model_key : tuple of str
        The key representing the model to plot.
    model_key_names : list
        A corresponding list of names for each value in the model key.
    label_for_effect : dict
        A dictionary mapping effect categories to labels.

    Returns
    -------
    str
    '''
    key_for = dict(zip(model_key_names, model_key))
    effect = key_for['target_effect']
    return label_for_effect[effect]
#endregion

#region: annotate_vertical_spans
def annotate_vertical_spans(ax, categories, category_kwargs, y_pos_axes=0.97):
    '''
    Annotate vertical spans and category labels in a matplotlib Axes object.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object in which to annotate vertical spans.
    categories : dict
        Dictionary defining the categories and their corresponding 
        vertical span limits.
    category_kwargs : dict
        Mapping of category names to key-word arguments for Axes.axvspan().
    y_pos_axes : float
        The y-position of the category labels in Axes fraction units.

    Returns
    -------
    None
    '''
    for category, (lower, upper) in categories.items():
        if lower == -np.inf:
            # Extend the lower limit to the xmin of the Axes
            lower = ax.get_xlim()[0]
        if upper == np.inf:
            upper = ax.get_xlim()[-1]

        ax.axvspan(lower, upper, **category_kwargs[category])

        x_pos_data = (lower + upper) / 2  # arithmetic mean in linear scale

        # Convert to data coordinates
        coords_axes = (0, y_pos_axes)
        _, y_pos_data = ax.transData.inverted().transform(
            ax.transAxes.transform(coords_axes))

        ax.text(
            x_pos_data,
            y_pos_data,
            category.replace(" ", "\n"),
            ha='center', 
            va='top',
            fontsize='small',
        )
#endregion

#region: set_even_ticks
def set_even_ticks(ax, axis_type='x', data_type='line'):
    '''
    Set the ticks on an axis of a matplotlib Axes object to be at even 
    intervals.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object on which to set the ticks.
    axis_type : str
        The axis on which to set the ticks. Should be either 'x' or 'y'.

    Returns
    -------
    None
    '''
    # Define the sequence of "nice" numbers
    nice_numbers = [
        1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

    data_min, data_max = get_data_limits(
        ax, 
        axis_type=axis_type, 
        data_type=data_type
        )
    
    # Handle when data_min or data_max is NaN
    if data_min is np.nan or data_max is np.nan:  
        return
    
    # Calculate the range of data and the raw step
    data_range = data_max - data_min
    raw_step = data_range / 10  # We desire around 10 ticks on the axis

    # Get the "nice" step by finding the closest nice number to the raw step
    nice_step = min(nice_numbers, key=lambda x:abs(x-raw_step))

    # Calculate new minimum and maximum values to be multiples of nice_step
    data_min = nice_step * np.floor(data_min/nice_step)
    data_max = nice_step * np.ceil(data_max/nice_step)
    
    if axis_type == 'x':
        ax.set_xlim(data_min, data_max)
        # +nice_step to include data_max in the ticks
        ax.set_xticks(
            np.arange(data_min, data_max + nice_step, step=nice_step))  
    elif axis_type == 'y':
        ax.set_ylim(data_min, data_max)
        # +nice_step to include data_max in the ticks
        ax.set_yticks(
            np.arange(data_min, data_max + nice_step, step=nice_step))  
#endregion

#region: set_even_log_ticks
def set_even_log_ticks(ax, axis_type='x', data_type='line'):
    '''
    Set the ticks on a logarithmic axis of a matplotlib Axes object to be at 
    even intervals.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object on which to set the ticks.
    axis_type : str
        The axis on which to set the ticks. Should be either 'x' or 'y'.

    Returns
    -------
    None
    '''
    # Get the limits of the data
    data_min, data_max = get_data_limits(
        ax, 
        axis_type=axis_type, 
        data_type=data_type
        )

    # Get the powers of 10 that bound the data
    data_min_pow = np.floor(np.log10(data_min))
    data_max_pow = np.ceil(np.log10(data_max))

    # Make sure the exponents are even
    if data_min_pow % 2 != 0:
        data_min_pow -= 1
    if data_max_pow % 2 != 0:
        data_max_pow += 1

    # Create the list of ticks
    # Step by 2 to get even exponents
    ticks = [
        10**i for i in range(int(data_min_pow), int(data_max_pow) + 1, 2)]  

    # Set the ticks on the appropriate axis
    if axis_type == 'x':
        ax.set_xticks(ticks)
    elif axis_type == 'y':
        ax.set_yticks(ticks)
#endregion

#region: get_data_limits
def get_data_limits(ax, axis_type='x', data_type='line'):
    '''
    Get the minimum and maximum limits of the data on a specific axis of a 
    matplotlib Axes object.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object containing the data.
    axis_type : str
        The axis for which to get the data limits. Should be either 'x' or 'y'.
    data_type : str
        The type of data for which to get the limits. Should be either 'line' 
        or 'fill'.

    Returns
    -------
    float, float
        The minimum and maximum data limits.
    '''
    if data_type == 'line':
        # Get the data from line plots
        if axis_type == 'x':
            lines = ax.get_lines()
            data = np.concatenate([line.get_xdata() for line in lines])
        elif axis_type == 'y':
            lines = ax.get_lines()
            data = np.concatenate([line.get_ydata() for line in lines])
    elif data_type == 'fill':
        # Get the data from filled areas
        collections = ax.collections
        axis_index = 0 if axis_type == 'x' else 1
        data = np.concatenate(
            [collection.get_paths()[0].vertices[:, axis_index] 
             for collection in collections]
             )
    else:
        raise ValueError(f"Invalid data_type: {data_type}. Choose either 'line' or 'fill'.")

    # Exclude NaN values
    data = data[~np.isnan(data)]
    
    # Check if data is empty after excluding NaN values
    if len(data) == 0:
        # Return None for both min and max if no valid data is available
        return None, None  

    # Get the min and max
    data_min = np.min(data)
    data_max = np.max(data)

    return data_min, data_max
#endregion

#region: group_model_keys
def group_model_keys(results_analyzer):
    '''
    Group model keys based on the target effect.

    Parameters
    ----------
    results_analyzer : ResultsAnalyzer
        An instance of ResultsAnalyzer used to read and group model keys.

    Returns
    -------
    tuple
        model_key_names, grouped_keys
    '''
    model_key_names = results_analyzer.read_model_key_names()
    grouped_keys = results_analyzer.group_model_keys('target_effect')
    return model_key_names, grouped_keys
#endregion