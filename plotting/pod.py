'''
Plotting module for Point of Departure (POD) data.

This module contains functions for plotting cumulative distribution functions
(CDFs) of POD data across various models and datasets.

Notes
-----
This module is part of a larger plotting sub-package focused on the 
visualization of results from the main package. It relies on external classes 
such as ResultsAnalyzer and PlotSetting for data processing and configuration.
'''

import matplotlib.pyplot as plt 
import matplotlib.lines as mlines
import seaborn as sns
import itertools

from . import utilities

#region: cumulative_pod_distributions
def cumulative_pod_distributions(results_analyzer, plot_settings):
    '''
    Generate a subplot with multiple Axes, each representing the cumulative
    distribution functions (CDFs) for different models and samples.

    The first row shows the CDFs for the intersection of samples across 
    models. The second row shows the CDFs as in the original data. Each
    column corresponds to an effect category.

    Parameters
    ----------
    results_analyzer : ResultsAnalyzer
        An instance of the ResultsAnalyzer class, which provides methods for
        data analysis and manipulation.
    plot_settings : PlotSettings
        An instance of the PlotSettings class, which contains various settings
        and configurations for the plot.

    Returns
    -------
    None
        The function creates and saves the subplots based on the provided
        results_analyzer and plot_settings.
    '''
    colors, linestyles = get_plot_styles()

    grouped_keys = results_analyzer.group_model_keys('target_effect')

    for grouping_key, model_keys in grouped_keys:

        fig, axs = plt.subplots(
            2, 
            len(model_keys), 
            figsize=(len(model_keys)*4, 8)
            )
        
        global_xlim = utilities.initialize_global_limits()

        for i, model_key in enumerate(model_keys):

            y_for_label = results_analyzer.get_pod_comparison_data(model_key)
            
            # Plot CDFs for intersection of samples in the first row
            plot_intersection_cdfs(
                axs[0, i], 
                y_for_label, 
                results_analyzer, 
                colors, 
                linestyles, 
                global_xlim
            )

            # Plot CDF for each distinct dataset in the second row
            plot_distinct_cdfs(
                axs[1, i], 
                y_for_label, 
                results_analyzer, 
                plot_settings,
                colors, 
                linestyles, 
                global_xlim
            )

            set_row_axs_properties(
                axs[:, i],
                global_xlim,
                i, 
                model_key, 
                results_analyzer, 
                plot_settings.label_for_effect
            )

        utilities.set_centralized_legend(
            fig, 
            axs[-1][-1], 
            bottom=0.1,
            bbox_to_anchor=(0.5, -0.01)
        )

        utilities.save_figure(
            fig, 
            cumulative_pod_distributions, 
            grouping_key
        )
#endregion

#region: single_model_cdfs
def single_model_cdfs(
        y_for_label, 
        results_analyzer, 
        plot_settings,
        title=None
        ):
    '''
    Plot the cumulative distribution functions (CDFs) for a single model.

    Parameters
    ----------
    y_for_label : dict
        A dictionary containing different data series for which the CDFs need
        to be plotted.
    results_analyzer : ResultsAnalyzer
        An instance of the ResultsAnalyzer class, which provides methods for
        data analysis and manipulation.
    title : str, optional
        The title to be set for the plot. If None, no title is set.

    Returns
    -------
    2-tuple:
        matplotlib.pyplot Figure and Axes
    '''
    fig, ax = plt.subplots()

    colors, linestyles = get_plot_styles()

    plot_distinct_cdfs(
        ax, 
        y_for_label, 
        results_analyzer, 
        plot_settings,
        colors, 
        linestyles
    )

    set_ax_properties(
        ax, 
        title=title,
        ylabel='Proportion of Chemicals'
    )

    utilities.set_centralized_legend(
        fig, 
        ax, 
        bottom=0.17,
        bbox_to_anchor=(0.5, -0.01)
        )

    return fig, ax
#endregion

#region: plot_intersection_cdfs
def plot_intersection_cdfs(
        ax, 
        y_for_label, 
        results_analyzer, 
        colors, 
        linestyles, 
        global_xlim=None
        ):
    '''
    Plot the cumulative distribution functions (CDFs) for the intersection of 
    samples across models.

    Parameters
    ----------
    ax : Axes
        The matplotlib axes object where the CDFs are to be plotted.
    y_for_label : dict
        A dictionary containing different data series for which the CDFs need
        to be plotted.
    results_analyzer : ResultsAnalyzer
        An instance of the ResultsAnalyzer class, which provides methods for
        data analysis and manipulation.
    colors : list
        A list of colors to be used for plotting the CDFs.
    linestyles : list
        A list of linestyles to be used for plotting the CDFs.
    global_xlim : list, optional
        A list containing the global x-axis limits. If None, limits are 
        determined automatically.

    Returns
    -------
    None
    '''
    line_cycle = itertools.cycle(linestyles)

    common_samples = list(
        set.intersection(*[set(y.index) for y in y_for_label.values()])
        )

    for j, (label, data_series) in enumerate(y_for_label.items()):
        plot_cdf(
            ax,
            data_series.loc[common_samples], 
            results_analyzer, 
            colors[j], 
            next(line_cycle), 
            label, 
            global_xlim
        )
#endregion

#region: plot_distinct_cdfs
def plot_distinct_cdfs(
        ax, 
        y_for_label, 
        results_analyzer, 
        plot_settings,
        colors, 
        linestyles, 
        global_xlim=None
        ):
    '''
    Plot the cumulative distribution functions (CDFs) for each data series 
    provided.

    Parameters
    ----------
    ax : Axes
        The matplotlib axes object where the CDFs are to be plotted.
    y_for_label : dict
        A dictionary containing different data series for which the CDFs need
        to be plotted.
    results_analyzer : ResultsAnalyzer
        An instance of the ResultsAnalyzer class, which provides methods for
        data analysis and manipulation.
    colors : list
        A list of colors to be used for plotting the CDFs.
    linestyles : list
        A list of linestyles to be used for plotting the CDFs.
    global_xlim : list, optional
        A list containing the global x-axis limits. If None, limits are 
        determined automatically.

    Returns
    -------
    None
    '''
    line_cycle = itertools.cycle(linestyles)

    datasets = [
        plot_settings.authoritative_label, 
        plot_settings.surrogate_label, 
        plot_settings.qsar_label
        ]

    for i, label in enumerate(datasets):
        distinct_series = y_for_label[label]
        
        # Drop samples that are in previously plotted datasets
        for previous_label in datasets[:i]:
            distinct_series = distinct_series.drop(
                y_for_label[previous_label].index,
                errors='ignore'
            )

        plot_cdf(
            ax, 
            distinct_series, 
            results_analyzer,
            colors[i], 
            next(line_cycle), 
            label, 
            global_xlim
        )
#endregion

#region: plot_cdf
def plot_cdf(
        ax, 
        data_series, 
        results_analyzer, 
        color, 
        linestyle, 
        label, 
        global_xlim=None
        ):
    '''
    Plot the cumulative distribution function (CDF) for a given data series on 
    the specified Axes.

    Parameters
    ----------
    ax : Axes
        The matplotlib axes object where the CDF is to be plotted.
    data_series : Series
        A pandas Series containing the data for which the CDF needs to be 
        plotted.
    results_analyzer : ResultsAnalyzer
        An instance of the ResultsAnalyzer class, which provides methods for
        data analysis and manipulation, including CDF generation.
    color : str
        The color to be used for plotting the CDF.
    linestyle : str
        The linestyle to be used for plotting the CDF.
    label : str
        The label to be used for the plotted CDF in the legend.
    global_xlim : list, optional
        A list containing the global x-axis limits. If provided, the function 
        updates these limits based on the data series. If None, limits are not 
        updated.

    Returns
    -------
    None
    '''
    sorted_values, cumulative_proportions = (
        results_analyzer.generate_cdf_data(data_series, normalize=True))
    
    ax.plot(
        sorted_values, 
        cumulative_proportions,
        color=color,
        linestyle=linestyle,
        label=label
    )
    
    if global_xlim:
        utilities.update_global_limits(global_xlim, ax.get_xlim())
#endregion

#region: set_row_axs_properties
def set_row_axs_properties(
        row_axs, 
        global_xlim, 
        i, 
        model_key, 
        results_analyzer, 
        label_for_effect
        ):
    '''
    Set properties for each Axes object in a row based on their position and 
    model key.

    Parameters
    ----------
    row_axs : array_like
        An array of Axes objects for a row in a subplot.
    global_xlim : list
        A list containing the global x-axis limits to be set for all Axes.
    i : int
        The column index of the current Axes in the subplot.
    model_key : tuple
        A tuple representing the key for the current model.
    results_analyzer : ResultsAnalyzer
        An instance of the ResultsAnalyzer class, which provides methods for
        data analysis and manipulation.
    label_for_effect : dict
        A dictionary mapping effect categories to their corresponding labels.

    Returns
    -------
    None
    '''
    for ax_index, ax in enumerate(row_axs):

        is_first_row = ax_index == 0
        if is_first_row:
            # Set the title as the effect category
            effect_index = (
                results_analyzer.read_model_key_names()
                .index('target_effect')
            )
            effect = model_key[effect_index]
            title = label_for_effect[effect]
        else:
            title = None

        is_first_column = i == 0
        if is_first_column:
            if is_first_row:
                ylabel = 'Proportion of Chemicals (Intersection)'
            else:
                ylabel = 'Proportion of Chemicals'
        else:
            ylabel = None
            
        set_ax_properties(
            ax, 
            global_xlim=global_xlim, 
            title=title,
            ylabel=ylabel
        )
#endregion

#region: set_ax_properties
def set_ax_properties(
        ax, 
        global_xlim=None, 
        title=None,
        ylabel=None
        ):
    '''
    Set various properties for a given Axes object.

    Parameters
    ----------
    ax : Axes
        The matplotlib Axes object whose properties are to be set.
    global_xlim : list, optional
        A list containing the global x-axis limits to be set for the Axes. If 
        None, x-limits are not set.
    title : str, optional
        The title to be set for the Axes. If None, no title is set.
    ylabel : str, optional
        The y-label to be set for the Axes. If None, no y-label is set.

    Returns
    -------
    None
    '''
    ax.set_xlabel("$log_{10}POD$")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    if global_xlim:
        ax.set_xlim(global_xlim)
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
#endregion

#region: get_plot_styles
def get_plot_styles():
    '''
    Retrieve and return plot styles including colors and linestyles.

    Returns
    -------
    colors : list
        A list of color codes representing the color palette used for plotting.
    linestyles : list
        A list of linestyles used for plotting lines in the plot.

    Notes
    -----
    This function utilizes seaborn's 'colorblind' color palette. For 
    linestyles, it extracts all string-type linestyles from matplotlib's lineStyles.
    '''
    colors = sns.color_palette('colorblind')
    linestyles = [
        style for style in mlines.lineStyles.keys() 
        if isinstance(style, str)
        ]
    return colors, linestyles
#endregion