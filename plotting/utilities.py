'''
This module contains various utility functions for general plotting tasks.
'''

import seaborn as sns
import os
from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np

#region: save_figure
def save_figure(fig, function, fig_label, extension='.png', bbox_inches=None):
    '''
    Save a matplotlib figure to a specified directory with a given label and 
    file extension.

    This function constructs the file path for the figure using the `function` 
    and `fig_label` arguments, then saves the figure to this path. The 
    directory in which the figure is saved is determined by the `function`'s 
    name.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to be saved.
    function : function
        The function based on which the directory for saving the figure is 
        determined.
    fig_label : str or tuple or list
        The label to be used for naming the saved figure file. If it is a 
        tuple or list, elements will be joined with a dash.
    extension : str, optional
        The file extension for the saved figure (default is '.png').
    bbox_inches : str, optional
        The parameter for `matplotlib.figure.Figure.savefig` method to set 
        bounding box in inches (default is None).

    Returns
    -------
    None

    Notes
    -----
    The function prints the path to the saved figure as feedback to the user.
    '''
    output_dir = function_directory(function)
    fig_path = figure_path(output_dir, fig_label, extension=extension)
    print(f'Saving figure --> "{fig_path}"')
    fig.savefig(fig_path, bbox_inches=bbox_inches)
#endregion

#region: figure_path
def figure_path(function_dir, fig_label, extension='.png'):
    '''
    Generate a file path for saving a figure.

    This function creates a filename from the `fig_label`, replacing spaces 
    and slashes with dashes, and appends the specified file `extension`. It 
    then constructs a full file path by joining this filename with the 
    `function_dir`.

    Parameters
    ----------
    function_dir : str
        The directory where the figure file will be saved.
    fig_label : str or tuple or list
        The label for the figure file. If it is a tuple or list, elements will 
        be joined with a dash. Spaces and slashes in the label are replaced 
        with dashes.
    extension : str, optional
        The file extension for the figure (default is '.png').

    Returns
    -------
    str
        The full file path for the figure.

    Notes
    -----
    This function ensures that the file name is safe for the file system by 
    avoiding certain characters like slashes that could be interpreted as 
    directory separators.
    '''
    if isinstance(fig_label, (tuple, list)):
        fig_label = '-'.join(map(str, fig_label))
    filename = fig_label.replace(' ', '-').replace('/', '-') + extension
    return os.path.join(function_dir, filename)
#endregion

#region: function_directory
def function_directory(function):
    '''
    Get the name of the function and create a directory in the output 
    directory for it
    '''
    output_dir = os.path.join('Figures', function.__name__)
    ensure_directory_exists(output_dir)
    return output_dir
#endregion

#region: ensure_directory_exists
def ensure_directory_exists(path):
    '''
    Check if the directory at `path` exists and if not, create it.
    '''
    if not os.path.exists(path):
        os.makedirs(path)
#endregion

#region: generate_scatterplot
def generate_scatterplot(
        ax, 
        y_true, 
        y_pred, 
        function_for_metric, 
        label_for_metric,
        with_sample_size=True,
        color='black', 
        size=None,
        alpha=0.7,
        highlight_size=None,
        main_size=None,
        highlight_color='#004488',  # blue
        highlight_indices=None, 
        main_label='Main',
        highlight_label='Highlight', 
        title='', 
        xlabel='', 
        ylabel=''
    ):
    '''
    Generate a scatterplot comparing two sets of data.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to plot the scatterplot on.
    y_true : pd.Series
        The comparison data (x).
    y_pred : pd.Series
        The evaluation data (y).
    workflow : object
        The workflow object containing additional information.
    color : str
        The color for the scatterplot points.
    highlight_color : str
        The color for the highlighted scatterplot points.
    highlight_indices : array-like
        Indices for which to use the highlight color.
    title : str
        The title of the scatterplot.
    xlabel : str
        The label for the x-axis.
    ylabel : str
        The label for the y-axis.
    '''

    chem_intersection = y_true.index.intersection(y_pred.index)
    y_true = y_true.loc[chem_intersection]
    y_pred = y_pred.loc[chem_intersection]

    if highlight_indices is not None:
        # Divide the data into two sets and plot separately.
        y_true_highlight = y_true.loc[highlight_indices]
        y_pred_highlight = y_pred.loc[highlight_indices]
        y_true_rest = y_true.drop(highlight_indices)
        y_pred_rest = y_pred.drop(highlight_indices)
        
        ax.scatter(
            y_true_rest, 
            y_pred_rest, 
            alpha=alpha, 
            s=main_size,
            color=color,
            label=main_label
            )
        ax.scatter(
            y_true_highlight, 
            y_pred_highlight, 
            alpha=alpha, 
            s=highlight_size,
            color=highlight_color,
            label=highlight_label,
            )
        
        ax.legend(
            loc='lower right', 
            fontsize='x-small', 
            markerscale=0.8
            )

    else:

        ax.scatter(
            y_true, 
            y_pred, 
            alpha=alpha, 
            color=color,
            s=size
            )

    ## Set the performance scores as text.

    float_to_string = lambda score : format(score, '.2f')  # limit precision
    dict_to_string = lambda d : '\n'.join([f'{k}: {v}' for k, v in d.items()])
    get_score = (
        lambda metric : function_for_metric[metric](y_true, y_pred))
    
    score_dict = {
        label : float_to_string(get_score(metric)) 
        for metric, label in label_for_metric.items()
        }
    if with_sample_size:
        score_dict['n'] = comma_separated(len(y_true))

    ax.text(
        0.05, 
        0.95, 
        dict_to_string(score_dict), 
        transform=ax.transAxes,
        va='top', 
        ha='left', 
        size='small'
        )

    fontsize = 'medium'
    if title:
        ax.set_title(title, size=fontsize)
    if xlabel:
        ax.set_xlabel(xlabel, size=fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, size=fontsize)
#endregion

#region: plot_one_one_line
def plot_one_one_line(ax, xmin, xmax, color='#BB5566'):  # red
    '''
    Add one-one line to a subplot

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object to add the line.
    xmin : float
        Minimum limit for x and y axes.
    xmax : float
        Maximum limit for x and y axes.
    '''

    # Set consistent limits.
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)

    # Set consistent tick locations
    locator = MaxNLocator(nbins=5)
    ax.xaxis.set_major_locator(locator)
    ax.yaxis.set_major_locator(locator)
        
    ax.plot([xmin, xmax], [xmin, xmax], color=color, 
            linestyle='--', linewidth=1)
#endregion

#region: compute_global_x_limits
def compute_global_x_limits(data_containers):
    '''
    Compute global x limits based on IQR for a list of pandas Series or DataFrames.
    
    Parameters:
    - data_containers: List containing pandas Series or DataFrames with the data.
    
    Returns:
    - xlim: Tuple containing the x limits.
    '''
    whisker_data = []

    # Loop through data_containers to determine whisker data
    for data in data_containers:
        if isinstance(data, pd.Series):
            whisker_data.append(calculate_whisker_data(data))
        elif isinstance(data, pd.DataFrame):
            for column in data.columns:
                whisker_data.append(calculate_whisker_data(data[column]))

    global_min = min(data[0] for data in whisker_data)
    global_max = max(data[1] for data in whisker_data)

    buffer = (global_max - global_min) * 0.05
    xlim = (global_min - buffer, global_max + buffer)

    return xlim
#endregion

#region: calculate_whisker_data
def calculate_whisker_data(predictions):
    '''
    Calculate whisker data based on the interquartile range (IQR).
    
    Parameters:
    - predictions: Series or array-like containing the predictions.
    
    Returns:
    - (whisker_min, whisker_max): Tuple containing the whisker endpoints.
    '''
    Q1, Q3 = np.percentile(predictions, [25, 75])
    IQR = Q3 - Q1
    whisker_min = Q1 - 1.5 * IQR
    whisker_max = Q3 + 1.5 * IQR
    return whisker_min, whisker_max
#endregion

#region: vertical_boxplots_subplot
def vertical_boxplots_subplot(
        axs,
        df_wide,
        sorting_level,
        evaluation_label_mapper,
        evaluation_level,
        do_sort=True,  # New parameter to control sorting
        ascending=False,
        xlim=(),
        ylabel=None, 
        start=0,
        palette='icefire',
        ):
    '''
    Plot boxplots for feature importances for all models in workflow.

    Parameters
    ----------
    axs : matplotlib axes object
        Axes on which to plot the boxplots.
    df_wide : DataFrame
        Wide-form DataFrame containing the data to plot.
    sorting_level : str
        The level in the DataFrame to sort by if sorting is enabled.
    evaluation_label_mapper : dict
        Dictionary mapping evaluation metrics to their labels.
    evaluation_level : str
        The level in the DataFrame corresponding to the evaluation metrics.
    do_sort : bool, optional
        Whether to sort the boxes according to the median of the first metric.
        Default is True.
    ascending : bool, optional
        Whether to sort the values in ascending order. Default is False.
    xlim : tuple, optional
        Limits for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    start : int, optional
        Starting index for plotting axes. Default is 0.
    palette : str, optional
        Color palette for the boxplots.

    Returns
    -------
    None
    '''
    if do_sort:
        # Compute the sorting order based on the median of the first metric
        first_metric_key = list(evaluation_label_mapper.keys())[0]
        medians_first_metric = (
            df_wide.xs(first_metric_key, axis=1, level=evaluation_level)
            .median(axis=0)
            .sort_values(ascending=ascending)
        )
        sorted_keys_first_metric = list(medians_first_metric.index)
    else:
        # Use the original order of keys if sorting is not required
        sorted_keys_first_metric = df_wide.columns.get_level_values(sorting_level).unique()

    reverse_metric = evaluation_label_mapper.get('r2_score', 'r2_score')
    
    for i, (k, metric) in enumerate(evaluation_label_mapper.items(), start=start):
        
        df_long = df_wide.xs(k, axis=1, level=evaluation_level).melt()
        
        if do_sort:
            # Use the sorting order from the first metric
            df_long[sorting_level] = pd.Categorical(
                df_long[sorting_level], 
                categories=sorted_keys_first_metric, 
                ordered=True
                )
            df_long = df_long.sort_values(by=sorting_level)
        
        _sns_boxplot_wrapper(
            axs[i], 
            df_long, 
            xlabel=metric, 
            palette=palette
            )

        if xlim:
            set_axis_limit(
                axs[i], 
                metric,
                limit_values=xlim,
                reverse_metric=reverse_metric
        )
        
        # Remove yticklabels for all axes except the first one
        axs[i].tick_params(axis='y', size=10)
        if i == 0:
            axs[i].set_ylabel(ylabel, size=12)
        else:
            axs[i].set_yticklabels([])  

    if ylabel:
        # Set ylabel, first column only.
        for i, ax in enumerate(axs.flatten()):
            ax.tick_params(axis='y', size=10)
            if i == 0:
                ax.set_ylabel(ylabel, size=12)
#endregion

#region: _sns_boxplot_wrapper
def _sns_boxplot_wrapper(
        ax, 
        df_long, 
        xlabel='',
        ylabel='', 
        title='',
        **kwargs
        ):
    '''Wrapper around seaborn.boxplot() with customization.
    '''
    y, x = list(df_long.columns)
    
    sns.boxplot(    
        x=x, 
        y=y, 
        data=df_long,
        linewidth=0.8,
        dodge=False,
        ax=ax,
        showfliers=False, 
        **kwargs
    )

    ## Set gridlines below all artists.
    ax.grid(axis='x', linestyle='--', linewidth=0.5)
    ax.grid(axis='y', linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
#endregion

#region: set_axis_limit
def set_axis_limit(
        ax, 
        metric, 
        limit_values=(0., 1.), 
        reverse_metric='r2_score', 
        orientation='x',
        num_ticks=3
        ):
    '''
    Set the axis limits of a given axis. If the metric matches the reverse_metric,
    the axis limits are reversed.
    '''
    if metric == reverse_metric:
        limits = limit_values[::-1] 
    else:
        limits = limit_values

    if orientation == 'x':
        ax.set_xlim(limits)
        ax.set_xticks(np.linspace(limits[0], limits[1], num_ticks))
    elif orientation == 'y':
        ax.set_ylim(limits)
        ax.set_yticks(np.linspace(limits[0], limits[1], num_ticks))
#endregion

#region: set_centralized_legend
def set_centralized_legend(fig, ax, bottom=None, bbox_to_anchor=None):
    '''
    Set a centralized legend for the entire subplot.

    This function also adjusts the figure layout to ensure that the legend 
    does not overlap with the plot.

    Parameters
    ----------
    fig : Figure
        The matplotlib figure object for which the legend is to be set.
    ax : Axes
        The matplotlib axes object from which the legend handles and labels 
        are to be extracted.
    bottom : float, optional
        The bottom padding of the subplot layout to accommodate the legend. If 
        None, the default padding is used.
    bbox_to_anchor : tuple, optional
        The bbox_to_anchor argument for the legend. If None, the legend is placed
        at the default position.

    Returns
    -------
    None
    '''
    # Adjust layout to accomodate the legend
    fig.tight_layout()
    if bottom:
        fig.subplots_adjust(bottom=bottom)
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, 
        labels, 
        loc='lower center', 
        fontsize='small',
        ncol=len(labels), 
        bbox_to_anchor=bbox_to_anchor
    )
#endregion

#region: comma_separated
def comma_separated(number):
    '''Convert float or int to a string with comma-separated thousands.
    '''
    return '{:,}'.format(number)
#endregion

#region: initialize_global_limits
def initialize_global_limits():
    '''
    Initialize the global limits for data.

    This function initializes the global limits as a list containing positive
    and negative infinity, which are placeholders to be updated with actual
    data limits.

    Returns
    -------
    list
        A list of two elements: [float('inf'), float('-inf')]
    '''
    return [float('inf'), float('-inf')]
#endregion

#region: update_global_limits
def update_global_limits(global_lim, new_lim):
    '''
    Update the global limits with new limits.

    This function updates the global limits for the data with new limits
    provided. The global limits are expected to be a list where the first
    element is the minimum (lower bound) and the second element is the maximum
    (upper bound).

    Parameters
    ----------
    global_lim : list
        The current global limits as a list [min, max].
    new_lim : list
        The new limits to update the global limits with, as a list [new_min, new_max].

    Returns
    -------
    None

    Example
    -------
    >>> ax.plot(x, y)
    >>> update_global_limits(global_xlim, ax.get_xlim())
    '''
    global_lim[0] = new_lim[0]
    global_lim[1] = new_lim[1]
#endregion