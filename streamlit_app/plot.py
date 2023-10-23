'''

Notes
-----
Most of this code was copied over from the plot.py module of the main project 
repository. This was a temporary solution while the repository was not 
properly packaged.
'''

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 

#region: margins_of_exposure_cumulative
def margins_of_exposure_cumulative(
        results_for_percentile, 
        label_for_exposure_column,
        effect_label,
        right_truncation=None
        ):
    '''
    Plots margins of exposure for different chemicals. 

    Parameters
    ----------
    right_truncation : float, optional
        If provided, sets the right truncation limit for x-axis.

    Returns
    -------
    None
    '''
    # Define the limits of the vertical spans in log10 units of MOE.
    # log10(0) is undefined and will be handled dynamically
    moe_categories = {
        'Potential Concern': (0., 2.),  # 1, 100
        'Definite Concern' : (-np.inf, 0.)  # 0, 1
    }
    moe_colors = sns.color_palette('Paired', len(moe_categories)+1)

    fig, ax = plt.subplots(figsize=(5, 5))

    percentile_colors = sns.color_palette('Set2', len(results_for_percentile))

    for j, (percentile, results) in enumerate(results_for_percentile.items()):

        plot_with_prediction_interval(
            ax, 
            results['moe'], 
            results['cum_count'], 
            results['lb'], 
            results['ub'], 
            percentile_colors[j], 
            label=label_for_exposure_column[percentile]
            )

    ## Update the limits.
    set_even_ticks(ax, axis_type='x', data_type='fill')
    set_even_log_ticks(ax, axis_type='y', data_type='fill')
    if right_truncation:
        ax.set_xlim(ax.get_xlim()[0], right_truncation)

    ## Set labels, etc. 
    ax.set_title(effect_label)
    ax.set_xlabel("$log_{10}MOE$")
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_ylabel('Cumulative Count of Chemicals')

    annotate_vertical_spans(
        ax, 
        moe_categories, 
        moe_colors
        )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)  

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, 
        labels, 
        loc='lower center', 
        fontsize='small',
        ncol=len(labels), 
        bbox_to_anchor=(0.5, -0.01)
    )

    return fig, ax
#endregion

#region: plot_with_prediction_interval
def plot_with_prediction_interval(
        ax, 
        sorted_values, 
        cumulative_counts, 
        lower_bound, 
        upper_bound, 
        color, 
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

#region: annotate_vertical_spans
def annotate_vertical_spans(ax, categories, colors, y_pos_axes=0.97):
    '''
    Annotate vertical spans and category labels in a matplotlib Axes object.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object in which to annotate vertical spans.
    categories : dict
        Dictionary defining the categories and their corresponding 
        vertical span limits.
    colors : list
        List of colors for the vertical spans.
    y_pos_axes : float
        The y-position of the category labels in Axes fraction units.

    Returns
    -------
    None
    '''
    for k, (category, (lower, upper)) in enumerate(categories.items()):
        if lower == -np.inf:
            # Extend the lower limit to the xmin of the Axes
            lower = ax.get_xlim()[0]

        ax.axvspan(lower, upper, alpha=0.2, color=colors[k])

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
        data = np.concatenate([collection.get_paths()[0].vertices[:, axis_index] for collection in collections])
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