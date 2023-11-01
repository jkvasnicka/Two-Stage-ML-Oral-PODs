'''
'''

import matplotlib.pyplot as plt 
import matplotlib.lines as mlines
import seaborn as sns
import itertools

from . import utilities

#region: cumulative_pod_distributions
def cumulative_pod_distributions(results_analyzer, plot_settings):
    '''
    '''
    colors, linestyles = get_plot_styles()

    grouped_keys = results_analyzer.group_model_keys('target_effect')

    for grouping_key, model_keys in grouped_keys:

        fig, axs = plt.subplots(
            2, 
            len(model_keys), 
            figsize=(len(model_keys)*4, 8)
            )
        
        global_xlim = [float('inf'), float('-inf')]  # initialize

        for i, model_key in enumerate(model_keys):

            y_for_label = results_analyzer.get_pod_comparison_data(model_key)

            # Compute intersection of samples
            common_samples = get_common_samples(y_for_label)
            
            # Plot CDFs for intersection of samples in the first row
            plot_intersection_cdfs(
                axs[0, i], 
                y_for_label, 
                results_analyzer, 
                colors, 
                linestyles, 
                common_samples, 
                global_xlim
            )

            # Plot CDFs as in original in the second row
            plot_original_cdfs(
                axs[1, i], 
                y_for_label, 
                results_analyzer, 
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

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.1)

        set_legend(fig, axs[-1][-1])

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
        title=None
        ):
    '''
    '''
    fig, ax = plt.subplots()

    colors, linestyles = get_plot_styles()

    plot_original_cdfs(
        ax, 
        y_for_label, 
        results_analyzer, 
        colors, 
        linestyles
    )

    set_ax_properties(
        ax, 
        title=title,
        ylabel='Proportion of Chemicals'
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
        common_samples, 
        global_xlim=None
        ):
    '''
    '''
    line_cycle = itertools.cycle(linestyles)

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

#region: plot_original_cdfs
def plot_original_cdfs(
        ax, 
        y_for_label, 
        results_analyzer, 
        colors, 
        linestyles, 
        global_xlim=None
        ):
    '''
    '''
    line_cycle = itertools.cycle(linestyles)

    for j, (label, data_series) in enumerate(y_for_label.items()):
        plot_cdf(
            ax, 
            data_series, 
            results_analyzer,
            colors[j], 
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
    Helper function to plot CDF and update global x limits.
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
        # Update the global x-limits
        global_xlim[0] = min(global_xlim[0], sorted_values.min())
        global_xlim[1] = max(global_xlim[1], sorted_values.max())
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
    '''
    colors = sns.color_palette('colorblind')
    linestyles = [
        style for style in mlines.lineStyles.keys() 
        if isinstance(style, str)
        ]
    return colors, linestyles
#endregion

#region: get_common_samples
def get_common_samples(y_for_label):
    '''
    '''
    common_samples = (
        y_for_label['Regulatory'].index
        .intersection(y_for_label['ToxValDB'].index)
        .intersection(y_for_label['QSAR'].index)
    )
    return common_samples
#endregion

#region: set_legend
def set_legend(fig, ax):
    '''
    '''
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, 
        labels, 
        loc='lower center', 
        fontsize='small',
        ncol=len(labels), 
        bbox_to_anchor=(0.5, -0.01)
    )
#endregion