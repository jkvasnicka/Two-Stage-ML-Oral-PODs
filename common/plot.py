'''Various functions for plotting.
'''

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import numpy as np 

#region: default_figsize
def default_figsize(scale_width=1., scale_height=1.):
    '''Return the default figsize from matplotlib with optional scaling.
    ''' 
    figsize = np.array(rcParams['figure.figsize'])
    figsize[0] *= scale_width
    figsize[-1] *= scale_height
    return figsize
#endregion

#region: histograms_for_columns
def histograms_for_columns(data, figsize=None, fontsize=8):
    '''Helper function to visualize the distributions of each column in a
    pandas.DataFrame object.
    '''
    axs = data.hist(
        xlabelsize=fontsize, 
        ylabelsize=fontsize, 
        figsize=figsize)

    for ax, k in zip(axs.flatten(), data.columns):
        ax.set_title(k, fontsize=fontsize)

    fig = plt.gcf()
    fig.tight_layout()
    
    return fig, axs
#endregion

#region: vertical_boxplots
def vertical_boxplots(
            data_for_key, x, y, xlabel, sharex=False, xlim=None, title_for_key=None, 
            figsize=None, write_path=None, **kwargs):
    '''Wrapper around seaborn.boxplot().

    Parameters
    ----------
    data_for_key : dict of pandas.DataFrame
        Datasets, in long form, to plot.
    x, y, huenames of variables in data or vector data, optional
        Inputs for plotting long-form data.
    xlabel : str
        Used for Axes.set_xlabel().
    sharex : bool
        Controls sharing of properties among x axes via matplotlib.subplots().
    xlim : 2-tuple (optional)
        Used for Axes.set_xlim().
    title_for_key : dict (optional)
        Mapping DataFrame keys to Axes titles.
    figsize : 2-tuple (optional)
    write_path : str (optional)
        If specified, the figure will be saved as Figure.savefig(write_path).
    kwargs : key-value mapping
        Other keyword arguments are passed to seaborn.boxplot().

    Returns
    -------
    matplotlib (Figure, Axes)
    '''
    fig, axs = plt.subplots(
        ncols=len(data_for_key),
        sharey=True,
        sharex=sharex,
        figsize=figsize
    )
        
    for i, (k, data) in enumerate(data_for_key.items()): 

        sns.boxplot(    
            x=x, 
            y=y, 
            data=data,
            dodge=False,
            ax=axs[i],
            **kwargs
        )

        axs[i].grid(axis='x', linestyle='--', linewidth=0.5)
        axs[i].grid(axis='y', linestyle='--', linewidth=0.5)
        # Set grid lines, etc., below all artists.
        axs[i].set_axisbelow(True)

        if xlim is not None:
            axs[i].set_xlim(xlim)
        axs[i].set_ylabel('')
        axs[i].set_xlabel(xlabel)
        title = k if title_for_key is None else title_for_key[k]
        axs[i].set_title(title)
        
    fig.tight_layout()

    if write_path is not None:
        fig.savefig(write_path)

    return fig, axs
#endregion

#region: model_scores_by_fold
def model_scores_by_fold(scores, ylabel, xlim=None):
    '''Helper function to plot the performance scores of models by 
    cross-validation fold (x-axis).
    
    Parameters
    ----------
    scores : pandas.DataFrame
        Index = fold, columns = model names.
    
    References
    ----------
    scikit-learn.org: Statistical comparison of models using grid search.
    '''
    greatest_to_least = (
        scores.mean()
        .sort_values(ascending=False)
        .index
    )
    scores = scores[greatest_to_least]

    fig, ax = plt.subplots()
    
    sns.lineplot(
        data=scores,
        dashes=False, 
        palette='Set1', 
        marker='o', 
        alpha=0.5, 
        ax=ax
    )
    ax.set_title('Comparison of Estimator Performance')
    ax.set_xlabel('Fold', size=12, labelpad=10)
    _ = ax.set_ylabel(ylabel, size=12)
    _ = ax.legend(loc='center left', bbox_to_anchor=(1, 0.85))
    
    if xlim is not None:
        ax.set_xlim(xlim)
    
    return fig, ax
#endregion 

#region: bivariate_scatterplot
def bivariate_scatterplot(
        x, y, add_oneone=True, score_text=None, title=None, xlabel=None, 
        ylabel=None, loglog=False, ax=None, **scatter_kwargs):
    '''Generate a scatterplot comparing, for example, observed vs. predicted
    data with the one-one line for comparison. 
    
    Parameters
    ----------
    score_text : str or dict of str
    '''
    if ax is None:
        ax = plt.axes()

    if loglog:
        ax.loglog()

    ax.scatter(x, y, **scatter_kwargs)

    if add_oneone:
        add_one_one_line(ax)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if score_text is not None:
        if isinstance(score_text, dict):
            score_text = dictionary_to_string(score_text)

        ## Add a text box to the right of the plot.
        bbox = dict(
            boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5
        )
        text_position = (1.05, 0.95)  # Sets the box to the right
        ax.text(
            *text_position, score_text, fontsize=10, bbox=bbox, 
            transform=ax.transAxes, horizontalalignment='left', 
            verticalalignment='top'
        )

    return ax
#endregion

# TODO: Could be more flexible with plot parameters.
#region: add_one_one_line
def add_one_one_line(ax):
    '''Helper function which plots y=x.
    '''
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]

    ax.plot(
        lims,
        lims,
        color='#808080',
        linestyle='--',
        linewidth=0.75,
        zorder=0
    )
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
#endregion

#region: dictionary_to_string
def dictionary_to_string(d):
    '''Convert a dictionary to string for plotting. 
    
    Input: {k1 : v1, k2 : v2, ...} 
    Output: 'k1 : v1 \n k2 : v2 \n ...' 
        where \n denotes a new line.
    '''
    return str(d).strip('{|}').replace("'", '').replace(',', ' \n')
#endregion