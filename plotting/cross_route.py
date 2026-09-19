'''Plot cross-route QSAR comparisons and Aurisano PODs.'''

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np

from config_management import UnifiedConfiguration
from cross_route import (
    ENDPOINTS,
    load_cross_route_comparison,
    summarize_cross_route,
)
from . import utilities


def cross_route_pod_comparison(
        oral_config_file,
        inhalation_config_file,
        plot_settings,
        output_dir=None,
        ):
    '''Generate the manuscript cross-route figure from persisted inputs.'''
    oral_config = UnifiedConfiguration(oral_config_file)
    inhalation_config = UnifiedConfiguration(inhalation_config_file)
    tables, source_tables, _ = load_cross_route_comparison(
        oral_config, inhalation_config,
    )
    summary = summarize_cross_route(tables, source_tables)
    figure = cross_route_comparison(
        tables, source_tables, summary,
        label_for_effect=plot_settings.label_for_effect,
        color_for_effect=plot_settings.color_for_effect,
    )
    utilities.save_figure(
        figure,
        cross_route_pod_comparison,
        'oral-inhalation-by-effect',
        bbox_inches='tight',
        output_dir=output_dir,
    )
    plt.close(figure)


def cross_route_comparison(
        tables, source_tables, summary, label_for_effect, color_for_effect,
        bin_width=0.25):
    '''Plot density and ratio panels with Spearman annotations.

    Parameters
    ----------
    tables : dict of pandas.DataFrame
        Matched QSAR calculations from ``build_cross_route_comparison``.
    source_tables : dict of pandas.DataFrame
        Independently paired original Table S5/S6 POD calculations.
    summary : pandas.DataFrame
        ``summarize_cross_route`` output. CI bounds may be populated later
        after an inferential method is selected. Finite supplied bounds
        appear in the A/B annotations; missing bounds are omitted.
    label_for_effect, color_for_effect : dict
        Existing endpoint labels and colors from the plot configuration.
    bin_width : float, optional
        Shared ratio-histogram bin width in log10 units.

    Returns
    -------
    matplotlib.figure.Figure
        Figure using all valid pairs, including final training chemicals.
        The smaller layer uses Aurisano PODs without requiring
        membership in the QSAR prediction universe.

    '''
    if not np.isfinite(bin_width) or bin_width <= 0:
        raise ValueError('bin_width must be finite and positive.')
    data = {
        effect: tables[effect].loc[tables[effect]['valid_pair']]
        for effect in ENDPOINTS
    }
    if any(frame.empty for frame in data.values()):
        raise ValueError('Each endpoint needs valid pairs for plotting.')
    source_data = {
        effect: source_tables[effect].loc[
            source_tables[effect]['valid_pair']
        ]
        for effect in ENDPOINTS
    }
    scatter_columns = ['log10_inhalation_dose', 'log10_oral_pod']
    prediction_min = np.floor(min(
        frame[scatter_columns].min().min() for frame in data.values()
    ))
    prediction_max = np.ceil(max(
        frame[scatter_columns].max().max() for frame in data.values()
    ))
    if prediction_min == prediction_max:
        prediction_min -= 1
        prediction_max += 1
    all_frames = [
        frame for frame in [*data.values(), *source_data.values()]
        if not frame.empty
    ]
    scatter_min = np.floor(min(
        frame[scatter_columns].min().min() for frame in all_frames
    ))
    scatter_max = np.ceil(max(
        frame[scatter_columns].max().max() for frame in all_frames
    ))
    if scatter_min == scatter_max:
        scatter_min -= 1
        scatter_max += 1
    ratio_min = min(-2, min(
        frame['log10_ratio'].min() for frame in all_frames
    ))
    ratio_max = max(2, max(
        frame['log10_ratio'].max() for frame in all_frames
    ))
    bin_edges = np.arange(
        np.floor(ratio_min / bin_width) * bin_width,
        (np.ceil(ratio_max / bin_width) + 1) * bin_width,
        bin_width,
    )
    figure, axes = plt.subplots(
        2, 2, figsize=(11, 9), dpi=300, constrained_layout=True,
        gridspec_kw={'height_ratios': [1.5, 1]},
    )
    source_color = '#E69F00'
    hexbins = []
    for column, effect in enumerate(ENDPOINTS):
        frame = data[effect]
        source = source_data[effect]
        axis = axes[0, column]
        hexbins.append(axis.hexbin(
            frame['log10_inhalation_dose'], frame['log10_oral_pod'],
            gridsize=90, mincnt=1, cmap='viridis',
            extent=(
                prediction_min, prediction_max,
                prediction_min, prediction_max,
            ),
        ))
        utilities.plot_one_one_line(
            axis, scatter_min, scatter_max, color='#444444'
        )
        axis.scatter(
            source['log10_inhalation_dose'], source['log10_oral_pod'],
            s=4, facecolors='#D55E00', edgecolors='white', linewidths=0.2,
            label='Aurisano PODs', zorder=3,
        )
        axis.set(
            aspect='equal',
            xlabel=r'$\log_{10}$(inhalation administered dose, '
            r'mg kg$^{-1}$ day$^{-1}$)',
            ylabel=r'$\log_{10}$(oral POD, mg kg$^{-1}$ day$^{-1}$)',
            title=(
                f'({chr(65 + column)}) {label_for_effect[effect]}\n'
                f'QSAR: n = {len(frame):,}; '
                f'Aurisano PODs: n = {len(source):,}'
            ),
        )
        axis.legend(
            loc='lower right', fontsize=7, framealpha=0.95, markerscale=2,
        )

        endpoint_summary = summary.loc[
            summary['effect'].eq(effect)
        ].set_index('population')
        annotation_lines = ['Spearman rho']
        for population, label in (
                ('full', 'QSAR'), ('source_overlap', 'Aurisano PODs')):
            row = endpoint_summary.loc[population]
            text = f'{label}: {row["spearman_rho"]:.3f}'
            lower, upper = (
                row['spearman_ci_lower'], row['spearman_ci_upper']
            )
            if np.isfinite([lower, upper]).all():
                text += f'; 95% CI {lower:.3f} to {upper:.3f}'
            annotation_lines.append(text)
        axis.text(
            0.03, 0.97, '\n'.join(annotation_lines),
            transform=axis.transAxes, ha='left', va='top', fontsize=8,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.85},
        )

        axis = axes[1, column]
        axis.hist(
            frame['log10_ratio'], bins=bin_edges,
            weights=np.full(len(frame), 100 / len(frame)),
            color=color_for_effect[effect], alpha=0.65,
            label='QSAR predictions',
        )
        if len(source):
            axis.hist(
                source['log10_ratio'], bins=bin_edges,
                weights=np.full(len(source), 100 / len(source)),
                histtype='step', color=source_color, linewidth=1.5,
                label='Aurisano PODs',
            )
        for threshold in (-2, -1, 0, 1, 2):
            axis.axvline(
                threshold, color='#444444', linewidth=0.8,
                linestyle={0: '-', 1: '--', 2: ':'}[abs(threshold)],
            )
        axis.set(
            xlim=(bin_edges[0], bin_edges[-1]),
            xlabel=r'$\log_{10}$(oral POD / inhalation administered dose)',
            ylabel='Chemicals within population (%)',
            title=f'({chr(67 + column)}) {label_for_effect[effect]}',
        )
        population_legend = axis.legend(loc='upper left', fontsize=7)
        axis.add_artist(population_legend)
        axis.legend(
            handles=[
                Line2D([], [], color='#444444', linewidth=0.8,
                       linestyle=style, label=label)
                for style, label in (
                    ('-', '0: equal PODs'),
                    ('--', '+/-1: 10-fold difference'),
                    (':', '+/-2: 100-fold difference'),
                )
            ],
            loc='upper right', fontsize=6.5,
        )
        axis.grid(axis='y', linestyle=':', linewidth=0.5)


    maximum_count = max(hexbin.get_array().max() for hexbin in hexbins)
    shared_norm = LogNorm(vmin=1, vmax=max(2, maximum_count))
    for hexbin in hexbins:
        hexbin.set_norm(shared_norm)
    colorbar = figure.colorbar(
        hexbins[0], ax=axes[0, :], shrink=0.8, pad=0.02
    )
    colorbar.set_label('Chemicals per hexagon (QSAR)')
    histogram_max = max(axis.get_ylim()[1] for axis in axes[1, :])
    for axis in axes[1, :]:
        axis.set_ylim(0, histogram_max)
    return figure
