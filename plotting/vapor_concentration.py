'''
Plot saturated vapor concentration screening results.
'''

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

from vapor_concentration import (
    ENDPOINTS,
    POINT_ABOVE,
    POINT_AT_OR_BELOW,
    build_vapor_concentration_table,
    summarize_vapor_concentration,
)

from . import utilities


def vapor_concentration_ceiling(
        features_file,
        predictions_file,
        surrogate_pods_file,
        plot_settings,
        output_dir=None,
        ):
    '''Generate the manuscript BMCh/SVC figure from persisted inputs.'''
    features = pd.read_parquet(
        features_file,
        columns=['VP_pred', 'MolWeight'],
    )
    predictions = pd.read_parquet(
        Path(predictions_file).with_suffix('.parquet')
    )
    surrogate_pods = pd.read_csv(surrogate_pods_file, index_col=0)
    training_chemicals_for_effect = {
        effect: surrogate_pods[effect].dropna().index
        for effect in ENDPOINTS
    }
    table = build_vapor_concentration_table(
        features,
        predictions,
        training_chemicals_for_effect=training_chemicals_for_effect,
    )
    figure = bmch_svc_by_effect(
        table,
        label_for_effect=plot_settings.label_for_effect,
        color_for_effect=plot_settings.color_for_effect,
    )
    utilities.save_figure(
        figure,
        vapor_concentration_ceiling,
        'bmch-svc-by-effect',
        bbox_inches='tight',
        output_dir=output_dir,
    )
    plt.close(figure)


def bmch_svc_by_effect(
        table,
        label_for_effect,
        color_for_effect,
        bin_width=0.25,
        ):
    '''Plot endpoint-specific BMCh/SVC densities and ratio distributions.'''
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(10, 9),
        dpi=300,
        constrained_layout=True,
    )

    log_svc_for_effect = {}
    log_bmch_for_effect = {}
    ratio_for_effect = {}
    for effect in ENDPOINTS:
        application = table.loc[~table[f'{effect}_training_chemical']]
        valid = application[f'{effect}_log10_bmch_svc_ratio'].notna()
        log_svc_for_effect[effect] = np.log10(
            application.loc[valid, 'svc_mg_m3']
        )
        log_bmch_for_effect[effect] = np.log10(
            application.loc[valid, f'{effect}_bmch_mg_m3']
        )
        ratio_for_effect[effect] = application.loc[
            valid,
            f'{effect}_log10_bmch_svc_ratio',
        ]

    summary = summarize_vapor_concentration(
        table,
        label_for_effect=label_for_effect,
    )
    point_summary_for_effect = {
        effect: (
            summary.loc[
                summary['effect'].eq(effect)
                & summary['summary_type'].eq('point_classification')
            ]
            .set_index('category')
        )
        for effect in ENDPOINTS
    }

    all_scatter_values = [
        *log_svc_for_effect.values(),
        *log_bmch_for_effect.values(),
    ]
    scatter_min = np.floor(min(values.min() for values in all_scatter_values))
    scatter_max = np.ceil(max(values.max() for values in all_scatter_values))
    scatter_extent = (
        scatter_min,
        scatter_max,
        scatter_min,
        scatter_max,
    )

    hexbins = []
    for column, effect in enumerate(ENDPOINTS):
        axis = axes[0, column]
        hexbin = axis.hexbin(
            log_svc_for_effect[effect],
            log_bmch_for_effect[effect],
            gridsize=90,
            extent=scatter_extent,
            mincnt=1,
            cmap='viridis',
        )
        hexbins.append(hexbin)
        utilities.plot_one_one_line(
            axis,
            scatter_min,
            scatter_max,
            color='#444444',
        )
        valid_n = int(
            point_summary_for_effect[effect]['denominator'].iloc[0]
        )
        axis.set(
            aspect='equal',
            xlabel=r'$\log_{10}$(SVC, mg m$^{-3}$)',
            ylabel=r'$\log_{10}$(predicted BMCh, mg m$^{-3}$)',
            title=(
                f'({chr(65 + column)}) {label_for_effect[effect]}\n'
                f'n = {valid_n:,}'
            ),
        )
        axis.text(
            0.97,
            0.04,
            'BMCh at or below SVC',
            transform=axis.transAxes,
            ha='right',
            va='bottom',
            fontsize=8,
            color='#333333',
        )
        axis.text(
            0.03,
            0.96,
            'BMCh above SVC',
            transform=axis.transAxes,
            ha='left',
            va='top',
            fontsize=8,
            color='#333333',
        )

    maximum_count = max(hexbin.get_array().max() for hexbin in hexbins)
    shared_norm = LogNorm(vmin=1, vmax=maximum_count)
    for hexbin in hexbins:
        hexbin.set_norm(shared_norm)
    colorbar = figure.colorbar(
        hexbins[0],
        ax=axes[0, :],
        shrink=0.8,
        pad=0.02,
    )
    colorbar.set_label('Chemicals per hexagon')

    combined_ratios = pd.concat(ratio_for_effect.values(), ignore_index=True)
    lower = np.floor(combined_ratios.min() / bin_width) * bin_width
    upper = np.ceil(combined_ratios.max() / bin_width) * bin_width
    bin_edges = np.arange(lower, upper + bin_width, bin_width)

    for column, effect in enumerate(ENDPOINTS):
        axis = axes[1, column]
        ratios = ratio_for_effect[effect]
        axis.hist(
            ratios,
            bins=bin_edges,
            weights=np.full(len(ratios), 100 / len(ratios)),
            color=color_for_effect[effect],
            edgecolor='white',
            linewidth=0.25,
        )
        axis.axvline(0, color='#444444', linestyle='--', linewidth=1)
        axis.set(
            xlabel=r'$\log_{10}$(BMCh/SVC)',
            ylabel='Chemicals (%)',
            title=f'({chr(67 + column)}) {label_for_effect[effect]}',
        )
        axis.grid(axis='y', linestyle=':', linewidth=0.5)
        point_summary = point_summary_for_effect[effect]
        valid_n = int(point_summary['denominator'].iloc[0])
        at_or_below = point_summary.loc[POINT_AT_OR_BELOW, 'percent']
        above = point_summary.loc[POINT_ABOVE, 'percent']
        summary_text = (
            f'n = {valid_n:,}\n'
            f'Median = {ratios.median():.2f}\n'
            f'At or below SVC = {at_or_below:.1f}%\n'
            f'Above SVC = {above:.1f}%'
        )
        axis.text(
            0.97,
            0.95,
            summary_text,
            transform=axis.transAxes,
            ha='right',
            va='top',
            fontsize=8,
            bbox={
                'boxstyle': 'round',
                'facecolor': 'white',
                'edgecolor': '#BBBBBB',
                'alpha': 0.9,
            },
        )

    return figure
