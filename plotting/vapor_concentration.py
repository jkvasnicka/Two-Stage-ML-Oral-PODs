'''
Plot saturated vapor concentration screening results.
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vapor_concentration import (
    ENDPOINTS,
    POINT_ABOVE,
    prepare_vapor_concentration_table,
    summarize_vapor_concentration,
)

from . import utilities


def vapor_concentration_ceiling(
        results_analyzer,
        path_settings,
        plot_settings,
        output_dir=None,
        ):
    '''Generate the manuscript BMCh/SVC figure from fitted predictions.'''
    table = prepare_vapor_concentration_table(results_analyzer, path_settings)
    figure = bmch_svc_by_effect(
        table,
        label_for_effect=plot_settings.label_for_effect,
        color_for_effect=plot_settings.color_for_effect,
    )
    utilities.save_figure(
        figure,
        vapor_concentration_ceiling,
        'bmch-svc-by-effect',
        output_dir=output_dir,
    )
    plt.close(figure)


@plt.rc_context({'font.family': 'Arial'})
def bmch_svc_by_effect(
        table,
        label_for_effect,
        color_for_effect,
        bin_width=0.25,
        ):
    '''Plot endpoint-specific BMCh/SVC ratio distributions.'''
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7, 3.2),
        dpi=600,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    ratio_for_effect = {}
    for effect in ENDPOINTS:
        application = table.loc[~table[f'{effect}_training_chemical']]
        valid = application[f'{effect}_log10_bmch_svc_ratio'].notna()
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

    combined_ratios = pd.concat(ratio_for_effect.values(), ignore_index=True)
    lower = np.floor(combined_ratios.min() / bin_width) * bin_width
    upper = np.ceil(combined_ratios.max() / bin_width) * bin_width
    bin_edges = np.arange(lower, upper + bin_width, bin_width)

    for column, effect in enumerate(ENDPOINTS):
        axis = axes[column]
        ratios = ratio_for_effect[effect]
        axis.hist(
            ratios,
            bins=bin_edges,
            weights=np.full(len(ratios), 100 / len(ratios)),
            color=color_for_effect[effect],
            edgecolor='none',
        )
        axis.axvline(0, color='#444444', linestyle='--', linewidth=1)
        axis.set(
            xlabel=r'$\log_{10}$(BMCh/SVC)',
            ylabel='Chemicals (%)',
            title=label_for_effect[effect],
        )
        axis.xaxis.label.set_size(9)
        axis.yaxis.label.set_size(9)
        axis.title.set_size(9)
        axis.tick_params(labelsize=8)
        axis.text(
            0, 0.98, 'BMCh = SVC',
            transform=axis.get_xaxis_transform(),
            ha='center', va='top', fontsize=8,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 2},
        )
        axis.grid(axis='y', linestyle=':', linewidth=0.5)
        point_summary = point_summary_for_effect[effect]
        valid_n = int(point_summary['denominator'].iloc[0])
        above = point_summary.loc[POINT_ABOVE, 'percent']
        summary_text = (
            f'n = {valid_n:,}\n'
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
