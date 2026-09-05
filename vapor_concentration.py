'''
Calculate and summarize saturated vapor concentration screening results.
'''

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_TEMPERATURE_K = 298.15
GAS_CONSTANT_PA_M3_MOL_K = 8.314462618
MMHG_TO_PA = 133.322387415
STANDARD_ATMOSPHERIC_PRESSURE_MMHG = 760.0

ENDPOINTS = ('general', 'repro_dev')
POINT_AT_OR_BELOW = 'bmch_at_or_below_svc'
POINT_ABOVE = 'bmch_above_svc'
UNCLASSIFIED = 'unclassified'
INTERVAL_AT_OR_BELOW = 'entire_interval_at_or_below_svc'
INTERVAL_CROSSES = 'interval_crosses_svc'
INTERVAL_ABOVE = 'entire_interval_above_svc'


def vapor_concentration_interval_classification_table(
        data_manager,
        path_settings,
        label_for_effect,
        ):
    '''Return the manuscript prediction-interval classification table.'''
    summary = vapor_concentration_summary_table(
        data_manager,
        path_settings,
        label_for_effect,
    )
    interval_summary = summary.loc[
        summary['summary_type'].eq('interval_classification')
    ]
    categories = (
        INTERVAL_AT_OR_BELOW,
        INTERVAL_CROSSES,
        INTERVAL_ABOVE,
    )
    category_labels = (
        'Entirely at or below SVC',
        'Crosses SVC',
        'Entirely above SVC',
    )
    table_data = {
        '90% BMCh prediction-interval classification': category_labels,
    }

    for effect in ENDPOINTS:
        effect_summary = (
            interval_summary.loc[
                interval_summary['effect_label'].eq(
                    label_for_effect[effect]
                )
            ]
            .set_index('category')
            .loc[list(categories)]
        )
        column = f'{label_for_effect[effect].capitalize()}, n (%)'
        table_data[column] = [
            f'{count:,} ({percent:.2f})'
            for count, percent in zip(
                effect_summary['count'],
                effect_summary['percent'],
            )
        ]

    return pd.DataFrame(table_data)


def vapor_concentration_summary_table(
        data_manager,
        path_settings,
        label_for_effect,
        ):
    '''Return the SVC analysis summary from configured inputs.

    Parameters
    ----------
    data_manager : DataManager
        Configured manager used to identify endpoint training chemicals.
    path_settings : SimpleNamespace
        Configured repository input paths.
    label_for_effect : dict
        Endpoint names mapped to manuscript-facing labels.

    Returns
    -------
    pandas.DataFrame
        Coverage and classification counts, denominators, and percentages.
    '''
    features = pd.read_parquet(
        path_settings.file_for_features_source['opera'],
        columns=['VP_pred', 'MolWeight'],
    )
    predictions = pd.read_parquet(
        Path(path_settings.pod_predictions_file).with_suffix('.parquet')
    )
    training_chemicals_for_effect = {
        effect: data_manager.load_target(target_effect=effect).index
        for effect in ENDPOINTS
    }
    summary = summarize_vapor_concentration_inputs(
        features,
        predictions,
        training_chemicals_for_effect=training_chemicals_for_effect,
        label_for_effect=label_for_effect,
    )
    summary_columns = [
        'effect_label',
        'summary_type',
        'category',
        'count',
        'denominator',
        'percent',
    ]
    return summary.loc[:, summary_columns]


def summarize_vapor_concentration_inputs(
        features,
        predictions,
        training_chemicals_for_effect=None,
        label_for_effect=None,
        temperature_k=DEFAULT_TEMPERATURE_K,
        pressure_cap_mmhg=STANDARD_ATMOSPHERIC_PRESSURE_MMHG,
        ):
    '''Calculate and summarize BMCh/SVC results without retaining a table.'''
    table = build_vapor_concentration_table(
        features,
        predictions,
        training_chemicals_for_effect=training_chemicals_for_effect,
        temperature_k=temperature_k,
        pressure_cap_mmhg=pressure_cap_mmhg,
    )
    return summarize_vapor_concentration(
        table,
        label_for_effect=label_for_effect,
    )


def build_vapor_concentration_table(
        features,
        predictions,
        training_chemicals_for_effect=None,
        temperature_k=DEFAULT_TEMPERATURE_K,
        pressure_cap_mmhg=STANDARD_ATMOSPHERIC_PRESSURE_MMHG,
        ):
    '''Build identifier-aligned SVC and BMCh comparison results.

    Parameters
    ----------
    features : pandas.DataFrame
        OPERA features indexed by DTXSID. ``VP_pred`` must be in linear
        mmHg and ``MolWeight`` must be in g/mol.
    predictions : pandas.DataFrame
        Persisted BMCh predictions with a unique ``DTXSID`` column.
    training_chemicals_for_effect : dict, optional
        Endpoint names mapped to training-chemical identifiers.
    temperature_k : float, optional
        Absolute temperature in kelvin.
    pressure_cap_mmhg : float, optional
        Upper vapor-pressure bound in mmHg. The default is one atmosphere.

    Returns
    -------
    pandas.DataFrame
        One row per DTXSID with SVC, endpoint-specific ratios,
        classifications, prediction-interval sensitivity, and training flags.
    '''
    _validate_positive_scalar(temperature_k, 'temperature_k')
    _validate_positive_scalar(pressure_cap_mmhg, 'pressure_cap_mmhg')
    _validate_inputs(features, predictions)

    prediction_table = predictions.set_index('DTXSID')
    missing_predictions = features.index.difference(prediction_table.index)
    extra_predictions = prediction_table.index.difference(features.index)
    if not missing_predictions.empty or not extra_predictions.empty:
        raise ValueError(
            'Feature and prediction identifier sets differ: '
            f'{len(missing_predictions)} missing predictions and '
            f'{len(extra_predictions)} extra predictions.'
        )

    table = features.loc[:, ['VP_pred', 'MolWeight']].rename(
        columns={
            'VP_pred': 'vapor_pressure_mmhg',
            'MolWeight': 'molecular_weight_g_mol',
        }
    )
    table = table.join(prediction_table, how='left', validate='one_to_one')

    table['svc_status'] = _svc_status(
        table['vapor_pressure_mmhg'],
        table['molecular_weight_g_mol'],
    )
    table['vapor_pressure_capped'] = (
        table['svc_status'].eq('calculable')
        & table['vapor_pressure_mmhg'].gt(pressure_cap_mmhg)
    )
    table['vapor_pressure_used_mmhg'] = (
        table['vapor_pressure_mmhg']
        .where(table['svc_status'].eq('calculable'))
        .clip(upper=pressure_cap_mmhg)
    )
    table['svc_mg_m3'] = saturated_vapor_concentration(
        table['vapor_pressure_mmhg'],
        table['molecular_weight_g_mol'],
        temperature_k=temperature_k,
        pressure_cap_mmhg=pressure_cap_mmhg,
    )

    if training_chemicals_for_effect is None:
        training_chemicals_for_effect = {}

    for effect in ENDPOINTS:
        _add_endpoint_results(
            table,
            effect,
            training_chemicals_for_effect.get(effect, ()),
        )

    source_prediction_columns = [
        f'{effect}_{statistic}'
        for effect in ENDPOINTS
        for statistic in ('pod', 'lb', 'ub')
    ]
    table = table.drop(columns=source_prediction_columns)
    table.index.name = 'DTXSID'
    return table.reset_index()


def saturated_vapor_concentration(
        vapor_pressure_mmhg,
        molecular_weight_g_mol,
        temperature_k=DEFAULT_TEMPERATURE_K,
        pressure_cap_mmhg=STANDARD_ATMOSPHERIC_PRESSURE_MMHG,
        ):
    '''Calculate saturated vapor concentration in mg/m3.

    Vapor pressure is capped at one atmosphere by default. Missing,
    nonfinite, and nonpositive inputs return missing results.
    '''
    _validate_positive_scalar(temperature_k, 'temperature_k')
    _validate_positive_scalar(pressure_cap_mmhg, 'pressure_cap_mmhg')

    vapor_pressure = _as_series(
        vapor_pressure_mmhg,
        'vapor_pressure_mmhg',
    ).astype(float)
    molecular_weight = _as_series(
        molecular_weight_g_mol,
        'molecular_weight_g_mol',
    ).astype(float)
    vapor_pressure, molecular_weight = vapor_pressure.align(
        molecular_weight,
        join='outer',
    )

    valid = (
        np.isfinite(vapor_pressure)
        & np.isfinite(molecular_weight)
        & vapor_pressure.gt(0)
        & molecular_weight.gt(0)
    )
    pressure_used_mmhg = vapor_pressure.clip(upper=pressure_cap_mmhg)
    concentration = (
        pressure_used_mmhg
        * MMHG_TO_PA
        * molecular_weight
        * 1000
        / (GAS_CONSTANT_PA_M3_MOL_K * temperature_k)
    )
    return concentration.where(valid).rename('svc_mg_m3')


def summarize_vapor_concentration(table, label_for_effect=None):
    '''Summarize coverage and classification counts by endpoint.'''
    label_for_effect = label_for_effect or {
        'general': 'General noncancer',
        'repro_dev': 'Reproductive/developmental',
    }
    rows = []

    for effect in ENDPOINTS:
        application = table.loc[~table[f'{effect}_training_chemical']]
        point_column = f'{effect}_bmch_svc_classification'
        interval_column = f'{effect}_bmch_svc_interval_classification'

        calculable = application['svc_status'].eq('calculable')
        _append_count_rows(
            rows,
            effect,
            label_for_effect[effect],
            'svc_coverage',
            {
                'svc_calculable': int(calculable.sum()),
                'svc_unavailable': int((~calculable).sum()),
            },
            len(application),
            'endpoint application population',
        )

        point_valid = application[point_column].ne(UNCLASSIFIED)
        point_counts = (
            application.loc[point_valid, point_column]
            .value_counts()
            .reindex([POINT_AT_OR_BELOW, POINT_ABOVE], fill_value=0)
        )
        _append_count_rows(
            rows,
            effect,
            label_for_effect[effect],
            'point_classification',
            point_counts.to_dict(),
            int(point_valid.sum()),
            'chemicals with calculable SVC and valid median BMCh',
        )

        interval_valid = application[interval_column].ne(UNCLASSIFIED)
        interval_counts = (
            application.loc[interval_valid, interval_column]
            .value_counts()
            .reindex(
                [
                    INTERVAL_AT_OR_BELOW,
                    INTERVAL_CROSSES,
                    INTERVAL_ABOVE,
                ],
                fill_value=0,
            )
        )
        _append_count_rows(
            rows,
            effect,
            label_for_effect[effect],
            'interval_classification',
            interval_counts.to_dict(),
            int(interval_valid.sum()),
            'chemicals with calculable SVC and valid 90% BMCh interval',
        )

    return pd.DataFrame(rows)


def _add_endpoint_results(table, effect, training_chemicals):
    source_columns = {
        f'{effect}_pod': f'{effect}_bmch_mg_m3',
        f'{effect}_lb': f'{effect}_bmch_lower_mg_m3',
        f'{effect}_ub': f'{effect}_bmch_upper_mg_m3',
    }
    for source, destination in source_columns.items():
        table[destination] = table[source]

    table[f'{effect}_training_chemical'] = table.index.isin(
        training_chemicals
    )
    bmch = table[f'{effect}_bmch_mg_m3']
    lower = table[f'{effect}_bmch_lower_mg_m3']
    upper = table[f'{effect}_bmch_upper_mg_m3']
    svc = table['svc_mg_m3']
    valid_point = _valid_positive(bmch) & _valid_positive(svc)
    valid_interval = (
        valid_point
        & _valid_positive(lower)
        & _valid_positive(upper)
        & lower.le(bmch)
        & bmch.le(upper)
    )

    ratio = pd.Series(np.nan, index=table.index, dtype=float)
    ratio.loc[valid_point] = np.log10(
        bmch.loc[valid_point] / svc.loc[valid_point]
    )
    table[f'{effect}_log10_bmch_svc_ratio'] = ratio

    point = pd.Series(UNCLASSIFIED, index=table.index, dtype='string')
    point.loc[valid_point & bmch.le(svc)] = POINT_AT_OR_BELOW
    point.loc[valid_point & bmch.gt(svc)] = POINT_ABOVE
    table[f'{effect}_bmch_svc_classification'] = point

    interval = pd.Series(UNCLASSIFIED, index=table.index, dtype='string')
    interval.loc[valid_interval & upper.le(svc)] = INTERVAL_AT_OR_BELOW
    interval.loc[valid_interval & lower.gt(svc)] = INTERVAL_ABOVE
    crosses = (
        valid_interval
        & lower.le(svc)
        & upper.gt(svc)
    )
    interval.loc[crosses] = INTERVAL_CROSSES
    table[f'{effect}_bmch_svc_interval_classification'] = interval


def _svc_status(vapor_pressure, molecular_weight):
    status = pd.Series(
        'calculable',
        index=vapor_pressure.index,
        dtype='string',
    )
    status.loc[molecular_weight.isna()] = 'missing_molecular_weight'
    status.loc[vapor_pressure.isna()] = 'missing_vapor_pressure'
    status.loc[
        vapor_pressure.notna() & ~np.isfinite(vapor_pressure)
    ] = 'nonfinite_vapor_pressure'
    status.loc[
        molecular_weight.notna() & ~np.isfinite(molecular_weight)
    ] = 'nonfinite_molecular_weight'
    status.loc[
        np.isfinite(molecular_weight) & molecular_weight.le(0)
    ] = 'nonpositive_molecular_weight'
    status.loc[
        np.isfinite(vapor_pressure) & vapor_pressure.le(0)
    ] = 'nonpositive_vapor_pressure'
    return status


def _append_count_rows(
        rows,
        effect,
        effect_label,
        summary_type,
        counts,
        denominator,
        denominator_definition,
        ):
    for category, count in counts.items():
        percent = count / denominator * 100 if denominator else np.nan
        rows.append({
            'effect': effect,
            'effect_label': effect_label,
            'summary_type': summary_type,
            'category': category,
            'count': int(count),
            'denominator': int(denominator),
            'percent': percent,
            'denominator_definition': denominator_definition,
        })


def _validate_inputs(features, predictions):
    feature_columns = {'VP_pred', 'MolWeight'}
    missing_features = feature_columns.difference(features.columns)
    if missing_features:
        raise ValueError(
            f'Missing feature columns: {sorted(missing_features)}'
        )
    if not features.index.is_unique:
        raise ValueError('Feature identifiers must be unique.')
    if 'DTXSID' not in predictions:
        raise ValueError("Predictions must contain a 'DTXSID' column.")
    if not predictions['DTXSID'].is_unique:
        raise ValueError('Prediction identifiers must be unique.')

    expected_predictions = {
        f'{effect}_{statistic}'
        for effect in ENDPOINTS
        for statistic in ('pod', 'lb', 'ub')
    }
    missing_predictions = expected_predictions.difference(
        predictions.columns
    )
    if missing_predictions:
        raise ValueError(
            f'Missing prediction columns: {sorted(missing_predictions)}'
        )


def _validate_positive_scalar(value, name):
    if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"'{name}' must be a positive finite scalar.")


def _valid_positive(values):
    return np.isfinite(values) & values.gt(0)


def _as_series(values, name):
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError(f"'{name}' must contain exactly one column.")
        values = values.iloc[:, 0]
    if isinstance(values, pd.Series):
        return values.copy()
    return pd.Series(values, name=name)
