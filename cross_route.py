'''Compare cross-route QSAR predictions and Aurisano PODs.'''

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ENDPOINTS = ('general', 'repro_dev')
POPULATION_LABELS = {
    'full': 'QSAR predictions',
    'source_overlap': 'Aurisano PODs',
}


def load_cross_route_comparison(oral_config, inhalation_config):
    '''Load predictions and unfiltered Table S5/S6 Aurisano PODs.

    Parameters
    ----------
    oral_config, inhalation_config : UnifiedConfiguration
        Route-specific path, workbook, metric, and endpoint settings.

    Returns
    -------
    tables : dict of pandas.DataFrame
        Endpoint-specific matched QSAR calculations indexed by DTXSID.
    source_tables : dict of pandas.DataFrame
        Original Table S5/S6 POD calculations indexed by CASRN field values.
    audit : pandas.DataFrame
        Prediction-join, source-overlap, and valid-pair counts by endpoint.

    Notes
    -----
    Aurisano PODs use the original workbook casrn field values verbatim,
    including NOCAS identifiers and qualified record keys. No DTXSID,
    study-count, or QSAR eligibility restriction is applied.
    '''
    inputs = []
    for config in (oral_config, inhalation_config):
        predictions = pd.read_parquet(
            Path(config.path.pod_predictions_file).with_suffix('.parquet')
        )
        source_pods = _read_aurisano_pods(config)
        inputs.append((predictions, source_pods))
    return build_cross_route_comparison(
        inputs[0][0], inputs[1][0], inputs[0][1], inputs[1][1]
    )


def build_cross_route_comparison(
        oral_predictions,
        inhalation_predictions,
        oral_source_pods,
        inhalation_source_pods,
        ):
    '''Pair QSAR predictions by DTXSID and Aurisano PODs by CASRN field.

    Parameters
    ----------
    oral_predictions, inhalation_predictions : pandas.DataFrame
        Unique DTXSID column and ``general_pod``/``repro_dev_pod`` columns
        in linear mg/(kg day) and mg/m3, respectively.
    oral_source_pods, inhalation_source_pods : pandas.DataFrame
        Original Table S5/S6 PODs indexed by unique original CASRN field
        values, including NOCAS identifiers, with endpoint columns. No
        DTXSID, study-count, or QSAR eligibility filters may be applied.
        Nonmissing POD pairs supply the comparator values in original
        linear mg/(kg day) and mg/m3 units, respectively. They need not
        occur in either prediction table.

    Returns
    -------
    tables : dict of pandas.DataFrame
        One table per endpoint containing every common prediction DTXSID,
        including training chemicals. Invalid POD pairs retain their rows
        but have missing logarithms and ratios.
    source_tables : dict of pandas.DataFrame
        Original oral/inhalation POD pairs by endpoint, with the same
        administered-dose normalization and ratio calculations.
    audit : pandas.DataFrame
        Counts accounting for unmatched predictions, source membership,
        and invalid pairs. Duplicate/missing identifiers raise an error
        rather than silently altering the population.
    '''
    oral = _prediction_index(oral_predictions, 'oral')
    inhalation = _prediction_index(inhalation_predictions, 'inhalation')
    for name, source_pods in (
            ('oral Aurisano PODs', oral_source_pods),
            ('inhalation Aurisano PODs', inhalation_source_pods)):
        _validate_identifiers(source_pods.index, name, identifier='CASRN')
        if not set(ENDPOINTS).issubset(source_pods.columns):
            raise ValueError(f'{name} must contain both endpoint columns.')
        if np.isinf(source_pods.loc[:, ENDPOINTS].to_numpy()).any():
            raise ValueError(f'{name} contains nonfinite PODs.')

    tables, source_tables, audit_rows = {}, {}, []
    for effect in ENDPOINTS:
        column = f'{effect}_pod'
        table = _pair_pods(oral[column], inhalation[column])
        source = _pair_pods(
            oral_source_pods[effect].dropna(),
            inhalation_source_pods[effect].dropna(),
        )
        tables[effect] = table
        source_tables[effect] = source
        audit_rows.append({
            'effect': effect,
            'oral_prediction_rows': len(oral),
            'inhalation_prediction_rows': len(inhalation),
            'duplicate_prediction_ids': 0,
            'oral_only_ids': len(oral.index.difference(inhalation.index)),
            'inhalation_only_ids': len(
                inhalation.index.difference(oral.index)
            ),
            'matched_ids': len(table),
            'valid_pairs': int(table['valid_pair'].sum()),
            'invalid_pairs': int((~table['valid_pair']).sum()),
            'oral_source_ids': int(oral_source_pods[effect].notna().sum()),
            'inhalation_source_ids': int(
                inhalation_source_pods[effect].notna().sum()
            ),
            'source_overlap_ids': len(source),
            'oral_source_only_ids': (
                int(oral_source_pods[effect].notna().sum()) - len(source)
            ),
            'inhalation_source_only_ids': (
                int(inhalation_source_pods[effect].notna().sum())
                - len(source)
            ),
            'source_overlap_valid_pairs': int(source['valid_pair'].sum()),
            'source_overlap_invalid_pairs': int(
                (~source['valid_pair']).sum()
            ),
        })
    return tables, source_tables, pd.DataFrame(audit_rows)


def summarize_cross_route(tables, source_tables):
    '''Summarize paired QSAR predictions and Aurisano PODs.

    Spearman point estimates use paired, finite positive PODs and average
    ranks for ties (SciPy). Distribution quantiles use pandas' default
    linear interpolation. Divergence is strictly abs(log10 ratio) > 1.
    All proportions use the corresponding valid-pair denominator.

    Parameters
    ----------
    tables, source_tables : dict of pandas.DataFrame
        QSAR and original-POD tables from ``build_cross_route_comparison``.
        An empty dictionary skips that population's summary.

    Returns
    -------
    pandas.DataFrame
        One summary row per endpoint and population, including denominator,
        quantiles, directional divergence, and Spearman point estimate.
    '''
    rows = []
    for effect in ENDPOINTS:
        for population, population_tables in (
                ('full', tables), ('source_overlap', source_tables)):
            if effect not in population_tables:
                continue
            table = population_tables[effect]
            data = table.loc[table['valid_pair']]
            n = len(data)
            rho = np.nan
            if (n >= 2 and data['log10_oral_pod'].nunique() > 1
                    and data['log10_inhalation_dose'].nunique() > 1):
                rho = spearmanr(
                    data['log10_oral_pod'], data['log10_inhalation_dose']
                ).statistic
            row = {
                'effect': effect,
                'population': population,
                'n': n,
                'invalid_pairs': len(table) - n,
                'spearman_rho': rho,
                'spearman_ci_lower': np.nan,
                'spearman_ci_upper': np.nan,
            }
            for column in (
                    'oral_pod_mg_kg_day', 'inhalation_dose_mg_kg_day',
                    'log10_ratio'):
                for quantile, value in data[column].quantile(
                        [0.05, 0.5, 0.95]).items():
                    row[f'{column}_p{int(quantile * 100):02d}'] = value
            ratio = data['log10_ratio']
            for name, condition in (
                    ('oral_higher', ratio.gt(1)),
                    ('inhalation_higher', ratio.lt(-1)),
                    ('either_direction', ratio.abs().gt(1))):
                count = int(condition.sum())
                row[f'{name}_count'] = count
                row[f'{name}_percent'] = 100 * count / n if n else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def cross_route_summary_table(summary, label_for_effect):
    '''Format the summary, showing CIs only when all bounds are finite.'''
    ci_columns = ['spearman_ci_lower', 'spearman_ci_upper']
    include_ci = (
        not summary.empty
        and np.isfinite(summary.reindex(columns=ci_columns).to_numpy()).all()
    )
    metrics = [
        'Paired PODs, n',
        'Oral POD, mg/(kg day), median (5th-95th percentile)',
        'Inhalation administered dose, mg/(kg day), '
        'median (5th-95th percentile)',
        'log10(oral/inhalation dose), median (5th-95th percentile)',
        'Spearman rho',
    ]
    if include_ci:
        metrics.append('Spearman 95% confidence interval')
    metrics.extend([
        'Oral POD >10 times inhalation dose, n (%)',
        'Inhalation dose >10 times oral POD, n (%)',
        '>10-fold divergence in either direction, n (%)',
    ])
    result = {'Statistic': metrics}
    for _, row in summary.iterrows():
        values = [f"{row['n']:,}"]
        for column in (
                'oral_pod_mg_kg_day', 'inhalation_dose_mg_kg_day',
                'log10_ratio'):
            values.append(
                f'{row[column + "_p50"]:.3g} '
                f'({row[column + "_p05"]:.3g} to '
                f'{row[column + "_p95"]:.3g})'
            )
        values.append(f"{row['spearman_rho']:.3f}")
        if include_ci:
            lower, upper = row['spearman_ci_lower'], row['spearman_ci_upper']
            values.append(f'{lower:.3f} to {upper:.3f}')
        for name in ('oral_higher', 'inhalation_higher', 'either_direction'):
            values.append(
                f'{row[name + "_count"]:,} '
                f'({row[name + "_percent"]:.2f}%)'
            )
        label = (
            f"{label_for_effect[row['effect']]}: "
            f"{POPULATION_LABELS[row['population']]}"
        )
        result[label] = values
    return pd.DataFrame(result)


def bmch_pod_to_hed(pods):
    '''Convert native log10 BMCh POD results to log10 POD_inh,HED.

    Parameters
    ----------
    pods : pandas.DataFrame
        Native BMCh results in log10 mg/m3, with pod/lb/ub columns.
        These must be concentration predictions, not oral dose predictions.

    Returns
    -------
    pandas.DataFrame
        A copy in log10 mg/(kg day); DTXSIDs and CDF columns are unchanged.
        The constant shift preserves log10 RMSE and interval widths.
    '''
    converted = pods.copy()
    columns = ['pod', 'lb', 'ub']
    converted[columns] += np.log10(bmch_to_administered_dose(1.0))
    return converted


def bmch_to_administered_dose(
        bmch, ventilation_m3_day=13.0, body_weight_kg=70.0):
    '''Normalize BMCh (mg/m3) to administered dose (mg/(kg day)).'''
    for name, value in (
            ('ventilation_m3_day', ventilation_m3_day),
            ('body_weight_kg', body_weight_kg)):
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be a finite positive scalar.')
    return bmch * ventilation_m3_day / body_weight_kg


def _read_aurisano_pods(config):
    '''Read original PODs with verbatim CASRN field keys and no filtering.'''
    raw = config.raw_data
    workbook = pd.read_excel(
        config.path.raw_surrogate_pods_file,
        **raw.surrogate_tox_data_kwargs,
    ).swaplevel(axis=1)
    pods = workbook[raw.tox_metric].rename(columns=raw.effect_mapper)
    pods.index = pd.Index(workbook['casrn'].iloc[:, 0], name='CASRN')
    return pods


def _prediction_index(predictions, name):
    columns = ['DTXSID', *(f'{effect}_pod' for effect in ENDPOINTS)]
    if not set(columns).issubset(predictions.columns):
        raise ValueError(f'{name} predictions must contain {columns}.')
    table = predictions.loc[:, columns].set_index('DTXSID')
    _validate_identifiers(table.index, f'{name} predictions')
    return table


def _pair_pods(oral_pods, inhalation_bmch):
    '''Join two POD series and apply the shared conversion and ratio.'''
    table = oral_pods.rename('oral_pod_mg_kg_day').to_frame().join(
        inhalation_bmch.rename('inhalation_bmch_mg_m3'),
        how='inner', validate='one_to_one',
    )
    table['inhalation_dose_mg_kg_day'] = bmch_to_administered_dose(
        table['inhalation_bmch_mg_m3']
    )
    pair_values = table[
        ['oral_pod_mg_kg_day', 'inhalation_dose_mg_kg_day']
    ]
    table['valid_pair'] = (
        np.isfinite(pair_values) & pair_values.gt(0)
    ).all(axis=1)
    table['log10_oral_pod'] = np.log10(
        table['oral_pod_mg_kg_day'].where(table['valid_pair'])
    )
    table['log10_inhalation_dose'] = np.log10(
        table['inhalation_dose_mg_kg_day'].where(table['valid_pair'])
    )
    table['log10_ratio'] = (
        table['log10_oral_pod'] - table['log10_inhalation_dose']
    )
    return table


def _validate_identifiers(index, name, identifier='DTXSID'):
    duplicates = int(index.duplicated().sum())
    missing = int(index.isna().sum())
    if duplicates or missing:
        raise ValueError(
            f'{name}: {duplicates} duplicate and {missing} '
            f'{identifier} keys.'
        )
    if identifier == 'CASRN':
        if not all(isinstance(key, str) and key.strip() for key in index):
            raise ValueError(f'{name} contains invalid CASRN field keys.')
    elif not index.astype(str).str.fullmatch(r'DTXSID\d+').all():
        raise ValueError(f'{name} contains malformed DTXSIDs.')
