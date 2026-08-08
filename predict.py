'''
Generate route-specific full-table point-of-departure prediction exports.
'''

from pathlib import Path

from config_management import config_from_cli_args
from data_management import DataManager
from results_analysis import ResultsAnalyzer
from results_management import ResultsManager


EXPECTED_EFFECTS = ('general', 'repro_dev')
PREDICTION_STATISTICS = ('pod', 'lb', 'ub')


def main():
    '''Load command-line configuration and generate prediction exports.'''
    config = config_from_cli_args()
    output_files = generate_prediction_exports(config)
    for output_file in output_files:
        print(f'Wrote {output_file}')


def generate_prediction_exports(config):
    '''Generate both configured export formats for one exposure route.'''
    data_manager = DataManager(config.data, config.path)
    results_manager = ResultsManager(
        output_dir=config.path.results_dir,
        results_file_type=config.data.file_type
    )
    results_analyzer = ResultsAnalyzer(
        results_manager,
        data_manager,
        config.plot
    )
    prediction_table = build_prediction_table(
        results_analyzer,
        config.plot.final_model_keys
    )
    return write_prediction_table(
        prediction_table,
        config.path.pod_predictions_file
    )


def build_prediction_table(results_analyzer, model_keys):
    '''
    Build an identifier-aligned table for the configured final models.

    The first final model defines the exported identifier set and POD-based
    row order. Subsequent model results are aligned to that order by DTXSID.

    Parameters
    ----------
    results_analyzer : ResultsAnalyzer
        Analyzer used to generate POD predictions and intervals.
    model_keys : list of tuple or list of list
        Final model keys, ordered with ``general`` first and ``repro_dev``
        second.

    Returns
    -------
    pandas.DataFrame
        Prediction table containing DTXSID as its first column.
    '''
    model_keys = ResultsAnalyzer.validate_model_keys(model_keys)
    model_key_names = results_analyzer.read_model_key_names()
    target_effect_index = model_key_names.index('target_effect')
    effects = tuple(key[target_effect_index] for key in model_keys)

    if effects != EXPECTED_EFFECTS:
        raise ValueError(
            "Final models must be ordered as 'general', then 'repro_dev'."
        )

    prediction_table = None
    first_index = None

    for model_key, effect in zip(model_keys, effects):
        endpoint_table = results_analyzer.pod_and_prediction_interval(
            model_key,
            inverse_transform=True,
            exclude_training=False
        )
        endpoint_table = endpoint_table.loc[:, PREDICTION_STATISTICS]

        if not endpoint_table.index.is_unique:
            raise ValueError(
                f"Predictions for '{effect}' contain duplicate DTXSIDs."
            )

        endpoint_table = endpoint_table.rename(
            columns={
                statistic: f'{effect}_{statistic}'
                for statistic in PREDICTION_STATISTICS
            }
        )

        if prediction_table is None:
            first_index = endpoint_table.index
            prediction_table = endpoint_table.copy()
        else:
            missing_dtxsids = first_index.difference(endpoint_table.index)
            extra_dtxsids = endpoint_table.index.difference(first_index)
            if not missing_dtxsids.empty or not extra_dtxsids.empty:
                raise ValueError(
                    f"Predictions for '{effect}' do not match the first "
                    'model identifier set: '
                    f'{len(missing_dtxsids)} missing and '
                    f'{len(extra_dtxsids)} extra DTXSIDs.'
                )
            endpoint_table = endpoint_table.reindex(first_index)
            prediction_table = prediction_table.join(endpoint_table)

    prediction_table.index.name = 'DTXSID'
    return prediction_table.reset_index()


def write_prediction_table(prediction_table, output_file):
    '''Write a prediction table to Parquet and zipped CSV files.'''
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    parquet_file = output_file.with_suffix('.parquet')
    csv_file = output_file.with_suffix('.csv.zip')

    prediction_table.to_parquet(parquet_file, index=False)
    prediction_table.to_csv(csv_file, compression='zip', index=False)

    return parquet_file, csv_file


if __name__ == '__main__':
    print('Generating prediction exports...')
    main()
    print('Prediction export completed.')
