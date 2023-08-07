import click
import pandas as pd
import numpy as np
import gzip
import pickle
from pathlib import Path
from funmap.funmap import validation_llr, predict_all_pairs, dataset_llr
from funmap.plotting import explore_data, plot_results, merge_and_delete
from funmap.funmap import prepare_features, train_ml_model, prepare_gs_data
from funmap.funmap import get_funmap
from funmap.data_urls import misc_urls as urls
from funmap.logger import setup_logging, setup_logger
from funmap.utils import setup_experiment, cleanup_experiment, check_gold_standard_file
from funmap.utils import check_extra_feature_file
from funmap import __version__

log = setup_logger(__name__)

@click.group(help='funmap command line interface')
@click.version_option(version=f'{__version__}')
def cli():
    """
    Command line interface for funmap.
    """
    click.echo("====== funmap =======")

@cli.command(help='check the data quality')
@click.option('--config-file', '-c', required=True, type=click.Path(exists=True),
            help='path to experiment configuration yaml file')
@click.option('--data-file', '-d', required=True, type=click.Path(exists=True),
            help='path to tar gzipped data file')
@click.option('--force-rerun', '-f', is_flag=True, default=False,
            help='if set, will remove results from previous run first')
def qc(config_file, data_file, force_rerun):
    if force_rerun:
        click.echo('Removing results from previous run')
        cleanup_experiment(config_file, data_file)

    setup_logging(config_file)
    log.info(f'Running QC...')
    cfg = setup_experiment(config_file, data_file)
    all_fig_names = []
    figure_dir = Path(cfg['results_dir']) / cfg['subdirs']['figure_dir']
    min_sample_count = cfg['min_sample_count']
    fig_names = explore_data(cfg, data_file, min_sample_count, figure_dir)
    all_fig_names.extend(fig_names)
    merge_and_delete(figure_dir, all_fig_names, 'qc.pdf')
    log.info('figure qc.pdf saved to {}'.format(figure_dir))
    log.info('QC complete')


@cli.command(help='run funmap')
@click.option('--config-file', '-c', required=True, type=click.Path(exists=True),
            help='path to experiment configuration yaml file')
@click.option('--data-file', '-d', required=True, type=click.Path(exists=True),
            help='path to tar gzipped data file')
@click.option('--extra-feature-file', '-e', required=False, type=click.Path(exists=True),
            help='path to optional extra feature file')
@click.option('--gold-standard-file', '-g', required=False, type=click.Path(exists=True),
            help='path to optional gold standard file')
@click.option('--force-rerun', '-f', is_flag=True, default=False,
            help='if set, will remove results from previous run first')
def run(config_file, data_file, extra_feature_file, gold_standard_file, force_rerun):
    click.echo(f'Running funmap...')
    if (extra_feature_file is not None) and (not check_extra_feature_file(extra_feature_file)):
        return

    if (gold_standard_file is not None) and (not check_gold_standard_file(gold_standard_file)):
        return

    if force_rerun:
        click.echo('Removing results from previous run')
        cleanup_experiment(config_file, data_file)

    setup_logging(config_file)

    cfg = setup_experiment(config_file, data_file)

    seed = cfg['seed']
    np.random.seed(seed)
    ml_type = cfg['ml_type']
    feature_type = cfg['feature_type']
    min_feature_count = cfg['min_feature_count']
    min_sample_count = cfg['min_sample_count']
    filter_before_prediction = cfg['filter_before_prediction']
    test_size = cfg['test_size']
    cor_type = cfg['cor_type']
    split_by = cfg['split_by']
    filter_after_prediction = cfg['filter_after_prediction']
    filter_criterion = cfg['filter_criterion']
    filter_threshold = cfg['filter_threshold']
    filter_blacklist = cfg['filter_blacklist']
    n_jobs = cfg['n_jobs']
    n_chunks = cfg['n_chunks']
    max_num_edges = cfg['max_num_edges']
    step_size = cfg['step_size']
    start_edge_num = cfg['start_edge_num']

    results_dir = Path(cfg['results_dir'])
    saved_data_dir = results_dir / cfg['subdirs']['saved_data_dir']
    model_dir = results_dir / cfg['subdirs']['model_dir']
    prediction_dir = results_dir / cfg['subdirs']['prediction_dir']
    network_dir = results_dir / cfg['subdirs']['network_dir']
    figure_dir = results_dir / cfg['subdirs']['figure_dir']

    ml_model_file = model_dir / 'model.pkl.gz'
    predicted_all_pairs_file = prediction_dir / 'predicted_all_pairs.pkl.gz'
    blacklist_file = urls['funmap_blacklist']

    llr_res_file = results_dir / f'llr_results.tsv'
    edge_list_file = network_dir/ f'network.tsv'
    # llr obtained with each invividual dataset
    llr_dataset_file = results_dir / 'llr_dataset.tsv'
    all_feature_df = None
    gs_train = gs_test_pos = gs_test_neg = None

    feature_args = {
        'saved_data_dir': saved_data_dir,
        'data_config': cfg,
        'data_file': data_file,
        'min_sample_count': min_sample_count,
        'cor_type': cor_type,
        'feature_type': feature_type,
        'extra_feature_file': extra_feature_file,
        'n_jobs': n_jobs,
        'n_chunks': n_chunks
    }

    all_feature_df, valid_gene_list = prepare_features(**feature_args)
    gs_args = {
        'saved_data_dir': saved_data_dir,
        'all_feature_df': all_feature_df,
        'valid_gene_list': valid_gene_list,
        'min_feature_count': min_feature_count,
        'test_size': test_size,
        'seed': seed,
        'split_by': split_by
    }

    gs_train, gs_test_pos, gs_test_neg = prepare_gs_data(**gs_args)
    # check if models and predictions are available from previous run with the
    # same configuration
    if predicted_all_pairs_file.exists():
        log.info(f'Loading predicted all pairs from {predicted_all_pairs_file}')
        with gzip.open(predicted_all_pairs_file, 'rb') as fh:
            predicted_all_pairs = pickle.load(fh)
        log.info(f'Loading predicted all pairs ... done')
    else:
        if ml_model_file.exists():
            log.info(f'Loading model from {ml_model_file} ...')
            with gzip.open(ml_model_file, 'rb') as fh:
                ml_model = pickle.load(fh)
            log.info(f'Loading model ... done')
        else:
            ml_model = train_ml_model(gs_train, ml_type, feature_type, seed, n_jobs)
            with gzip.open(ml_model_file, 'wb') as fh:
                pickle.dump(ml_model, fh)

        log.info('Predicting for all pairs ...')
        predicted_all_pairs = predict_all_pairs(ml_model, all_feature_df,
                                                min_feature_count,
                                                filter_before_prediction,
                                                predicted_all_pairs_file)
        with gzip.open(predicted_all_pairs_file, 'wb') as fh:
            pickle.dump(predicted_all_pairs, fh)
        log.info('Predicting for all pairs ... done.')

    for i in predicted_all_pairs:
        predicted_all_pairs[i] = predicted_all_pairs[i].astype('float32')
    gs_test_pos_set = set(gs_test_pos.index)
    gs_test_neg_set = set(gs_test_neg.index)

    # check to see if all files in the llr_res_files list exist
    llr_res_exist = llr_res_file.exists()
    edge_list_file_exist = edge_list_file.exists()

    if llr_res_exist and edge_list_file_exist:
        log.info('validation results already exist.')
        validation_res = {
            'llr_res_file': llr_res_file,
            'edge_list_file': edge_list_file
        }
    else:
        log.info('Computing LLR with trained model ...')
        validation_res = validation_llr(all_feature_df, predicted_all_pairs,
                    filter_after_prediction, filter_criterion, filter_threshold,
                    filter_blacklist, blacklist_file,
                    max_num_edges, step_size, gs_test_pos_set,
                    gs_test_neg_set, results_dir, network_dir)
        log.info('Done.')

    if not llr_dataset_file.exists():
        log.info('Computing LLR for each dataset ...')
        step_size = int(start_edge_num / 10)
        llr_ds = dataset_llr(all_feature_df, gs_test_pos_set, gs_test_neg_set,
            start_edge_num, step_size, max_num_edges,  llr_dataset_file)
        log.info('Done.')
    else:
        llr_ds = pd.read_csv(llr_dataset_file, sep='\t')

    get_funmap(validation_res, cfg, network_dir)
    all_fig_names = []
    fig_names = plot_results(cfg, validation_res, llr_ds, gs_train, figure_dir)
    all_fig_names.extend(fig_names)

    merge_and_delete(figure_dir, all_fig_names, 'results.pdf')


if __name__ == '__main__':
    cli()
