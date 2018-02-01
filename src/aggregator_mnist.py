# pylint: disable = C0103, C0111, C0301, R0913, R0903, R0914

"""Aggregate hparams and metrics for experiments"""

import os
import pandas as pd

from commons import basic_utils


def get_pkl_filepaths(root_dir):
    pkl_filepaths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == 'hparams.pkl':
                pkl_filepath = dirpath + '/' + filename
                pkl_filepaths.append(pkl_filepath)
    return pkl_filepaths


def get_metrics(hparams):
    incpt_pkl = hparams.incpt_pkl
    data = basic_utils.load_if_pickled(incpt_pkl)
    metric_names = [
        'inception_score_mean',
    ]
    metrics = {metric_name: data[metric_name] for metric_name in metric_names}
    return metrics


def get_all_values(hparams, metrics):
    all_values = {}
    for key, value in hparams.values().iteritems():
        all_values[key] = value
    for key, value in metrics.iteritems():
        all_values['m_' + key] = value
    return all_values


def get_df(hparams_list, metrics_list):
    all_values_list = [get_all_values(hparams, metrics) for (hparams, metrics) in zip(hparams_list, metrics_list)]
    df = pd.DataFrame(all_values_list)
    return df


def main():
    root_dir = './results/hparams/mnist/'
    pkl_filepaths = get_pkl_filepaths(root_dir)

    hparams_list = [basic_utils.read_hparams(pkl_filepath) for pkl_filepath in pkl_filepaths]
    metrics_list = [get_metrics(hparams) for hparams in hparams_list]

    df = get_df(hparams_list, metrics_list)
    df_pkl_path = './results/df_mnist.pkl'
    basic_utils.save_to_pickle(df, df_pkl_path)


if __name__ == '__main__':
    main()
