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


# def get_amb_hparams(inception_hparams):
#     temp1 = inception_hparams.pretrained_gen_base.split('/')

#     temp1[-2] = 'hparams'
#     temp1 = '/'.join(temp1)
#     temp2 = inception_hparams.pretrained_gen_expt_dir
#     pkl_filepath = temp1 + temp2 + 'hparams.pkl'
#     amb_hparams = basic_utils.read_hparams(pkl_filepath)
#     return amb_hparams


def get_metrics(hparams):
    # ckpt_dir = hparams.incpt_dir
    incpt_pkl = hparams.incpt_pkl
    data = basic_utils.load_if_pickled(incpt_pkl)
    metric_names = [
        'inception_score_mean',
        # 'inception_score_std'
    ]
    metrics = {metric_name: data[metric_name] for metric_name in metric_names}
    return metrics


def get_df(inception_hparams_list, metrics_list):
    all_values_list = []
    for inception_hparams, metrics in zip(inception_hparams_list,
                                                       metrics_list):
        all_values = {}
        for key, value in inception_hparams.values().iteritems():
            all_values[key] = value
        # for key, value in amb_hparams.values().iteritems():
            # all_values['amb_' + key] = value
        for key, value in metrics.iteritems():
            all_values['m_' + key] = value

        all_values_list.append(all_values)

    df = pd.DataFrame(all_values_list)
    return df


def main():
    inception_root_dir = './results/hparams/'
    pkl_filepaths = get_pkl_filepaths(inception_root_dir)

    inception_hparams_list = [basic_utils.read_hparams(pkl_filepath) for pkl_filepath in pkl_filepaths]
    # # amb_hparams_list = [get_amb_hparams(inception_hparams) for inception_hparams in inception_hparams_list]
    metrics_list = [get_metrics(inception_hparams) for inception_hparams in inception_hparams_list]

    df = get_df(inception_hparams_list, metrics_list)
    df_pkl_path = './results/df_hparams_metrics.pkl'
    basic_utils.save_to_pickle(df, df_pkl_path)


if __name__ == '__main__':
    main()
