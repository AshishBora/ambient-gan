# # pylint: disable = C0103, C0111, C0301, R0913, R0903, R0914

# """Aggregate hparams and metrics for experiments"""

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


# def get_amb_hparams(hparams):
#     temp1 = hparams.pretrained_gen_base.split('/')
#     temp1[-2] = 'hparams'
#     temp1 = '/'.join(temp1)
#     temp2 = hparams.pretrained_gen_expt_dir
#     pkl_filepath = temp1 + temp2 + 'hparams.pkl'
#     amb_hparams = basic_utils.read_hparams(pkl_filepath)
#     return amb_hparams


def get_metrics(hparams):
    data = basic_utils.load_if_pickled(hparams.incpt_pkl)

    if not data:
        return None

    metrics = {'inception': data}
    metrics['max_iter'] = data[-1][0]
    metrics['final_score'] = data[-1][1][0]
    metrics['final_std'] = data[-1][1][1]

    # iters, scores = zip(*data)
    # mean, std = zip(*scores)
    # print iters
    # print mean
    # print std

    # for iter_, (mean, std) in data:
    #     print data

    # metrics = {}
    # for iter_ in iters:
    #     metrics['mean_{}'.format(iter_)] = mean

    # inception_score = data['inception_score']
    # metrics = {'inception_score' : inception_score}
    return metrics


def get_df(hparams_list, metrics_list):
    all_values_list = []
    for hparams, metrics in zip(hparams_list, metrics_list):
        all_values = {}
        for key, value in hparams.values().iteritems():
            all_values[key] = value
        for key, value in metrics.iteritems():
            all_values['m_'+ key] = value

        all_values_list.append(all_values)

    df = pd.DataFrame(all_values_list)
    return df


def main():
    inception_root_dir = './results/hparams/cifar10/'
    pkl_filepaths = get_pkl_filepaths(inception_root_dir)

    hparams_list = []
    metrics_list = []
    for pkl_filepath in pkl_filepaths:
        hparams = basic_utils.read_hparams(pkl_filepath)
        metrics = get_metrics(hparams)
        if metrics is not None:
            hparams_list.append(hparams)
            metrics_list.append(metrics)

    df = get_df(hparams_list, metrics_list)
    df_pkl_path = './results/df_cifar.pkl'
    basic_utils.save_to_pickle(df, df_pkl_path)


if __name__ == '__main__':
    main()
