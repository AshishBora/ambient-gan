# # pylint: disable = C0103, C0111, C0301, R0913, R0903, R0914

# """Functions to compute expt_dir from hparams"""

def get_task_dir(hparams):

    if hparams.measurement_type in ['drop_independent', 'drop_row', 'drop_col', 'drop_rowcol']:
        task_dir = '{}_{}_p{}/'.format(
            hparams.dataset,
            hparams.measurement_type,
            hparams.drop_prob,
        )
    elif hparams.measurement_type in ['drop_patch', 'keep_patch', 'extract_patch']:
        task_dir = '{}_{}_k{}/'.format(
            hparams.dataset,
            hparams.measurement_type,
            hparams.patch_size,
        )
    elif hparams.measurement_type in ['blur_addnoise']:
        task_dir = '{}_{}_br{}_bfs{}_anstd{}/'.format(
            hparams.dataset,
            hparams.measurement_type,
            hparams.blur_radius,
            hparams.blur_filter_size,
            hparams.additive_noise_std,
        )
    elif hparams.measurement_type in ['pad_rotate_project', 'pad_rotate_project_with_theta']:
        task_dir = '{}_{}_na{}/'.format(
            hparams.dataset,
            hparams.measurement_type,
            hparams.num_angles,
        )
    else:
        raise NotImplementedError

    return task_dir


def get_mode_dir(hparams):

    if hparams.train_mode in ['ambient', 'baseline']:
        mode_dir = '{}/'.format(
            hparams.train_mode,
        )
    elif hparams.train_mode == 'unmeasure':
        mode_dir = '{}_{}/'.format(
            hparams.train_mode,
            hparams.unmeasure_type,
        )
    else:
        raise NotImplementedError

    return mode_dir


# def get_model_dir(hparams):

#     if hparams.model_type == 'dcgan':
#         model_dir = '{}_{}_{}_zd{}/'.format(
#             hparams.model_class,
#             hparams.model_type,
#             hparams.z_dist,
#             hparams.z_dim,
#         )
#     elif hparams.model_type == 'wgangp':
#         model_dir = '{}_{}_{}_zd{}_gpl{}/'.format(
#             hparams.model_class,
#             hparams.model_type,
#             hparams.z_dist,
#             hparams.z_dim,
#             hparams.gp_lambda,
#         )
#     elif hparams.model_type == 'acwgangp':
#         model_dir = '{}_{}_{}_zd{}_gpl{}_dacl{}_gacl{}/'.format(
#             hparams.model_class,
#             hparams.model_type,
#             hparams.z_dist,
#             hparams.z_dim,
#             hparams.gp_lambda,
#             hparams.d_ac_lambda,
#             hparams.g_ac_lambda,
#         )
#     else:
#         raise NotImplementedError

#     return model_dir


# def get_opt_dir(hparams):

#     if hparams.lr_decay == 'false':
#         opt_dir = '{}_bs{}_glr{}_dlr{}_{}_p{}_p{}_gi{}_di{}/'.format(
#             hparams.opt_type,
#             hparams.batch_size,
#             hparams.g_lr,
#             hparams.d_lr,
#             hparams.lr_decay,
#             hparams.opt_param1,
#             hparams.opt_param2,
#             hparams.g_iters,
#             hparams.d_iters,
#         )
#     elif hparams.lr_decay == 'linear':
#         opt_dir = '{}_bs{}_glr{}_dlr{}_{}{}_p{}_p{}_gi{}_di{}/'.format(
#             hparams.opt_type,
#             hparams.batch_size,
#             hparams.g_lr,
#             hparams.d_lr,
#             hparams.lr_decay,
#             hparams.linear_decay_max_iter,
#             hparams.opt_param1,
#             hparams.opt_param2,
#             hparams.g_iters,
#             hparams.d_iters,
#         )
#     return opt_dir


def get_expt_dir(hparams):
    task_dir = get_task_dir(hparams)
    mode_dir = get_mode_dir(hparams)
    # model_dir = get_model_dir(hparams)
    # opt_dir = get_opt_dir(hparams)
    expt_dir = task_dir + mode_dir
    # + model_dir + opt_dir
    return expt_dir
