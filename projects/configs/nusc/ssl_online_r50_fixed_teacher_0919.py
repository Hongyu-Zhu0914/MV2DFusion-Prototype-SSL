"""Scaffold: fixed online teacher, same geometry, no temporal memory.

Uses the saved 150-whole-scene selection (6000 labeled + 6008 unlabeled).
Use the dedicated smoke script before deciding a formal training schedule.
"""
_base_ = ['./ssl_offline_r50_labeled_stage0_0914.py']
num_iters_per_epoch = 12008  # nominal dataset-length reference, not a sampling-ratio guarantee
custom_imports = dict(imports=[
    'projects.mmdet3d_plugin.datasets.online_nuscenes_dataset_0919',
    'projects.mmdet3d_plugin.models.detectors.mv2dfusion_online_ssl_0919',
], allow_failed_imports=False)
model = dict(
    _delete_=True, type='MV2DFusionOnlineSSL0919',
    detector_config='projects/configs/nusc/ssl_offline_r50_labeled_stage0_0914.py',
    initial_checkpoint='weights/r50.pth', score_threshold=0.8,
    pseudo_loss_weight=0.25, teacher_update='fixed', prototype_filter=False)
data = dict(samples_per_gpu=1, workers_per_gpu=0, train=dict(
    type='OnlineNuScenesDataset0919',
    ann_file='/root/autodl-tmp/data/nuscenes/ssl_splits/nuscenes_ssl_train_labeled_150scenes_6000_0919.pkl',
    unlabeled_ann_file='/root/autodl-tmp/data/nuscenes/ssl_splits/nuscenes_ssl_train_unlabeled_official_test.pkl',
    seq_mode=True, seq_split_num=2, filter_empty_gt=False))
# Wrapper loads both networks explicitly. Resuming requires a wrapper checkpoint,
# not a historical single-model baseline or prototype checkpoint.
load_from = None
resume_from = None
# A bounded scaffold default, not a selected scientific training schedule.
runner = dict(type='IterBasedRunner', max_iters=2)
evaluation = dict(interval=1000)
checkpoint_config = dict(interval=1000)
# FP32 is verified on the 24GB card. AMP's GradScaler needs a special empty-step
# hook before enabling it: an all-skipped pseudo batch has no parameter grads.
optimizer_config = dict(_delete_=True, type='OptimizerHook',
                        grad_clip=dict(max_norm=35, norm_type=2))
work_dir = 'work_dirs/ssl_online_r50_fixed_teacher_0919'
