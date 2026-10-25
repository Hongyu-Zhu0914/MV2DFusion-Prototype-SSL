"""Stage-0 labeled-data configuration for the offline SSL pipeline.

This config establishes a reproducible R50 baseline on the selected 15 train
logs.  It does not consume the official-test unlabeled split and does not yet
implement pseudo-label training.
"""

_base_ = [
    './mv2dfusion-fsd_freeze-r50_1600_gridmask-ep24_nusc.py',
]

ssl_pkl_root = '/root/autodl-tmp/data/nuscenes/ssl_splits/'

# Start conservatively because the current data-loader cgroup has only 2 GiB.
num_gpus = 1
batch_size = 1
num_iters_per_epoch = 13008 // (num_gpus * batch_size)
num_epochs = 1

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=0,
    train=dict(
        ann_file=ssl_pkl_root + 'nuscenes_ssl_train_labeled_15log.pkl',
    ),
    val=dict(
        ann_file=ssl_pkl_root + 'nuscenes_ssl_val_split.pkl',
    ),
    test=dict(
        ann_file=ssl_pkl_root + 'nuscenes_ssl_test_split.pkl',
    ),
)

# Full MV2DFusion-R50 detector baseline, not the Mask R-CNN backbone checkpoint.
load_from = 'weights/r50.pth'
resume_from = None

runner = dict(type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
evaluation = dict(interval=1000)
checkpoint_config = dict(interval=1000)

work_dir = 'work_dirs/ssl_offline_r50_labeled_stage0_0914'

