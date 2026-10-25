"""First EMA baseline. Not a proven optimal schedule; formal launch is manual."""
_base_ = ['./ssl_online_r50_fixed_teacher_0919.py']
custom_imports = dict(imports=[
    'projects.mmdet3d_plugin.datasets.online_nuscenes_dataset_0919',
    'projects.mmdet3d_plugin.models.detectors.mv2dfusion_online_ema_1011',
], allow_failed_imports=False)
model = dict(type='MV2DFusionOnlineEMA1011', teacher_update='ema', ema_decay=0.999)
# One optimizer attempt per frame, alternating labeled/unlabeled. 6000 of each.
# This traverses labeled data once, but does NOT traverse all 6008 unlabeled frames.
runner = dict(type='IterBasedRunner', max_iters=12000)
optimizer = dict(lr=1e-4)  # inherited img_backbone multiplier 0.1
lr_config = dict(warmup_iters=500)
checkpoint_config = dict(interval=1000, by_epoch=False, max_keep_ckpts=3)
custom_hooks = [dict(type='OnlineEMAHook1011', priority=45)]
evaluation = dict(interval=12000)
work_dir = 'work_dirs/ssl_online_r50_ema_1011/formal_train'
ssl_seed = 1011
ssl_final_evaluation = True
