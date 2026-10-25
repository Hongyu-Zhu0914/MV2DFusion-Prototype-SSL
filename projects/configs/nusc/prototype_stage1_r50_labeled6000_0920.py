"""Restart the historical 0723/0819 stage-1 on the saved 6000-frame split."""
_base_ = ['./prototype_finetune_sinkhorn_stage1_0723_retrain_r50_0819.py']
custom_imports = dict(imports=[
    'projects.mmdet3d_plugin.models.dense_heads.prototype_stage1_head_0920',
    'projects.mmdet3d_plugin.models.detectors.prototype_only_detector_0920',
], allow_failed_imports=False)
model = dict(type='PrototypeOnlyDetector0920', baseline_checkpoint='weights/r50.pth',
             fusion_bbox_head=dict(type='PrototypeStage1Head0920',
                 prototype_save_dir='work_dirs/prototype_stage1_r50_labeled6000_0920/prototypes'))
data = dict(samples_per_gpu=1, workers_per_gpu=0,
            train=dict(ann_file='/root/autodl-tmp/data/nuscenes/ssl_splits/nuscenes_ssl_train_labeled_150scenes_6000_0919.pkl'),
            val=dict(ann_file='/root/autodl-tmp/data/nuscenes/ssl_splits/nuscenes_ssl_val_split.pkl'),
            test=dict(ann_file='/root/autodl-tmp/data/nuscenes/ssl_splits/nuscenes_ssl_test_split.pkl'))
load_from = None  # detector.init_weights validates and loads full R50 baseline itself
resume_from = None
runner = dict(type='IterBasedRunner', max_iters=5000)
optimizer_config = dict(_delete_=True, type='OptimizerHook', grad_clip=dict(max_norm=35, norm_type=2))
checkpoint_config = dict(interval=1000, by_epoch=False, save_last=True, max_keep_ckpts=5)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
work_dir = 'work_dirs/prototype_stage1_r50_labeled6000_0920'
# Dedicated entry trains with no evaluation hook; custom-val integration is separate.
