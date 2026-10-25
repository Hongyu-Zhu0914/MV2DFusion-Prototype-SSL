"""Independent labeled-only Stage 2: one finite pass, 6000 new optimizer steps."""
_base_ = ['./prototype_stage1_r50_labeled6000_10cls_0923.py']

custom_imports = dict(imports=[
    'projects.mmdet3d_plugin.models.dense_heads.prototype_stage1_head_0920',
    'projects.mmdet3d_plugin.models.detectors.prototype_only_detector_0920',
    'projects.mmdet3d_plugin.models.dense_heads.prototype_stage2_adaptive_head_20261023',
    'projects.mmdet3d_plugin.models.detectors.prototype_only_stage2_20261025',
], allow_failed_imports=False)

work_dir = 'work_dirs/prototype_stage2_r50_labeled6000_10cls_20261025'
model = dict(
    type='PrototypeOnlyStage2Detector20261025',
    baseline_checkpoint=None,  # Override inherited R50 label; Stage 2 loads full Stage 1.
    stage1_checkpoint=(
        'work_dirs/prototype_stage1_r50_labeled6000_10cls_0923/checkpoints/iter_6000.pth'),
    fusion_bbox_head=dict(
        type='PrototypeStage2AdaptiveHead20261023',
        proto_loss_weight=0.5, gamma=0.99,
        proto_inter_upper=0.4, proto_inter_weight=0.01,
        neg_proto_margin=0.7, neg_proto_gt_dist_th=2.0,
        neg_proto_loss_weight=0.05, neg_proto_dynamic_weight=True,
        neg_proto_loss_weight_min=0.005, neg_proto_loss_weight_max=0.10,
        neg_proto_weight_step=0.02,
        neg_proto_monitor_interval=500, neg_proto_weight_update_interval=1000,
        neg_proto_min_mean_improvement=0.01, neg_proto_min_ratio_improvement=0.05,
        neg_proto_monitor_patience=1, neg_proto_positive_floor=0.80,
        neg_proto_positive_class_floor=0.80, neg_proto_positive_drop_tolerance=0.03,
        neg_proto_min_monitor_samples=20,
        prototype_save_dir=None,  # The entry saves small state snapshots after optimizer.step.
    ),
)

optimizer = dict(_delete_=True, type='AdamW', lr=2e-4, weight_decay=0.01)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(_delete_=True, policy='CosineAnnealing', warmup='linear',
                 warmup_iters=500, warmup_ratio=1.0 / 3, min_lr_ratio=1e-3)
runner = dict(type='IterBasedRunner', max_iters=6000)
checkpoint_config = dict(_delete_=True, interval=1000, max_keep_ckpts=3)
custom_subset_validation = None  # Detector frozen; no repeated full-set detection validation.
load_from = None
resume_from = None
stage2_training = dict(seed=920, expected_frames=6000, expected_scenes=150,
                       audit_interval=100, log_interval=50,
                       full_detector_evaluation_during_training=False)
