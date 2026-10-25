"""Conservative candidate parameters, not experimentally validated best settings."""
_base_ = ['./ssl_online_r50_ema_1011.py']
custom_imports = dict(imports=[
    'projects.mmdet3d_plugin.datasets.online_nuscenes_dataset_0919',
    'projects.mmdet3d_plugin.models.detectors.mv2dfusion_online_ema_aug_2026_10_14',
], allow_failed_imports=False)
model = dict(type='MV2DFusionOnlineEMAAug20261014', augment_labeled=False,
             disable_student_gridmask=True,
             student_augmentation=dict(enabled=True,
                 photo_probability=0.5, photo_views=2,
                 brightness=(0.9, 1.1), contrast=(0.9, 1.1), saturation=(0.9, 1.1),
                 object_probability=0.3, max_objects=8, image_modality_probability=0.5,
                 patch_area=(0.05, 0.10), max_image_coverage=0.10,
                 noise_sigma=0.005, noise_clip=0.02, dropout_ratio=(0.05, 0.10),
                 min_object_points=20, min_remaining_points=16,
                 jitter_probability=0.5, blur_sigma=(0.5, 0.8), mosaic_blocks=8))
work_dir = '/root/autodl-tmp/data/mv2dfusion_results/ssl_online_r50_ema_aug_2026-10-14'
# The original 12000-attempt schedule is inherited for comparison, not launched.
# Start from the baseline for a fresh comparison; no automatic resume from old run.
resume_from = None
load_from = None
