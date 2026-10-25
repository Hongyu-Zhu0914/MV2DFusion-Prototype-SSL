"""Confirmed global student perturbations, 2026-10-19.

Uses the verified 2026-10-14 implementation. Select this configuration for
the agreed single strategy; no pseudo-label coordinate perturbation.
"""
_base_ = ['./ssl_online_r50_ema_aug_global_v2_2026_10_14.py']

model = dict(
    augmentation_scope='global',
    augment_labeled=False,
    disable_student_gridmask=True,
    student_augmentation=dict(
        enabled=True,
        photo_probability=0.5, photo_views=6,
        brightness=(0.9, 1.1), contrast=(0.9, 1.1), saturation=(0.9, 1.1),
        max_image_coverage=0.10, mosaic_blocks=8, blur_sigma=(0.5, 0.8),
        noise_sigma=0.005, noise_clip=0.02,
        dropout_ratio=(0.05, 0.10), min_remaining_points=16,
        jitter_probability=0.5,
    ),
    global_augmentation=dict(
        image_probability=0.5, image_views=6, photo_views=6,
        image_patch_area=(0.05, 0.10), point_probability=0.5,
    ),
)

work_dir = '/root/autodl-tmp/data/mv2dfusion_results/ssl_aug_global_2026-10-19'
resume_from = None
load_from = None
