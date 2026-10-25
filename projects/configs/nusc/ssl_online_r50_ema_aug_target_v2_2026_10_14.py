"""Scheme B: preserve pseudo-target-guided image/LiDAR perturbations."""
_base_ = ['./ssl_online_r50_ema_aug_2026_10_14.py']
custom_imports = dict(imports=[
    'projects.mmdet3d_plugin.datasets.online_nuscenes_dataset_0919',
    'projects.mmdet3d_plugin.models.detectors.mv2dfusion_online_ema_aug_ab_v2_2026_10_14'],
    allow_failed_imports=False)
model = dict(type='MV2DFusionOnlineEMAAugABV220261014', augmentation_scope='target')
work_dir = '/root/autodl-tmp/data/mv2dfusion_results/ssl_aug_target_v2_2026-10-14'
