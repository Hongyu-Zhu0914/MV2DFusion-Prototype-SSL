"""Export projected proposal features from the 5000-step prototype model."""
_base_ = ['./prototype_finetune_sinkhorn_stage1_0723_retrain_r50_0819.py']

analysis_root = (
    'work_dirs/prototype_stage1_r50_labeled6000_0920/'
    'analysis_val_iter5000'
)

model = dict(
    type='MV2DFusion',
    fusion_bbox_head=dict(
        type='PrototypeMV2DFusionHead',
        save_dir=analysis_root + '/features',
    ),
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    test=dict(
        ann_file=(
            '/root/autodl-tmp/data/nuscenes/ssl_splits/'
            'nuscenes_ssl_val_split.pkl'
        ),
        test_mode=True,
    ),
)

load_from = None
resume_from = None
work_dir = analysis_root
