"""Export all valid current and temporal queries from the 10-class 6000-step model."""

_base_ = ['./prototype_stage1_r50_labeled6000_0920_export.py']

analysis_root = (
    'work_dirs/prototype_stage1_r50_labeled6000_10cls_0923/'
    'analysis_val_iter6000_fixed'
)

model = dict(
    fusion_bbox_head=dict(
        save_dir=analysis_root + '/features',
        active_proto_class_ids=list(range(10)),
        greedy_dist_th_by_class=[
            1.0, 1.2, 1.5, 1.5, 1.5,
            0.5, 0.5, 0.4, 0.35, 0.25,
        ],
        greedy_angle_class_ids=[0, 1, 2, 3, 4, 6, 7],
    ),
)

data = dict(samples_per_gpu=1, workers_per_gpu=2)
work_dir = analysis_root
