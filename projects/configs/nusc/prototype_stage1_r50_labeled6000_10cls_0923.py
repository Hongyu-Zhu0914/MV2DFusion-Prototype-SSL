"""Full nuScenes 10-class prototype stage 1 on the labeled 6000-frame split.

Keep the original 0920 seven-class experiment intact for comparison.
The detector is initialized from the same R50 baseline and remains frozen.
"""

_base_ = ['./prototype_stage1_r50_labeled6000_0920.py']

model = dict(
    fusion_bbox_head=dict(
        # nuScenes order: car, truck, construction_vehicle, bus, trailer,
        # barrier, motorcycle, bicycle, pedestrian, traffic_cone.
        active_proto_class_ids=list(range(10)),
        greedy_dist_th_by_class=[
            1.0, 1.2, 1.5, 1.5, 1.5,
            0.5, 0.5, 0.4, 0.35, 0.25,
        ],
        # Heading matters for vehicles; barrier, pedestrian, and cone are
        # matched by class, confidence, and center distance without yaw gating.
        greedy_angle_class_ids=[0, 1, 2, 3, 4, 6, 7],
        prototype_save_dir=(
            'work_dirs/prototype_stage1_r50_labeled6000_10cls_0923/prototypes'
        ),
    ),
)

# With one GPU and one sample per step, the grouped sampler visits each of
# the 6000 training frames exactly once before repeating.
runner = dict(type='IterBasedRunner', max_iters=6000)

work_dir = 'work_dirs/prototype_stage1_r50_labeled6000_10cls_0923'

# The dedicated training entry invokes subset-aware nuScenes evaluation after
# each saved checkpoint. The inherited stock evaluation setting alone does not
# register an EvalHook and would reject this 3014-frame custom val split.
custom_subset_validation = dict(
    interval=1000,
    expected_samples=3014,
    # R50 detector is frozen; intermediate full-set mAP/NDS is redundant.
    # Keep per-1000 checkpoint/integrity audits and validate the final model.
    steps=[6000],
)
