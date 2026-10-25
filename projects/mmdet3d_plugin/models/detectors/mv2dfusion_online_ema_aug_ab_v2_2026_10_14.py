"""Selects a global or target-level perturbation scope; EMA logic is inherited."""
from mmdet.models import DETECTORS
from .mv2dfusion_online_ema_aug_2026_10_14 import MV2DFusionOnlineEMAAug20261014
from ..utils.student_augment_ab_v2_2026_10_14 import StudentPerturbationsAB20261014V2


@DETECTORS.register_module()
class MV2DFusionOnlineEMAAugABV220261014(MV2DFusionOnlineEMAAug20261014):
    def __init__(self, augmentation_scope='target', global_augmentation=None, **kwargs):
        super().__init__(**kwargs)
        self.student_perturbations = StudentPerturbationsAB20261014V2(
            self.student_perturbations.cfg, self.point_cloud_range,
            scope=augmentation_scope, global_cfg=global_augmentation)
