"""Raw-unlabeled input for online SSL; no offline pseudo annotations are used."""
import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from .pseudo_nuscenes_dataset_0915 import PseudoAwareNuScenesDataset0915


@DATASETS.register_module()
class OnlineNuScenesDataset0919(PseudoAwareNuScenesDataset0915):
    def __init__(self, unlabeled_ann_file, *args, **kwargs):
        if 'pseudo_ann_file' in kwargs:
            raise ValueError('Use raw unlabeled_ann_file, not an offline pseudo PKL')
        super().__init__(*args, pseudo_ann_file=unlabeled_ann_file,
                         require_pseudo_flag=False, **kwargs)
        if self.queue_length != 1 or not self.seq_mode or self.filter_empty_gt:
            raise ValueError('Online v1 requires seq_mode=True, queue_length=1, filter_empty_gt=False')
        for info in self.data_infos[self.supervised_length:]:
            # Only placeholders for the shared data pipeline. Teacher predictions
            # replace them at runtime; any annotations on disk are ignored.
            for key, shape, dtype in (
                ('bboxes2d', (0, 4), np.float32),
                ('labels2d', (0,), np.int64),
                ('bboxes3d_cams', (0, 7), np.float32),
                ('centers2d', (0, 2), np.float32),
                ('depths', (0,), np.float32),
                ('bboxes_ignore', (0, 4), np.float32),
            ):
                info[key] = [np.empty(shape, dtype=dtype) for _ in range(6)]

    def get_ann_info(self, index):
        if index < self.supervised_length:
            return super().get_ann_info(index)
        return dict(gt_bboxes_3d=LiDARInstance3DBoxes(np.empty((0, 9), np.float32), box_dim=9),
                    gt_labels_3d=np.empty((0,), np.int64),
                    gt_names=np.empty((0,), dtype=str))
