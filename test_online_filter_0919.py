import unittest
import torch
import projects.mmdet3d_plugin
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from projects.mmdet3d_plugin.models.detectors.mv2dfusion_online_ssl_0919 import select_pseudo


class FilterTest(unittest.TestCase):
    def test_strict_threshold_geometry_and_alignment(self):
        boxes = torch.tensor([[0., 0., 0., 1., 1., 1., 0., 0., 0.]]).repeat(7, 1)
        boxes[3, 0] = 100
        boxes[4, 3] = -1
        boxes[5, 0] = float('nan')
        pred = dict(boxes_3d=LiDARInstance3DBoxes(boxes, box_dim=9),
                    scores_3d=torch.tensor([.81, .8, .79, .99, .99, .99, .99]),
                    labels_3d=torch.tensor([0, 1, 2, 3, 4, 5, 10]))
        mask = select_pseudo(pred, .8, 10, [-54.4, -54.4, -5, 54.4, 54.4, 3])
        self.assertEqual(mask.tolist(), [True, False, False, False, False, False, False])
        pred['labels_3d'] = pred['labels_3d'][:1]
        with self.assertRaises(ValueError):
            select_pseudo(pred, .8, 10, [-54.4, -54.4, -5, 54.4, 54.4, 3])

    def test_empty(self):
        pred = dict(boxes_3d=LiDARInstance3DBoxes(torch.empty(0, 9), box_dim=9),
                    scores_3d=torch.empty(0), labels_3d=torch.empty(0, dtype=torch.long))
        self.assertEqual(select_pseudo(pred, .8, 10, [-1, -1, -1, 1, 1, 1]).numel(), 0)


if __name__ == '__main__':
    unittest.main()
