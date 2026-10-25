"""Opt-in MV2DFusion detector branch for offline pseudo-label training.

Real labelled samples use the untouched :class:`MV2DFusion` implementation.
Pseudo samples keep image-proposal inference and multimodal fusion, but do not
train the 2D detector, do not add 2D GT proposals, and do not use mono/depth GT
auxiliary losses.  The module is registered only when explicitly imported by
the dated SSL configuration.
"""

import torch

from mmdet.core import bbox2roi
from mmdet.models import DETECTORS

from .mv2dfusion import MV2DFusion


@DETECTORS.register_module()
class MV2DFusionOfflineSSL0915(MV2DFusion):
    """MV2DFusion with a conservative offline-pseudo training branch."""

    def __init__(self, pseudo_loss_weight=0.25, *args, **kwargs):
        if pseudo_loss_weight <= 0:
            raise ValueError("pseudo_loss_weight must be positive")
        self.pseudo_loss_weight = float(pseudo_loss_weight)
        super().__init__(*args, **kwargs)

    @staticmethod
    def _pseudo_batch_flag(img_metas):
        flags = [bool(meta.get("is_pseudo", False)) for meta in img_metas]
        if flags and any(flag != flags[0] for flag in flags):
            raise RuntimeError(
                "mixed real/pseudo samples in one batch are unsupported; "
                "use samples_per_gpu=1"
            )
        return bool(flags and flags[0])

    def forward_pts_train(self,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_bboxes,
                          gt_labels,
                          img_metas,
                          centers2d,
                          depths,
                          requires_grad=True,
                          return_losses=False,
                          **data):
        is_pseudo = self._pseudo_batch_flag(img_metas)
        if not is_pseudo:
            return super().forward_pts_train(
                gt_bboxes_3d=gt_bboxes_3d,
                gt_labels_3d=gt_labels_3d,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                img_metas=img_metas,
                centers2d=centers2d,
                depths=depths,
                requires_grad=requires_grad,
                return_losses=return_losses,
                **data,
            )

        if not requires_grad:
            raise NotImplementedError(
                "offline pseudo branch currently requires seq_mode=True/queue_length=1"
            )
        if not self.with_img_roi_head:
            raise RuntimeError("pseudo branch requires the image ROI head for image queries")

        # Keep these keys because the stock query generator expects them while
        # training.  Pseudo PKLs intentionally provide six empty 2D arrays, so
        # its depth matching returns a zero auxiliary term rather than treating
        # pseudo 3D boxes as fabricated 2D supervision.
        data["gt_bboxes"] = gt_bboxes
        data["gt_labels"] = gt_labels
        data["gt_bboxes_3d"] = gt_bboxes_3d
        data["gt_labels_3d"] = gt_labels_3d
        data["depths"] = depths

        imgs_det, feats_det, img_metas_det, imgs_shape = \
            self.prepare_detection_data(img_metas, **data)
        batch_size, num_views = imgs_shape[:2]

        # Preserve the pretrained 2D detector as a proposal generator.  Unlike
        # the supervised branch, this does not call forward_roi_head_train and
        # does not complement detections with 2D GT boxes.
        self.eval()
        with torch.no_grad():
            _, dets = self.forward_roi_head(imgs_det, feats_det, img_metas_det)
            if self.use_2d_proposal:
                if "proposals" not in data:
                    raise KeyError("use_2d_proposal=True but no proposals were provided")
                dets = sum([item for item in data["proposals"]], [])
            if sum(len(proposal) for proposal in dets) == 0:
                proposal = torch.tensor(
                    [[10, 20, 30, 40, 0, 1]],
                    dtype=dets[0].dtype,
                    device=dets[0].device,
                )
                dets = [proposal] + dets[1:]
            rois = bbox2roi(dets)
        self.train()

        roi_feats = self.extract_roi_feats(feats_det, rois, **data)
        n_rois_per_view = [len(proposal) for proposal in dets]
        n_rois_per_batch = [
            sum(n_rois_per_view[index * num_views:(index + 1) * num_views])
            for index in range(batch_size)
        ]
        dyn_query, dyn_feats_pred = self.img_query_generator(
            roi_feats,
            dets,
            img_metas_det,
            n_rois_per_view=n_rois_per_view,
            n_rois_per_batch=n_rois_per_batch,
            data=data,
        )

        fsd_gt_bboxes_3d, fsd_gt_labels_3d = [], []
        for batch_index in range(batch_size):
            box, label = self.convert_to_fsd_anno(
                gt_bboxes_3d[batch_index], gt_labels_3d[batch_index]
            )
            fsd_gt_bboxes_3d.append(box)
            fsd_gt_labels_3d.append(label)

        out_dict = self.pts_backbone.forward_train(
            data["pts_feats"],
            img_metas,
            fsd_gt_bboxes_3d,
            fsd_gt_labels_3d,
        )
        pts_feat, pts_pos, pts_query_feat, pts_query_center = self.pts_query_generator(
            out_dict["voxel_feats"],
            out_dict["voxel_coors"],
            out_dict["voxel_xyz"],
            out_dict["query_feats"],
            out_dict["query_xyz"],
            out_dict["query_pred"],
            out_dict["query_cat"],
            batch_size,
        )
        outs = self.fusion_bbox_head(
            img_metas,
            dyn_query=dyn_query,
            dyn_feats=dyn_feats_pred,
            pts_query_center=pts_query_center,
            pts_query_feat=pts_query_feat,
            pts_feat=pts_feat,
            pts_pos=pts_pos,
            pts_shape=None,
            **data,
        )

        if not return_losses:
            return None

        # Pseudo supervision is deliberately restricted to the point detector
        # and fused 3D detector.  No det2d.* or imgqg.* auxiliary losses are
        # emitted for pseudo samples.
        losses = self.fusion_bbox_head.loss(gt_bboxes_3d, gt_labels_3d, outs)
        for key, value in out_dict["losses"].items():
            if "loss" in key:
                value = value * self.loss_weight_pts
            losses["pts." + key] = value

        for key, value in list(losses.items()):
            if "loss" in key:
                losses[key] = value * self.loss_weight_3d * self.pseudo_loss_weight
        return losses
