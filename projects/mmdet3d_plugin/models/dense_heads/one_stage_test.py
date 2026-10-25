import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmdet.models import HEADS
from mmdet.core import multi_apply
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core import nms_bev
from mmdet3d.core.bbox.structures import xywhr2xyxyr
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
from projects.mmdet3d_plugin.models.utils.positional_encoding import pos2posemb3d

import os
import numpy as np

from .sinkhorn import distributed_sinkhorn
from .contrast import momentum_update, l2_normalize, ProjectionHead
from .mv2dfusion_head import MV2DFusionHead


@HEADS.register_module()
class PrototypeMV2DFusionHead(MV2DFusionHead):
    def __init__(self,
                 num_classes,
                 in_channels=256,
                 embed_dims=256,
                 num_query=100,
                 num_reg_fcs=2,
                 memory_len=6 * 256,
                 topk_proposals=256,
                 num_propagated=256,
                 with_dn=True,
                 with_ego_pos=True,
                 match_with_velo=True,
                 match_costs=None,
                 sync_cls_avg_factor=False,
                 code_weights=None,
                 bbox_coder=None,
                 transformer=None,
                 normedlinear=False,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 train_cfg=dict(
                     assigner=dict(
                         type='HungarianAssigner3D',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(
                             type='IoUCost', iou_mode='giou', weight=2.0)), ),
                 test_cfg=dict(max_per_img=100),
                 scalar=5,
                 noise_scale=0.4,
                 noise_trans=0.0,
                 dn_weight=1.0,
                 split=0.5,
                 prob_bin=50,
                 post_bev_nms_thr=0.25,  # 0.2
                 post_bev_nms_score=0.0,
                 post_bev_nms_ops=[],
                 init_cfg=None,
                 debug=False,
                 num_prototype=10,
                 update_prototype=True,
                 gamma=0.999,
                 warm_up=2000,
                 sim_thr=0.7,
                 score_thr=0.1,
                 proto_loss_weight=0.1,
                 save_dir=None,  # 新增参数
                 **kwargs):

        # Training-only prototype options may be present in dated configs.
        # The export head does not use them, and the base detector head does
        # not accept them, so consume them here for test-time compatibility.
        test_compatibility_keys = (
            'proto_intra_lower',
            'proto_intra_upper',
            'proto_intra_weight',
            'sinkhorn_min_samples',
            'sinkhorn_queue_size',
            'greedy_dist_th',
            'greedy_dist_th_by_class',
            'greedy_conf_thr',
            'greedy_angle_thr',
            'greedy_angle_class_ids',
            'active_proto_class_ids',
            'neg_proto_margin',
            'neg_proto_loss_weight',
            'prototype_save_dir',
            'prototype_save_interval',
        )
        for compatibility_key in test_compatibility_keys:
            kwargs.pop(compatibility_key, None)

        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            embed_dims=embed_dims,
            num_query=num_query,
            num_reg_fcs=num_reg_fcs,
            memory_len=memory_len,
            topk_proposals=topk_proposals,
            num_propagated=num_propagated,
            with_dn=with_dn,
            with_ego_pos=with_ego_pos,
            match_with_velo=match_with_velo,
            match_costs=match_costs,
            sync_cls_avg_factor=sync_cls_avg_factor,
            code_weights=code_weights,
            bbox_coder=bbox_coder,
            transformer=transformer,
            normedlinear=normedlinear,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_iou=loss_iou,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            scalar=scalar,
            noise_scale=noise_scale,
            noise_trans=noise_trans,
            dn_weight=dn_weight,
            split=split,
            prob_bin=prob_bin,
            post_bev_nms_thr=post_bev_nms_thr,
            post_bev_nms_score=post_bev_nms_score,
            post_bev_nms_ops=post_bev_nms_ops,
            init_cfg=init_cfg,
            debug=debug,
            **kwargs
        )

        self.num_prototype = num_prototype
        self.update_prototype = update_prototype
        self.gamma = gamma
        self.warm_up = warm_up
        self.sim_thr = sim_thr
        self.score_thr = score_thr
        self.proto_loss_weight = proto_loss_weight

        self.register_buffer('local_iter', torch.zeros(1))
        self.prototypes = nn.Parameter(
            torch.zeros(self.num_classes, self.num_prototype, self.embed_dims),
            requires_grad=False)
        nn.init.trunc_normal_(self.prototypes, std=0.02)

        self.proto_proj = ProjectionHead(dim_in=self.embed_dims, proj_dim=self.embed_dims)
        self.feat_norm = nn.LayerNorm(self.embed_dims)

        self.sample_counter = 0
        self.saved_samples = []

        # ========== 设置保存目录 ==========
        if save_dir is None:
            self.save_dir = '/root/MV2DFusion-main/work_dirs/prototype_mv2dfusion-fsd_freeze-r50_1600_gridmask-ep24_nusc/features/legacy_nms_features_all'
        else:
            self.save_dir = save_dir
        # =================================

    def get_saved_samples(self):
        """返回保存的样本信息并清空"""
        data = self.saved_samples.copy()
        self.saved_samples = []
        return data

    def prototype_learning(self, out_feat, gt_labels, feat_proto_sim, pred_scores):
        label = gt_labels
        proto_logits = feat_proto_sim
        proto_target = label.clone().float()
        protos = self.prototypes.data.clone()
        pred_seg = pred_scores.argmax(dim=-1)
        correct_mask = (pred_seg == gt_labels)
        unique_labels = torch.unique(label)

        for k in unique_labels:
            k = k.long().item()
            if k >= self.num_classes: continue

            feat_proto_sim_reshape = feat_proto_sim.view(-1, self.num_classes, self.num_prototype)
            init_q = feat_proto_sim_reshape[:, k, :]
            class_mask = (label == k)
            init_q_k = init_q[class_mask, :]

            if init_q_k.shape[0] == 0: continue

            q, indexs = distributed_sinkhorn(init_q_k)
            c_k = out_feat[class_mask, ...]
            proto_target[class_mask] = indexs.float() + (self.num_prototype * k)

            correct_k = correct_mask[class_mask]
            if correct_k.sum() == 0:
                continue

            q_correct = q[correct_k, :]
            c_k_correct = c_k[correct_k, :]
            f = q_correct.t() @ c_k_correct
            n = q_correct.sum(dim=0)

            if torch.sum(n) > 0 and self.update_prototype:
                f = l2_normalize(f)
                active_mask = (n != 0)
                if active_mask.any():
                    updated_proto = momentum_update(
                        old_value=protos[k, active_mask, :],
                        new_value=f[active_mask, :],
                        momentum=self.gamma,
                        iter=self.local_iter,
                        warm_up=self.warm_up
                    )
                    protos[k, active_mask, :] = updated_proto

            assigned_proto_ids = indexs.float() + (self.num_prototype * k)
            proto_target[class_mask] = assigned_proto_ids

        self.prototypes.data.copy_(l2_normalize(protos))
        return proto_logits, proto_target

    def forward(self, img_metas, dyn_query=None, dyn_feats=None,
                pts_query_center=None, pts_query_feat=None, pts_feat=None, pts_pos=None, **data):
        # ==========================================================
        # 1. 数据准备
        # ==========================================================
        self.pre_update_memory(data)

        intrinsics = data['intrinsics'] / 1e3
        extrinsics = data['extrinsics'][..., :3, :]

        mln_input = torch.cat([intrinsics[..., 0, 0:1], intrinsics[..., 1, 1:2], extrinsics.flatten(-2)], dim=-1)
        mln_input = mln_input.flatten(0, 1).unsqueeze(1)
        mlvl_feats = data['img_feats_for_det']
        B, N, _, _, _ = mlvl_feats[0].shape

        # ==================== 1.1 图像特征处理 ====================
        feat_flatten_img = []
        spatial_flatten_img = []
        for i in range(1, len(mlvl_feats)):
            B, N, C, H, W = mlvl_feats[i].shape
            mlvl_feat = mlvl_feats[i].reshape(B * N, C, -1).transpose(1, 2)
            mlvl_feat = self.spatial_alignment(mlvl_feat, mln_input)
            feat_flatten_img.append(mlvl_feat.to(torch.float))
            spatial_flatten_img.append((H, W))

        feat_flatten_img = torch.cat(feat_flatten_img, dim=1)
        spatial_flatten_img = torch.as_tensor(spatial_flatten_img, dtype=torch.long, device=mlvl_feats[0].device)
        level_start_index_img = torch.cat(
            (spatial_flatten_img.new_zeros((1,)), spatial_flatten_img.prod(1).cumsum(0)[:-1]))

        # ==================== 1.2 点云特征处理 ====================
        feat_flatten_pts = self.pts_embed(pts_feat)
        pos_flatten_pts = pts_pos

        # ==================== 1.3 查询生成 ====================
        reference_points, query_coords, query_probs, query_feats, query_mask = \
            self.gen_dynamic_query(self.reference_points.weight, dyn_query, dyn_feats.get('query_feats', None))

        pts_ref = self.gen_pts_query(pts_query_center)
        query_mask = torch.cat([torch.ones_like(pts_ref[..., 0]).bool(), query_mask], dim=1)
        reference_points = torch.cat([pts_ref, reference_points], dim=1)

        num_query_img = int(self.num_query - pts_ref.size(1))
        num_query_pts = pts_ref.size(1)

        # ==================== 1.4 去噪训练准备 ====================
        reference_points, attn_mask, mask_dict = self.prepare_for_dn(B, reference_points, img_metas)

        # ==================== 1.5 注意力掩码准备 ====================
        tgt_size = self.num_query + self.num_propagated
        src_size = self.num_query + self.memory_len

        if attn_mask is None:
            attn_mask = torch.zeros((tgt_size, src_size), dtype=torch.bool, device=reference_points.device)

        pad_size = attn_mask.size(0) - tgt_size
        if mask_dict is not None:
            assert pad_size == mask_dict['pad_size']

        attn_mask = attn_mask.repeat(B, 1, 1)

        tgt_query_mask = torch.cat([query_mask, self.memory_query_mask[:, :self.num_propagated, 0]], dim=1)
        src_query_mask = torch.cat([query_mask, self.memory_query_mask[:, :, 0]], dim=1)

        attn_mask[:, :, pad_size:] = ~src_query_mask[:, None]

        num_heads = self.transformer.decoder.layers[0].attentions[0].num_heads
        attn_mask = attn_mask.repeat_interleave(num_heads, dim=0)

        # ==================== 1.6 查询特征准备 ====================
        tgt = self.dyn_q_embed.weight.repeat(B, num_query_img, 1)
        pts_tgt = self.pts_q_embed.weight.repeat(B, num_query_pts, 1)
        tgt = torch.cat([tgt.new_zeros((B, pad_size, self.embed_dims)), pts_tgt, tgt], dim=1)

        pad_query_feats = query_feats.new_zeros([B, pad_size + self.num_query, self.embed_dims])
        pts_query_feat = self.pts_query_embed(pts_query_feat)
        pad_query_feats[:, pad_size:pad_size + num_query_pts] = pts_query_feat
        pad_query_feats[:, pad_size + num_query_pts:pad_size + self.num_query] = query_feats
        tgt = self.dyn_q_enc(tgt, pad_query_feats)

        # ==================== 1.7 位置编码 ====================
        query_pos = self.query_embedding(pos2posemb3d(reference_points))

        # ==================== 1.8 时序对齐 ====================
        tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose = \
            self.temporal_alignment(query_pos, tgt, reference_points)

        # ==================== 1.9 动态查询位置编码 ====================
        query_pos_det = self.dyn_q_pos(query_coords.flatten(-2, -1))
        query_pos_det = self.dyn_q_pos_with_prob(query_pos_det, query_probs)
        query_pos[:, pad_size + num_query_pts:pad_size + self.num_query] = query_pos_det

        dyn_q_mask = torch.zeros_like(tgt[..., 0]).bool()
        dyn_q_mask[:, pad_size + num_query_pts:pad_size + self.num_query] = 1
        dyn_q_mask[:, pad_size:] &= tgt_query_mask
        dyn_q_mask_img = dyn_q_mask[:, pad_size + num_query_pts:pad_size + self.num_query]

        dyn_q_coords = query_coords[dyn_q_mask_img]
        dyn_q_probs = query_probs[dyn_q_mask_img]

        # ==========================================================
        # 2. Transformer 计算
        # ==========================================================
        outs_dec, reference_points, dyn_q_logits = self.transformer(
            tgt, query_pos, attn_mask,
            feat_flatten_img, spatial_flatten_img, level_start_index_img, self.pc_range, img_metas, data['lidar2img'],
            feat_flatten_pts=feat_flatten_pts, pos_flatten_pts=pos_flatten_pts,
            temp_memory=temp_memory, temp_pos=temp_pos,
            cross_attn_masks=None, reference_points=reference_points,
            dyn_q_coords=dyn_q_coords, dyn_q_probs=dyn_q_probs, dyn_q_mask=dyn_q_mask, dyn_q_pos_branch=self.dyn_q_pos,
            dyn_q_pos_with_prob_branch=self.dyn_q_pos_with_prob, dyn_q_prob_branch=self.dyn_q_prob_branch,
        )

        # ==========================================================
        # 3. 结果生成
        # ==========================================================
        outs_dec = torch.nan_to_num(outs_dec)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points[lvl].clone())
            outputs_class = self.cls_branches[lvl](outs_dec[lvl])
            tmp = self.reg_branches[lvl](outs_dec[lvl])
            tmp[..., 0:3] += reference[..., 0:3]
            tmp[..., 0:3] = tmp[..., 0:3].sigmoid()
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)

        all_bbox_preds[..., 0:3] = (
                all_bbox_preds[..., 0:3] * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3])

        print(f"self.num_propagated = {self.num_propagated}")
        print(f"all_bbox_preds[-1].shape: {all_bbox_preds[-1].shape}")  # 应为 [B, N, code_size]

        # ==========================================================
        # 4. 原型学习特征计算
        # ==========================================================
        proto_sims = None
        projected_feats = None

        self.local_iter += 1

        # Use the class-head semantic feature before the final classifier.
        # This matches clsfeat prototype training: decoder feature -> cls branch
        # hidden layers -> proto projection, while keeping the feature dim 256.
        last_feat = outs_dec[-1]
        for layer in list(self.cls_branches[-1].children())[:-1]:
            last_feat = layer(last_feat)

        B, N, C = last_feat.shape
        feat_flat = last_feat.reshape(B * N, C)
        proj_flat = self.proto_proj(feat_flat)
        proj_feat = proj_flat.reshape(B, N, -1)
        proj_feat = self.feat_norm(proj_feat)
        proj_feat_norm = l2_normalize(proj_feat)
        projected_feats = proj_feat_norm

        flat_protos = self.prototypes.view(-1, self.embed_dims)
        flat_protos_norm = l2_normalize(flat_protos)
        proto_sims = torch.matmul(proj_feat_norm, flat_protos_norm.t())

        # ==========================================================
        # 5. 后处理
        # ==========================================================
        iou_thr = self.post_bev_nms_thr
        score_thr = self.post_bev_nms_score
        ops = self.post_bev_nms_ops

        if len(ops) > 0:
            bbox_output = denormalize_bbox(all_bbox_preds[-1, :, pad_size:], None)
            bbox_bev = bbox_output[..., [0, 1, 3, 4, 6]]
            score_bev = all_cls_scores[-1, :, pad_size:].sigmoid().max(-1).values.clone()
            score_bev[~tgt_query_mask] = 0
            nms_tgt_query_mask = torch.zeros_like(tgt_query_mask)

            for i in range(B):
                if 0 in ops:
                    assert len(ops) == 1
                    all_boxes = bbox_bev[i]
                    all_scores = score_bev[i]
                    keep = nms_bev(xywhr2xyxyr(all_boxes), all_scores, iou_thr, pre_max_size=None, post_max_size=None)
                    nms_tgt_query_mask[i, :][keep] = 1
                    nms_tgt_query_mask[i, :][all_scores == 0] = 0

                    # Export both current-frame and propagated temporal queries.
                    start = pad_size
                    num_export_queries = self.num_query + self.num_propagated
                    end = pad_size + num_export_queries

                    # 取出当前样本的正常查询特征、分数、框
                    current_frame_features = projected_feats[i][start:end]
                    current_frame_scores = all_cls_scores[-1][i][start:end]
                    current_frame_boxes = all_bbox_preds[-1][i][start:end]

                    # 计算每个查询的最大置信度（sigmoid 后）
                    max_conf = current_frame_scores.sigmoid().max(dim=-1)[0]

                    # 结合 NMS 的 keep 和置信度阈值进行过滤
                    keep_nms = keep.cpu().numpy() if torch.is_tensor(keep) else keep
                    # The NMS index is relative to the non-DN query block.
                    valid_nms = (keep_nms >= 0) & (keep_nms < num_export_queries)
                    keep_nms = keep_nms[valid_nms]

                    # 置信度阈值
                    conf_thr = 0.00
                    high_conf_mask = max_conf > conf_thr
                    # Never reintroduce padded/invalid queries through their
                    # unmasked raw class scores.  Save only effective NMS keeps.
                    final_keep = [
                        int(idx) for idx in keep_nms
                        if tgt_query_mask[i, int(idx)].item()
                        and nms_tgt_query_mask[i, int(idx)].item()
                        and high_conf_mask[int(idx)].item()
                    ]

                    # Save zero-detection frames too, preserving PKL index order.
                    if final_keep is not None:
                        nms_features = current_frame_features[final_keep].cpu().numpy()
                        nms_scores = current_frame_scores[final_keep].cpu().numpy()
                        nms_labels = nms_scores.argmax(axis=1)
                        nms_boxes = bbox_output[i][final_keep].cpu().numpy()

                        print(f"pad_size={pad_size}, self.num_query={self.num_query}, start={start}, end={end}")

                        print(f"Before NMS: {current_frame_boxes.shape[0]}")
                        print(f"After NMS: {len(keep_nms)}")
                        print(f"After conf_thr: {len(final_keep)}")

                        # 保存文件（目录、索引等沿用原有逻辑）
                        save_dir = self.save_dir
                        os.makedirs(save_dir, exist_ok=True)
                        sample_idx = self.sample_counter
                        np.save(f'{save_dir}/sample_{sample_idx}_features.npy', nms_features)
                        np.save(f'{save_dir}/sample_{sample_idx}_labels.npy', nms_labels)
                        np.save(f'{save_dir}/sample_{sample_idx}_scores.npy', nms_scores)
                        np.save(f'{save_dir}/sample_{sample_idx}_boxes.npy', nms_boxes)
                        np.save(f'{save_dir}/sample_{sample_idx}_query_indices.npy',
                                np.asarray(final_keep, dtype=np.int64))

                        # 可选：保存 token
                        sample_token = data.get('sample_tokens', [None])[i] if 'sample_tokens' in data else None
                        if sample_token:
                            np.save(f'{save_dir}/sample_{sample_idx}_token.npy', np.array([sample_token], dtype=object))

                        # 保存到内存列表
                        self.saved_samples.append({
                            'sample_idx': sample_idx,
                            'sample_token': sample_token,
                            'features': nms_features,
                            'boxes': nms_boxes
                        })

                        print(f"\n✅ 样本 {sample_idx} 保存成功")
                        print(f"   保留查询数: {len(final_keep)}")
                        print(f"   特征形状: {nms_features.shape}")
                        print(f"   标签分布: {np.bincount(nms_labels)}")
                        if sample_token:
                            print(f"   sample_token: {sample_token}")

                        self.sample_counter += 1

            nms_tgt_query_mask &= all_cls_scores[-1, :, pad_size:].sigmoid().max(-1).values > score_thr
            tgt_query_mask &= nms_tgt_query_mask
            # 将 NMS 抑制的查询的分数设为 -40，框设为 0
            all_cls_scores[:, :, pad_size:][:, ~tgt_query_mask] = -40
            all_bbox_preds[:, :, pad_size:][:, ~tgt_query_mask] = 0

        self.post_update_memory(data, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec, mask_dict,
                                query_mask=tgt_query_mask, instance_inds=None)

        if mask_dict and mask_dict['pad_size'] > 0:
            output_known_class = all_cls_scores[:, :, :mask_dict['pad_size'], :]
            output_known_coord = all_bbox_preds[:, :, :mask_dict['pad_size'], :]
            all_cls_scores = all_cls_scores[:, :, mask_dict['pad_size']:, :]
            all_bbox_preds = all_bbox_preds[:, :, mask_dict['pad_size']:, :]
            mask_dict['output_known_lbs_bboxes'] = (output_known_class, output_known_coord)

            if self.training:
                projected_feats = projected_feats[:, mask_dict['pad_size']:, :]
                proto_sims = proto_sims[:, mask_dict['pad_size']:, :]
        else:
            mask_dict = None

        outs = {
            'all_cls_scores': all_cls_scores,
            'all_bbox_preds': all_bbox_preds,
            'dyn_cls_scores': all_cls_scores,
            'dyn_bbox_preds': all_bbox_preds,
            'dn_mask_dict': mask_dict,
            'projected_feats': projected_feats,
            'proto_sims': proto_sims
        }
        return outs

    def loss_detection_only(self, gt_bboxes_list, gt_labels_list, preds_dicts, gt_bboxes_ignore=None):
        return super().loss(gt_bboxes_list, gt_labels_list, preds_dicts, gt_bboxes_ignore)

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_list, gt_labels_list, preds_dicts, gt_bboxes_ignore=None):
        labeled_idx = [i for i, labels in enumerate(gt_labels_list) if len(labels) > 0]
        if len(labeled_idx) == 0:
            return dict(loss_dummy=preds_dicts['all_cls_scores'].sum() * 0)

        gt_bboxes_list = [gt_bboxes_list[i] for i in labeled_idx]
        gt_labels_list = [gt_labels_list[i] for i in labeled_idx]

        loss_dict = super().loss(gt_bboxes_list, gt_labels_list, preds_dicts, gt_bboxes_ignore)

        if not self.training:
            return loss_dict

        cls_scores = preds_dicts['all_cls_scores'][-1][labeled_idx]
        bbox_preds = preds_dicts['all_bbox_preds'][-1][labeled_idx]
        projected_feats = preds_dicts['projected_feats'][labeled_idx]
        proto_sims = preds_dicts['proto_sims'][labeled_idx]

        if projected_feats is None or proto_sims is None:
            return loss_dict

        num_imgs = len(labeled_idx)
        device = cls_scores.device

        gt_bboxes_list_tensor = [
            torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1).to(device)
            for gt_bboxes in gt_bboxes_list
        ]

        batch_pos_feats = []
        batch_pos_labels = []
        batch_pos_sims = []
        batch_pos_pred_scores = []

        for i in range(num_imgs):
            assign_result = self.assigner.assign(
                bbox_preds[i], cls_scores[i], gt_bboxes_list_tensor[i], gt_labels_list[i], gt_bboxes_ignore
            )
            sampling_result = self.sampler.sample(assign_result, bbox_preds[i], gt_bboxes_list_tensor[i])
            pos_inds = sampling_result.pos_inds

            if pos_inds.numel() > 0:
                pos_gt_labels = gt_labels_list[i][sampling_result.pos_assigned_gt_inds]
                pos_pred_scores = cls_scores[i][pos_inds]
                pos_feat = projected_feats[i][pos_inds]
                pos_sim = proto_sims[i][pos_inds]

                batch_pos_feats.append(pos_feat)
                batch_pos_labels.append(pos_gt_labels)
                batch_pos_sims.append(pos_sim)
                batch_pos_pred_scores.append(pos_pred_scores)

        if len(batch_pos_feats) > 0:
            all_pos_feats = torch.cat(batch_pos_feats, dim=0)
            all_pos_labels = torch.cat(batch_pos_labels, dim=0)
            all_pos_sims = torch.cat(batch_pos_sims, dim=0)
            all_pos_pred_scores = torch.cat(batch_pos_pred_scores, dim=0)

            self.prototypes.data.copy_(l2_normalize(self.prototypes))

            contrast_logits, contrast_targets = self.prototype_learning(
                all_pos_feats, all_pos_labels, all_pos_sims, all_pos_pred_scores
            )

            temperature = 0.07
            loss_proto = F.cross_entropy(
                contrast_logits / temperature,
                contrast_targets.long(),
                ignore_index=-100
            )

            loss_dict['loss_proto'] = loss_proto * self.proto_loss_weight
        else:
            loss_dict['loss_proto'] = cls_scores.sum() * 0.0

        if self.training and self.local_iter % 500 == 0:
            save_dir = 'work_dirs/prototype_mv2dfusion-fsd_freeze-r50_1600_gridmask-ep24_nusc/prototypes'
            os.makedirs(save_dir, exist_ok=True)
            proto_data = self.prototypes.data.cpu().clone()
            torch.save({
                'iter': self.local_iter.item(),
                'prototypes': proto_data,
                'num_classes': self.num_classes,
                'num_prototype': self.num_prototype
            }, f'{save_dir}/prototypes_iter_{int(self.local_iter.item())}.pth')

        return loss_dict
