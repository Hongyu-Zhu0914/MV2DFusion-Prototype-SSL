import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mmcv.runner import force_fp32
from mmdet.models import HEADS
from mmdet.core import multi_apply
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core import nms_bev
from mmdet3d.core.bbox.structures import xywhr2xyxyr
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
from projects.mmdet3d_plugin.models.utils.positional_encoding import pos2posemb3d
import os

# 外部依赖导入
from .sinkhorn_balanced_0712 import distributed_sinkhorn_balanced
from .contrast import momentum_update, l2_normalize, ProjectionHead

# 导入父类
from .mv2dfusion_head import MV2DFusionHead


@HEADS.register_module()
class PrototypeMV2DFusionHead(MV2DFusionHead):
    def __init__(self,
                 # 1. MV2DFusionHead 原有参数
                 num_classes,  # 检测类别数（如10类）
                 in_channels=256,  # 输入特征通道数
                 embed_dims=256,  # 特征嵌入维度
                 num_query=100,  # Transformer查询数量
                 num_reg_fcs=2,  # 回归分支的全连接层数
                 memory_len=6 * 256,  # 记忆特征长度
                 topk_proposals=256,  # Top-K提案数
                 num_propagated=256,  # 传播的特征数
                 with_dn=True,  # 是否使用去噪训练
                 with_ego_pos=True,  # 是否使用自车位置编码
                 match_with_velo=True,  # 是否匹配速度信息
                 match_costs=None,  # 匹配损失权重
                 sync_cls_avg_factor=False,  # 是否同步分类平均因子
                 code_weights=None,  # BBox编码权重
                 bbox_coder=None,  # BBox编码器
                 transformer=None,  # Transformer配置
                 normedlinear=False,  # 是否使用归一化线性层
                 loss_cls=dict(  # 分类损失配置
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),  # 回归损失
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),  # IoU损失
                 train_cfg=dict(  # 训练配置
                     assigner=dict(
                         type='HungarianAssigner3D',  # 匈牙利匹配器
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(
                             type='IoUCost', iou_mode='giou', weight=2.0)), ),
                 test_cfg=dict(max_per_img=100),  # 测试配置
                 scalar=5,  # 标量参数
                 noise_scale=0.4,  # 噪声尺度
                 noise_trans=0.0,  # 噪声平移
                 dn_weight=1.0,  # 去噪权重
                 split=0.5,  # 分割比例
                 prob_bin=50,  # 概率分桶数
                 post_bev_nms_thr=0.2,  # BEV NMS阈值
                 post_bev_nms_score=0.0,  # BEV NMS分数阈值
                 post_bev_nms_ops=[],  # BEV NMS操作
                 init_cfg=None,  # 初始化配置
                 debug=False,  # 调试模式

                 # =========================================================
                 # 2. 原型学习特有参数
                 # =========================================================
                 num_prototype=10,  # 每个类别的原型数量
                 update_prototype=True,  # 是否更新原型
                 gamma=0.999,  # 动量更新系数
                 warm_up=0,  # fixed EMA momentum; required by current contrast.py
                 sim_thr=0.7,  # 相似度阈值
                 score_thr=0.1,  # 得分阈值
                 proto_loss_weight=0.1,  # 原型 Loss 权重
                 proto_intra_lower=None,
                 proto_intra_upper=0.8,
                 proto_intra_weight=0.05,
                 proto_inter_upper=0.4,
                 proto_inter_weight=0.01,
                 sinkhorn_min_samples=20,
                 sinkhorn_queue_size=20,
                 greedy_dist_th=2.0,
                 greedy_dist_th_by_class=None,
                 greedy_conf_thr=0.2,
                 greedy_angle_thr=45.0,
                 greedy_angle_class_ids=None,
                 active_proto_class_ids=None,
                 neg_proto_margin=0.7,
                 neg_proto_gt_dist_th=2.0,
                 neg_proto_loss_weight=0.02,
                 neg_proto_dynamic_weight=True,
                 neg_proto_loss_weight_min=0.005,
                 neg_proto_loss_weight_max=0.10,
                 neg_proto_weight_step=0.01,
                 neg_proto_monitor_interval=500,
                 neg_proto_weight_update_interval=1000,
                 neg_proto_min_mean_improvement=0.01,
                 neg_proto_min_ratio_improvement=0.05,
                 neg_proto_monitor_patience=1,
                 neg_proto_positive_floor=0.80,
                 neg_proto_positive_class_floor=0.80,
                 neg_proto_positive_drop_tolerance=0.03,
                 neg_proto_min_monitor_samples=20,
                 prototype_save_dir=(
                     'work_dirs/prototype_vis_sinkhorn_stage2_gradneg_0726'),
                 prototype_save_interval=1000,
                 **kwargs):  # 其他参数

        # 调用父类初始化
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

        # -----------------------------------------------------------
        # Prototype Module Initialization
        # -----------------------------------------------------------
        self.num_prototype = num_prototype
        self.update_prototype = update_prototype
        self.gamma = gamma
        self.warm_up = warm_up
        self.sim_thr = sim_thr
        self.score_thr = score_thr
        self.proto_loss_weight = proto_loss_weight
        self.proto_intra_lower = (
            None if proto_intra_lower is None else float(proto_intra_lower))
        self.proto_intra_upper = float(proto_intra_upper)
        self.proto_intra_weight = proto_intra_weight
        self.proto_inter_upper = float(proto_inter_upper)
        self.proto_inter_weight = float(proto_inter_weight)
        if not -1.0 < self.proto_intra_upper <= 1.0:
            raise ValueError('Expected -1 < proto_intra_upper <= 1')
        if (self.proto_intra_lower is not None
                and not -1.0 <= self.proto_intra_lower < self.proto_intra_upper):
            raise ValueError(
                'Expected -1 <= proto_intra_lower < proto_intra_upper')
        self.sinkhorn_min_samples = int(sinkhorn_min_samples)
        self.sinkhorn_queue_size = int(sinkhorn_queue_size)
        if self.sinkhorn_min_samples <= 0:
            raise ValueError('sinkhorn_min_samples must be positive')
        if self.sinkhorn_queue_size < self.sinkhorn_min_samples:
            raise ValueError(
                'sinkhorn_queue_size must be at least sinkhorn_min_samples')
        if greedy_dist_th_by_class is None:
            greedy_dist_th_by_class = [greedy_dist_th] * self.num_classes
        if len(greedy_dist_th_by_class) != self.num_classes:
            raise ValueError(
                'greedy_dist_th_by_class must contain one value per class')

        if active_proto_class_ids is None:
            active_proto_class_ids = list(range(self.num_classes))
        if greedy_angle_class_ids is None:
            greedy_angle_class_ids = []

        self.greedy_dist_th_by_class = tuple(
            float(value) for value in greedy_dist_th_by_class)
        self.greedy_conf_thr = float(greedy_conf_thr)
        self.greedy_angle_thr = math.radians(float(greedy_angle_thr))
        self.greedy_angle_class_ids = tuple(
            int(class_id) for class_id in greedy_angle_class_ids)
        self.active_proto_class_ids = tuple(
            int(class_id) for class_id in active_proto_class_ids)
        self.neg_proto_margin = float(neg_proto_margin)
        self.neg_proto_loss_weight = float(neg_proto_loss_weight)
        self.neg_proto_gt_dist_th = float(neg_proto_gt_dist_th)
        self.neg_proto_dynamic_weight = bool(neg_proto_dynamic_weight)
        self.neg_proto_loss_weight_min = float(neg_proto_loss_weight_min)
        self.neg_proto_loss_weight_max = float(neg_proto_loss_weight_max)
        self.neg_proto_weight_step = float(neg_proto_weight_step)
        self.neg_proto_monitor_interval = int(neg_proto_monitor_interval)
        self.neg_proto_weight_update_interval = int(
            neg_proto_weight_update_interval)
        self.neg_proto_min_mean_improvement = float(
            neg_proto_min_mean_improvement)
        self.neg_proto_min_ratio_improvement = float(
            neg_proto_min_ratio_improvement)
        self.neg_proto_monitor_patience = int(neg_proto_monitor_patience)
        self.neg_proto_positive_floor = float(neg_proto_positive_floor)
        self.neg_proto_positive_class_floor = float(
            neg_proto_positive_class_floor)
        self.neg_proto_positive_drop_tolerance = float(
            neg_proto_positive_drop_tolerance)
        self.neg_proto_min_monitor_samples = int(
            neg_proto_min_monitor_samples)
        self.prototype_save_dir = prototype_save_dir
        self.prototype_save_interval = int(prototype_save_interval)

        for class_id in self.active_proto_class_ids:
            if class_id < 0 or class_id >= self.num_classes:
                raise ValueError(f'Invalid active prototype class id: {class_id}')
            if self.greedy_dist_th_by_class[class_id] <= 0:
                raise ValueError(
                    f'Distance threshold for active class {class_id} must be positive')
        if not set(self.greedy_angle_class_ids).issubset(
                set(self.active_proto_class_ids)):
            raise ValueError(
                'greedy_angle_class_ids must be a subset of active_proto_class_ids')
        if self.prototype_save_interval <= 0:
            raise ValueError('prototype_save_interval must be positive')
        if not -1.0 < self.proto_inter_upper <= 1.0:
            raise ValueError('Expected -1 < proto_inter_upper <= 1')
        if self.proto_inter_weight < 0:
            raise ValueError('proto_inter_weight must be non-negative')
        if not -1.0 < self.neg_proto_margin < 1.0:
            raise ValueError('Expected -1 < neg_proto_margin < 1')
        if self.neg_proto_gt_dist_th <= 0:
            raise ValueError('neg_proto_gt_dist_th must be positive')
        if not (
                0 <= self.neg_proto_loss_weight_min
                <= self.neg_proto_loss_weight
                <= self.neg_proto_loss_weight_max):
            raise ValueError(
                'Expected min <= initial <= max negative loss weight')
        if self.neg_proto_weight_step <= 0:
            raise ValueError('neg_proto_weight_step must be positive')
        if self.neg_proto_monitor_interval <= 0:
            raise ValueError('neg_proto_monitor_interval must be positive')
        if (
                self.neg_proto_weight_update_interval
                < self.neg_proto_monitor_interval
                or self.neg_proto_weight_update_interval
                % self.neg_proto_monitor_interval != 0):
            raise ValueError(
                'neg_proto_weight_update_interval must be a positive '
                'multiple of neg_proto_monitor_interval')
        if self.neg_proto_monitor_patience <= 0:
            raise ValueError('neg_proto_monitor_patience must be positive')
        if not (
                -1.0
                <= self.neg_proto_positive_class_floor
                <= self.neg_proto_positive_floor
                <= 1.0):
            raise ValueError(
                'Expected -1 <= positive class floor '
                '<= positive overall floor <= 1')
        if self.neg_proto_positive_drop_tolerance < 0:
            raise ValueError(
                'neg_proto_positive_drop_tolerance must be non-negative')
        if self.neg_proto_min_monitor_samples <= 0:
            raise ValueError(
                'neg_proto_min_monitor_samples must be positive')

        # 本地迭代计数器（用于预热）
        self.register_buffer('local_iter', torch.zeros(1))
        self.register_buffer(
            'proto_update_count',
            torch.zeros(self.num_classes, self.num_prototype, dtype=torch.long))
        self.register_buffer(
            'proto_assignment_mass',
            torch.zeros(self.num_classes, self.num_prototype))
        self.register_buffer(
            'proto_hard_assignment_count',
            torch.zeros(
                self.num_classes, self.num_prototype, dtype=torch.long))
        self.register_buffer(
            'proto_feature_queue',
            torch.zeros(
                self.num_classes, self.sinkhorn_queue_size, self.embed_dims))
        self.register_buffer(
            'proto_feature_queue_count',
            torch.zeros(self.num_classes, dtype=torch.long))
        self.register_buffer(
            'proto_feature_queue_ptr',
            torch.zeros(self.num_classes, dtype=torch.long))
        negative_weights = torch.zeros(
            self.num_classes, dtype=torch.float32)
        negative_weights[list(self.active_proto_class_ids)] = (
            self.neg_proto_loss_weight)
        self.register_buffer(
            'neg_proto_class_weight',
            negative_weights)
        self.register_buffer(
            'neg_monitor_similarity_sum',
            torch.zeros(self.num_classes))
        self.register_buffer(
            'neg_monitor_sample_count',
            torch.zeros(self.num_classes))
        self.register_buffer(
            'neg_monitor_hard_count',
            torch.zeros(self.num_classes))
        self.register_buffer(
            'neg_monitor_positive_similarity_sum',
            torch.zeros(self.num_classes))
        self.register_buffer(
            'neg_monitor_positive_count',
            torch.zeros(self.num_classes))
        self.register_buffer(
            'neg_monitor_previous_mean_by_class',
            torch.full((self.num_classes,), float('nan')))
        self.register_buffer(
            'neg_monitor_previous_ratio_by_class',
            torch.full((self.num_classes,), float('nan')))
        self.register_buffer(
            'neg_monitor_previous_positive_mean_by_class',
            torch.full((self.num_classes,), float('nan')))
        self.register_buffer(
            'neg_monitor_bad_windows_by_class',
            torch.zeros(self.num_classes, dtype=torch.long))

        # 初始化原型 Bank: [Class, Prototype, Dim]
        self.prototypes = nn.Parameter(
            torch.zeros(self.num_classes, self.num_prototype, self.embed_dims),
            requires_grad=False)  # 不通过反向传播更新

        # 截断正态分布初始化，std=0.02是Transformer常用初始化
        nn.init.trunc_normal_(self.prototypes, std=0.02)

        # 初始化投影头，将特征映射到更适合原型学习的空间
        self.proto_proj = ProjectionHead(dim_in=self.embed_dims, proj_dim=self.embed_dims)

        # 特征归一化层
        # 注意：这里的 in_channels 对应特征维度，在 Transformer Head 中通常是 embed_dims
        self.feat_norm = nn.LayerNorm(self.embed_dims)

    def _greedy_match_by_distance(self,
                                  pred_boxes,
                                  pred_scores,
                                  gt_centers,
                                  gt_labels,
                                  gt_boxes=None,
                                  return_matched_ids=True):
        """
        保守的同类别一对一贪心匹配：
          1. 只允许高于置信度阈值、且属于有效类别的预测框参与匹配。
          2. 预测框按置信度降序处理，只寻找尚未占用的同类别 GT。
          3. 先选最近的同类别 GT，再检查分类别距离阈值和方向类别角度阈值。
          4. 最近 GT 不满足条件时保持未匹配，不跳转到更远 GT。
        返回:
            pos_inds: 正样本索引
            pos_assigned_gt_inds: 正样本对应的 GT 索引
            matched_gt_ids: 每个预测框匹配的 GT 索引（-1 表示未匹配）
        """
        device = pred_boxes.device
        N = pred_boxes.shape[0]
        M = gt_centers.shape[0]

        if M == 0:
            pos_inds = torch.empty(0, dtype=torch.long, device=device)
            pos_assigned_gt_inds = torch.empty(0, dtype=torch.long, device=device)
            matched_gt_ids = torch.full((N,), -1, dtype=torch.long, device=device)
            if return_matched_ids:
                return pos_inds, pos_assigned_gt_inds, matched_gt_ids
            else:
                return pos_inds, pos_assigned_gt_inds

        pred_prob = pred_scores.sigmoid()
        pred_conf, pred_labels = pred_prob.max(dim=-1)

        active_class_mask = torch.zeros(
            self.num_classes, dtype=torch.bool, device=device)
        active_class_mask[list(self.active_proto_class_ids)] = True
        high_conf_mask = (
            (pred_conf > self.greedy_conf_thr)
            & active_class_mask[pred_labels])
        if not high_conf_mask.any():
            matched_gt_ids = torch.full((N,), -1, dtype=torch.long, device=device)
            pos_inds = torch.empty(0, dtype=torch.long, device=device)
            pos_assigned_gt_inds = torch.empty(0, dtype=torch.long, device=device)
            if return_matched_ids:
                return pos_inds, pos_assigned_gt_inds, matched_gt_ids
            else:
                return pos_inds, pos_assigned_gt_inds

        high_conf_idx = torch.where(high_conf_mask)[0]  # 原始索引
        high_conf_boxes = pred_boxes[high_conf_mask]  # [H, code_size]
        high_conf_conf = pred_conf[high_conf_mask]  # [H]
        high_conf_labels = pred_labels[high_conf_mask]

        H = high_conf_boxes.shape[0]

        pred_centers_2d = high_conf_boxes[:, :2]  # [H, 2]
        gt_centers_2d = gt_centers[:, :2]  # [M, 2]
        dist_mat = torch.cdist(pred_centers_2d, gt_centers_2d, p=2)  # [H, M]

        if gt_boxes is None:
            raise ValueError('gt_boxes are required for angle-aware matching')
        if high_conf_boxes.size(-1) <= 7:
            raise ValueError(
                'Expected predicted boxes to contain sine and cosine yaw channels')
        pred_yaws = torch.atan2(
            high_conf_boxes[:, 6], high_conf_boxes[:, 7])
        gt_yaws = gt_boxes[:, 6]

        high_matched_gt_ids = torch.full((H,), -1, dtype=torch.long, device=device)
        gt_matched = torch.zeros(M, dtype=torch.bool, device=device)

        sorted_idx = torch.argsort(high_conf_conf, descending=True)  # [H]
        for li in sorted_idx.tolist():
            pred_class = int(high_conf_labels[li].item())
            candidate_mask = (~gt_matched) & (gt_labels == pred_class)
            if not candidate_mask.any():
                continue

            candidate_indices = torch.where(candidate_mask)[0]
            candidate_distances = dist_mat[li, candidate_indices]
            min_local_idx = torch.argmin(candidate_distances)
            global_gt_idx = int(candidate_indices[min_local_idx].item())
            min_distance = candidate_distances[min_local_idx]

            distance_threshold = self.greedy_dist_th_by_class[pred_class]
            if min_distance >= distance_threshold:
                continue

            if pred_class in self.greedy_angle_class_ids:
                angle_delta = pred_yaws[li] - gt_yaws[global_gt_idx]
                angle_delta = torch.atan2(
                    torch.sin(angle_delta), torch.cos(angle_delta)).abs()
                if angle_delta > self.greedy_angle_thr:
                    continue

            high_matched_gt_ids[li] = global_gt_idx
            gt_matched[global_gt_idx] = True

        matched_gt_ids = torch.full((N,), -1, dtype=torch.long, device=device)
        for local_i, global_i in enumerate(high_conf_idx):
            matched_gt_ids[global_i] = high_matched_gt_ids[local_i]

        pos_mask = matched_gt_ids != -1
        pos_inds = torch.where(pos_mask)[0]
        pos_assigned_gt_inds = matched_gt_ids[pos_inds]

        if return_matched_ids:
            return pos_inds, pos_assigned_gt_inds, matched_gt_ids
        else:
            return pos_inds, pos_assigned_gt_inds

    #pos_inds 记录了哪些预测框被判定为正样本（即成功匹配到了某个 GT）。
    ##pos_assigned_gt_inds 记录了每个正样本对应的 GT 的索引（即该 GT 在 gt_bboxes_list[i] 中的位置）。

    # ======================================原型学习==================================
    def _apply_proto_intra_band(self, proto_k):
        """Only repel same-class prototypes above the configured upper bound."""
        proto_k = l2_normalize(proto_k)
        band_loss = proto_k.new_zeros(())
        pair_count = 0

        for first_idx in range(self.num_prototype):
            for second_idx in range(first_idx + 1, self.num_prototype):
                similarity = torch.dot(proto_k[first_idx], proto_k[second_idx])
                high_violation = torch.clamp(
                    similarity - self.proto_intra_upper, min=0.0)
                band_loss = band_loss + high_violation.square()
                pair_count += 1

                if self.proto_intra_weight <= 0:
                    continue
                direction = proto_k[second_idx] - proto_k[first_idx]
                if high_violation.item() > 0:
                    step = self.proto_intra_weight * high_violation
                    proto_k[first_idx] = proto_k[first_idx] - step * direction
                    proto_k[second_idx] = proto_k[second_idx] + step * direction
                proto_k = l2_normalize(proto_k)

        if pair_count:
            band_loss = band_loss / pair_count
        return l2_normalize(proto_k), band_loss.detach()

    def _apply_proto_inter_upper(self, protos):
        """Repel only cross-class prototype pairs above the upper bound."""
        protos = l2_normalize(protos)
        active_classes = list(self.active_proto_class_ids)
        active_protos = protos[active_classes]
        flat_protos = active_protos.reshape(-1, self.embed_dims)
        class_positions = torch.arange(
            len(active_classes), device=protos.device)
        class_positions = class_positions.repeat_interleave(
            self.num_prototype)
        cross_class_mask = (
            class_positions[:, None] != class_positions[None, :])
        similarities = flat_protos @ flat_protos.t()
        violations = torch.clamp(
            similarities - self.proto_inter_upper, min=0.0)
        violations = violations * cross_class_mask
        unique_pair_mask = torch.triu(cross_class_mask, diagonal=1)
        inter_value = violations[unique_pair_mask].square().mean()

        if self.proto_inter_weight > 0 and violations.any():
            weighted_degree = violations.sum(dim=1, keepdim=True)
            repulsion = (
                weighted_degree * flat_protos
                - violations @ flat_protos)
            flat_protos = l2_normalize(
                flat_protos + self.proto_inter_weight * repulsion)
            protos[active_classes] = flat_protos.view_as(active_protos)
        return l2_normalize(protos), inter_value.detach()

    @torch.no_grad()
    def _accumulate_negative_monitor(
            self, negative_similarity_by_class,
            positive_similarity_by_class):
        for class_id, similarities in negative_similarity_by_class.items():
            if similarities.numel() == 0:
                continue
            similarities = similarities.detach().float()
            self.neg_monitor_similarity_sum[class_id] += similarities.sum()
            self.neg_monitor_sample_count[class_id] += similarities.numel()
            self.neg_monitor_hard_count[class_id] += (
                similarities > self.neg_proto_margin).sum()

        for class_id, similarities in positive_similarity_by_class.items():
            if similarities.numel() == 0:
                continue
            similarities = similarities.detach().float()
            self.neg_monitor_positive_similarity_sum[
                class_id] += similarities.sum()
            self.neg_monitor_positive_count[class_id] += similarities.numel()

    @torch.no_grad()
    def _maybe_adjust_negative_weight(self):
        current_iter = int(self.local_iter.item())
        if (
                not self.neg_proto_dynamic_weight
                or current_iter <= 0
                or current_iter % self.neg_proto_monitor_interval != 0):
            return None
        can_update_weight = (
            current_iter % self.neg_proto_weight_update_interval == 0)

        reports = []
        for class_id in self.active_proto_class_ids:
            negative_count = int(
                self.neg_monitor_sample_count[class_id].item())
            positive_count = int(
                self.neg_monitor_positive_count[class_id].item())
            old_weight = float(
                self.neg_proto_class_weight[class_id].item())

            if (
                    negative_count < self.neg_proto_min_monitor_samples
                    or positive_count < self.neg_proto_min_monitor_samples):
                reports.append({
                    'class_id': class_id,
                    'negative_count': negative_count,
                    'positive_count': positive_count,
                    'negative_mean': float('nan'),
                    'negative_ratio': float('nan'),
                    'positive_mean': float('nan'),
                    'positive_drop': float('nan'),
                    'mean_improvement': float('nan'),
                    'ratio_improvement': float('nan'),
                    'old_weight': old_weight,
                    'new_weight': old_weight,
                    'action': 'hold_insufficient_samples',
                    'guard_reason': 'insufficient_samples',
                })
                continue

            negative_mean = (
                self.neg_monitor_similarity_sum[class_id]
                / self.neg_monitor_sample_count[class_id])
            negative_ratio = (
                self.neg_monitor_hard_count[class_id]
                / self.neg_monitor_sample_count[class_id])
            positive_mean = (
                self.neg_monitor_positive_similarity_sum[class_id]
                / self.neg_monitor_positive_count[class_id])
            previous_mean = (
                self.neg_monitor_previous_mean_by_class[class_id].clone())
            previous_ratio = (
                self.neg_monitor_previous_ratio_by_class[class_id].clone())
            previous_positive = (
                self.neg_monitor_previous_positive_mean_by_class[
                    class_id].clone())
            mean_improvement = negative_mean.new_tensor(float('nan'))
            ratio_improvement = negative_ratio.new_tensor(float('nan'))
            positive_drop = positive_mean.new_tensor(float('nan'))
            action = 'baseline'
            guard_reasons = []
            new_weight = old_weight

            has_previous = (
                torch.isfinite(previous_mean)
                and torch.isfinite(previous_ratio)
                and torch.isfinite(previous_positive))
            if positive_mean < self.neg_proto_positive_class_floor:
                guard_reasons.append('class_below_floor')
            if has_previous:
                mean_improvement = previous_mean - negative_mean
                ratio_improvement = previous_ratio - negative_ratio
                positive_drop = previous_positive - positive_mean
                if positive_drop > self.neg_proto_positive_drop_tolerance:
                    guard_reasons.append('period_drop_exceeded')

            if not can_update_weight:
                action = 'monitor_only'
            elif guard_reasons:
                new_weight = max(
                    old_weight * 0.5,
                    self.neg_proto_loss_weight_min)
                self.neg_monitor_bad_windows_by_class[class_id] = 0
                action = 'decrease_positive_guard'
            elif has_previous:
                if (
                        mean_improvement
                        < self.neg_proto_min_mean_improvement
                        and ratio_improvement
                        < self.neg_proto_min_ratio_improvement):
                    self.neg_monitor_bad_windows_by_class[class_id] += 1
                    action = 'wait_insufficient_improvement'
                    if (
                            int(self.neg_monitor_bad_windows_by_class[
                                class_id].item())
                            >= self.neg_proto_monitor_patience):
                        new_weight = min(
                            old_weight + self.neg_proto_weight_step,
                            self.neg_proto_loss_weight_max)
                        self.neg_monitor_bad_windows_by_class[class_id] = 0
                        action = 'increase'
                else:
                    self.neg_monitor_bad_windows_by_class[class_id] = 0
                    action = 'hold_improving'

            self.neg_proto_class_weight[class_id] = new_weight
            self.neg_monitor_previous_mean_by_class[
                class_id].copy_(negative_mean)
            self.neg_monitor_previous_ratio_by_class[
                class_id].copy_(negative_ratio)
            self.neg_monitor_previous_positive_mean_by_class[
                class_id].copy_(positive_mean)
            reports.append({
                'class_id': class_id,
                'negative_count': negative_count,
                'positive_count': positive_count,
                'negative_mean': float(negative_mean.item()),
                'negative_ratio': float(negative_ratio.item()),
                'positive_mean': float(positive_mean.item()),
                'positive_drop': float(positive_drop.item()),
                'mean_improvement': float(mean_improvement.item()),
                'ratio_improvement': float(ratio_improvement.item()),
                'old_weight': old_weight,
                'new_weight': new_weight,
                'action': action,
                'guard_reason': (
                    '+'.join(guard_reasons)
                    if guard_reasons else 'none'),
            })

        self.neg_monitor_similarity_sum.zero_()
        self.neg_monitor_sample_count.zero_()
        self.neg_monitor_hard_count.zero_()
        self.neg_monitor_positive_similarity_sum.zero_()
        self.neg_monitor_positive_count.zero_()
        return reports

    def _get_proto_feature_queue(self, class_idx, dtype):
        queue_count = int(self.proto_feature_queue_count[class_idx].item())
        if queue_count == 0:
            return self.proto_feature_queue.new_empty(
                (0, self.embed_dims), dtype=dtype)
        return self.proto_feature_queue[
            class_idx, :queue_count].to(dtype=dtype)

    @torch.no_grad()
    def _enqueue_proto_features(self, class_idx, features):
        features = l2_normalize(features.detach()).to(
            dtype=self.proto_feature_queue.dtype)
        num_features = features.shape[0]
        if num_features == 0:
            return

        if num_features >= self.sinkhorn_queue_size:
            self.proto_feature_queue[class_idx].copy_(
                features[-self.sinkhorn_queue_size:])
            self.proto_feature_queue_count[class_idx] = (
                self.sinkhorn_queue_size)
            self.proto_feature_queue_ptr[class_idx] = 0
            return

        queue_ptr = int(self.proto_feature_queue_ptr[class_idx].item())
        write_indices = (
            torch.arange(num_features, device=features.device) + queue_ptr
        ) % self.sinkhorn_queue_size
        self.proto_feature_queue[class_idx, write_indices] = features
        self.proto_feature_queue_ptr[class_idx] = (
            queue_ptr + num_features) % self.sinkhorn_queue_size
        self.proto_feature_queue_count[class_idx] = min(
            self.sinkhorn_queue_size,
            int(self.proto_feature_queue_count[class_idx].item())
            + num_features)

    def prototype_learning(self, out_feat, gt_labels, feat_proto_sim, pred_scores):
        """Update prototypes after each class has enough recent positives."""
        proto_logits = feat_proto_sim
        proto_target = torch.full_like(gt_labels, -100, dtype=torch.long)
        protos = self.prototypes.data.clone()
        pred_labels = pred_scores.argmax(dim=-1)
        correct_mask = pred_labels == gt_labels
        band_values = []

        for class_label in torch.unique(gt_labels):
            class_idx = int(class_label.long().item())
            if class_idx < 0 or class_idx >= self.num_classes:
                continue

            valid_mask = (gt_labels == class_idx) & correct_mask
            if not valid_mask.any():
                continue

            valid_feat = out_feat[valid_mask]
            cached_feat = self._get_proto_feature_queue(
                class_idx, dtype=valid_feat.dtype)
            combined_feat = torch.cat(
                [cached_feat, valid_feat.detach()], dim=0)

            if combined_feat.shape[0] < self.sinkhorn_min_samples:
                self._enqueue_proto_features(class_idx, valid_feat)
                protos[class_idx], class_band = self._apply_proto_intra_band(
                    protos[class_idx])
                band_values.append(class_band)
                continue

            class_prototypes = l2_normalize(protos[class_idx]).float()
            combined_sim = combined_feat.float() @ class_prototypes.t()
            assignment, assigned_indices = distributed_sinkhorn_balanced(
                combined_sim)

            current_start = cached_feat.shape[0]
            current_assignment = assignment[current_start:]
            current_assigned_indices = assigned_indices[current_start:]

            assignment_mass = assignment.sum(dim=0)
            weighted_centers = assignment.t() @ combined_feat.float()
            active_mask = assignment_mass > 1e-6

            if self.update_prototype and active_mask.any():
                weighted_centers = weighted_centers[active_mask]
                weighted_centers = weighted_centers / assignment_mass[active_mask].unsqueeze(-1)
                weighted_centers = l2_normalize(weighted_centers).to(
                    dtype=protos.dtype)
                protos[class_idx, active_mask] = momentum_update(
                    old_value=protos[class_idx, active_mask],
                    new_value=weighted_centers,
                    momentum=self.gamma,
                    iter=self.local_iter,
                    warm_up=self.warm_up,
                )
                self.proto_update_count[class_idx, active_mask] += 1
                self.proto_assignment_mass[class_idx] += (
                    current_assignment.sum(dim=0).detach().to(
                        self.proto_assignment_mass.dtype))
                self.proto_hard_assignment_count[class_idx] += torch.bincount(
                    current_assigned_indices,
                    minlength=self.num_prototype).to(
                        self.proto_hard_assignment_count.dtype)

            protos[class_idx], class_band = self._apply_proto_intra_band(
                protos[class_idx])
            band_values.append(class_band)
            proto_target[valid_mask] = (
                current_assigned_indices + self.num_prototype * class_idx)
            self._enqueue_proto_features(class_idx, valid_feat)

        protos, proto_inter_value = self._apply_proto_inter_upper(protos)
        self.prototypes.data.copy_(l2_normalize(protos))
        if band_values:
            proto_band_value = torch.stack(band_values).mean()
        else:
            proto_band_value = proto_logits.new_zeros(())
        return (
            proto_logits,
            proto_target,
            proto_band_value,
            proto_inter_value,
        )

    def forward(self, img_metas, dyn_query=None, dyn_feats=None,
                pts_query_center=None, pts_query_feat=None, pts_feat=None, pts_pos=None, **data):
        """
                前向传播主函数

                参数说明:
                - img_metas: 图像元信息（尺寸、相机参数等）
                - dyn_query: 动态查询（来自图像）
                - dyn_feats: 动态特征
                - pts_query_center: 点云查询中心
                - pts_query_feat: 点云查询特征
                - pts_feat: 点云特征
                - pts_pos: 点云位置编码
                - data: 其他数据（通过**kwargs传递）

                返回值:
                - outs: 包含预测结果的字典
         """
        # ==========================================================
        # 1. 数据准备 (与父类保持一致)
        # ==========================================================
        self.pre_update_memory(data)  # 更新记忆（如果有）

        # 相机内外参处理
        intrinsics = data['intrinsics'] / 1e3  # 内参归一化
        extrinsics = data['extrinsics'][..., :3, :]  # 外参（3×4矩阵）

        # 多尺度特征提取
        mln_input = torch.cat([intrinsics[..., 0, 0:1], intrinsics[..., 1, 1:2], extrinsics.flatten(-2)], dim=-1)
        mln_input = mln_input.flatten(0, 1).unsqueeze(1)
        mlvl_feats = data['img_feats_for_det']  # 多尺度图像特征
        B, N, _, _, _ = mlvl_feats[0].shape  # B: batch, N: 相机数

        # ==================== 1.1 图像特征处理 ====================
        feat_flatten_img = []  # 存储展平的图像特征
        spatial_flatten_img = []  # 存储特征图尺寸
        for i in range(1, len(mlvl_feats)):  # 遍历多尺度特征（跳过第一层？）
            B, N, C, H, W = mlvl_feats[i].shape
            # 重塑：[B, N, C, H, W] → [B*N, C, H*W] → [B*N, H*W, C]
            mlvl_feat = mlvl_feats[i].reshape(B * N, C, -1).transpose(1, 2)
            # 空间对齐（根据相机参数变换）
            mlvl_feat = self.spatial_alignment(mlvl_feat, mln_input)
            feat_flatten_img.append(mlvl_feat.to(torch.float))
            spatial_flatten_img.append((H, W))

        # 拼接所有尺度的特征
        feat_flatten_img = torch.cat(feat_flatten_img, dim=1)  # [B*N, 总点数, C]
        spatial_flatten_img = torch.as_tensor(spatial_flatten_img, dtype=torch.long, device=mlvl_feats[0].device)
        level_start_index_img = torch.cat(
            (spatial_flatten_img.new_zeros((1,)), spatial_flatten_img.prod(1).cumsum(0)[:-1]))

        # ==================== 1.2 点云特征处理 ====================
        feat_flatten_pts = self.pts_embed(pts_feat)  # 点云特征嵌入
        pos_flatten_pts = pts_pos  # 点云位置编码

        # ==================== 1.3 查询生成 ====================
        # 生成动态查询（来自图像）
        reference_points, query_coords, query_probs, query_feats, query_mask = \
            self.gen_dynamic_query(self.reference_points.weight, dyn_query, dyn_feats.get('query_feats', None))

        # 生成点云查询
        pts_ref = self.gen_pts_query(pts_query_center)
        # 合并查询：点云查询 + 图像查询
        query_mask = torch.cat([torch.ones_like(pts_ref[..., 0]).bool(), query_mask], dim=1)
        reference_points = torch.cat([pts_ref, reference_points], dim=1)

        # 统计查询数量
        num_query_img = int(self.num_query - pts_ref.size(1))  # 图像查询数
        num_query_pts = pts_ref.size(1)  # 点云查询数

        # ==================== 1.4 去噪训练准备 ====================
        reference_points, attn_mask, mask_dict = self.prepare_for_dn(B, reference_points, img_metas)

        # ==================== 1.5 注意力掩码准备 ====================
        tgt_size = self.num_query + self.num_propagated  # 目标大小
        src_size = self.num_query + self.memory_len  # 源大小

        # 创建或扩展注意力掩码
        if attn_mask is None:
            attn_mask = torch.zeros((tgt_size, src_size), dtype=torch.bool, device=reference_points.device)

        # 处理去噪训练的padding
        pad_size = attn_mask.size(0) - tgt_size
        if mask_dict is not None:
            assert pad_size == mask_dict['pad_size']

        # 扩展掩码到batch维度
        attn_mask = attn_mask.repeat(B, 1, 1)

        # 创建查询掩码
        tgt_query_mask = torch.cat([query_mask, self.memory_query_mask[:, :self.num_propagated, 0]], dim=1)
        src_query_mask = torch.cat([query_mask, self.memory_query_mask[:, :, 0]], dim=1)

        # 设置注意力掩码：无效位置为True（需要mask掉）
        attn_mask[:, :, pad_size:] = ~src_query_mask[:, None]

        # 扩展到多头注意力
        num_heads = self.transformer.decoder.layers[0].attentions[0].num_heads
        attn_mask = attn_mask.repeat_interleave(num_heads, dim=0)

        # ==================== 1.6 查询特征准备 ====================
        tgt = self.dyn_q_embed.weight.repeat(B, num_query_img, 1)  # 动态查询嵌入
        pts_tgt = self.pts_q_embed.weight.repeat(B, num_query_pts, 1)  # 点云查询嵌入
        # 拼接：padding + 点云查询 + 图像查询
        tgt = torch.cat([tgt.new_zeros((B, pad_size, self.embed_dims)), pts_tgt, tgt], dim=1)

        # 查询特征处理
        pad_query_feats = query_feats.new_zeros([B, pad_size + self.num_query, self.embed_dims])
        pts_query_feat = self.pts_query_embed(pts_query_feat)
        pad_query_feats[:, pad_size:pad_size + num_query_pts] = pts_query_feat
        pad_query_feats[:, pad_size + num_query_pts:pad_size + self.num_query] = query_feats
        # 查询特征编码
        tgt = self.dyn_q_enc(tgt, pad_query_feats)

        # ==================== 1.7 位置编码 ====================
        query_pos = self.query_embedding(pos2posemb3d(reference_points))

        # ==================== 1.8 时序对齐（多帧融合） ====================
        tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose = \
            self.temporal_alignment(query_pos, tgt, reference_points)

        # ==================== 1.9 动态查询位置编码 ====================
        query_pos_det = self.dyn_q_pos(query_coords.flatten(-2, -1))
        query_pos_det = self.dyn_q_pos_with_prob(query_pos_det, query_probs)
        query_pos[:, pad_size + num_query_pts:pad_size + self.num_query] = query_pos_det

        # 创建动态查询掩码
        dyn_q_mask = torch.zeros_like(tgt[..., 0]).bool()
        dyn_q_mask[:, pad_size + num_query_pts:pad_size + self.num_query] = 1
        dyn_q_mask[:, pad_size:] &= tgt_query_mask
        dyn_q_mask_img = dyn_q_mask[:, pad_size + num_query_pts:pad_size + self.num_query]

        # 提取动态查询的坐标和概率
        dyn_q_coords = query_coords[dyn_q_mask_img]
        dyn_q_probs = query_probs[dyn_q_mask_img]

        # ==========================================================
        # 2. Transformer 计算
        # ==========================================================
        outs_dec, reference_points, dyn_q_logits = self.transformer(
            tgt, query_pos, attn_mask,  # 查询特征和位置
            # 图像特征
            feat_flatten_img, spatial_flatten_img, level_start_index_img, self.pc_range, img_metas, data['lidar2img'],
            # 点云特征
            feat_flatten_pts=feat_flatten_pts, pos_flatten_pts=pos_flatten_pts,
            # 时序记忆
            temp_memory=temp_memory, temp_pos=temp_pos,
            # 其他参数
            cross_attn_masks=None, reference_points=reference_points,
            # 动态查询相关
            dyn_q_coords=dyn_q_coords, dyn_q_probs=dyn_q_probs, dyn_q_mask=dyn_q_mask, dyn_q_pos_branch=self.dyn_q_pos,
            dyn_q_pos_with_prob_branch=self.dyn_q_pos_with_prob, dyn_q_prob_branch=self.dyn_q_prob_branch,
        )

        # ==========================================================
        # 3. 结果生成 (Head Predictions)
        # ==========================================================
        outs_dec = torch.nan_to_num(outs_dec)  # 处理NaN值
        outputs_classes = []  # 存储各层分类结果
        outputs_coords = []  # 存储各层回归结果
        for lvl in range(outs_dec.shape[0]):  # 遍历Transformer解码层
            reference = inverse_sigmoid(reference_points[lvl].clone())  # 获取当前层的参考点（归一化坐标）
            outputs_class = self.cls_branches[lvl](outs_dec[lvl])  # 分类分支
            tmp = self.reg_branches[lvl](outs_dec[lvl])  # 回归分支
            tmp[..., 0:3] += reference[..., 0:3]  # 将相对偏移加到参考点上
            tmp[..., 0:3] = tmp[..., 0:3].sigmoid()  # sigmoid激活，将坐标归一化到[0,1]
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        # 堆叠各层结果
        all_cls_scores = torch.stack(outputs_classes)  # [层数, B, N_query, num_classes]
        all_bbox_preds = torch.stack(outputs_coords)  # [层数, B, N_query, 7] (xyzwhdl+旋转)

        # 将归一化坐标还原到真实世界坐标
        # pc_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        all_bbox_preds[..., 0:3] = (
                all_bbox_preds[..., 0:3] * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3])

        # ==========================================================
        # 4. 原型学习特征计算（仅训练时）   新增
        # 逻辑顺序：Proj -> Norm -> L2 Norm -> Sim
        # ==========================================================
        proto_sims = None  # 原型相似度
        projected_feats = None  # 投影后的特征

        if self.training:  # 仅在训练时计算原型特征
            self.local_iter += 1  # 更新迭代计数器

            # 取最后一层分类头最终Linear之前的语义特征: [B, N, C]
            # 保持256维，并与0710分类特征原型训练的数据流一致
            last_feat = outs_dec[-1]
            for layer in list(self.cls_branches[-1].children())[:-1]:
                last_feat = layer(last_feat)

            # 1. 投影 (Projection)
            B, N, C = last_feat.shape  # last_feat: [B, N, C]
            feat_flat = last_feat.reshape(B * N, C)  # [B*N, C]
            proj_flat = self.proto_proj(feat_flat)  # [B*N, proj_dim]
            # 置换回原始维度
            proj_feat = proj_flat.reshape(B, N, -1)  # [B, N, proj_dim]

            # 2. 层归一化 (LayerNorm)
            proj_feat = self.feat_norm(proj_feat)

            # 3. L2 归一化（用于余弦相似度计算）
            # 使用 contrast.py 中的 l2_normalize 或者 F.normalize
            proj_feat_norm = l2_normalize(proj_feat)

            # 保存用于 Loss 计算的特征 (是经过 Proj+Norm+L2 后的特征)
            projected_feats = proj_feat_norm  # [B, N_query, embed_dims]

            # 4. 计算特征与原型的相似度
            # prototypes: [num_classes, num_prototype, embed_dims]
            # 展平: [num_classes * num_prototype, embed_dims]
            flat_protos = self.prototypes.view(-1, self.embed_dims)
            flat_protos_norm = l2_normalize(flat_protos)  # 原型也进行L2归一化

            # 矩阵乘法计算余弦相似度
            # [B, N, C] @ [C, K] -> [B, N, K]
            proto_sims = torch.matmul(proj_feat_norm, flat_protos_norm.t())

        # ==========================================================
        # 5. 后处理与输出整理
        # ==========================================================
        # 将无效查询的分类分数设置为很低的值
        # 确保在推理阶段（Inference），这些Padding出来的无效Query永远不会被选为目标

        # NMS后处理
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

            nms_tgt_query_mask &= all_cls_scores[-1, :, pad_size:].sigmoid().max(-1).values > score_thr
            tgt_query_mask &= nms_tgt_query_mask

            # 应用NMS后重新掩码
            all_cls_scores[:, :, pad_size:][:, ~tgt_query_mask] = -40
            all_bbox_preds[:, :, pad_size:][:, ~tgt_query_mask] = 0

        # 更新记忆（用于下一帧）
        self.post_update_memory(data, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec, mask_dict,
                                query_mask=tgt_query_mask, instance_inds=None)

        # ==================== 5.1 处理去噪训练padding ====================
        if mask_dict and mask_dict['pad_size'] > 0:
            # 分离已知样本（去噪训练使用的噪声样本）
            output_known_class = all_cls_scores[:, :, :mask_dict['pad_size'], :]
            output_known_coord = all_bbox_preds[:, :, :mask_dict['pad_size'], :]
            # 裁剪掉padding部分
            all_cls_scores = all_cls_scores[:, :, mask_dict['pad_size']:, :]
            all_bbox_preds = all_bbox_preds[:, :, mask_dict['pad_size']:, :]
            # 保存到mask_dict供损失计算使用
            mask_dict['output_known_lbs_bboxes'] = (output_known_class, output_known_coord)

            if self.training:
                # 裁剪原型特征对应的padding部分
                projected_feats = projected_feats[:, mask_dict['pad_size']:, :]
                proto_sims = proto_sims[:, mask_dict['pad_size']:, :]
        else:
            mask_dict = None

        # ==================== 5.2 整理输出字典 ====================
        outs = {
            'all_cls_scores': all_cls_scores,  # 所有层的分类分数
            'all_bbox_preds': all_bbox_preds,  # 所有层的边界框预测
            'dyn_cls_scores': all_cls_scores,  # 动态查询的分类分数（别名）
            'dyn_bbox_preds': all_bbox_preds,  # 动态查询的回归结果（别名）
            'dn_mask_dict': mask_dict,  # 去噪训练相关信息
            'projected_feats': projected_feats,  # 投影后的融合特征（训练用）
            'proto_sims': proto_sims  # 原型相似度（训练用）
        }
        return outs

    def loss_detection_only(self, gt_bboxes_list, gt_labels_list, preds_dicts, gt_bboxes_ignore=None):
        """
        只计算检测损失（分类+回归），不计算原型损失
        用于无标签样本的伪标签监督
        """
        # 直接调用父类的 loss（MV2DFusionHead.loss）
        # 父类的 loss 只计算检测损失，没有原型损失
        return super().loss(gt_bboxes_list, gt_labels_list, preds_dicts, gt_bboxes_ignore)

    @force_fp32(apply_to=('preds_dicts'))
    # 装饰器：强制使用FP32精度计算损失，防止数值问题
    def loss(self, gt_bboxes_list, gt_labels_list, preds_dicts, gt_bboxes_ignore=None):
        """
                计算总损失，包括检测损失和原型损失
                参数:
                - gt_bboxes_list: 真实边界框列表（每个样本一个）
                - gt_labels_list: 真实标签列表
                - preds_dicts: forward的输出字典
                - gt_bboxes_ignore: 需要忽略的真实框
                返回:
                - loss_dict: 包含各项损失的字典
                """
        # 过滤掉没有标签的样本索引
        labeled_idx = [i for i, labels in enumerate(gt_labels_list) if len(labels) > 0]
        # 如果整个 Batch 都没有标签，直接返回空 Loss 字典
        if len(labeled_idx) == 0:
            return dict(loss_dummy=preds_dicts['all_cls_scores'].sum() * 0)

        gt_bboxes_list = [gt_bboxes_list[i] for i in labeled_idx]
        gt_labels_list = [gt_labels_list[i] for i in labeled_idx]
        # ==================== 1. 基础检测损失 ====================
        # 调用父类的loss方法计算检测损失
        loss_dict = super().loss(gt_bboxes_list, gt_labels_list, preds_dicts, gt_bboxes_ignore)

        # 如果不是训练模式，只返回检测损失
        if not self.training:
            return loss_dict

        # ==================== 2. 原型损失准备 ====================
        # 获取最后一层的预测结果。只取有标签的
        cls_scores = preds_dicts['all_cls_scores'][-1][labeled_idx]  # 只取有标签的
        bbox_preds = preds_dicts['all_bbox_preds'][-1][labeled_idx]  # 只取有标签的
        projected_feats = preds_dicts['projected_feats'][labeled_idx]
        proto_sims = preds_dicts['proto_sims'][labeled_idx]

        # 如果没有原型特征，直接返回检测损失
        if projected_feats is None or proto_sims is None:
            return loss_dict

        # ==================== 3. 基于距离的贪心匹配获取正样本 ====================
        num_imgs = len(labeled_idx)
        device = cls_scores.device

        # 准备存储正样本
        batch_pos_feats = []
        batch_pos_labels = []
        batch_pos_sims = []
        batch_pos_pred_scores = []
        all_matched_gt_ids_list = []  # 新增

        for i in range(num_imgs):
            # 1. 准备贪心匹配需要的数据
            # 注意：bbox_preds 必须是反归一化后的真实世界坐标
            cur_bbox_preds = bbox_preds[i]  # [N_query, 10+]
            cur_cls_scores = cls_scores[i]  # [N_query, num_classes]
            # GT 处理：提取 3D 中心点坐标
            cur_gt_centers = gt_bboxes_list[i].gravity_center.to(device)  # [M, 3]
            cur_gt_labels = gt_labels_list[i]

            cur_gt_boxes = gt_bboxes_list[i].tensor.to(device)  # 形状 [M, 7] 或 [M, 9]

            pos_inds, pos_assigned_gt_inds, matched_gt_ids = self._greedy_match_by_distance(
                pred_boxes=cur_bbox_preds,
                pred_scores=cur_cls_scores,
                gt_centers=cur_gt_centers,
                gt_labels=cur_gt_labels,
                gt_boxes=cur_gt_boxes,
                return_matched_ids=True
            )
            all_matched_gt_ids_list.append(matched_gt_ids)  # 保存

            if pos_inds.numel() == 0:
                continue

            # 3. 获取正样本对应的真实标签
            pos_gt_labels = cur_gt_labels[pos_assigned_gt_inds]

            # 4. 获取正样本的特征和相似度
            final_pos_feat = projected_feats[i][pos_inds]
            final_pos_sim = proto_sims[i][pos_inds]
            final_pos_labels = pos_gt_labels
            final_pos_scores = cur_cls_scores[pos_inds]

            batch_pos_feats.append(final_pos_feat)
            batch_pos_labels.append(final_pos_labels)
            batch_pos_sims.append(final_pos_sim)
            batch_pos_pred_scores.append(final_pos_scores)

        # ==================== 4. 原型损失计算（带预热） ====================
        positive_similarity_by_class = {}
        if len(batch_pos_feats) > 0:
            # 拼接所有正样本
            all_pos_feats = torch.cat(batch_pos_feats, dim=0)  # [总正样本数, D]
            all_pos_labels = torch.cat(batch_pos_labels, dim=0)  # [总正样本数]
            all_pos_sims = torch.cat(batch_pos_sims, dim=0)  # [总正样本数, total_prototypes]
            all_pos_pred_scores = torch.cat(batch_pos_pred_scores, dim=0)  # [total_pos, num_classes]
            all_pos_sims_by_class = all_pos_sims.view(
                -1, self.num_classes, self.num_prototype)
            positive_rows = torch.arange(
                all_pos_labels.shape[0], device=all_pos_labels.device)
            positive_own_max = all_pos_sims_by_class[
                positive_rows, all_pos_labels].max(dim=-1).values
            for class_id in self.active_proto_class_ids:
                class_mask = all_pos_labels == class_id
                if class_mask.any():
                    positive_similarity_by_class[class_id] = (
                        positive_own_max[class_mask])

            # 核心原型学习
            (
                contrast_logits,
                contrast_targets,
                proto_band_value,
                proto_inter_value,
            ) = self.prototype_learning(
                all_pos_feats,
                all_pos_labels,
                all_pos_sims,
                all_pos_pred_scores,
            )

            # 计算交叉熵损失
            temperature = 0.07  # 沿用 CPDet3D 默认值
            valid_proto_targets = contrast_targets != -100
            if valid_proto_targets.any():
                loss_proto = F.cross_entropy(
                    contrast_logits[valid_proto_targets] / temperature,
                    contrast_targets[valid_proto_targets],
                )
            else:
                loss_proto = contrast_logits.sum() * 0.0

            loss_dict['loss_proto'] = loss_proto * self.proto_loss_weight
            loss_dict['loss_proto_intra_band'] = (
                contrast_logits.sum() * 0.0
                + proto_band_value * self.proto_intra_weight)
            loss_dict['loss_proto_inter_upper'] = (
                contrast_logits.sum() * 0.0
                + proto_inter_value * self.proto_inter_weight)
        else:
            # 如果没有正样本，原型损失为0（但需要保持计算图防止DDP死锁）
            loss_dict['loss_proto'] = cls_scores.sum() * 0.0
            loss_dict['loss_proto_intra_band'] = cls_scores.sum() * 0.0
            loss_dict['loss_proto_inter_upper'] = cls_scores.sum() * 0.0

        # ========================== 负样本损失（Margin Ranking Loss） ==========================
        # 目标：让负样本与所有原型的相似度尽可能低于 margin
        # 获取最后一层的 proto_sims 和 cls_scores（已经去除了 denoising 部分）
        proto_sims = preds_dicts['proto_sims'][labeled_idx]  # [B, N, total_proto]
        cls_scores = preds_dicts['all_cls_scores'][-1][labeled_idx]  # [B, N, num_classes]

        negative_similarity_parts = {
            class_id: [] for class_id in self.active_proto_class_ids
        }
        for i in range(len(labeled_idx)):
            matched_ids = all_matched_gt_ids_list[i]
            cur_proto_sims = proto_sims[i].view(
                -1, self.num_classes, self.num_prototype)
            cur_cls_scores = cls_scores[i]
            cur_pred_cls = cur_cls_scores.argmax(dim=-1)
            cur_bbox_preds = bbox_preds[i]
            cur_gt_centers = gt_bboxes_list[i].gravity_center.to(device)

            if cur_gt_centers.numel() == 0:
                nearest_gt_distance = cur_bbox_preds.new_full(
                    (cur_bbox_preds.shape[0],), float('inf'))
            else:
                nearest_gt_distance = torch.cdist(
                    cur_bbox_preds[:, :2],
                    cur_gt_centers[:, :2],
                    p=2,
                ).min(dim=1).values

            active_class_mask = torch.zeros(
                self.num_classes,
                dtype=torch.bool,
                device=cur_pred_cls.device,
            )
            active_class_mask[list(self.active_proto_class_ids)] = True
            clear_negative_mask = (
                (matched_ids == -1)
                & (nearest_gt_distance > self.neg_proto_gt_dist_th)
                & active_class_mask[cur_pred_cls]
            )

            for class_id in self.active_proto_class_ids:
                class_mask = clear_negative_mask & (cur_pred_cls == class_id)
                if not class_mask.any():
                    continue
                class_similarity = cur_proto_sims[
                    class_mask, class_id].max(dim=-1).values
                negative_similarity_parts[class_id].append(class_similarity)

        negative_similarity_by_class = {}
        class_negative_losses = []
        for class_id, similarity_parts in negative_similarity_parts.items():
            if not similarity_parts:
                continue
            class_similarity = torch.cat(similarity_parts, dim=0)
            negative_similarity_by_class[class_id] = class_similarity
            hard_mask = class_similarity > self.neg_proto_margin
            if not hard_mask.any():
                continue
            violation = (
                class_similarity[hard_mask] - self.neg_proto_margin)
            class_weight = self.neg_proto_class_weight[
                class_id].detach().clone()
            class_negative_losses.append(
                violation.square().mean() * class_weight)

        if class_negative_losses:
            loss_dict['loss_neg_proto'] = torch.stack(
                class_negative_losses).mean()
        else:
            loss_dict['loss_neg_proto'] = cls_scores.sum() * 0.0

        self._accumulate_negative_monitor(
            negative_similarity_by_class,
            positive_similarity_by_class,
        )
        neg_monitor_report = self._maybe_adjust_negative_weight()
        clear_negative_count = sum(
            similarities.numel()
            for similarities in negative_similarity_by_class.values())
        hard_negative_count = sum(
            (similarities > self.neg_proto_margin).sum().item()
            for similarities in negative_similarity_by_class.values())
        if negative_similarity_by_class:
            batch_negative_class_means = torch.stack([
                similarities.detach().mean()
                for similarities in negative_similarity_by_class.values()
            ])
            batch_negative_mean = float(
                batch_negative_class_means.mean().item())
        else:
            batch_negative_mean = float('nan')

        if self.training and self.local_iter % 100 == 0:
            from mmcv.utils import get_logger
            logger = get_logger('mmdet')

            # 1. 基础统计
            total_gt = sum(len(gt_labels) for gt_labels in gt_labels_list)
            total_pos = len(torch.cat(batch_pos_labels)) if batch_pos_labels else 0
            total_neg = sum((matched_ids == -1).sum().item() for matched_ids in all_matched_gt_ids_list)
            neg_loss_val = loss_dict.get('loss_neg_proto', 0).item()
            proto_loss_val = loss_dict.get('loss_proto', 0).item()

            # 统计 NMS 后置信度为0的查询向量数量
            # 注意：NMS 将抑制框的分数设为 -40，sigmoid 后 ≈ 0
            zero_conf_count = 0
            for i in range(len(labeled_idx)):
                cur_cls_scores = cls_scores[i]  # [N, num_classes]
                max_prob = cur_cls_scores.sigmoid().max(dim=-1)[0]  # [N]
                zero_conf_count += (max_prob < 0.01).sum().item()  # 极低置信度视为被抑制

            class_weight_text = ','.join(
                f'{class_id}:'
                f'{float(self.neg_proto_class_weight[class_id].item()):.4f}'
                for class_id in self.active_proto_class_ids)
            logger.info(f"\n[Iter {int(self.local_iter.item())}] "
                        f"GT: {total_gt} | Pos: {total_pos} | Neg: {total_neg} | "
                        f"ClearNeg: {clear_negative_count} | "
                        f"HardNeg: {hard_negative_count} | "
                        f"NegMeanSim: {batch_negative_mean:.4f} | "
                        f"NegWeights: [{class_weight_text}] | "
                        f"ZeroConf (suppressed): {zero_conf_count} | "
                        f"ProtoLoss: {proto_loss_val:.4f} | NegLoss: {neg_loss_val:.4f}")

            if neg_monitor_report:
                for report in neg_monitor_report:
                    logger.info(
                        "[NegativeWeightMonitor] "
                        f"class={report['class_id']} | "
                        f"NegSamples={report['negative_count']} | "
                        f"PosSamples={report['positive_count']} | "
                        f"NegMean={report['negative_mean']:.4f} | "
                        f"NegRatio>{self.neg_proto_margin:.2f}="
                        f"{report['negative_ratio']:.4f} | "
                        f"PosMean={report['positive_mean']:.4f} | "
                        f"PosDrop={report['positive_drop']:.4f} | "
                        f"MeanImprove="
                        f"{report['mean_improvement']:.4f} | "
                        f"RatioImprove="
                        f"{report['ratio_improvement']:.4f} | "
                        f"Weight={report['old_weight']:.4f}"
                        f"->{report['new_weight']:.4f} | "
                        f"Action={report['action']} | "
                        f"Guard={report['guard_reason']}")

            # 2. 详细匹配信息：每个样本中匹配成功的查询向量（按置信度排序，并显示距离）
            logger.info("Detailed matching info (per sample):")
            for i in range(len(labeled_idx)):
                matched_ids = all_matched_gt_ids_list[i]
                pos_mask = (matched_ids != -1)
                if not pos_mask.any():
                    continue

                cur_cls_scores = cls_scores[i]
                cur_gt_labels = gt_labels_list[i]
                cur_bbox_preds = bbox_preds[i]
                cur_gt_centers = gt_bboxes_list[i].gravity_center.to(device)

                # 计算所有预测框与所有 GT 中心的距离矩阵（用于快速获取匹配距离）
                pred_centers_2d = cur_bbox_preds[:, :2]
                gt_centers_2d = cur_gt_centers[:, :2]
                dist_mat = torch.cdist(pred_centers_2d, gt_centers_2d, p=2)  # [N, M]

                match_info = []
                for idx in torch.where(pos_mask)[0]:
                    pred_cls = cur_cls_scores[idx].argmax(dim=-1).item()
                    gt_idx = matched_ids[idx].item()
                    gt_cls = cur_gt_labels[gt_idx].item()
                    # conf = cur_cls_scores[idx].softmax(dim=-1)[pred_cls].item()
                    conf = cur_cls_scores[idx].sigmoid()[pred_cls].item()
                    # 获取预测框与匹配 GT 的中心距离
                    dist = dist_mat[idx, gt_idx].item()
                    match_info.append((conf, pred_cls, gt_cls, dist))

                match_info.sort(key=lambda x: x[0], reverse=True)

                logger.info(f"  Sample {i}: {len(match_info)} matches")
                for conf, pred_cls, gt_cls, dist in match_info:
                    logger.info(f"    conf={conf:.4f}, pred_cls={pred_cls}, gt_cls={gt_cls}, dist={dist:.2f}m")

            # 3. 未匹配GT分析：列出没有匹配到任何查询向量的GT及其最近查询信息
            logger.info("Unmatched GT analysis (per sample):")
            for i in range(len(labeled_idx)):
                matched_ids = all_matched_gt_ids_list[i]
                cur_gt_centers = gt_bboxes_list[i].gravity_center.to(device)
                cur_gt_labels = gt_labels_list[i]
                cur_bbox_preds = bbox_preds[i]
                cur_cls_scores = cls_scores[i]

                # 已匹配的GT索引集合
                matched_gt_set = set(matched_ids[matched_ids != -1].cpu().tolist())
                unmatched_gt_indices = [j for j in range(len(cur_gt_labels)) if j not in matched_gt_set]

                if not unmatched_gt_indices:
                    logger.info(f"  Sample {i}: all GT matched.")
                    continue

                # 计算所有预测框与所有GT的中心距离（BEV）
                pred_centers_2d = cur_bbox_preds[:, :2]  # [N,2]
                gt_centers_2d = cur_gt_centers[:, :2]  # [M,2]
                dist_mat = torch.cdist(pred_centers_2d, gt_centers_2d, p=2)  # [N, M]

                for gt_idx in unmatched_gt_indices:
                    gt_cls = cur_gt_labels[gt_idx].item()
                    distances = dist_mat[:, gt_idx]  # [N]
                    min_dist, nearest_pred_idx = torch.min(distances, dim=0)
                    nearest_pred_idx = nearest_pred_idx.item()

                    pred_cls = cur_cls_scores[nearest_pred_idx].argmax(dim=-1).item()
                    # pred_conf = cur_cls_scores[nearest_pred_idx].softmax(dim=-1)[pred_cls].item()
                    pred_conf = cur_cls_scores[nearest_pred_idx].sigmoid()[pred_cls].item()
                    matched_gt_of_this_pred = matched_ids[nearest_pred_idx].item()
                    is_matched = "✓" if matched_gt_of_this_pred != -1 else "✗"
                    matched_gt_cls = cur_gt_labels[
                        matched_gt_of_this_pred].item() if matched_gt_of_this_pred != -1 else None

                    logger.info(
                        f"  Sample {i}: GT_cls={gt_cls} | "
                        f"nearest pred: conf={pred_conf:.4f}, pred_cls={pred_cls}, "
                        f"dist={min_dist:.2f}m, this pred matched by GT={is_matched} "
                        f"(matched_GT_cls={matched_gt_cls if matched_gt_cls is not None else 'None'})"
                    )
            logger.info("")  # 空行分隔

        if (self.training
                and self.local_iter % self.prototype_save_interval == 0):
            save_dir = self.prototype_save_dir
            os.makedirs(save_dir, exist_ok=True)
            # 保存当前原型，形状 [num_classes, num_prototype, embed_dims]
            proto_data = self.prototypes.data.cpu().clone()
            torch.save({
                'iter': self.local_iter.item(),
                'prototypes': proto_data,
                'num_classes': self.num_classes,
                'num_prototype': self.num_prototype,
                'proto_update_count': self.proto_update_count.cpu().clone(),
                'proto_assignment_mass': (
                    self.proto_assignment_mass.cpu().clone()),
                'proto_hard_assignment_count': (
                    self.proto_hard_assignment_count.cpu().clone()),
                'proto_feature_queue_count': (
                    self.proto_feature_queue_count.cpu().clone()),
                'neg_proto_class_weight': (
                    self.neg_proto_class_weight.cpu().clone()),
            }, f'{save_dir}/prototypes_iter_{int(self.local_iter.item())}.pth')

        return loss_dict

    # 原始融合特征 → 投影头 → LayerNorm → L2归一化 → 计算相似度 → Sinkhorn分配 → 动量更新
