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
from mmcv.ops import box_iou_rotated

import os
from collections import deque

# 外部依赖导入
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
                 sim_thr=0.7,  # 相似度阈值
                 score_thr=0.1,  # 得分阈值
                 proto_loss_weight=0.1,  # 原型 Loss 权重
                 proto_intra_weight=0.05,
                 active_proto_classes=(0, 1, 3, 6, 7, 8, 9),
                 overlap_thresholds=(0.7, 0.8, 1.0, 0.9, 1.0, 1.0, 0.1, 0.4, 0.6, 0.1),
                 small_object_classes=(6, 7, 8, 9),
                 proto_intra_upper_thresholds=(0.87, 0.95, 1.0, 0.99, 1.0, 1.0,
                                               0.97, 0.96, 0.87, 0.88),
                 inter_class_margin=0.20,
                 inter_class_loss_weight=0.05,
                 proto_inter_upper=0.40,
                 proto_inter_weight=0.01,
                 neg_proto_loss_weight=0.0,
                 neg_proto_score_threshold=0.2,
                 neg_proto_similarity_margin=0.7,
                 neg_proto_repulsion_rate=0.001,
                 positive_similarity_thresholds=(0.95, 0.95, 1.0, 0.95, 1.0,
                                                 1.0, 0.94, 0.93, 0.95, 0.92),
                 positive_similarity_floor=0.7,
                 dead_proto_check_interval=1000,
                 dead_proto_min_samples=50,
                 dead_proto_bank_size=2048,
                 dead_proto_patience_windows=2,
                 proto_probation_max_windows=3,
                 proto_maturity_min_age=1000,
                 proto_maturity_active_windows=2,
                 proto_maturity_history_windows=3,
                 proto_maturity_min_wins=3,
                 proto_maturity_max_wins=20,
                 proto_maturity_win_ratio=0.01,
                 reset_knn=5,
                 reset_density_threshold=0.85,
                 reset_max_coverage=0.92,
                 reset_candidate_fraction=0.25,
                 prototype_save_dir='work_dirs/prototype_vis_stage3_0718',
                 prototype_save_interval=2500,
                 greedy_dist_th=2,#贪心距离
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
        self.sim_thr = sim_thr
        self.score_thr = score_thr
        self.proto_loss_weight = proto_loss_weight
        self.proto_intra_weight = proto_intra_weight
        self.active_proto_classes = tuple(int(idx) for idx in active_proto_classes)
        self.small_object_classes = tuple(int(idx) for idx in small_object_classes)
        self.inter_class_margin = float(inter_class_margin)
        self.inter_class_loss_weight = float(inter_class_loss_weight)
        self.proto_inter_upper = float(proto_inter_upper)
        self.proto_inter_weight = float(proto_inter_weight)
        self.neg_proto_loss_weight = float(neg_proto_loss_weight)
        self.neg_proto_score_threshold = float(neg_proto_score_threshold)
        self.neg_proto_similarity_margin = float(
            neg_proto_similarity_margin)
        self.neg_proto_repulsion_rate = float(neg_proto_repulsion_rate)
        self.positive_similarity_floor = float(positive_similarity_floor)
        self.dead_proto_check_interval = int(dead_proto_check_interval)
        self.dead_proto_min_samples = int(dead_proto_min_samples)
        self.dead_proto_bank_size = int(dead_proto_bank_size)
        self.dead_proto_patience_windows = int(dead_proto_patience_windows)
        self.proto_probation_max_windows = int(proto_probation_max_windows)
        self.proto_maturity_min_age = int(proto_maturity_min_age)
        self.proto_maturity_active_windows = int(
            proto_maturity_active_windows)
        self.proto_maturity_history_windows = int(
            proto_maturity_history_windows)
        self.proto_maturity_min_wins = int(proto_maturity_min_wins)
        self.proto_maturity_max_wins = int(proto_maturity_max_wins)
        self.proto_maturity_win_ratio = float(proto_maturity_win_ratio)
        self.reset_knn = int(reset_knn)
        self.reset_density_threshold = float(reset_density_threshold)
        self.reset_max_coverage = float(reset_max_coverage)
        self.reset_candidate_fraction = float(reset_candidate_fraction)
        self.prototype_save_dir = str(prototype_save_dir)
        self.prototype_save_interval = int(prototype_save_interval)
        if len(overlap_thresholds) != self.num_classes:
            raise ValueError('overlap_thresholds must contain one value per class')
        if len(proto_intra_upper_thresholds) != self.num_classes:
            raise ValueError(
                'proto_intra_upper_thresholds must contain one value per class')
        if len(positive_similarity_thresholds) != self.num_classes:
            raise ValueError(
                'positive_similarity_thresholds must contain one value per class')
        if any(idx < 0 or idx >= self.num_classes for idx in self.active_proto_classes):
            raise ValueError('active_proto_classes contains an invalid class index')
        if any(idx < 0 or idx >= self.num_classes for idx in self.small_object_classes):
            raise ValueError('small_object_classes contains an invalid class index')
        self.greedy_dist_th = greedy_dist_th

        # local_iter兼容旧checkpoint；stage2_iter从0开始统计Stage2训练。
        self.register_buffer('local_iter', torch.zeros(1))
        self.register_buffer('stage2_iter', torch.zeros(1))
        self.register_buffer('stage3_iter', torch.zeros(1))
        self.register_buffer(
            'overlap_threshold_tensor',
            torch.tensor(overlap_thresholds, dtype=torch.float32))
        self.register_buffer(
            'proto_intra_upper_tensor',
            torch.tensor(proto_intra_upper_thresholds, dtype=torch.float32))
        self.register_buffer(
            'positive_similarity_threshold_tensor',
            torch.tensor(positive_similarity_thresholds, dtype=torch.float32))
        active_mask = torch.zeros(self.num_classes, dtype=torch.bool)
        active_mask[list(self.active_proto_classes)] = True
        self.register_buffer('active_proto_class_mask', active_mask)
        small_mask = torch.zeros(self.num_classes, dtype=torch.bool)
        small_mask[list(self.small_object_classes)] = True
        self.register_buffer('small_object_class_mask', small_mask)
        self.register_buffer(
            'proto_update_count',
            torch.zeros(self.num_classes, self.num_prototype, dtype=torch.long))
        self.register_buffer(
            'proto_assignment_mass',
            torch.zeros(self.num_classes, self.num_prototype))
        self.register_buffer(
            'stage2_winner_count_window',
            torch.zeros(self.num_classes, self.num_prototype, dtype=torch.long))
        self.register_buffer(
            'stage2_winner_count_total',
            torch.zeros(self.num_classes, self.num_prototype, dtype=torch.long))
        self.register_buffer(
            'stage2_weighted_update_mass',
            torch.zeros(self.num_classes, self.num_prototype))
        self.register_buffer(
            'stage2_overlap_sum',
            torch.zeros(self.num_classes, self.num_prototype))
        self.register_buffer(
            'stage2_overlap_count',
            torch.zeros(self.num_classes, self.num_prototype, dtype=torch.long))
        self.register_buffer(
            'stage2_reset_count',
            torch.zeros(self.num_classes, self.num_prototype, dtype=torch.long))
        self.register_buffer(
            'stage2_valid_sample_count_window',
            torch.zeros(self.num_classes, dtype=torch.long))
        self.register_buffer(
            'stage2_last_window_winner_count',
            torch.zeros(self.num_classes, self.num_prototype, dtype=torch.long))
        self.register_buffer(
            'stage2_last_window_valid_sample_count',
            torch.zeros(self.num_classes, dtype=torch.long))
        self.register_buffer(
            'proto_positive_anchor',
            torch.zeros(self.num_classes, self.num_prototype, self.embed_dims))
        self.register_buffer(
            'proto_positive_anchor_valid',
            torch.zeros(self.num_classes, self.num_prototype, dtype=torch.bool))
        self.register_buffer(
            'proto_state',
            torch.zeros(self.num_classes, self.num_prototype, dtype=torch.long))
        self.register_buffer(
            'proto_age_since_reset',
            torch.zeros(self.num_classes, self.num_prototype, dtype=torch.long))
        self.register_buffer(
            'proto_winner_since_reset',
            torch.zeros(self.num_classes, self.num_prototype, dtype=torch.long))
        self.register_buffer(
            'proto_support_mass_since_reset',
            torch.zeros(self.num_classes, self.num_prototype))
        self.register_buffer(
            'proto_active_window_history',
            torch.zeros(self.num_classes, self.num_prototype, dtype=torch.long))
        self.register_buffer(
            'proto_empty_window_count',
            torch.zeros(self.num_classes, self.num_prototype, dtype=torch.long))
        self.register_buffer(
            'proto_probation_window_count',
            torch.zeros(self.num_classes, self.num_prototype, dtype=torch.long))
        self.register_buffer(
            'stage3_state_initialized',
            torch.zeros(1, dtype=torch.bool))
        self._recent_feature_bank = {
            class_idx: deque() for class_idx in self.active_proto_classes
        }
        self._recent_feature_bank_size = {
            class_idx: 0 for class_idx in self.active_proto_classes
        }

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
                                  dist_th=2.0,
                                  iou_th=0.25,
                                  conf_thr=0.3,  # 新增：置信度阈值
                                  return_matched_ids=True):
        """
        全局贪心匹配（不区分类别）：
          1. 计算预测置信度，筛选出 conf_thr 以上的高置信度框。
          2. 高置信度框按置信度降序排序，依次匹配未占用的 GT。
          3. 匹配条件：中心距离 < dist_th，且（如果启用 IoU）IoU > iou_th。
          4. 每个 GT 最多匹配一个预测框，每个预测框最多匹配一个 GT。
          5. 低置信度框直接标记为负样本（不参与匹配）。
        返回:
            pos_inds: 正样本索引（仅来自高置信度且匹配成功的框）
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

        # 1. 计算预测置信度
        # pred_conf = pred_scores.softmax(dim=-1).max(dim=-1)[0]  # [N]
        pred_conf = pred_scores.sigmoid().max(dim=-1)[0]  # [N]

        # 2. 根据置信度阈值筛选高置信度框
        high_conf_mask = pred_conf > conf_thr
        if not high_conf_mask.any():
            # 没有高置信度框，所有预测框都是负样本
            matched_gt_ids = torch.full((N,), -1, dtype=torch.long, device=device)
            pos_inds = torch.empty(0, dtype=torch.long, device=device)
            pos_assigned_gt_inds = torch.empty(0, dtype=torch.long, device=device)
            if return_matched_ids:
                return pos_inds, pos_assigned_gt_inds, matched_gt_ids
            else:
                return pos_inds, pos_assigned_gt_inds

        high_conf_idx = torch.where(high_conf_mask)[0]  # 原始索引
        high_conf_boxes = pred_boxes[high_conf_mask]  # [H, code_size]
        high_conf_scores = pred_scores[high_conf_mask]  # [H, num_classes]
        high_conf_conf = pred_conf[high_conf_mask]  # [H]

        H = high_conf_boxes.shape[0]

        # 3. 计算距离矩阵和 IoU 矩阵（仅针对高置信度框）
        pred_centers_2d = high_conf_boxes[:, :2]  # [H, 2]
        gt_centers_2d = gt_centers[:, :2]  # [M, 2]
        dist_mat = torch.cdist(pred_centers_2d, gt_centers_2d, p=2)  # [H, M]

        iou_mat = None
        if gt_boxes is not None:
            pred_bev = torch.stack([high_conf_boxes[:, 0], high_conf_boxes[:, 1],
                                    high_conf_boxes[:, 3], high_conf_boxes[:, 4],
                                    high_conf_boxes[:, 6]], dim=-1)  # [H,5]
            gt_bev = torch.stack([gt_boxes[:, 0], gt_boxes[:, 1],
                                  gt_boxes[:, 3], gt_boxes[:, 4],
                                  gt_boxes[:, 6]], dim=-1)  # [M,5]
            iou_mat = box_iou_rotated(pred_bev, gt_bev)  # [H, M]

        # 4. 贪心匹配
        high_matched_gt_ids = torch.full((H,), -1, dtype=torch.long, device=device)
        gt_matched = torch.zeros(M, dtype=torch.bool, device=device)

        # 按置信度降序排序（在高置信度框内）
        sorted_idx = torch.argsort(high_conf_conf, descending=True)  # [H]
        for li in sorted_idx.tolist():
            avail_gt = ~gt_matched
            if not avail_gt.any():
                break
            dist_i = dist_mat[li, avail_gt]  # [num_avail]
            dist_ok = dist_i < dist_th
            if not dist_ok.any():
                continue
            if iou_mat is not None:
                iou_i = iou_mat[li, avail_gt]  # [num_avail]
                iou_ok = iou_i > iou_th
                mask = dist_ok & iou_ok
            else:
                mask = dist_ok
            if not mask.any():
                continue
            valid_dist = dist_i.clone()
            valid_dist[~mask] = float('inf')
            min_dist_val, min_local_idx = torch.min(valid_dist, dim=0)
            global_gt_idx = torch.where(avail_gt)[0][min_local_idx].item()
            high_matched_gt_ids[li] = global_gt_idx
            gt_matched[global_gt_idx] = True

        # 5. 映射回原始索引
        matched_gt_ids = torch.full((N,), -1, dtype=torch.long, device=device)
        for local_i, global_i in enumerate(high_conf_idx):
            matched_gt_ids[global_i] = high_matched_gt_ids[local_i]

        # 6. 提取正样本索引（仅来自高置信度且匹配成功的框）
        pos_mask = (matched_gt_ids != -1) & high_conf_mask
        pos_inds = torch.where(pos_mask)[0]
        pos_assigned_gt_inds = matched_gt_ids[pos_inds]

        if return_matched_ids:
            return pos_inds, pos_assigned_gt_inds, matched_gt_ids
        else:
            return pos_inds, pos_assigned_gt_inds

    #pos_inds 记录了哪些预测框被判定为正样本（即成功匹配到了某个 GT）。
    ##pos_assigned_gt_inds 记录了每个正样本对应的 GT 的索引（即该 GT 在 gt_bboxes_list[i] 中的位置）。

    # ======================================Stage2原型学习==================================
    def _matched_gt_overlap_ratio(self, pred_boxes, gt_boxes,
                                  pos_inds, assigned_gt_inds):
        """Return BEV intersection area divided by matched GT BEV area."""
        if pos_inds.numel() == 0:
            return pred_boxes.new_zeros((0,))
        decoded_pred = denormalize_bbox(pred_boxes[pos_inds], None)
        matched_gt = gt_boxes[assigned_gt_inds]
        pred_bev = decoded_pred[:, [0, 1, 3, 4, 6]]
        gt_bev = matched_gt[:, [0, 1, 3, 4, 6]]
        pairwise_iou = box_iou_rotated(pred_bev, gt_bev)
        pair_iou = pairwise_iou.diagonal()
        pred_area = pred_bev[:, 2].clamp_min(1e-6) * pred_bev[:, 3].clamp_min(1e-6)
        gt_area = gt_bev[:, 2].clamp_min(1e-6) * gt_bev[:, 3].clamp_min(1e-6)
        intersection = pair_iou * (pred_area + gt_area) / (
            1.0 + pair_iou).clamp_min(1e-6)
        return (intersection / gt_area).clamp_(0.0, 1.0)

    def _quality_weights(self, overlaps, labels):
        small_mask = self.small_object_class_mask[labels]
        weights = overlaps.clone()
        weights[small_mask] = overlaps[small_mask].clamp_min(0.0).sqrt()
        return weights

    def _append_recent_features(self, class_idx, features):
        if class_idx not in self._recent_feature_bank or features.numel() == 0:
            return
        chunk = features.detach().float().cpu()
        bank = self._recent_feature_bank[class_idx]
        bank.append(chunk)
        self._recent_feature_bank_size[class_idx] += len(chunk)
        while self._recent_feature_bank_size[class_idx] > self.dead_proto_bank_size:
            overflow = (
                self._recent_feature_bank_size[class_idx]
                - self.dead_proto_bank_size)
            first = bank[0]
            if len(first) <= overflow:
                bank.popleft()
                self._recent_feature_bank_size[class_idx] -= len(first)
            else:
                bank[0] = first[overflow:]
                self._recent_feature_bank_size[class_idx] -= overflow

    def _initialize_stage3_proto_states(self):
        if bool(self.stage3_state_initialized.item()):
            return
        with torch.no_grad():
            for class_idx in self.active_proto_classes:
                recent_usage = self.stage2_last_window_winner_count[class_idx]
                mature_mask = recent_usage > 0
                if not mature_mask.any():
                    total_usage = self.stage2_winner_count_total[class_idx]
                    mature_mask[torch.argmax(total_usage)] = True
                self.proto_state[class_idx, mature_mask] = 2
                self.proto_state[class_idx, ~mature_mask] = 0
            self.stage3_state_initialized.fill_(True)

    def _mature_mask(self, class_idx=None):
        mask = self.proto_state == 2
        if class_idx is None:
            return mask
        return mask[class_idx]

    @staticmethod
    def _count_active_windows(history, window_count):
        active_count = torch.zeros_like(history)
        for bit_idx in range(window_count):
            active_count += (history >> bit_idx) & 1
        return active_count

    def _density_reset_candidate(self, features, references):
        if features.numel() == 0:
            return None
        if references.numel():
            coverage = (
                features @ l2_normalize(references).t()
            ).max(dim=1).values
        else:
            coverage = features.new_full((len(features),), -1.0)
        uncovered = torch.where(coverage <= self.reset_max_coverage)[0]
        if uncovered.numel() < max(2, self.reset_knn):
            return None

        pool_size = max(
            self.reset_knn * 4,
            int(round(len(features) * self.reset_candidate_fraction)))
        pool_size = min(pool_size, int(uncovered.numel()))
        uncovered_coverage = coverage[uncovered]
        pool_local = torch.topk(
            uncovered_coverage, k=pool_size, largest=False).indices
        pool_indices = uncovered[pool_local]
        pool = features[pool_indices]
        pairwise = pool @ pool.t()
        pairwise.fill_diagonal_(-1.0)
        neighbor_count = min(self.reset_knn, max(len(pool) - 1, 1))
        neighbor_sims, neighbor_indices = torch.topk(
            pairwise, k=neighbor_count, dim=1)
        density = neighbor_sims.mean(dim=1)
        best_local = int(torch.argmax(density).item())
        if density[best_local].item() < self.reset_density_threshold:
            return None

        cluster_local = torch.cat([
            pool.new_tensor([best_local], dtype=torch.long),
            neighbor_indices[best_local],
        ])
        cluster = pool[cluster_local]
        center = l2_normalize(cluster.mean(dim=0, keepdim=True))[0]
        if references.numel():
            center_coverage = (
                center.unsqueeze(0) @ l2_normalize(references).t()
            ).max()
            if center_coverage.item() > self.reset_max_coverage:
                return None
        return center

    def _apply_proto_intra_upper(self, proto_k, class_idx):
        proto_k = l2_normalize(proto_k)
        upper = self.proto_intra_upper_tensor[class_idx]
        anchors = self.proto_positive_anchor[class_idx]
        anchor_valid = self.proto_positive_anchor_valid[class_idx]
        mature_mask = self._mature_mask(class_idx)
        value = proto_k.new_zeros(())
        pair_count = 0
        for first_idx in range(self.num_prototype):
            for second_idx in range(first_idx + 1, self.num_prototype):
                if not (mature_mask[first_idx] and mature_mask[second_idx]):
                    continue
                similarity = torch.dot(proto_k[first_idx], proto_k[second_idx])
                violation = torch.clamp(similarity - upper, min=0.0)
                value = value + violation.square()
                pair_count += 1
                if (self.proto_intra_weight <= 0
                        or violation.item() <= 0
                        or not anchor_valid[first_idx]
                        or not anchor_valid[second_idx]):
                    continue
                direction = anchors[second_idx] - anchors[first_idx]
                direction_norm = direction.norm()
                if direction_norm.item() <= 1e-6:
                    continue
                direction = direction / direction_norm
                step = self.proto_intra_weight * violation
                proto_k[first_idx] = proto_k[first_idx] - step * direction
                proto_k[second_idx] = proto_k[second_idx] + step * direction
                proto_k = l2_normalize(proto_k)
        if pair_count:
            value = value / pair_count
        return l2_normalize(proto_k), value.detach()

    def _apply_proto_inter_upper(self, protos):
        protos = l2_normalize(protos)
        value = protos.new_zeros(())
        pair_count = 0
        active_classes = list(self.active_proto_classes)
        for first_pos, first_class in enumerate(active_classes):
            for second_class in active_classes[first_pos + 1:]:
                for first_idx in range(self.num_prototype):
                    for second_idx in range(self.num_prototype):
                        if (not self._mature_mask(first_class)[first_idx]
                                or not self._mature_mask(second_class)[second_idx]):
                            continue
                        similarity = torch.dot(
                            protos[first_class, first_idx],
                            protos[second_class, second_idx])
                        violation = torch.clamp(
                            similarity - self.proto_inter_upper, min=0.0)
                        value = value + violation.square()
                        pair_count += 1
                        if (self.proto_inter_weight <= 0
                                or violation.item() <= 0
                                or not self.proto_positive_anchor_valid[
                                    first_class, first_idx]
                                or not self.proto_positive_anchor_valid[
                                    second_class, second_idx]):
                            continue
                        direction = (
                            self.proto_positive_anchor[second_class, second_idx]
                            - self.proto_positive_anchor[first_class, first_idx])
                        direction_norm = direction.norm()
                        if direction_norm.item() <= 1e-6:
                            continue
                        direction = direction / direction_norm
                        step = self.proto_inter_weight * violation
                        protos[first_class, first_idx] = (
                            protos[first_class, first_idx] - step * direction)
                        protos[second_class, second_idx] = (
                            protos[second_class, second_idx] + step * direction)
                        protos[first_class] = l2_normalize(protos[first_class])
                        protos[second_class] = l2_normalize(protos[second_class])
        if pair_count:
            value = value / pair_count
        return l2_normalize(protos), value.detach()

    def _maybe_reset_dead_prototypes(self, protos):
        if (self.dead_proto_check_interval <= 0
                or int(self.stage3_iter.item()) % self.dead_proto_check_interval != 0):
            return protos
        self.stage2_last_window_winner_count.copy_(
            self.stage2_winner_count_window)
        self.stage2_last_window_valid_sample_count.copy_(
            self.stage2_valid_sample_count_window)
        for class_idx in self.active_proto_classes:
            winner_count = self.stage2_winner_count_window[class_idx]
            state = self.proto_state[class_idx]
            active_this_window = winner_count > 0
            history_mask = (1 << self.proto_maturity_history_windows) - 1
            self.proto_active_window_history[class_idx] = (
                (self.proto_active_window_history[class_idx] << 1)
                | active_this_window.long()) & history_mask
            self.proto_empty_window_count[class_idx, active_this_window] = 0
            self.proto_empty_window_count[class_idx, ~active_this_window] += 1

            probation_mask = state == 1
            self.proto_age_since_reset[class_idx, probation_mask] += (
                self.dead_proto_check_interval)
            self.proto_probation_window_count[class_idx, probation_mask] += 1
            active_windows = self._count_active_windows(
                self.proto_active_window_history[class_idx],
                self.proto_maturity_history_windows)
            valid_count = int(
                self.stage2_valid_sample_count_window[class_idx].item())
            adaptive_min_wins = int(round(
                valid_count * self.proto_maturity_win_ratio))
            adaptive_min_wins = max(
                self.proto_maturity_min_wins, adaptive_min_wins)
            adaptive_min_wins = min(
                self.proto_maturity_max_wins, adaptive_min_wins)
            mature_now = (
                probation_mask
                & (self.proto_age_since_reset[class_idx]
                   >= self.proto_maturity_min_age)
                & (self.proto_winner_since_reset[class_idx]
                   >= adaptive_min_wins)
                & (self.proto_support_mass_since_reset[class_idx]
                   >= 0.5 * adaptive_min_wins)
                & (active_windows >= self.proto_maturity_active_windows))
            self.proto_state[class_idx, mature_now] = 2

            probation_failed = (
                (self.proto_state[class_idx] == 1)
                & (self.proto_probation_window_count[class_idx]
                   >= self.proto_probation_max_windows))
            self.proto_state[class_idx, probation_failed] = 0

            mature_stale = (
                (self.proto_state[class_idx] == 2)
                & (self.proto_empty_window_count[class_idx]
                   >= self.dead_proto_patience_windows))
            self.proto_state[class_idx, mature_stale] = 0

            reset_indices = torch.where(self.proto_state[class_idx] == 0)[0]
            enough_samples = valid_count >= self.dead_proto_min_samples
            bank = self._recent_feature_bank[class_idx]
            if enough_samples and reset_indices.numel() and len(bank):
                features = l2_normalize(
                    torch.cat(list(bank), dim=0).to(protos.device))
                alive_indices = torch.where(
                    self.proto_state[class_idx] > 0)[0]
                references = (
                    protos[class_idx, alive_indices]
                    if alive_indices.numel()
                    else protos.new_zeros((0, self.embed_dims)))
                selected = []
                for dead_idx in reset_indices.tolist():
                    current_refs = references
                    if selected:
                        current_refs = torch.cat(
                            [current_refs, torch.stack(selected)], dim=0)
                    candidate = self._density_reset_candidate(
                        features, current_refs)
                    if candidate is None:
                        break
                    selected.append(candidate)
                    protos[class_idx, dead_idx] = candidate
                    self.proto_positive_anchor[class_idx, dead_idx] = candidate
                    self.proto_positive_anchor_valid[class_idx, dead_idx] = True
                    self.stage2_reset_count[class_idx, dead_idx] += 1
                    self.proto_state[class_idx, dead_idx] = 1
                    self.proto_age_since_reset[class_idx, dead_idx] = 0
                    self.proto_winner_since_reset[class_idx, dead_idx] = 0
                    self.proto_support_mass_since_reset[
                        class_idx, dead_idx] = 0
                    self.proto_active_window_history[class_idx, dead_idx] = 0
                    self.proto_empty_window_count[class_idx, dead_idx] = 0
                    self.proto_probation_window_count[class_idx, dead_idx] = 0
            self.stage2_winner_count_window[class_idx].zero_()
            self.stage2_valid_sample_count_window[class_idx].zero_()
            bank.clear()
            self._recent_feature_bank_size[class_idx] = 0
        return l2_normalize(protos)

    def _negative_proto_loss_and_update(self, projected_feats, proto_sims,
                                        cls_scores, bbox_preds,
                                        gt_bboxes_list,
                                        matched_gt_ids_list):
        self._initialize_stage3_proto_states()
        loss_sum = proto_sims.sum() * 0.0
        hard_negative_count = 0
        repulsion_features = {}
        repulsion_weights = {}

        reshaped_sims = proto_sims.view(
            proto_sims.shape[0], proto_sims.shape[1],
            self.num_classes, self.num_prototype)
        for batch_idx in range(len(cls_scores)):
            probabilities = cls_scores[batch_idx].sigmoid()
            confidence, pred_labels = probabilities.max(dim=-1)
            unmatched = matched_gt_ids_list[batch_idx] == -1
            gt_centers = gt_bboxes_list[batch_idx].gravity_center.to(
                bbox_preds.device)
            if len(gt_centers):
                nearest_gt_distance = torch.cdist(
                    bbox_preds[batch_idx, :, :2], gt_centers[:, :2], p=2
                ).min(dim=1).values
                clear_negative = unmatched & (
                    nearest_gt_distance > self.greedy_dist_th)
            else:
                clear_negative = unmatched
            clear_negative &= confidence >= self.neg_proto_score_threshold

            for class_idx in self.active_proto_classes:
                mature_mask = self._mature_mask(class_idx)
                if not mature_mask.any():
                    continue
                class_mask = clear_negative & (pred_labels == class_idx)
                if not class_mask.any():
                    continue
                class_sims = reshaped_sims[
                    batch_idx, class_mask, class_idx, :][:, mature_mask]
                max_similarity, local_proto_idx = class_sims.max(dim=1)
                hard_mask = (
                    max_similarity > self.neg_proto_similarity_margin)
                if not hard_mask.any():
                    continue
                hard_similarity = max_similarity[hard_mask]
                hard_confidence = confidence[class_mask][hard_mask]
                confidence_weight = (
                    (hard_confidence - self.neg_proto_score_threshold)
                    / max(1.0 - self.neg_proto_score_threshold, 1e-6)
                ).clamp(0.0, 1.0)
                violation = (
                    hard_similarity - self.neg_proto_similarity_margin)
                loss_sum = loss_sum + (
                    confidence_weight * violation.square()).sum()
                hard_negative_count += int(hard_mask.sum().item())

                mature_indices = torch.where(mature_mask)[0]
                selected_proto_idx = mature_indices[local_proto_idx[hard_mask]]
                selected_features = projected_feats[
                    batch_idx, class_mask][hard_mask].detach()
                selected_weights = (
                    confidence_weight * violation.detach()).detach()
                for proto_idx in torch.unique(selected_proto_idx):
                    proto_mask = selected_proto_idx == proto_idx
                    key = (class_idx, int(proto_idx.item()))
                    repulsion_features.setdefault(key, []).append(
                        selected_features[proto_mask])
                    repulsion_weights.setdefault(key, []).append(
                        selected_weights[proto_mask])

        if hard_negative_count:
            loss = loss_sum / hard_negative_count
        else:
            loss = loss_sum

        if self.update_prototype and self.neg_proto_repulsion_rate > 0:
            with torch.no_grad():
                protos = self.prototypes.data.clone()
                for key, feature_parts in repulsion_features.items():
                    class_idx, proto_idx = key
                    if self.proto_state[class_idx, proto_idx].item() != 2:
                        continue
                    features = l2_normalize(torch.cat(feature_parts, dim=0))
                    weights = torch.cat(repulsion_weights[key], dim=0)
                    if weights.sum().item() <= 0:
                        continue
                    negative_center = (
                        weights.unsqueeze(-1) * features).sum(dim=0)
                    negative_center = l2_normalize(
                        negative_center.unsqueeze(0))[0]
                    prototype = l2_normalize(
                        protos[class_idx, proto_idx].unsqueeze(0))[0]
                    tangent = negative_center - (
                        torch.dot(prototype, negative_center) * prototype)
                    severity = weights.mean().clamp(0.0, 1.0)
                    prototype = prototype - (
                        self.neg_proto_repulsion_rate * severity * tangent)
                    protos[class_idx, proto_idx] = l2_normalize(
                        prototype.unsqueeze(0))[0]
                self.prototypes.data.copy_(l2_normalize(protos))
        return loss, hard_negative_count

    def prototype_learning(self, out_feat, gt_labels, feat_proto_sim,
                           pred_scores, quality_weights, overlaps):
        self._initialize_stage3_proto_states()
        proto_logits = feat_proto_sim
        proto_target = torch.full_like(gt_labels, -100, dtype=torch.long)
        effective_weights = torch.zeros_like(quality_weights)
        pred_labels = pred_scores.argmax(dim=-1)
        correct_mask = pred_labels == gt_labels
        active_mask = self.active_proto_class_mask[gt_labels]
        reshaped_sim = feat_proto_sim.view(
            -1, self.num_classes, self.num_prototype)
        inter_loss_sum = proto_logits.new_zeros(())
        inter_sample_count = 0
        intra_values = []
        with torch.no_grad():
            protos = self.prototypes.data.clone()

        for class_label in torch.unique(gt_labels):
            class_idx = int(class_label.long().item())
            if (class_idx < 0 or class_idx >= self.num_classes
                    or class_idx not in self.active_proto_classes):
                continue
            valid_mask = (
                (gt_labels == class_idx) & correct_mask & active_mask)
            if not valid_mask.any():
                continue

            valid_sim = reshaped_sim[valid_mask, class_idx, :]
            valid_feat = out_feat[valid_mask]
            valid_weights = quality_weights[valid_mask]
            valid_overlaps = overlaps[valid_mask]
            assigned_indices = valid_sim.detach().argmax(dim=-1)
            winner_similarity = valid_sim.gather(
                1, assigned_indices.unsqueeze(1)).squeeze(1).detach()
            positive_threshold = self.positive_similarity_threshold_tensor[
                class_idx]
            attraction_denom = max(
                float(positive_threshold.item())
                - self.positive_similarity_floor, 1e-6)
            attraction_scale = (
                (positive_threshold - winner_similarity) / attraction_denom
            ).clamp(0.0, 1.0)
            update_weights = valid_weights * attraction_scale
            proto_target[valid_mask] = (
                assigned_indices + self.num_prototype * class_idx)
            effective_weights[valid_mask] = update_weights

            own_max = valid_sim.max(dim=-1).values
            wrong_parts = [
                reshaped_sim[valid_mask, other_class, :][
                    :, self._mature_mask(other_class)]
                for other_class in self.active_proto_classes
                if (other_class != class_idx
                    and self._mature_mask(other_class).any())
            ]
            if wrong_parts:
                wrong_max = torch.cat(
                    wrong_parts, dim=-1).max(dim=-1).values
                inter_hinge = torch.clamp(
                    self.inter_class_margin + wrong_max - own_max, min=0.0)
                inter_loss_sum = inter_loss_sum + (
                    inter_hinge * valid_weights).sum()
                inter_sample_count += int(valid_mask.sum().item())

            with torch.no_grad():
                detached_feat = valid_feat.detach()
                self._append_recent_features(class_idx, detached_feat)
                self.stage2_valid_sample_count_window[class_idx] += len(detached_feat)
                for proto_idx in range(self.num_prototype):
                    assigned_mask = assigned_indices == proto_idx
                    if not assigned_mask.any():
                        continue
                    proto_features = detached_feat[assigned_mask]
                    proto_weights = update_weights[assigned_mask].detach()
                    support_weights = valid_weights[assigned_mask].detach()
                    proto_overlaps = valid_overlaps[assigned_mask].detach()
                    support_center = (
                        support_weights.unsqueeze(-1) * proto_features
                    ).sum(dim=0)
                    support_center = support_center / (
                        support_weights.sum().clamp_min(1e-6))
                    support_center = l2_normalize(
                        support_center.unsqueeze(0))[0]
                    if proto_weights.sum().item() > 0:
                        weighted_center = (
                            proto_weights.unsqueeze(-1) * proto_features
                        ).sum(dim=0)
                        weighted_center = weighted_center / (
                            proto_weights.sum().clamp_min(1e-6))
                        weighted_center = l2_normalize(
                            weighted_center.unsqueeze(0))[0]
                        mean_weight = proto_weights.mean().clamp(0.0, 1.0)
                        effective_gamma = (
                            1.0 - (1.0 - self.gamma) * mean_weight)
                        if self.update_prototype:
                            protos[class_idx, proto_idx] = momentum_update(
                                old_value=protos[class_idx, proto_idx],
                                new_value=weighted_center,
                                momentum=effective_gamma,
                                iter=self.stage2_iter,
                            )
                    winner_num = int(assigned_mask.sum().item())
                    self.stage2_winner_count_window[
                        class_idx, proto_idx] += winner_num
                    self.stage2_winner_count_total[
                        class_idx, proto_idx] += winner_num
                    self.stage2_weighted_update_mass[
                        class_idx, proto_idx] += proto_weights.sum()
                    self.proto_winner_since_reset[
                        class_idx, proto_idx] += winner_num
                    self.proto_support_mass_since_reset[
                        class_idx, proto_idx] += support_weights.sum()
                    self.stage2_overlap_sum[
                        class_idx, proto_idx] += proto_overlaps.sum()
                    self.stage2_overlap_count[
                        class_idx, proto_idx] += winner_num
                    self.proto_positive_anchor[
                        class_idx, proto_idx] = support_center
                    self.proto_positive_anchor_valid[
                        class_idx, proto_idx] = True

                protos[class_idx], intra_value = self._apply_proto_intra_upper(
                    protos[class_idx], class_idx)
                intra_values.append(intra_value)

        with torch.no_grad():
            protos, proto_inter_value = self._apply_proto_inter_upper(protos)
            protos = self._maybe_reset_dead_prototypes(protos)
            self.prototypes.data.copy_(l2_normalize(protos))

        intra_value = (
            torch.stack(intra_values).mean()
            if intra_values else proto_logits.new_zeros(()))
        inter_sample_loss = inter_loss_sum / max(inter_sample_count, 1)
        return (proto_logits, proto_target, effective_weights, intra_value,
                inter_sample_loss, proto_inter_value)

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
            self.stage2_iter += 1
            self.stage3_iter += 1

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
        batch_pos_weights = []
        batch_pos_overlaps = []
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
                gt_boxes=None,  # 新增
                dist_th=self.greedy_dist_th,
                iou_th=0.25,
                conf_thr=0.01,  # 新增
                return_matched_ids=True
            )
            all_matched_gt_ids_list.append(matched_gt_ids)  # 保存

            if pos_inds.numel() == 0:
                continue

            # 3. 获取正样本对应的真实标签
            pos_gt_labels = cur_gt_labels[pos_assigned_gt_inds]

            # Stage2仅允许重叠质量达标的活跃类别正样本更新原型。
            pos_overlaps = self._matched_gt_overlap_ratio(
                cur_bbox_preds, cur_gt_boxes, pos_inds,
                pos_assigned_gt_inds)
            active_mask = self.active_proto_class_mask[pos_gt_labels]
            overlap_threshold = self.overlap_threshold_tensor[pos_gt_labels]
            quality_mask = active_mask & (pos_overlaps >= overlap_threshold)
            pos_inds = pos_inds[quality_mask]
            pos_assigned_gt_inds = pos_assigned_gt_inds[quality_mask]
            pos_gt_labels = pos_gt_labels[quality_mask]
            pos_overlaps = pos_overlaps[quality_mask]
            if pos_inds.numel() == 0:
                continue
            pos_weights = self._quality_weights(pos_overlaps, pos_gt_labels)

            # 4. 获取正样本的特征和相似度
            final_pos_feat = projected_feats[i][pos_inds]
            final_pos_sim = proto_sims[i][pos_inds]
            final_pos_labels = pos_gt_labels
            final_pos_scores = cur_cls_scores[pos_inds]

            batch_pos_feats.append(final_pos_feat)
            batch_pos_labels.append(final_pos_labels)
            batch_pos_sims.append(final_pos_sim)
            batch_pos_pred_scores.append(final_pos_scores)
            batch_pos_weights.append(pos_weights)
            batch_pos_overlaps.append(pos_overlaps)

        # ==================== 4. 原型损失计算（带预热） ====================
        if len(batch_pos_feats) > 0:
            # 拼接所有正样本
            all_pos_feats = torch.cat(batch_pos_feats, dim=0)  # [总正样本数, D]
            all_pos_labels = torch.cat(batch_pos_labels, dim=0)  # [总正样本数]
            all_pos_sims = torch.cat(batch_pos_sims, dim=0)  # [总正样本数, total_prototypes]
            all_pos_pred_scores = torch.cat(batch_pos_pred_scores, dim=0)  # [total_pos, num_classes]
            all_pos_weights = torch.cat(batch_pos_weights, dim=0)
            all_pos_overlaps = torch.cat(batch_pos_overlaps, dim=0)

            # 核心原型学习
            (contrast_logits, contrast_targets, effective_weights,
             proto_intra_value, inter_sample_loss,
             proto_inter_value) = self.prototype_learning(
                all_pos_feats,  # 正样本特征
                all_pos_labels,  # 真实标签
                all_pos_sims,  # 原型相似度
                all_pos_pred_scores,  # 预测分数
                all_pos_weights,
                all_pos_overlaps,
            )

            # 计算交叉熵损失
            temperature = 0.07  # 沿用 CPDet3D 默认值
            valid_proto_targets = contrast_targets != -100
            if valid_proto_targets.any():
                valid_logits = contrast_logits[valid_proto_targets]
                valid_targets = contrast_targets[valid_proto_targets]
                allowed_prototypes = self._mature_mask().reshape(-1)
                allowed_logits = allowed_prototypes.unsqueeze(0).expand(
                    len(valid_logits), -1).clone()
                allowed_logits.scatter_(
                    1, valid_targets.unsqueeze(1), True)
                valid_logits = valid_logits.masked_fill(
                    ~allowed_logits, -1e4)
                proto_ce = F.cross_entropy(
                    valid_logits / temperature,
                    valid_targets,
                    reduction='none',
                )
                loss_proto = (
                    proto_ce * effective_weights[valid_proto_targets]
                ).sum() / valid_proto_targets.sum().clamp_min(1)
            else:
                loss_proto = contrast_logits.sum() * 0.0

            loss_dict['loss_proto'] = loss_proto * self.proto_loss_weight
            loss_dict['loss_proto_intra_upper'] = (
                contrast_logits.sum() * 0.0
                + proto_intra_value * self.proto_intra_weight)
            loss_dict['loss_inter_class'] = (
                inter_sample_loss * self.inter_class_loss_weight)
            loss_dict['loss_proto_inter'] = (
                contrast_logits.sum() * 0.0
                + proto_inter_value * self.proto_inter_weight)
        else:
            # 如果没有正样本，原型损失为0（但需要保持计算图防止DDP死锁）
            zero_loss = cls_scores.sum() * 0.0
            loss_dict['loss_proto'] = zero_loss
            loss_dict['loss_proto_intra_upper'] = zero_loss
            loss_dict['loss_inter_class'] = zero_loss
            loss_dict['loss_proto_inter'] = zero_loss

        negative_proto_loss, hard_negative_count = (
            self._negative_proto_loss_and_update(
                projected_feats,
                proto_sims,
                cls_scores,
                bbox_preds,
                gt_bboxes_list,
                all_matched_gt_ids_list,
            ))
        loss_dict['loss_neg_proto'] = (
            negative_proto_loss * self.neg_proto_loss_weight)

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

            logger.info(f"\n[Iter {int(self.local_iter.item())}] "
                        f"GT: {total_gt} | Pos: {total_pos} | Neg: {total_neg} | "
                        f"ZeroConf (suppressed): {zero_conf_count} | "
                        f"HardNeg: {hard_negative_count} | "
                        f"ProtoLoss: {proto_loss_val:.4f} | NegLoss: {neg_loss_val:.4f}")

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

        if self.training and self.stage3_iter % 1000 == 0:
            from mmcv.utils import get_logger
            logger = get_logger('mmdet')
            logger.info(
                '[Stage3 Iter %d] WTA usage and prototype state:',
                int(self.stage3_iter.item()))
            for class_idx in self.active_proto_classes:
                usage = self.stage2_last_window_winner_count[class_idx]
                valid_count = int(
                    self.stage2_last_window_valid_sample_count[class_idx].item())
                reset_count = self.stage2_reset_count[class_idx]
                logger.info(
                    '  class=%d valid=%d usage=%s states=%s resets=%s',
                    class_idx,
                    valid_count,
                    usage.cpu().tolist(),
                    self.proto_state[class_idx].cpu().tolist(),
                    reset_count.cpu().tolist())

        if (self.training and self.prototype_save_interval > 0
                and self.stage3_iter % self.prototype_save_interval == 0):
            save_dir = self.prototype_save_dir
            os.makedirs(save_dir, exist_ok=True)
            # 保存当前原型，形状 [num_classes, num_prototype, embed_dims]
            proto_data = self.prototypes.data.cpu().clone()
            torch.save({
                'iter': self.stage3_iter.item(),
                'prototypes': proto_data,
                'num_classes': self.num_classes,
                'num_prototype': self.num_prototype,
                'winner_count_total': self.stage2_winner_count_total.cpu().clone(),
                'weighted_update_mass': self.stage2_weighted_update_mass.cpu().clone(),
                'overlap_sum': self.stage2_overlap_sum.cpu().clone(),
                'overlap_count': self.stage2_overlap_count.cpu().clone(),
                'reset_count': self.stage2_reset_count.cpu().clone(),
                'proto_state': self.proto_state.cpu().clone(),
                'proto_age_since_reset': (
                    self.proto_age_since_reset.cpu().clone()),
                'proto_winner_since_reset': (
                    self.proto_winner_since_reset.cpu().clone()),
                'proto_support_mass_since_reset': (
                    self.proto_support_mass_since_reset.cpu().clone()),
                'proto_active_window_history': (
                    self.proto_active_window_history.cpu().clone()),
            }, f'{save_dir}/prototypes_iter_{int(self.stage3_iter.item())}.pth')

        return loss_dict

    # 原始融合特征 → 投影头 → LayerNorm → L2归一化 → 计算相似度 → Sinkhorn分配 → 动量更新
