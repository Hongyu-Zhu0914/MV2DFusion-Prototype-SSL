"""Ten-class stage-2 prototype head, adapted from the validated mini 0820 head.

Import explicitly with ``custom_imports`` and set the head type to
``PrototypeStage2AdaptiveHead20261023``. This does not replace either old head.
Load a learned stage-1 *full model/head checkpoint* before training. A complete
stage-2 checkpoint resumes its per-class weights, windows and stage-2 counter;
stage-1 state initializes only the newly introduced stage-2 buffers.

The head enables negative supervision immediately in stage 2. Monitoring counts
only loss calls with real, complete annotations, starting at zero after stage-1
initialization. Stage-1 ``local_iter`` is retained for historical audit/snapshots.
It does not choose detector freezing, optimizer, or Teacher/Student integration.
Pseudo examples must carry ``is_pseudo=True`` in img_metas (or an explicit false
``prototype_supervised`` flag); they receive detection losses only.
"""
import math
import os

import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmcv.utils import get_logger
from mmdet.models import HEADS

from .contrast import l2_normalize
from .prototype_stage1_head_0920 import PrototypeStage1Head0920


@HEADS.register_module()
class PrototypeStage2AdaptiveHead20261023(PrototypeStage1Head0920):
    """Stage-1 positive learning plus mini-style adaptive hard negatives.

    Defaults use the nuScenes ten-class order. Match thresholds, active classes
    and heading gates can still be explicitly overridden by the experiment.
    ``proto_loss_weight`` is a head-local coefficient; the detector's existing
    ``loss_weight_3d`` scaling still applies outside this head.

    ``prototype_save_dir=None`` disables optional diagnostic snapshots. Supply
    a new experiment's ``prototypes/`` directory to enable them; full runner
    checkpoints are still required for optimizer/RNG/model resume.
    """

    def __init__(self, num_classes=10, proto_loss_weight=0.5, gamma=0.99,
                 proto_inter_upper=0.4, proto_inter_weight=0.01,
                 neg_proto_margin=0.7, neg_proto_gt_dist_th=2.0,
                 neg_proto_loss_weight=0.05, neg_proto_dynamic_weight=True,
                 neg_proto_loss_weight_min=0.005,
                 neg_proto_loss_weight_max=0.10, neg_proto_weight_step=0.02,
                 neg_proto_monitor_interval=500,
                 neg_proto_weight_update_interval=1000,
                 neg_proto_min_mean_improvement=0.01,
                 neg_proto_min_ratio_improvement=0.05,
                 neg_proto_monitor_patience=1,
                 neg_proto_positive_floor=0.80,
                 neg_proto_positive_class_floor=0.80,
                 neg_proto_positive_drop_tolerance=0.03,
                 neg_proto_min_monitor_samples=20,
                 prototype_save_dir=None, require_stage1_prototypes=True,
                 allow_stage1_initialization=True, **kwargs):
        if num_classes == 10:
            kwargs.setdefault('greedy_dist_th_by_class', [
                1.0, 1.2, 1.5, 1.5, 1.5, 0.5, 0.5, 0.4, 0.35, 0.25])
            active = kwargs.get('active_proto_class_ids', list(range(10)))
            if active is None:
                active = list(range(10))
            kwargs.setdefault('greedy_angle_class_ids', [
                c for c in [0, 1, 2, 3, 4, 6, 7] if c in active])
        super().__init__(num_classes=num_classes,
                         proto_loss_weight=proto_loss_weight, gamma=gamma,
                         neg_proto_margin=neg_proto_margin,
                         neg_proto_loss_weight=neg_proto_loss_weight,
                         prototype_save_dir=prototype_save_dir, **kwargs)
        values = locals()
        for name in [
                'proto_inter_upper', 'proto_inter_weight',
                'neg_proto_gt_dist_th', 'neg_proto_loss_weight_min',
                'neg_proto_loss_weight_max', 'neg_proto_weight_step',
                'neg_proto_min_mean_improvement',
                'neg_proto_min_ratio_improvement', 'neg_proto_positive_floor',
                'neg_proto_positive_class_floor',
                'neg_proto_positive_drop_tolerance']:
            setattr(self, name, float(values[name]))
        for name in ['neg_proto_monitor_interval',
                     'neg_proto_weight_update_interval',
                     'neg_proto_monitor_patience',
                     'neg_proto_min_monitor_samples']:
            setattr(self, name, int(values[name]))
        self.neg_proto_dynamic_weight = bool(neg_proto_dynamic_weight)
        self.require_stage1_prototypes = bool(require_stage1_prototypes)
        self.allow_stage1_initialization = bool(allow_stage1_initialization)
        self._validate_stage2_config()
        # These buffers are the only state initialized during stage-1 loading.
        weights = torch.zeros(self.num_classes)
        weights[list(self.active_proto_class_ids)] = self.neg_proto_loss_weight
        self.register_buffer('neg_proto_class_weight', weights)
        for name in ['neg_monitor_similarity_sum', 'neg_monitor_sample_count',
                     'neg_monitor_hard_count',
                     'neg_monitor_positive_similarity_sum',
                     'neg_monitor_positive_count']:
            self.register_buffer(name, torch.zeros(self.num_classes))
        for name in ['neg_monitor_previous_mean_by_class',
                     'neg_monitor_previous_ratio_by_class',
                     'neg_monitor_previous_positive_mean_by_class']:
            self.register_buffer(name, torch.full((self.num_classes,), float('nan')))
        self.register_buffer('neg_monitor_bad_windows_by_class',
                             torch.zeros(self.num_classes, dtype=torch.long))
        self.register_buffer('stage2_steps', torch.zeros((), dtype=torch.long))
        self.register_buffer('stage2_last_local_iter',
                             torch.full((), -1, dtype=torch.long))
        self._stage2_buffer_names = tuple(
            n for n in self._buffers if n.startswith(('neg_', 'stage2_')))
        self.last_negative_monitor = None
        self.stage2_load_kind = 'fresh'

    def _validate_stage2_config(self):
        finite = [self.proto_inter_upper, self.proto_inter_weight,
                  self.neg_proto_gt_dist_th, self.neg_proto_loss_weight,
                  self.neg_proto_loss_weight_min, self.neg_proto_loss_weight_max,
                  self.neg_proto_weight_step, self.neg_proto_min_mean_improvement,
                  self.neg_proto_min_ratio_improvement, self.neg_proto_margin,
                  self.neg_proto_positive_floor, self.neg_proto_positive_class_floor,
                  self.neg_proto_positive_drop_tolerance]
        if not all(math.isfinite(v) for v in finite):
            raise ValueError('Stage-2 hyperparameters must be finite')
        if not -1 < self.proto_inter_upper <= 1 or self.proto_inter_weight < 0:
            raise ValueError('Invalid inter-class upper bound or update strength')
        if not -1 < self.neg_proto_margin < 1 or self.neg_proto_gt_dist_th <= 0:
            raise ValueError('Invalid negative similarity margin or GT distance')
        if not (0 <= self.neg_proto_loss_weight_min <= self.neg_proto_loss_weight
                <= self.neg_proto_loss_weight_max):
            raise ValueError('Expected min <= initial <= max negative weight')
        if self.neg_proto_weight_step <= 0:
            raise ValueError('Negative-weight increment must be positive')
        if (self.neg_proto_monitor_interval <= 0 or
                self.neg_proto_weight_update_interval < self.neg_proto_monitor_interval or
                self.neg_proto_weight_update_interval % self.neg_proto_monitor_interval):
            raise ValueError('Weight interval must be a positive multiple of monitor interval')
        if self.neg_proto_monitor_patience < 1 or self.neg_proto_min_monitor_samples < 1:
            raise ValueError('Patience and minimum samples must be positive')
        if not (-1 <= self.neg_proto_positive_class_floor <=
                self.neg_proto_positive_floor <= 1):
            raise ValueError('Invalid positive-similarity floor')
        if min(self.neg_proto_positive_drop_tolerance,
               self.neg_proto_min_mean_improvement,
               self.neg_proto_min_ratio_improvement) < 0:
            raise ValueError('Improvement and drop tolerances must be non-negative')

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        keys = [prefix + n for n in self._stage2_buffer_names]
        present = [k in state_dict for k in keys]
        if not any(present):
            has_stage1 = (prefix + 'prototypes' in state_dict and
                          prefix + 'proto_proj.proj.0.weight' in state_dict)
            if has_stage1 and self.allow_stage1_initialization:
                for name in self._stage2_buffer_names:
                    state_dict[prefix + name] = self._buffers[name].detach().clone()
                self.stage2_load_kind = 'stage1_initialization'
        elif not all(present):
            error_msgs.append('Incomplete stage-2 adaptive state at ' + prefix +
                              '; missing: ' + ', '.join(k for k in keys if k not in state_dict))
        else:
            self.stage2_load_kind = 'stage2_resume'
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def forward(self, img_metas, *args, **kwargs):
        outputs = super().forward(img_metas, *args, **kwargs)
        # The SSL wrapper already marks pseudo frames. Never treat those boxes
        # as complete GT to infer reliable background or update prototype state.
        # The inherited NMS/padding path marks invalid queries with -40 logits
        # and zeroed boxes. Those coordinates must never infer background.
        outputs['prototype_query_valid_mask'] = ~(
            outputs['all_cls_scores'][-1].detach() == -40).all(dim=-1)
        outputs['prototype_supervised_mask'] = tuple(
            not bool(m.get('is_pseudo', False)) and
            bool(m.get('prototype_supervised', True)) for m in img_metas)
        return outputs

    def prototype_learning(self, out_feat, gt_labels, feat_proto_sim, pred_scores):
        logits, targets, intra = super().prototype_learning(
            out_feat, gt_labels, feat_proto_sim, pred_scores)
        with torch.no_grad():
            protos, inter = self._apply_proto_inter_upper(
                self.prototypes.detach().clone())
            self.prototypes.copy_(protos)
        return logits, targets, intra, inter

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
        current_iter = int(self.stage2_steps.item())
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

    def _negative_loss(self, similarities_by_class, zero):
        # Same reduction as mini: hard-only mean per class, then mean over
        # classes that contain hard negatives (not over all query/prototype pairs).
        losses = []
        for c, sims in similarities_by_class.items():
            hard = sims > self.neg_proto_margin
            if hard.any():
                weight = self.neg_proto_class_weight[c].detach().clone()
                losses.append((sims[hard] - self.neg_proto_margin).square().mean() * weight)
        return torch.stack(losses).mean() if losses else zero

    def _check_stage1_ready(self):
        if self.require_stage1_prototypes:
            counts = self.proto_update_count.sum(dim=-1)
            missing = [c for c in self.active_proto_class_ids if counts[c].item() == 0]
            if missing:
                raise RuntimeError('Stage 2 requires learned stage-1 prototypes; '
                                   'no updates for active classes: ' + str(missing))

    @force_fp32(apply_to=('preds_dicts',))
    def loss(self, gt_bboxes_list, gt_labels_list, preds_dicts, gt_bboxes_ignore=None):
        # Keep full-batch detection targets, including empty-GT/pseudo frames.
        # Subsetting only the targets would misalign them with decoder outputs.
        losses = self.loss_detection_only(gt_bboxes_list, gt_labels_list,
                                          preds_dicts, gt_bboxes_ignore)
        if not self.training:
            return losses
        feats = preds_dicts.get('projected_feats')
        sims = preds_dicts.get('proto_sims')
        if feats is None or sims is None:
            raise ValueError('Stage-2 training requires projected_feats and proto_sims')
        cls = preds_dicts['all_cls_scores'][-1]
        boxes = preds_dicts['all_bbox_preds'][-1]
        batch = len(gt_labels_list)
        mask = preds_dicts.get('prototype_supervised_mask', (True,) * batch)
        if len(mask) != batch or cls.shape[0] != batch or feats.shape[:2] != cls.shape[:2]:
            raise ValueError('Prototype metadata/features and detection batch mismatch')
        query_valid = preds_dicts.get('prototype_query_valid_mask')
        if query_valid is None:
            # Compatibility with direct loss calls; same sentinel as forward.
            query_valid = ~(cls.detach() == -40).all(dim=-1)
        if query_valid.shape != cls.shape[:2] or query_valid.dtype != torch.bool:
            raise ValueError('Prototype query-valid mask must be a B x N bool tensor')
        zero = feats.sum() * 0.0
        for key in ['loss_proto', 'loss_proto_intra_band',
                    'loss_proto_inter_upper', 'loss_neg_proto']:
            losses[key] = zero
        supervised = [i for i in range(batch) if bool(mask[i])]
        if not supervised:
            return losses
        self._check_stage1_ready()
        local_iter = int(self.local_iter.item())
        if local_iter <= int(self.stage2_last_local_iter.item()):
            raise RuntimeError('Prototype loss already evaluated for this forward; '
                               'do not update prototype state twice')
        self.stage2_last_local_iter.fill_(local_iter)
        self.stage2_steps.add_(1)

        pos_feats, pos_labels, pos_sims, pos_scores = [], [], [], []
        negative_parts = {c: [] for c in self.active_proto_class_ids}
        active = cls.new_zeros(self.num_classes, dtype=torch.bool)
        active[list(self.active_proto_class_ids)] = True
        for i in supervised:
            centers = gt_bboxes_list[i].gravity_center.to(boxes.device)
            labels = gt_labels_list[i].to(device=cls.device, dtype=torch.long)
            pos, assigned, matched = self._greedy_match_by_distance(
                pred_boxes=boxes[i], pred_scores=cls[i], gt_centers=centers,
                gt_labels=labels, gt_boxes=gt_bboxes_list[i].tensor.to(boxes.device),
                return_matched_ids=True)
            if pos.numel():
                pos_feats.append(feats[i][pos])
                pos_labels.append(labels[assigned])
                pos_sims.append(sims[i][pos])
                pos_scores.append(cls[i][pos])
            if centers.numel():
                distance = torch.cdist(boxes[i, :, :2], centers[:, :2]).min(dim=1).values
            else:
                distance = boxes.new_full((boxes.shape[1],), float('inf'))
            pred_cls = cls[i].argmax(dim=-1)
            by_class = sims[i].view(-1, self.num_classes, self.num_prototype)
            finite = (torch.isfinite(boxes[i]).all(dim=-1) &
                      torch.isfinite(cls[i]).all(dim=-1) &
                      torch.isfinite(by_class).all(dim=-1).all(dim=-1))
            clear = ((matched == -1) & (distance > self.neg_proto_gt_dist_th) &
                     active[pred_cls] & finite & query_valid[i])
            for c in self.active_proto_class_ids:
                selected = clear & (pred_cls == c)
                if selected.any():
                    negative_parts[c].append(by_class[selected, c].max(dim=-1).values)

        positive_by_class = {}
        if pos_feats:
            features = torch.cat(pos_feats)
            labels = torch.cat(pos_labels)
            similarities = torch.cat(pos_sims)
            scores = torch.cat(pos_scores)
            grouped = similarities.view(-1, self.num_classes, self.num_prototype)
            own_max = grouped[torch.arange(len(labels), device=labels.device), labels].max(dim=-1).values
            for c in self.active_proto_class_ids:
                selected = labels == c
                if selected.any():
                    positive_by_class[c] = own_max[selected]
            logits, targets, intra, inter = self.prototype_learning(
                features, labels, similarities, scores)
            valid = targets != -100
            if valid.any():
                losses['loss_proto'] = F.cross_entropy(
                    logits[valid] / 0.07, targets[valid]) * self.proto_loss_weight
            # Manual prototype repulsion; these detached values are diagnostics,
            # not additional gradients into the projection network.
            losses['loss_proto_intra_band'] = zero + intra * self.proto_intra_weight
            losses['loss_proto_inter_upper'] = zero + inter * self.proto_inter_weight
        negative_by_class = {c: torch.cat(parts) for c, parts in negative_parts.items() if parts}
        losses['loss_neg_proto'] = self._negative_loss(negative_by_class, zero)
        if self.neg_proto_dynamic_weight:
            self._accumulate_negative_monitor(negative_by_class, positive_by_class)
            self.last_negative_monitor = self._maybe_adjust_negative_weight()
            if self.last_negative_monitor:
                logger = get_logger('mmdet')
                for report in self.last_negative_monitor:
                    logger.info('[Stage2NegativeWeight] local_iter=%d stage2_step=%d report=%s',
                                local_iter, int(self.stage2_steps), report)
        self._save_stage2_snapshot()
        return losses

    def _save_stage2_snapshot(self):
        step = int(self.stage2_steps.item())
        if self.prototype_save_dir is None or step == 0 or step % self.prototype_save_interval:
            return
        os.makedirs(self.prototype_save_dir, exist_ok=True)
        # Full model/optimizer/RNG resume belongs to the runner checkpoint.
        state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                 if k.startswith(('prototypes', 'proto_', 'feat_norm.', 'neg_', 'stage2_', 'local_iter'))}
        path = os.path.join(self.prototype_save_dir,
                            'stage2_prototypes_step_%d.pth' % step)
        # Never overwrite a previous run's diagnostic snapshot.
        with open(path, 'xb') as stream:
            torch.save({'local_iter': int(self.local_iter), 'stage2_step': step,
                        'num_classes': self.num_classes,
                        'num_prototype': self.num_prototype,
                        'state_dict': state}, stream)
