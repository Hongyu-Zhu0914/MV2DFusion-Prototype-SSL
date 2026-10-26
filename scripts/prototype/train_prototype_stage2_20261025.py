"""Finite-pass prototype-only Stage 2 with strict resume and frozen-state audits.

Formal execution requires --launch-formal. --stop-after performs a small,
isolated run under --work-dir; it never consumes the formal experiment directory.
"""

# Resolve imports and relative CLI paths after the publication-layout migration.
import sys as _bootstrap_sys
import os as _bootstrap_os
from pathlib import Path as _BootstrapPath
REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
if str(REPO_ROOT) not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, str(REPO_ROOT))
if __name__ == '__main__':
    _bootstrap_os.chdir(str(REPO_ROOT))

import argparse
from collections import OrderedDict
from datetime import datetime, timezone, timedelta
from functools import partial
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.utils import import_modules_from_strings
from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from torch.utils.data import DataLoader

ROOT = REPO_ROOT
SOURCE_FILES = [
    'scripts/prototype/train_prototype_stage2_20261025.py',
    'projects/configs/nusc/prototype_stage2_labeled6000_10cls_20261025.py',
    'projects/mmdet3d_plugin/models/detectors/prototype_only_stage2_20261025.py',
    'projects/mmdet3d_plugin/models/detectors/prototype_only_detector_0920.py',
    'projects/mmdet3d_plugin/models/dense_heads/prototype_stage2_adaptive_head_20261023.py',
    'projects/mmdet3d_plugin/models/dense_heads/prototype_stage1_head_0920.py',
    'projects/mmdet3d_plugin/models/dense_heads/mv2dfusion_head.py',
]
TEMPORAL_NAMES = [
    'memory_embedding', 'memory_reference_point', 'memory_timestamp',
    'memory_egopose', 'memory_velo', 'memory_query_mask', 'memory_instance_inds',
]


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                         encoding='utf-8')
    os.replace(str(temporary), str(path))


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(x) for x in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(x) for x in value)
    return value


def state_hash(model, prototype=False):
    from projects.mmdet3d_plugin.models.detectors.prototype_only_stage2_20261025 import is_stage2_state
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        if is_stage2_state(name) == prototype:
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def sample_order(dataset, seed):
    """Match the old single-GPU grouped sampler, but stop before it repeats."""
    groups = [np.flatnonzero(dataset.flag == c).tolist()
              for c in range(int(dataset.flag.max()) + 1)]
    generator = torch.Generator().manual_seed(seed)
    order = [i for group in torch.randperm(len(groups), generator=generator).tolist()
             for i in groups[group]]
    if len(order) != len(dataset) or sorted(order) != list(range(len(dataset))):
        raise RuntimeError('The finite training order must visit every frame exactly once')
    return order


def lr_for_step(cfg, step, total):
    base = cfg.optimizer.lr
    ratio = cfg.lr_config.min_lr_ratio
    rate = base * (ratio + 0.5 * (1 - ratio) * (1 + math.cos(math.pi * step / total)))
    if step < cfg.lr_config.warmup_iters:
        rate *= cfg.lr_config.warmup_ratio + (1 - cfg.lr_config.warmup_ratio) * step / cfg.lr_config.warmup_iters
    return rate


def bank_stats(head):
    with torch.no_grad():
        p = torch.nn.functional.normalize(head.prototypes.detach().float(), dim=-1)
        s = p.flatten(0, 1) @ p.flatten(0, 1).t()
        c = torch.arange(head.num_classes, device=p.device).repeat_interleave(head.num_prototype)
        cross = c[:, None] != c[None, :]
        same = ~cross & ~torch.eye(len(c), device=p.device, dtype=torch.bool)
        return dict(cross_class_mean=float(s[cross].mean()),
                    cross_class_max=float(s[cross].max()),
                    cross_class_above04=float((s[cross] > 0.4).float().mean()),
                    within_class_mean=float(s[same].mean()))


class SimilarityAudit:
    """Detached diagnostics; never call head.loss/prototype_learning a second time."""
    def __init__(self, head):
        self.head = head
        self.original = head.loss
        self.last = None
        self.window = np.zeros((10, 8), dtype=np.float64)
        head.loss = self.loss

    def loss(self, gt_boxes, gt_labels, preds, gt_bboxes_ignore=None):
        with torch.no_grad():
            cls = preds['all_cls_scores'][-1].detach()
            boxes = preds['all_bbox_preds'][-1].detach()
            sims = preds['proto_sims'].detach().view(cls.shape[0], cls.shape[1], 10, 10)
            valid = preds['prototype_query_valid_mask']
            self.last = dict(valid_queries=int(valid.sum()), masked_queries=int((~valid).sum()),
                             positives=0, negatives=0, hard_negatives=0, masked_negatives=0)
            for i, labels in enumerate(gt_labels):
                labels = labels.to(cls.device)
                gt = gt_boxes[i]
                centers = gt.gravity_center.to(cls.device)
                positive, assigned, matched = self.head._greedy_match_by_distance(
                    pred_boxes=boxes[i], pred_scores=cls[i], gt_centers=centers,
                    gt_labels=labels, gt_boxes=gt.tensor.to(cls.device), return_matched_ids=True)
                confidence, predicted = cls[i].sigmoid().max(-1)
                own = sims[i][torch.arange(len(predicted), device=cls.device), predicted].max(-1).values
                distance = (torch.cdist(boxes[i, :, :2], centers[:, :2]).min(1).values
                            if len(centers) else boxes.new_full((len(predicted),), float('inf')))
                finite = (torch.isfinite(boxes[i]).all(-1) & torch.isfinite(cls[i]).all(-1)
                          & torch.isfinite(sims[i]).all(-1).all(-1))
                negative = (matched == -1) & (distance > self.head.neg_proto_gt_dist_th) & valid[i] & finite
                pos_mask = torch.zeros_like(negative)
                pos_mask[positive] = True
                if (pos_mask & ~valid[i]).any():
                    raise RuntimeError('Masked query entered strict positives')
                self.last['positives'] += len(positive)
                self.last['negatives'] += int(negative.sum())
                self.last['hard_negatives'] += int((negative & (own > self.head.neg_proto_margin)).sum())
                self.last['masked_negatives'] += int((negative & ~valid[i]).sum())
                band = (confidence > 0.2) & (confidence < 0.8)
                for c in range(10):
                    for j, mask in enumerate([pos_mask, negative, pos_mask & band, negative & band]):
                        values = own[mask & (predicted == c)]
                        self.window[c, 2 * j] += int(values.numel())
                        self.window[c, 2 * j + 1] += float(values.sum())
        return self.original(gt_boxes, gt_labels, preds, gt_bboxes_ignore)

    def report(self, step):
        rows = []
        for c, row in enumerate(self.window):
            result = dict(class_id=c)
            for j, label in enumerate(['positive', 'clear_negative', 'positive_conf02_08', 'clear_negative_conf02_08']):
                count, total = row[2*j:2*j+2]
                result[label] = dict(count=int(count), mean=float(total/count) if count else None)
            rows.append(result)
        self.window.fill(0)
        return dict(stage2_step=step, matching='strict one-to-one positives; clear negatives nearest ANY GT >2m; ambiguous/duplicates excluded',
                    confidence_band='0.2<score<0.8 AFTER all-score strict matching', per_class=rows)


def save_checkpoint(model, optimizer, directory, completed, contract, order, audit):
    path = directory / 'checkpoints' / ('iter_%d.pth' % completed)
    if path.exists():
        raise FileExistsError(path)
    source = model.state_dict()
    state = OrderedDict((k, cpu_copy(v)) for k, v in source.items())
    if hasattr(source, '_metadata'):
        state._metadata = source._metadata.copy()
    temporary = path.with_suffix('.pth.tmp')
    head = model.fusion_bbox_head
    torch.save(dict(state_dict=state, optimizer=cpu_copy(optimizer.state_dict()),
                    rng=rng_state(), temporal={k: cpu_copy(getattr(head, k, None)) for k in TEMPORAL_NAMES},
                    similarity_window=audit.window.copy(), sample_order=order,
                    meta=dict(stage2_format='prototype_stage2_20261025_v1', completed_steps=completed,
                              contract=contract, local_iter=int(head.local_iter),
                              stage2_steps=int(head.stage2_steps))), str(temporary))
    os.replace(str(temporary), str(path))
    link = path.parent / '.latest.tmp'
    if link.is_symlink():
        link.unlink()
    link.symlink_to(path.name)
    os.replace(str(link), str(path.parent / 'latest.pth'))
    # Small snapshots are not sufficient for full resume; keep them for all milestones.
    small = OrderedDict((k, v) for k, v in state.items() if k.startswith('fusion_bbox_head.') and
                        k.split('.', 1)[1].startswith(('prototypes', 'proto_', 'feat_norm.', 'local_iter', 'neg_', 'stage2_')))
    torch.save(dict(state_dict=small, stage2_step=completed, local_iter=int(head.local_iter)),
               str(directory / 'prototypes' / ('stage2_step_%d.pth' % completed)))
    return path


def run(args):
    import projects.mmdet3d_plugin
    import projects.fsdv2
    cfg = Config.fromfile(args.config)
    import_modules_from_strings(**cfg.custom_imports)
    directory = Path(args.work_dir or cfg.work_dir).resolve()
    if ROOT / 'work_dirs' not in directory.parents:
        raise ValueError('Experiment outputs must stay under this project work_dirs')
    formal = Path(cfg.work_dir).resolve()
    if args.stop_after and directory == formal:
        raise ValueError('Use an independent --work-dir under preflight for smoke testing')
    if not args.stop_after and not args.launch_formal:
        raise ValueError('Formal training requires explicit --launch-formal')
    directory.mkdir(parents=True, exist_ok=True)
    for name in ['checkpoints', 'configs', 'logs', 'analysis', 'prototypes', 'visualisation', 'preflight']:
        (directory / name).mkdir(exist_ok=True)
    if not args.resume_checkpoint and (directory / 'initialization.json').exists():
        raise RuntimeError('Existing run; resume explicitly instead of overwriting')
    if not args.resume_checkpoint and any((directory / 'checkpoints').iterdir()):
        raise RuntimeError('Refusing to initialize over an existing checkpoint directory')
    if args.resume_checkpoint and not (directory / 'initialization.json').is_file():
        raise RuntimeError('Missing initialization record for resume')
    logger = logging.getLogger('mmdet')
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(str(directory / 'logs/train.log')))
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    seed = cfg.stage2_training.seed
    set_random_seed(seed, deterministic=False)
    dataset = build_dataset(cfg.data.train)
    total = int(cfg.runner.max_iters)
    if len(dataset) != 6000 or total != 6000 or dataset.queue_length != 1:
        raise ValueError('This entry is a single finite pass through the labeled 6000-frame split')
    if len({x['token'] for x in dataset.data_infos}) != 6000:
        raise ValueError('Duplicate training tokens')
    if len({x['scene_token'] for x in dataset.data_infos}) != 150:
        raise ValueError('Unexpected number of training scenes')
    order = sample_order(dataset, seed)
    model = build_model(cfg.model)
    model.init_weights()
    initial = model.stage2_initialization
    base_local = initial['stage1_local_iter']
    frozen = state_hash(model)
    model.cuda().train()
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if not trainable or any(not n.startswith(('fusion_bbox_head.proto_proj.', 'fusion_bbox_head.feat_norm.')) for n in trainable):
        raise RuntimeError('Invalid trainable parameter set')
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=cfg.optimizer.lr,
                                 weight_decay=cfg.optimizer.weight_decay)
    source_hashes = {name: sha_file(ROOT / name) for name in SOURCE_FILES}
    contract = dict(sources=source_hashes, config_sha256=hashlib.sha256(cfg.pretty_text.encode()).hexdigest(),
                    annotation_sha256=sha_file(cfg.data.train.ann_file),
                    stage1_sha256=sha_file(cfg.model.stage1_checkpoint), frozen_sha256=frozen,
                    seed=seed, total_steps=total, stage1_local_iter=base_local,
                    order_sha256=hashlib.sha256(np.asarray(order, dtype=np.int64).tobytes()).hexdigest())
    report = dict(status='INITIALIZED', formal_training_started=not bool(args.stop_after),
                  initialization=initial, contract=contract, frames=len(dataset), scenes=150,
                  trainable={n: p.numel() for n, p in trainable.items()},
                  trainable_parameters=sum(p.numel() for p in trainable.values()),
                  finite_single_pass=True, bank=bank_stats(model.fusion_bbox_head))
    if args.resume_checkpoint:
        previous = json.loads((directory / 'initialization.json').read_text())
        if previous['contract'] != contract:
            raise RuntimeError('Source/config/data/Stage1 contract changed; refuse silent resume')
    else:
        cfg.dump(str(directory / 'configs/effective_config.py'))
        json_write(directory / 'initialization.json', report)
        logger.info('STAGE2_INITIALIZATION %s', report)
    head = model.fusion_bbox_head
    completed = 0
    raw_resume = None
    if args.resume_checkpoint:
        path = Path(args.resume_checkpoint).resolve()
        if path.parent != directory / 'checkpoints':
            raise ValueError('Resume checkpoint must belong to this experiment')
        raw_resume = torch.load(str(path), map_location='cpu')
        if raw_resume['meta'].get('stage2_format') != 'prototype_stage2_20261025_v1':
            raise RuntimeError('Not a complete Stage-2 runner checkpoint')
        if raw_resume['meta']['contract'] != contract or raw_resume['sample_order'] != order:
            raise RuntimeError('Checkpoint contract/order mismatch')
        model.load_state_dict(raw_resume['state_dict'], strict=True)
        optimizer.load_state_dict(raw_resume['optimizer'])
        completed = int(raw_resume['meta']['completed_steps'])
        if int(head.stage2_steps) != completed or int(head.local_iter) != base_local + completed:
            raise RuntimeError('Resume counter mismatch')
        for name, value in raw_resume['temporal'].items():
            setattr(head, name, value.to('cuda') if torch.is_tensor(value) else value)
        if state_hash(model) != frozen:
            raise RuntimeError('Resume altered frozen detector state')
        logger.info('STAGE2_RESUME completed=%d local_iter=%d', completed, int(head.local_iter))
    audit = SimilarityAudit(head)
    if raw_resume:
        audit.window = raw_resume['similarity_window'].copy()
    batches = [[i] for i in order[completed:]]
    loader = DataLoader(dataset, batch_sampler=batches, num_workers=0,
                        collate_fn=partial(collate, samples_per_gpu=1), pin_memory=False)
    iterator = iter(loader)
    # DataLoader iterator creation consumes a torch RNG draw; restore AFTER it.
    if raw_resume:
        restore_rng(raw_resume['rng'])
        del raw_resume
    parallel = MMDataParallel(model, device_ids=[0])
    target = int(args.stop_after or total)
    if not completed < target <= total:
        raise ValueError('stop-after must exceed completed steps and not exceed 6000')
    start = time.time()
    initial_proto = state_hash(model, True)
    before_updates = head.proto_update_count.detach().cpu().clone()
    first_completed = completed
    for data in iterator:
        lr = lr_for_step(cfg, completed, total)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        output = parallel.train_step(data, optimizer)
        loss = output['loss']
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        loss.backward()
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise RuntimeError('Frozen detector received gradients')
        gradients = [p.grad for p in trainable.values() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError('Missing or nonfinite prototype gradients')
        norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), cfg.optimizer_config.grad_clip.max_norm)
        if not torch.isfinite(norm):
            raise RuntimeError('Nonfinite gradient norm')
        optimizer.step()
        completed += 1
        if int(head.stage2_steps) != completed or int(head.local_iter) != base_local + completed:
            raise RuntimeError('Stage-2/local iteration drift')
        if any(m.training for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)):
            raise RuntimeError('BatchNorm running state is not frozen')
        row = dict(step=completed, local_iter=int(head.local_iter), lr=lr, loss=float(loss.detach()),
                   grad_norm=float(norm), losses=output.get('log_vars', {}),
                   query_audit=audit.last, negative_weights=head.neg_proto_class_weight.cpu().tolist())
        with (directory / 'logs/iterations.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        if completed % cfg.stage2_training.log_interval == 0 or args.stop_after:
            logger.info('STAGE2_STEP %d/%d loss=%.6f lr=%.8g grad=%.5f elapsed=%.1fs',
                        completed, total, row['loss'], lr, float(norm), time.time()-start)
        audit_due = completed % cfg.stage2_training.audit_interval == 0 or completed == target
        if audit_due:
            if state_hash(model) != frozen:
                raise RuntimeError('Frozen detector/BN state changed')
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise RuntimeError('Nonfinite model parameter')
            if not (head.proto_update_count.sum(1) > 0).all():
                raise RuntimeError('One or more classes have no prototype history')
            integrity = dict(stage2_step=completed, local_iter=int(head.local_iter), frozen_unchanged=True,
                             update_counts=head.proto_update_count.sum(1).cpu().tolist(),
                             queue_counts=head.proto_feature_queue_count.cpu().tolist(),
                             bank=bank_stats(head), peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
            with (directory / 'analysis/integrity.jsonl').open('a') as stream:
                stream.write(json.dumps(integrity) + '\n')
            with (directory / 'analysis/similarity_windows.jsonl').open('a') as stream:
                stream.write(json.dumps(audit.report(completed), allow_nan=False) + '\n')
        if completed % cfg.checkpoint_config.interval == 0 or completed == target:
            checkpoint = save_checkpoint(model, optimizer, directory, completed, contract, order, audit)
            logger.info('STAGE2_CHECKPOINT %s', checkpoint)
            # Keep only this run's recent full checkpoints, never touch historical experiments.
            saved = sorted((directory / 'checkpoints').glob('iter_*.pth'), key=lambda p: int(p.stem.split('_')[-1]))
            for obsolete in saved[:-cfg.checkpoint_config.max_keep_ckpts]:
                obsolete.unlink()
        if completed == target:
            break
    if completed != target or state_hash(model) != frozen:
        raise RuntimeError('Incomplete pass or detector integrity failure')
    if state_hash(model, True) == initial_proto:
        raise RuntimeError('Prototype state did not change')
    report.update(status='COMPLETE' if completed == total else 'PREFLIGHT_PAUSED',
                  completed_steps=completed, local_iter=int(head.local_iter), stage2_steps=int(head.stage2_steps),
                  frozen_detector_unchanged=True, prototype_changed=True,
                  class_update_increments=(head.proto_update_count.cpu()-before_updates).sum(1).tolist(),
                  negative_weights=head.neg_proto_class_weight.cpu().tolist(),
                  optimizer_state_entries=len(optimizer.state), final_bank=bank_stats(head),
                  new_steps_this_process=completed-first_completed, runtime_seconds=time.time()-start,
                  peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
                  checkpoint=str(directory / 'checkpoints' / ('iter_%d.pth' % completed)))
    json_write(directory / 'run_summary.json', report)
    logger.info('STAGE2_%s %s', report['status'], report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='projects/configs/nusc/prototype_stage2_labeled6000_10cls_20261025.py')
    parser.add_argument('--work-dir')
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--resume-checkpoint')
    parser.add_argument('--launch-formal', action='store_true')
    args = parser.parse_args()
    os.chdir(str(ROOT))
    with tempfile.TemporaryDirectory(prefix='prototype_stage2_20261025_') as tmp:
        try:
            torch.cuda.set_device(0)
            torch.distributed.init_process_group('nccl', init_method='file://' + tmp + '/rendezvous', rank=0, world_size=1)
            run(args)
        finally:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
