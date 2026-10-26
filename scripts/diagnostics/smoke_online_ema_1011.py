"""Bounded in-memory test. No checkpoint saving and no formal training."""

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
import copy
import gc
import hashlib
import logging
import json
import os
import sys
import tempfile
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter, MMDataParallel
from mmcv.runner import build_optimizer, build_runner
from mmcv.utils import import_modules_from_strings
from mmdet3d.datasets import build_dataset
from mmdet.models import build_detector


def digest(model):
    result = hashlib.sha256()
    for name, value in model.state_dict().items():
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='projects/configs/nusc/ssl_online_r50_ema_1011.py')
    parser.add_argument('--dataset-only', action='store_true')
    parser.add_argument('--runner-smoke', action='store_true')
    parser.add_argument('--expected-labeled', type=int)
    args = parser.parse_args()
    if args.runner_smoke:
        raise ValueError('Use train_online_ema_1011.py --max-iters 2 for AdamW runner testing')
    sys.path.insert(0, os.getcwd())
    import projects.mmdet3d_plugin
    import projects.fsdv2
    cfg = Config.fromfile(args.config)
    import_modules_from_strings(**cfg.custom_imports)
    torch.manual_seed(1919)
    dataset = build_dataset(cfg.data.train)
    print('DATASET', len(dataset), dataset.supervised_length, dataset.pseudo_length, flush=True)
    if args.expected_labeled is not None:
        assert dataset.supervised_length == args.expected_labeled
        assert dataset.pseudo_length == 6008
    batches = {}
    for name, idx in [('labeled', 0), ('unlabeled', dataset.supervised_length)]:
        sample = dataset[idx]
        batch = collate([sample], samples_per_gpu=1)
        batches[name] = batch
        print('LOADED', name, idx, flush=True)
    if args.dataset_only:
        return
    rendezvous = tempfile.TemporaryDirectory(prefix='mv2d_online_smoke_')
    torch.distributed.init_process_group('nccl', init_method='file://' + rendezvous.name + '/rendezvous',
                                         rank=0, world_size=1)
    model = build_detector(cfg.model)
    model.init_weights()
    model.cuda().train()
    teacher_before = digest(model.teacher)
    print('TEACHER_HASH', teacher_before, flush=True)
    print('PARAMETERS', {key: sum(p.numel() for p in module.parameters() if p.requires_grad)
                         for key, module in [('teacher', model.teacher), ('student', model.student),
                                             ('student_pts', model.student.pts_backbone)]}, flush=True)
    if args.runner_smoke:
        parallel = MMDataParallel(model, device_ids=[0])
        optimizer = build_optimizer(parallel, cfg.optimizer)
        runner = build_runner(dict(type='IterBasedRunner', max_iters=3), default_args=dict(
            model=parallel, optimizer=optimizer, work_dir=rendezvous.name,
            logger=logging.getLogger('online_smoke')))
        runner.register_training_hooks(None, cfg.optimizer_config, None, None)
        runner.run([[batches['unlabeled'], batches['labeled'], batches['unlabeled']]], [('train', 1)])
        assert digest(model.teacher) == teacher_before
        # Fourth isolated step tests the hook with no grads (no AdamW decay).
        student_before = digest(model.student)
        model.score_threshold = 1.0
        runner._max_iters = 4
        runner.run([[batches['unlabeled']]], [('train', 1)])
        assert digest(model.student) == student_before
        assert digest(model.teacher) == teacher_before
        print('RUNNER_PASS: pseudo/real/pseudo AdamW steps; empty step unchanged', flush=True)
        torch.distributed.destroy_process_group()
        return
    # SGD avoids allocating full Adam moments during the bounded 24GB test.
    optimizer = torch.optim.SGD([p for p in model.student.parameters() if p.requires_grad], lr=1e-6)
    for name in ['unlabeled', 'labeled']:
        data = scatter(batches[name], [0])[0]
        optimizer.zero_grad(set_to_none=True)
        losses = model(return_loss=True, **data)
        loss, logs = model._parse_losses(losses)
        assert torch.isfinite(loss), logs
        if name == 'unlabeled':
            assert model.last_online_stats['selected'] > 0
            assert not any('det2d.' in k or 'imgqg.' in k for k in losses)
        else:
            assert any('det2d.' in k for k in losses)
        loss.backward()
        grads = [(n, p) for n, p in model.student.named_parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(p.grad).all() for _, p in grads)
        assert any(p.grad.abs().sum() > 0 for _, p in grads)
        assert all(p.grad is None for p in model.teacher.parameters())
        chosen_name, chosen = next((n, p) for n, p in grads if p.grad.abs().max() > 1e-5)
        before = chosen.detach().clone()
        teacher_chosen = dict(model.teacher.named_parameters())[chosen_name]
        teacher_value = teacher_chosen.detach().clone()
        torch.nn.utils.clip_grad_norm_(model.student.parameters(), 35)
        optimizer.step()
        assert not torch.equal(before, chosen)
        expected = teacher_value * model.ema_decay + chosen.detach() * (1-model.ema_decay)
        assert model.update_teacher()
        assert torch.allclose(teacher_chosen, expected, atol=2e-7, rtol=1e-5)
        assert not model.teacher.training
        print('STEP_PASS', name, json.dumps(model.last_online_stats), 'loss', float(loss),
              'grad_parameters', len(grads), 'changed', chosen_name,
              'peak_gib', torch.cuda.max_memory_allocated()/2**30, flush=True)
        del data, losses, loss, grads, before
        optimizer.zero_grad(set_to_none=True)
        model.reset_frame(model.student)
        gc.collect()
        torch.cuda.empty_cache()
    assert digest(model.teacher) != teacher_before, 'EMA teacher did not update'
    assert int(model.ema_updates) == 2
    teacher_after = digest(model.teacher)
    student_after = digest(model.student)
    model.score_threshold = 1.0
    data = scatter(batches['unlabeled'], [0])[0]
    optimizer.zero_grad(set_to_none=True)
    losses = model(return_loss=True, **data)
    assert model.last_online_stats['selected'] == 0
    loss, _ = model._parse_losses(losses)
    loss.backward()
    assert all(p.grad is None for p in model.student.parameters())
    optimizer.step()
    assert not model.update_teacher()
    assert digest(model.teacher) == teacher_after
    assert digest(model.student) == student_after
    assert int(model.ema_updates) == 2
    print('EMPTY_PASS; EMA_MATH_PASS; SMOKE_PASS', flush=True)
    torch.distributed.destroy_process_group()
    rendezvous.cleanup()


if __name__ == '__main__':
    main()
