"""Single-GPU, alternating-frame runner with explicit final subset evaluation."""

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
import functools
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from mmcv import Config
from mmcv.parallel import collate, MMDataParallel
from mmcv.runner import Hook, build_optimizer, build_runner
from mmcv.utils import import_modules_from_strings
from mmdet.models import build_detector
from mmdet3d.datasets import build_dataset


def digest(module, frozen_only=False):
    h = hashlib.sha256()
    for name, value in module.named_parameters():
        if frozen_only and value.requires_grad:
            continue
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class AlternatingSceneSampler(Sampler):
    """Deterministic scene blocks, frame order retained, exact 1:1 attempts.

    queue=1 and wrapper resets temporal memory every frame. Alternating domains
    would not be valid for a future persistent-temporal implementation.
    """
    def __init__(self, dataset, steps, start=0, seed=1011):
        self.steps, self.start = steps, start
        self.orders = []
        for domain in range(2):
            groups = {}
            lo = 0 if domain == 0 else dataset.supervised_length
            hi = dataset.supervised_length if domain == 0 else len(dataset)
            for idx in range(lo, hi):
                row = dataset.data_infos[idx]
                groups.setdefault(row['scene_token'], []).append(idx)
            keys = sorted(groups)
            random.Random(seed+domain).shuffle(keys)
            order = []
            for key in keys:
                order.extend(sorted(groups[key], key=lambda i: dataset.data_infos[i]['timestamp']))
            if not order:
                raise ValueError('Empty sampling domain')
            self.orders.append(order)

    def __iter__(self):
        for step in range(self.start, self.steps):
            order = self.orders[step % 2]
            yield order[(step // 2) % len(order)]

    def __len__(self):
        return self.steps-self.start


class AuditHook(Hook):
    def before_run(self, runner):
        m = runner.model.module
        self.frozen = digest(m.student, frozen_only=True)
        if any(p.requires_grad for p in m.teacher.parameters()):
            raise AssertionError('Teacher has trainable parameters')

    def after_train_iter(self, runner):
        if (runner.iter+1) % 1000 and runner.iter+1 != runner.max_iters:
            return
        m = runner.model.module
        valid = self.frozen == digest(m.student, frozen_only=True)
        row = dict(iter=runner.iter+1, frozen_unchanged=valid,
                   ema_updates=int(m.ema_updates), stats=m.last_online_stats)
        with open(Path(runner.work_dir)/'integrity.jsonl', 'a') as f:
            f.write(json.dumps(row)+'\n')
        if not valid:
            raise AssertionError('Frozen student parameters changed')


class RNGHook(Hook):
    def after_train_iter(self, runner):
        # Stored in checkpoint meta BEFORE checkpoint hook, for exact RNG recovery.
        if runner.meta is None:
            runner.meta = {}
        runner.meta['ssl_rng'] = dict(python=random.getstate(), numpy=np.random.get_state(),
                                     torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())
        runner.meta['ssl_completed_iters'] = runner.iter+1


def setup(cfg):
    import projects.mmdet3d_plugin
    import projects.fsdv2
    import_modules_from_strings(**cfg.custom_imports)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='projects/configs/nusc/ssl_online_r50_ema_1011.py')
    parser.add_argument('--resume')
    parser.add_argument('--max-iters', type=int)
    parser.add_argument('--work-dir')
    parser.add_argument('--skip-final-eval', action='store_true')
    args = parser.parse_args()
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Single GPU only')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    cfg = Config.fromfile(args.config)
    setup(cfg)
    seed = cfg.ssl_seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    steps = args.max_iters or cfg.runner.max_iters
    directory = Path(args.work_dir or cfg.work_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(directory.glob('iter_*.pth')):
        raise RuntimeError('Existing checkpoints: use --resume or new --work-dir')
    cfg.dump(str(directory/'resolved_config.py'))
    dataset = build_dataset(cfg.data.train)
    if (dataset.supervised_length, dataset.pseudo_length) != (6000, 6008):
        raise AssertionError('Unexpected split lengths')
    torch.cuda.set_device(0)
    with tempfile.TemporaryDirectory(prefix='online_ema_nccl_') as rendezvous:
        torch.distributed.init_process_group('nccl', init_method='file://'+rendezvous+'/init', rank=0, world_size=1)
        try:
            model = build_detector(cfg.model)
            model.init_weights(); model.cuda().train()
            parallel = MMDataParallel(model, device_ids=[0])
            # Only student optimizer: teacher parameters are excluded completely.
            optimizer = build_optimizer(model.student, cfg.optimizer)
            logger = logging.getLogger('online_ema'); logger.setLevel(logging.INFO)
            logger.addHandler(logging.StreamHandler())
            runner = build_runner(dict(type='IterBasedRunner', max_iters=steps), default_args=dict(
                model=parallel, optimizer=optimizer, work_dir=str(directory), logger=logger,
                meta=dict(config=cfg.pretty_text, seed=seed, sampling='alternating_scene_blocks')))
            runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                           cfg.checkpoint_config, cfg.log_config)
            from projects.mmdet3d_plugin.models.detectors.mv2dfusion_online_ema_1011 import OnlineEMAHook1011
            runner.register_hook(OnlineEMAHook1011(), priority=45)
            runner.register_hook(RNGHook(), priority=46)
            runner.register_hook(AuditHook(), priority=49)
            if args.resume:
                # This installed MMCV saves meta.iter as zero-based current iter
                # and does not restore custom meta in IterBasedRunner.resume.
                # Restore explicitly from our completed-step field, strictly.
                checkpoint = torch.load(args.resume, map_location='cpu')
                meta = checkpoint['meta']
                if 'ssl_completed_iters' not in meta:
                    raise RuntimeError('Use a new EMA checkpoint with completed-step metadata')
                model.load_state_dict(checkpoint['state_dict'], strict=True)
                optimizer.load_state_dict(checkpoint['optimizer'])
                runner._iter = int(meta['ssl_completed_iters'])
                runner._inner_iter = runner._iter
                runner._epoch = int(meta['epoch'])
                # TextLogger dumps meta to JSON before_run; RNG tensors are
                # restored separately and only reattached after the first step.
                runner.meta.update({k: v for k,v in meta.items() if k != 'ssl_rng'})
                state = meta.get('ssl_rng')
                if state is None:
                    raise RuntimeError('Resume checkpoint lacks RNG state')
                random.setstate(state['python']); np.random.set_state(state['numpy'])
                torch.set_rng_state(state['torch'].cpu())
                torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
                del checkpoint
            sampler = AlternatingSceneSampler(dataset, steps, start=runner.iter, seed=seed)
            loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0,
                                collate_fn=functools.partial(collate, samples_per_gpu=1), pin_memory=False,
                                generator=torch.Generator().manual_seed(seed))
            print('SSL_START', json.dumps(dict(start=runner.iter, steps=steps, labeled=6000,
                                               unlabeled=6008, ema=cfg.model.ema_decay)), flush=True)
            runner.run([loader], [('train', 1)])
            print('SSL_TRAIN_PASS', flush=True)
        finally:
            torch.distributed.destroy_process_group()
    if not args.skip_final_eval and cfg.ssl_final_evaluation:
        # Release optimizer/model GPU memory before separate full subset evaluation.
        del runner, optimizer, parallel, model, loader, dataset
        gc.collect(); torch.cuda.empty_cache()
        subprocess.run([sys.executable, 'scripts/evaluation/validate_online_ema_1011.py', '--config', args.config,
                        '--checkpoint', str(directory/'latest.pth'), '--branch', 'student',
                        '--output-dir', str(directory/'validation_final_student')], check=True)


if __name__ == '__main__':
    main()
