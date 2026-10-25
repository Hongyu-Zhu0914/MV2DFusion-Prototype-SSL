"""Single-GPU prototype-only training, with frozen-state integrity audits."""
import argparse
import hashlib
import itertools
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Sampler
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import Hook, build_optimizer, build_runner
from mmcv.utils import import_modules_from_strings
from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


def state_hash(model, want_proto=False):
    from projects.mmdet3d_plugin.models.detectors.prototype_only_detector_0920 import is_proto_state
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        if is_proto_state(name) != want_proto:
            continue
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class IntegrityHook(Hook):
    def __init__(self, model, frozen_hash, work_dir, interval):
        self.model, self.expected = model, frozen_hash
        self.path, self.interval = Path(work_dir) / 'integrity.jsonl', interval

    def after_train_iter(self, runner):
        loss = runner.outputs['loss']
        assert torch.isfinite(loss).all(), 'Nonfinite training loss'
        step = runner.iter + 1
        if step % self.interval and step != runner.max_iters:
            return
        head = self.model.fusion_bbox_head
        assert state_hash(self.model) == self.expected, 'Frozen detector/BN state changed'
        assert all(torch.isfinite(p).all() for p in self.model.parameters()), 'Nonfinite parameter'
        assert int(head.local_iter) == step, (head.local_iter, step)
        report = dict(step=step, frozen_detector_unchanged=True,
                      update_counts=head.proto_update_count.sum(1).cpu().tolist(),
                      queue_counts=head.proto_feature_queue_count.cpu().tolist(),
                      loss=float(loss.detach()),
                      peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
        with self.path.open('a') as f:
            f.write(json.dumps(report) + '\n')
        runner.logger.info('PROTOTYPE_AUDIT %s', report)


class SubsetValidationHook(Hook):
    """Evaluate the saved checkpoint in an isolated process every N steps.

    Isolating inference keeps nuScenes temporal memory and model eval/train state
    from contaminating the training process.
    """

    def __init__(self, config_path, work_dir, interval, expected_samples,
                 limit=None, steps=None):
        self.config_path = str(Path(config_path).resolve())
        self.work_dir = Path(work_dir).resolve()
        self.interval = interval
        self.expected_samples = expected_samples
        self.limit = limit
        self.steps = set(steps) if steps is not None else None
        if interval < 1 or expected_samples < 1:
            raise ValueError('Invalid custom subset validation settings')

    def after_train_iter(self, runner):
        step = runner.iter + 1
        if step % self.interval or (self.steps is not None and step not in self.steps):
            return
        checkpoint = self.work_dir / 'iter_{}.pth'.format(step)
        if not checkpoint.is_file():
            raise RuntimeError('Checkpoint was not saved before validation: {}'.format(checkpoint))
        output_dir = self.work_dir / 'validation' / 'iter_{}'.format(step)
        if output_dir.exists():
            raise RuntimeError('Refusing to overwrite existing validation: {}'.format(output_dir))
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        log_path = output_dir.parent / 'iter_{}.log'.format(step)
        if log_path.exists():
            raise RuntimeError('Refusing to overwrite existing validation log: {}'.format(log_path))
        command = [
            sys.executable, str(Path(__file__).resolve().with_name('validate_prototype_subset_0924.py')),
            '--config', self.config_path,
            '--checkpoint', str(checkpoint),
            '--output-dir', str(output_dir),
            '--expected-samples', str(self.expected_samples),
        ]
        if self.limit is not None:
            command.extend(['--limit', str(self.limit)])
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = '1'
        runner.logger.info('CUSTOM_VAL_START step=%d samples=%d', step, self.expected_samples)
        with log_path.open('w') as log:
            process = subprocess.run(command, cwd=str(Path(__file__).resolve().parent),
                                     env=env, stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            raise RuntimeError('Custom val failed at step {} (exit {}): {}'.format(
                step, process.returncode, log_path))
        summary_path = output_dir / 'subset_validation_summary.json'
        summary = json.loads(summary_path.read_text())
        if summary['frames'] != (self.limit or self.expected_samples):
            raise RuntimeError('Custom val sample count mismatch: {}'.format(summary_path))
        runner.logger.info('CUSTOM_VAL_PASS step=%d mAP=%.4f NDS=%.4f path=%s',
                           step, summary['mAP'], summary['NDS'], summary_path)


class SkipBatchSampler(Sampler):
    """Skip already trained batches without reloading images or point clouds."""

    def __init__(self, base_sampler, count):
        self.base_sampler = base_sampler
        self.count = count
        if not 0 <= count < len(base_sampler):
            raise ValueError('Invalid resume offset')

    def __iter__(self):
        return itertools.islice(iter(self.base_sampler), self.count, None)

    def __len__(self):
        return len(self.base_sampler) - self.count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='projects/configs/nusc/prototype_stage1_r50_labeled6000_0920.py')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-subset-validation', action='store_true',
                        help='Eight training steps followed by a four-frame validation hook test')
    parser.add_argument('--resume-checkpoint',
                        help='Resume optimizer/model state and the deterministic sample order')
    args = parser.parse_args()
    if args.smoke and args.smoke_subset_validation:
        parser.error('Choose only one smoke mode')
    if args.resume_checkpoint and (args.smoke or args.smoke_subset_validation):
        parser.error('Resume cannot be combined with a smoke mode')
    resume_step = None
    if args.resume_checkpoint:
        checkpoint_path = Path(args.resume_checkpoint).resolve()
        if not checkpoint_path.is_file() or not checkpoint_path.stem.startswith('iter_'):
            parser.error('Resume checkpoint must be an existing iter_N.pth')
        resume_step = int(checkpoint_path.stem.rsplit('_', 1)[1])
    import projects.mmdet3d_plugin
    import projects.fsdv2
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    cfg = Config.fromfile(args.config)
    import_modules_from_strings(**cfg.custom_imports)
    set_random_seed(920, deterministic=False)
    if args.smoke or args.smoke_subset_validation:
        cfg.runner.max_iters = 8
        cfg.work_dir += '/smoke_subset_validation_0924' if args.smoke_subset_validation else '/smoke'
        cfg.model.fusion_bbox_head.prototype_save_dir = cfg.work_dir + '/prototypes'
    if args.smoke_subset_validation:
        cfg.checkpoint_config.interval = 8
        cfg.checkpoint_config.max_keep_ckpts = 1
        cfg.custom_subset_validation.interval = 8
        cfg.custom_subset_validation.steps = [8]
    directory = Path(cfg.work_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if resume_step is None and (directory / 'initialization.json').exists():
        raise RuntimeError('Existing run directory; do not overwrite a run')
    if resume_step is not None and not (directory / 'initialization.json').exists():
        raise RuntimeError('Cannot resume without the original initialization record')
    if resume_step is not None and checkpoint_path.parent != directory.resolve():
        raise RuntimeError('Resume checkpoint must belong to the configured work directory')
    logger = logging.getLogger('mmdet')
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(directory / 'train.log'))
    torch.cuda.set_device(0)
    with tempfile.TemporaryDirectory(prefix='proto0920_') as tmp:
        torch.distributed.init_process_group('nccl', init_method='file://' + tmp + '/rendezvous', rank=0, world_size=1)
        try:
            dataset = build_dataset(cfg.data.train)
            assert len(dataset) == 6000 and dataset.queue_length == 1
            dataloader = build_dataloader(dataset, 1, 0, num_gpus=1, dist=False, seed=920,
                shuffler_sampler=cfg.data.shuffler_sampler,
                nonshuffler_sampler=cfg.data.nonshuffler_sampler, runner_type=cfg.runner)
            if resume_step is not None:
                dataloader = DataLoader(
                    dataset,
                    batch_sampler=SkipBatchSampler(dataloader.batch_sampler, resume_step),
                    num_workers=0, collate_fn=dataloader.collate_fn,
                    pin_memory=False)
                assert len(dataloader) == len(dataset) - resume_step
            model = build_model(cfg.model)
            model.init_weights()
            model.cuda().train()
            initial_frozen = state_hash(model)
            initial_proto = state_hash(model, True)
            trainable = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
            assert trainable and all('.proto_proj.' in n or '.feat_norm.' in n for n in trainable)
            report = dict(frames=len(dataset), baseline=cfg.model.baseline_checkpoint,
                          load_report=model.baseline_load_report, frozen_sha256=initial_frozen,
                          prototype_initial_sha256=initial_proto, trainable=trainable,
                          active_classes=list(model.fusion_bbox_head.active_proto_class_ids), seed=920)
            if resume_step is None:
                (directory / 'initialization.json').write_text(json.dumps(report, indent=2))
                cfg.dump(str(directory / 'effective_config.py'))
                logger.info('INITIALIZATION %s', report)
            else:
                cfg.dump(str(directory / 'effective_config_resume_{}.py'.format(resume_step)))
            model.CLASSES = dataset.CLASSES
            parallel = MMDataParallel(model, device_ids=[0])
            optimizer = build_optimizer(parallel, cfg.optimizer)
            runner = build_runner(cfg.runner, default_args=dict(model=parallel, optimizer=optimizer,
                work_dir=str(directory), logger=logger, meta=dict(CLASSES=dataset.CLASSES, config=cfg.pretty_text)))
            runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                None if args.smoke else cfg.checkpoint_config, cfg.log_config)
            runner.register_hook(IntegrityHook(model, initial_frozen, directory,
                                               4 if (args.smoke or args.smoke_subset_validation) else 100),
                                 priority='LOW')
            subset_eval = cfg.get('custom_subset_validation')
            if subset_eval and not args.smoke:
                if cfg.checkpoint_config.interval != subset_eval['interval']:
                    raise ValueError('Checkpoint and custom validation intervals must match')
                runner.register_hook(SubsetValidationHook(
                    args.config, directory, subset_eval['interval'],
                    subset_eval['expected_samples'],
                    limit=4 if args.smoke_subset_validation else None,
                    steps=subset_eval.get('steps')), priority='LOWEST')
            if resume_step is not None:
                runner.resume(str(checkpoint_path))
                if runner.iter != resume_step - 1:
                    raise RuntimeError('Unexpected checkpoint iteration: {}'.format(runner.iter))
                if int(model.fusion_bbox_head.local_iter) != resume_step:
                    raise RuntimeError('Prototype iteration differs from checkpoint name')
                # MMCV writes meta.iter before incrementing the just-completed step.
                runner._iter = resume_step
                runner._inner_iter = resume_step
                resume_report = dict(checkpoint=str(checkpoint_path),
                                     completed_steps=resume_step,
                                     remaining_batches=len(dataloader),
                                     frozen_sha256=state_hash(model),
                                     update_counts=model.fusion_bbox_head.proto_update_count.sum(1).cpu().tolist())
                (directory / 'resume_{}_0924.json'.format(resume_step)).write_text(
                    json.dumps(resume_report, indent=2))
                logger.info('RESUME_READY %s', resume_report)
            runner.run([dataloader], [('train', 1)])
            assert state_hash(model, True) != initial_proto
            assert model.fusion_bbox_head.proto_update_count.sum() > 0
            if args.smoke or args.smoke_subset_validation:
                logger.info('SMOKE_PASS')
            else:
                logger.info('TRAINING_COMPLETE_%d', runner.max_iters)
        finally:
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
