"""Versioned global-vs-target student augmentation ablation."""
import math
import torch
import torch.nn.functional as F
from .student_augment_2026_10_14 import (
    StudentPerturbations20261014, valid_view_shape, points_in_box,
)


class StudentPerturbationsAB20261014V2:
    """Select label-independent global or pseudo-target perturbations.

    The target branch delegates to the original implementation unchanged.
    The global branch does not inspect boxes: patches cover full camera frames
    and point perturbations sample the whole LiDAR cloud.
    """
    GLOBAL_DEFAULTS = dict(
        image_probability=0.5, image_views=6, photo_views=6,
        image_patch_area=(0.05, 0.10), point_probability=0.5,
    )

    def __init__(self, cfg=None, point_cloud_range=None, scope='target', global_cfg=None):
        if scope not in ('global', 'target'):
            raise ValueError('scope must be global or target')
        self.scope = scope
        self.target = StudentPerturbations20261014(cfg, point_cloud_range)
        self.global_cfg = dict(self.GLOBAL_DEFAULTS)
        unknown = set(global_cfg or {}) - set(self.global_cfg)
        if unknown:
            raise ValueError('Unknown global settings: '+str(sorted(unknown)))
        self.global_cfg.update(global_cfg or {})
        for key in ('image_probability', 'point_probability'):
            if not 0 <= self.global_cfg[key] <= 1:
                raise ValueError('Invalid probability: '+key)
        lo, hi = self.global_cfg['image_patch_area']
        if not 0 < lo <= hi <= self.target.cfg['max_image_coverage']:
            raise ValueError('Global patch exceeds image coverage budget')
        for key in ('image_views', 'photo_views'):
            if self.global_cfg[key] < 1:
                raise ValueError('View limits must be positive')
        if scope == 'global':
            self.target.cfg['photo_views'] = self.global_cfg['photo_views']

    @staticmethod
    def _patch(image, area, cfg):
        height, width = image.shape[-2:]
        ratio = StudentPerturbations20261014._draw(area)
        ph = max(1, min(height, round(height * math.sqrt(ratio))))
        pw = max(1, min(width, round(width * math.sqrt(ratio))))
        x = int(torch.randint(width-pw+1, ()))
        y = int(torch.randint(height-ph+1, ()))
        patch = image[:, y:y+ph, x:x+pw]
        operation = int(torch.randint(3, ()))
        if operation == 0:
            patch.zero_()  # normalized zero equals the per-channel mean fill
            key = 'erase'
        elif operation == 1:
            blocks = cfg['mosaic_blocks']
            small = F.interpolate(patch[None], size=(max(1, ph//blocks), max(1, pw//blocks)),
                                  mode='area')
            patch.copy_(F.interpolate(small, size=(ph, pw), mode='nearest')[0])
            key = 'mosaic'
        else:
            sigma = StudentPerturbations20261014._draw(cfg['blur_sigma'])
            axis = patch.new_tensor([-1., 0., 1.])
            weight = torch.exp(-axis**2/(2*sigma**2))
            weight = weight/weight.sum()
            kernel = (weight[:, None]*weight[None, :]).expand(3, 1, 3, 3)
            patch.copy_(F.conv2d(F.pad(patch[None], (1, 1, 1, 1), mode='replicate'),
                                 kernel, groups=3)[0])
            key = 'blur'
        return ph*pw, key

    @torch.no_grad()
    def __call__(self, data, boxes):
        if self.scope == 'target':
            out, stats = self.target(data, boxes)
            stats['aug_scope_global'] = False
            return out, stats

        # The empty box slice lets the shared helper clone inputs and apply
        # photometric transforms while keeping the global branch box-agnostic.
        out, stats = self.target(data, boxes[:0])
        stats.update(aug_scope_global=True, aug_global_patches=0,
                     aug_image_affected_pixels=0,
                     aug_points_before=len(out['points'][0][0]),
                     aug_drop_fraction=0.0)
        if not self.target.cfg['enabled']:
            return out, stats

        images = out['img']
        meta = out['img_metas'][0][0]
        views, height, width = images.shape[2], images.shape[-2], images.shape[-1]
        shapes = [valid_view_shape(meta, view, height, width) for view in range(views)]
        if float(torch.rand(())) < self.global_cfg['image_probability']:
            candidates = torch.randperm(views)[:min(views, self.global_cfg['image_views'])]
            for view in candidates.tolist():
                h, w = shapes[view]
                count, key = self._patch(images[0, 0, view, :, :h, :w],
                                         self.global_cfg['image_patch_area'], self.target.cfg)
                stats['aug_global_patches'] += 1
                stats['aug_image_affected_pixels'] += count
                stats['aug_'+key+'_patches'] += 1
                stats['aug_image_patches'] += 1

        points = out['points'][0][0]
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError('Invalid point layout')
        # This draw is global and does not use pseudo-box membership. Sparse
        # objects can therefore lose a few returns; the fraction remains small.
        if len(points) and float(torch.rand(())) < self.global_cfg['point_probability']:
            if float(torch.rand(())) < self.target.cfg['jitter_probability']:
                noise = (torch.randn_like(points[:, :3])*self.target.cfg['noise_sigma']).clamp(
                    -self.target.cfg['noise_clip'], self.target.cfg['noise_clip'])
                candidate = points[:, :3] + noise
                valid = torch.isfinite(candidate).all(dim=1)
                if self.target.point_cloud_range is not None:
                    limits = candidate.new_tensor(self.target.point_cloud_range)
                    valid &= ((candidate >= limits[:3]) & (candidate <= limits[3:])).all(dim=1)
                points[valid, :3] = candidate[valid]
                stats['aug_jitter_points'] += int(valid.sum())
                stats['aug_global_jitter_points'] = int(valid.sum())
            else:
                count = min(int(len(points)*self.target._draw(self.target.cfg['dropout_ratio'])),
                            max(0, len(points)-self.target.cfg['min_remaining_points']))
                keep = torch.ones(len(points), dtype=torch.bool, device=points.device)
                drop = torch.randperm(len(points), device=points.device)[:count]
                keep[drop] = False
                points = points[keep]
                out['points'][0][0] = points
                stats['aug_dropped_points'] = count
                stats['aug_global_dropped_points'] = count
                stats['aug_drop_fraction'] = count/max(1, stats['aug_points_before'])
        if not torch.isfinite(images).all() or not torch.isfinite(out['points'][0][0]).all():
            raise ValueError('Non-finite global augmentation output')
        out['img_metas'][0][0]['student_augmentation_2026_10_14'] = dict(stats)
        return out, stats
