"""Student-only, label-preserving perturbations. No model parameters or EMA state."""
import copy
import math
import torch
import torch.nn.functional as F


def points_in_box(points, center, dimensions, yaw):
    """Oriented LiDAR box membership, using gravity center and local box axes."""
    relative = points[:, :3] - center
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    local = torch.stack((relative[:, 0]*cosine + relative[:, 1]*sine,
                         -relative[:, 0]*sine + relative[:, 1]*cosine,
                         relative[:, 2]), dim=-1)
    return (local.abs() <= dimensions/2).all(dim=-1)


def valid_view_shape(meta, view, height, width):
    shape = meta.get('img_shape')
    if shape is None:
        raise ValueError('Missing img_shape: do not perturb unknown padding')
    if isinstance(shape[0], (tuple, list)) or getattr(shape[0], 'ndim', 0) > 0:
        shape = shape[view]
    h, w = min(int(shape[0]), height), min(int(shape[1]), width)
    if h <= 0 or w <= 0:
        raise ValueError('Invalid effective image shape')
    return h, w


def project_box(corners, matrix, height, width):
    """Conservative projection: skip boxes crossing the camera near plane."""
    corners = corners.to(device=matrix.device, dtype=matrix.dtype)
    homogeneous = torch.cat((corners, torch.ones_like(corners[:, :1])), dim=1)
    projected = homogeneous @ matrix.T
    if not torch.isfinite(projected).all() or (projected[:, 2] <= 1e-3).any():
        return None
    xy = projected[:, :2] / projected[:, 2:3]
    x0 = max(0, math.floor(float(xy[:, 0].min())))
    y0 = max(0, math.floor(float(xy[:, 1].min())))
    x1 = min(width, math.ceil(float(xy[:, 0].max())))
    y1 = min(height, math.ceil(float(xy[:, 1].max())))
    if x1-x0 < 16 or y1-y0 < 16:
        return None
    return x0, y0, x1, y1


class StudentPerturbations20261014:
    DEFAULTS = dict(enabled=True, photo_probability=0.5, photo_views=2,
                    brightness=(0.9, 1.1), contrast=(0.9, 1.1), saturation=(0.9, 1.1),
                    object_probability=0.3, max_objects=8, image_modality_probability=0.5,
                    patch_area=(0.05, 0.10), max_image_coverage=0.10,
                    noise_sigma=0.005, noise_clip=0.02, dropout_ratio=(0.05, 0.10),
                    min_object_points=20, min_remaining_points=16,
                    jitter_probability=0.5, blur_sigma=(0.5, 0.8), mosaic_blocks=8)

    def __init__(self, cfg=None, point_cloud_range=None):
        self.cfg = dict(self.DEFAULTS)
        unknown = set(cfg or {}) - set(self.cfg)
        if unknown:
            raise ValueError('Unknown augmentation settings: '+str(sorted(unknown)))
        self.cfg.update(cfg or {})
        self.point_cloud_range = point_cloud_range
        for key in ('photo_probability', 'object_probability', 'image_modality_probability',
                    'jitter_probability'):
            if not 0 <= self.cfg[key] <= 1:
                raise ValueError('Probability outside [0,1]: '+key)
        for key in ('brightness', 'contrast', 'saturation', 'blur_sigma'):
            lo, hi = self.cfg[key]
            if not 0 < lo <= hi:
                raise ValueError('Invalid positive interval: '+key)
        for key in ('patch_area', 'dropout_ratio'):
            lo, hi = self.cfg[key]
            if not 0 <= lo <= hi < 1:
                raise ValueError('Invalid fraction interval: '+key)
        if not 0 < self.cfg['max_image_coverage'] <= 1:
            raise ValueError('Invalid image coverage budget')
        if self.cfg['noise_sigma'] < 0 or self.cfg['noise_clip'] <= 0:
            raise ValueError('Invalid noise limits')
        if any(self.cfg[k] < 1 for k in ('photo_views', 'max_objects', 'min_object_points',
                                        'min_remaining_points', 'mosaic_blocks')):
            raise ValueError('Counts must be positive')

    @staticmethod
    def _draw(interval):
        return interval[0] + float(torch.rand(()))*(interval[1]-interval[0])

    @staticmethod
    def _clone_points(value):
        if torch.is_tensor(value):
            return value.clone()
        if isinstance(value, list):
            return [StudentPerturbations20261014._clone_points(x) for x in value]
        if isinstance(value, tuple):
            return [StudentPerturbations20261014._clone_points(x) for x in value]
        raise TypeError('Expected nested point tensors')

    @staticmethod
    def _norm(meta, image):
        norm = meta['img_norm_cfg']
        mean = image.new_tensor(norm['mean']).view(3, 1, 1)
        std = image.new_tensor(norm['std']).view(3, 1, 1)
        if not (std > 0).all():
            raise ValueError('Invalid image normalization')
        return mean, std, bool(norm.get('to_rgb', True))

    @torch.no_grad()
    def __call__(self, data, boxes):
        student = dict(data)
        student['img'] = data['img'].clone()
        student['points'] = self._clone_points(data['points'])
        student['img_metas'] = copy.deepcopy(data['img_metas'])
        stats = dict(aug_enabled=bool(self.cfg['enabled']), aug_photo_views=0,
                     aug_targets=0, aug_image_patches=0, aug_jitter_points=0,
                     aug_dropped_points=0, aug_skipped_sparse=0,
                     aug_erase_patches=0, aug_mosaic_patches=0, aug_blur_patches=0)
        if not self.cfg['enabled']:
            return student, stats
        images = student['img']
        if images.ndim != 6 or tuple(images.shape[:2]) != (1, 1) or images.shape[3] != 3:
            raise ValueError('Augmentation requires B=1,T=1,3-channel images')
        meta = student['img_metas'][0][0]
        views, height, width = images.shape[2], images.shape[-2], images.shape[-1]
        shapes = [valid_view_shape(meta, v, height, width) for v in range(views)]
        # Operate on raw-range pixels, not normalized values or padded pixels.
        if float(torch.rand(())) < self.cfg['photo_probability']:
            indices = torch.randperm(views)[:min(views, self.cfg['photo_views'])].tolist()
            for view in indices:
                h, w = shapes[view]
                image = images[0, 0, view, :, :h, :w]
                mean, std, rgb = self._norm(meta, image)
                raw = image*std + mean
                raw = raw*self._draw(self.cfg['brightness'])
                average = raw.mean(dim=(-2, -1), keepdim=True)
                raw = (raw-average)*self._draw(self.cfg['contrast']) + average
                weights = raw.new_tensor([0.299, 0.587, 0.114] if rgb else
                                         [0.114, 0.587, 0.299]).view(3, 1, 1)
                gray = (raw*weights).sum(dim=0, keepdim=True)
                raw = gray + (raw-gray)*self._draw(self.cfg['saturation'])
                images[0, 0, view, :, :h, :w] = (raw.clamp(0, 255)-mean)/std
                stats['aug_photo_views'] += 1
        if not len(boxes):
            return student, stats
        points = student['points'][0][0]
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError('Invalid point layout')
        centers = boxes.gravity_center.to(points)
        dimensions = boxes.tensor[:, 3:6].to(points)
        yaw = boxes.tensor[:, 6].to(points)
        corners = boxes.corners.to(device=images.device, dtype=images.dtype)
        selected = (torch.rand(len(boxes)) < self.cfg['object_probability']).nonzero().flatten()
        if len(selected) > self.cfg['max_objects']:
            selected = selected[torch.randperm(len(selected))[:self.cfg['max_objects']]]
        stats['aug_targets'] = len(selected)
        consumed = torch.zeros(len(points), dtype=torch.bool, device=points.device)
        retained = torch.ones_like(consumed)
        coverage = [0 for _ in range(views)]
        for idx in selected.tolist():
            # Extra information degradation is exclusive per object: image OR LiDAR.
            if float(torch.rand(())) < self.cfg['image_modality_probability']:
                matrix = data['lidar2img'][0, 0].to(device=images.device, dtype=images.dtype)
                available = []
                for view in range(views):
                    h, w = shapes[view]
                    rectangle = project_box(corners[idx], matrix[view], h, w)
                    if rectangle:
                        available.append((view, rectangle))
                if not available:
                    continue
                view, (x0, y0, x1, y1) = available[int(torch.randint(len(available), ()))]
                ratio = self._draw(self.cfg['patch_area'])
                ph = max(1, min(y1-y0, round((y1-y0)*math.sqrt(ratio))))
                pw = max(1, min(x1-x0, round((x1-x0)*math.sqrt(ratio))))
                h, w = shapes[view]
                if coverage[view]+ph*pw > self.cfg['max_image_coverage']*h*w:
                    continue
                px = x0 + int(torch.randint(x1-x0-pw+1, ()))
                py = y0 + int(torch.randint(y1-y0-ph+1, ()))
                patch = images[0, 0, view, :, py:py+ph, px:px+pw]
                operation = int(torch.randint(3, ()))
                if operation == 0:
                    # Normalized zero is channel-mean fill, not raw black.
                    patch.zero_()
                    stats['aug_erase_patches'] += 1
                elif operation == 1:
                    mh, mw = max(1, ph//self.cfg['mosaic_blocks']), max(1, pw//self.cfg['mosaic_blocks'])
                    small = F.interpolate(patch[None], size=(mh, mw), mode='area')
                    patch.copy_(F.interpolate(small, size=(ph, pw), mode='nearest')[0])
                    stats['aug_mosaic_patches'] += 1
                else:
                    sigma = self._draw(self.cfg['blur_sigma'])
                    axis = patch.new_tensor([-1., 0., 1.])
                    weight = torch.exp(-axis**2/(2*sigma**2))
                    weight = weight/weight.sum()
                    kernel = (weight[:, None]*weight[None, :]).expand(3, 1, 3, 3)
                    result = F.conv2d(F.pad(patch[None], (1, 1, 1, 1), mode='replicate'),
                                      kernel, groups=3)[0]
                    patch.copy_(result)
                    stats['aug_blur_patches'] += 1
                coverage[view] += ph*pw
                stats['aug_image_patches'] += 1
            else:
                mask = points_in_box(points, centers[idx], dimensions[idx], yaw[idx]) & ~consumed
                ids = mask.nonzero().flatten()
                consumed |= mask  # Avoid repeatedly degrading overlaps.
                if len(ids) < self.cfg['min_object_points']:
                    stats['aug_skipped_sparse'] += 1
                    continue
                if float(torch.rand(())) < self.cfg['jitter_probability']:
                    noise = (torch.randn_like(points[ids, :3])*self.cfg['noise_sigma']).clamp(
                        -self.cfg['noise_clip'], self.cfg['noise_clip'])
                    candidate = points[ids, :3]+noise
                    valid = torch.isfinite(candidate).all(dim=1)
                    if self.point_cloud_range is not None:
                        limits = candidate.new_tensor(self.point_cloud_range)
                        valid &= ((candidate >= limits[:3]) & (candidate <= limits[3:])).all(dim=1)
                    points[ids[valid], :3] = candidate[valid]
                    stats['aug_jitter_points'] += int(valid.sum())
                else:
                    count = min(int(len(ids)*self._draw(self.cfg['dropout_ratio'])),
                                max(0, len(ids)-self.cfg['min_remaining_points']))
                    drop = ids[torch.randperm(len(ids), device=points.device)[:count]]
                    retained[drop] = False
                    stats['aug_dropped_points'] += count
        student['points'][0][0] = points[retained]
        if not torch.isfinite(images).all() or not torch.isfinite(student['points'][0][0]).all():
            raise ValueError('Non-finite student perturbation output')
        # Prefix is intentionally separate from normal augmentation metadata.
        student['img_metas'][0][0]['student_augmentation_2026_10_14'] = dict(stats)
        return student, stats
