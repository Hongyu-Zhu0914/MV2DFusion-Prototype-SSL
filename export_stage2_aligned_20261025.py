"""Read-only Stage2 evaluation and aligned candidate/final-detection export.

No head swapping; current checkpoint projection and prototypes are used together.
Candidate statistics follow the raw ten-class metadata GT convention of the
0929 visualisation, distinct from the official nuScenes evaluation filters.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np

CLASSES = ('car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
           'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone')
DIST = np.array([1., 1.2, 1.5, 1.5, 1.5, .5, .5, .4, .35, .25])
DIRECTIONAL = {0, 1, 2, 3, 4, 6, 7}


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def match(boxes, labels, confidence, gt_boxes, gt_labels, upper=None):
    """Strict same-class, confidence-ordered, one-to-one BEV/yaw matching."""
    count = len(boxes)
    matched = np.full(count, -1, dtype=np.int64)
    if not count or not len(gt_boxes):
        return matched, np.full(count, np.inf, dtype=np.float32)
    distances = np.linalg.norm(boxes[:, None, :2] - gt_boxes[None, :, :2], axis=2)
    for cls in range(10):
        candidates = np.where((labels == cls) & (confidence > .2) &
                              ((confidence < upper) if upper is not None else True))[0]
        used = set()
        gt_ids = np.where(gt_labels == cls)[0]
        for idx in sorted(candidates.tolist(), key=lambda j: (-float(confidence[j]), j)):
            eligible = []
            for g in gt_ids:
                if int(g) in used or distances[idx, g] >= DIST[cls]:
                    continue
                if cls in DIRECTIONAL:
                    angle = abs(math.atan2(math.sin(float(boxes[idx, 6] - gt_boxes[g, 6])),
                                           math.cos(float(boxes[idx, 6] - gt_boxes[g, 6]))))
                    if angle > math.pi / 4:
                        continue
                eligible.append((float(distances[idx, g]), int(g)))
            if eligible:
                _, g = min(eligible)
                used.add(g)
                matched[idx] = g
    return matched, distances.min(axis=1).astype(np.float32)


def batch_token(batch, filename_to_token=None):
    def visit(x, depth=0):
        if depth > 8:
            return None
        if isinstance(x, dict):
            for key in ('sample_idx', 'sample_token', 'token'):
                if isinstance(x.get(key), str):
                    return x[key]
            filenames = x.get('filename')
            if (filename_to_token is not None and isinstance(filenames, (list, tuple))
                    and filenames and all(isinstance(v, str) for v in filenames)):
                answer = filename_to_token.get(tuple(sorted(filenames)))
                if answer is not None:
                    return answer
            for value in x.values():
                answer = visit(value, depth + 1)
                if answer is not None:
                    return answer
        elif isinstance(x, (list, tuple)):
            for value in x:
                answer = visit(value, depth + 1)
                if answer is not None:
                    return answer
        elif x.__class__.__name__ == 'DataContainer':
            return visit(x.data, depth + 1)
        return None
    return visit(batch['img_metas'])


def save_arrays(directory, index, values):
    prefix = directory / ('sample_%d' % index)
    for name, array in values.items():
        path = Path(str(prefix) + '_' + name + '.npy')
        np.save(str(path), array, allow_pickle=False)
        reread = np.load(str(path), allow_pickle=False)
        if not np.array_equal(reread, array):
            raise AssertionError('NPY readback mismatch: %s' % path)


def self_test():
    boxes = np.zeros((5, 9), dtype=np.float32)
    boxes[:, 0] = [.1, .2, 1.9, 3., .1]
    boxes[4, 6] = math.pi / 2
    labels = np.zeros(5, dtype=np.int64)
    conf = np.array([.95, .5, .6, .4, .7])
    gt = np.zeros((1, 7), dtype=np.float32)
    full, nearest = match(boxes, labels, conf, gt, np.array([0]))
    band, _ = match(boxes, labels, conf, gt, np.array([0]), upper=.8)
    assert full.tolist() == [0, -1, -1, -1, -1]
    assert band.tolist() == [-1, 0, -1, -1, -1]
    assert ((full < 0) & (nearest > 2)).tolist() == [False, False, False, True, False]
    empty, distance = match(boxes, labels, conf, gt[:0], np.array([], dtype=np.int64))
    assert (empty == -1).all() and np.isinf(distance).all()
    assert batch_token({'img_metas': [[{'filename': ['b.jpg', 'a.jpg']}]]},
                       {('a.jpg', 'b.jpg'): 'frame-token'}) == 'frame-token'
    print('MATCHING_SELF_TEST_PASS', flush=True)


def main(args):
    import torch
    import torch.nn.functional as F
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint
    from mmcv.utils import import_modules_from_strings
    from mmdet3d.core.bbox.structures import xywhr2xyxyr  # noqa: F401
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    import projects.mmdet3d_plugin  # noqa: F401
    import projects.fsdv2  # noqa: F401
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
    from validate_prototype_subset_0924 import SubsetDetectionEval
    from nuscenes import NuScenes

    started = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    cfg = Config.fromfile(args.config)
    import_modules_from_strings(**cfg.custom_imports)
    data_cfg = copy.deepcopy(cfg.data[args.split])
    data_cfg.test_mode = True
    dataset = build_dataset(data_cfg)
    expected = {'val': 3014, 'test': 3005}[args.split]
    assert len(dataset) == expected, (args.split, len(dataset))
    if args.limit:
        assert 0 < args.limit <= expected
        dataset.data_infos = dataset.data_infos[:args.limit]
    tokens = [x['token'] for x in dataset.data_infos]
    assert len(tokens) == len(set(tokens)) == len(dataset)
    # The existing Collect3D config retains six camera filenames, but does not
    # retain sample_idx. Resolve that *actual batch identity* back to the PKL
    # token; never simply assume that the DataLoader follows metadata order.
    filename_to_token = {
        tuple(sorted(str(cam['data_path']) for cam in info['cams'].values())): info['token']
        for info in dataset.data_infos
    }
    assert len(filename_to_token) == len(tokens), 'Ambiguous camera filename identity'
    out = Path(args.output_dir).resolve()
    feature_root = Path(args.feature_dir).resolve()
    for path in (out, feature_root):
        if path.exists() and any(path.iterdir()):
            raise RuntimeError('Refusing nonempty output: %s' % path)
        path.mkdir(parents=True, exist_ok=True)
    directories = {key: feature_root / key for key in ('candidates', 'detections', 'gt')}
    for path in directories.values():
        path.mkdir()
    model = build_model(cfg.model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu', strict=True)
    meta = checkpoint['meta']
    assert meta['completed_steps'] == meta['stage2_steps'] == 6000 and meta['local_iter'] == 12000
    assert hasattr(checkpoint['state_dict'], '_metadata')
    del checkpoint
    model.CLASSES = dataset.CLASSES
    model.cuda().eval()
    head = model.fusion_bbox_head
    assert head.stage2_load_kind == 'stage2_resume'
    assert head.post_bev_nms_ops == [0]
    assert head.bbox_coder.__class__.__name__ == 'NMSFreeCoder'
    prototypes = F.normalize(head.prototypes.detach(), dim=-1).float().cpu().numpy()
    assert prototypes.shape == (10, 10, 256) and np.isfinite(prototypes).all()
    np.save(str(feature_root / 'prototypes.npy'), prototypes, allow_pickle=False)
    head.reset_memory()
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0, num_gpus=1,
                             dist=False, shuffle=False, nonshuffler_sampler=cfg.data.nonshuffler_sampler)
    hidden_inputs, head_outputs = [], []
    def capture_input(_, inputs):
        hidden_inputs.append(inputs[0].detach())
    def capture_output(_, __, output):
        head_outputs.append(output)
    hooks = [head.cls_branches[-1][-1].register_forward_pre_hook(capture_input),
             head.register_forward_hook(capture_output)]
    wrapped = MMDataParallel(model, device_ids=[0])
    outputs, manifest = [], []
    npy_files = 1
    try:
        for index, batch in enumerate(loader):
            token = batch_token(batch, filename_to_token)
            assert token == tokens[index], ('DataLoader token/order mismatch', index, token, tokens[index])
            if args.temporal_mode == 'reset-each-frame':
                head.reset_memory()
            hidden_inputs.clear()
            head_outputs.clear()
            with torch.no_grad():
                result = wrapped(return_loss=False, rescale=True, **batch)
                assert len(result) == len(head_outputs) == 1 and hidden_inputs
                logits = head_outputs[0]['all_cls_scores'][-1, 0]
                normalized_boxes = head_outputs[0]['all_bbox_preds'][-1, 0]
                hidden = hidden_inputs[-1]
                b, n, channels = hidden.shape
                assert b == 1 and n == len(logits) == len(normalized_boxes)
                projected = head.proto_proj(hidden.reshape(b*n, channels)).reshape(b, n, -1)
                projected = F.normalize(head.feat_norm(projected), dim=-1)[0]
                keep = ~(logits <= -39.5).all(-1)
                raw_logits = head.cls_branches[-1][-1](hidden)[0]
                assert torch.allclose(raw_logits[keep], logits[keep], rtol=1e-3, atol=1e-3)
                query_indices = torch.where(keep)[0]
                boxes = denormalize_bbox(normalized_boxes, head.bbox_coder.pc_range)
                confidence, labels = logits.sigmoid().max(-1)
                coder = head.bbox_coder
                scores, flattened = logits.sigmoid().reshape(-1).topk(min(coder.max_num, logits.numel()))
                decoded_query = torch.div(flattened, 10, rounding_mode='floor')
                decoded_labels = flattened % 10
                decoded_boxes = boxes[decoded_query]
                limits = torch.as_tensor(coder.post_center_range, device=logits.device)
                mask = (decoded_boxes[:, :3] >= limits[:3]).all(-1) & (decoded_boxes[:, :3] <= limits[3:]).all(-1)
                if coder.score_threshold:
                    mask &= scores >= coder.score_threshold
                decoded_query, decoded_labels = decoded_query[mask], decoded_labels[mask]
                decoded_boxes, scores = decoded_boxes[mask], scores[mask]
                actual = result[0]['pts_bbox']
                actual_boxes = actual['boxes_3d'].tensor.to(logits.device).clone()
                actual_boxes[:, 2] += actual_boxes[:, 5] * .5
                assert decoded_boxes.shape == actual_boxes.shape
                assert torch.allclose(decoded_boxes, actual_boxes, rtol=1e-4, atol=1e-4)
                assert torch.allclose(scores, actual['scores_3d'].to(logits.device), rtol=1e-5, atol=1e-6)
                assert torch.equal(decoded_labels, actual['labels_3d'].to(logits.device))
                valid_decoded = keep[decoded_query]
                # If a decoder ever fills low-score slots with masked padding, retain
                # the original evaluation output, but do not export padding as a feature.
                result_indices = torch.where(valid_decoded)[0]
                decoded_query = decoded_query[valid_decoded]
                decoded_labels = decoded_labels[valid_decoded]
                decoded_boxes = decoded_boxes[valid_decoded]
                scores = scores[valid_decoded]
                info = dataset.data_infos[index]
                names = np.asarray(info['gt_names'])
                known = np.isin(names, CLASSES)
                gt_boxes = np.asarray(info['gt_boxes'], dtype=np.float32)[known]
                gt_labels = np.asarray([CLASSES.index(str(v)) for v in names[known]], dtype=np.int64)
                gt_values = dict(boxes=gt_boxes, labels=gt_labels,
                                 raw_indices=np.where(known)[0].astype(np.int64),
                                 valid_flag=np.asarray(info.get('valid_flag', np.asarray(info['num_lidar_pts']) > 0), dtype=bool)[known],
                                 token=np.asarray([token]))
                save_arrays(directories['gt'], index, gt_values)
                npy_files += len(gt_values)
                for kind, qid, bb, lab, conf, output_ids in (
                    ('candidates', query_indices, boxes[keep], labels[keep], confidence[keep], query_indices),
                    ('detections', decoded_query, decoded_boxes, decoded_labels, scores, result_indices)):
                    features = projected[qid].float().cpu().numpy()
                    bb = bb.float().cpu().numpy()
                    lab = lab.cpu().numpy().astype(np.int64)
                    conf = conf.float().cpu().numpy()
                    cls_logits = logits[qid].float().cpu().numpy()
                    assert features.shape == (len(bb), 256) and cls_logits.shape == (len(bb), 10)
                    assert all(np.isfinite(v).all() for v in (features, bb, conf, cls_logits))
                    assert np.allclose(np.linalg.norm(features, axis=-1), 1., atol=2e-3)
                    matched, nearest = match(bb, lab, conf, gt_boxes, gt_labels)
                    band_matched, _ = match(bb, lab, conf, gt_boxes, gt_labels, upper=.8)
                    similarity = np.einsum('nd,nkd->nk', features, prototypes[lab]).max(axis=1) if len(bb) else np.empty(0, dtype=np.float32)
                    values = dict(features=features, boxes=bb, labels=lab, scores=cls_logits,
                                  confidence=conf, query_indices=qid.cpu().numpy(),
                                  result_indices=output_ids.cpu().numpy(), token=np.asarray([token]),
                                  matched_gt_indices=matched, matched_gt_indices_band_rematch=band_matched,
                                  nearest_gt_distance=nearest,
                                  clear_negative=(matched < 0) & (nearest > 2.), similarity=similarity)
                    save_arrays(directories[kind], index, values)
                    npy_files += len(values)
                outputs.extend(result)
                manifest.append(dict(index=index, token=token, all_queries=len(logits),
                                     valid_candidates=len(query_indices), final_detections=len(decoded_query),
                                     raw_decoder_detections=len(valid_decoded), masked_decoder_detections=int((~valid_decoded).sum()),
                                     normal_candidates=int((query_indices < head.num_query).sum()),
                                     propagated_candidates=int((query_indices >= head.num_query).sum())))
            if (index + 1) % 100 == 0 or index + 1 == len(dataset):
                progress = dict(frames=index+1, total_frames=len(dataset), npy_files=npy_files,
                                elapsed_seconds=time.time()-started)
                write_json(out / 'progress.json', progress)
                print('STAGE2_EXPORT_PROGRESS', json.dumps(progress), flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    assert len(outputs) == len(manifest) == len(tokens)
    prediction_pkl = out / ('stage2_new6000_local12000_%s_predictions.pkl' % args.split)
    import pickle
    with prediction_pkl.open('wb') as stream:
        pickle.dump(outputs, stream, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(feature_root / 'feature_manifest.json', manifest)
    summary = dict(status='PASS', frames=len(tokens), split=args.split, stage2_steps=6000, local_iter=12000,
                   checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=sha_file(args.checkpoint),
                   annotation=str(data_cfg.ann_file), annotation_sha256=sha_file(data_cfg.ann_file),
                   prediction_pkl=str(prediction_pkl), feature_dir=str(feature_root), npy_files=npy_files,
                   valid_candidates=sum(x['valid_candidates'] for x in manifest),
                   final_detections=sum(x['final_detections'] for x in manifest),
                   masked_decoder_detections=sum(x['masked_decoder_detections'] for x in manifest),
                   seed=args.seed, temporal_mode=args.temporal_mode,
                   batch_identity='actual six camera filenames uniquely resolved to PKL sample token',
                   gt_for_similarity='raw ten-class PKL GT, same as 0929 visualisation; valid_flag exported separately',
                   band_matching='both global match then band filter and band-only rematch are exported separately',
                   coordinate_system='LiDAR, gravity-center boxes; candidate/final feature groups separate',
                   token_sha256=hashlib.sha256('\n'.join(sorted(tokens)).encode()).hexdigest(),
                   runtime_seconds=time.time()-started, gpu_bitwise_reproduction_not_guaranteed=True)
    if not args.skip_metrics:
        assert not args.limit, 'Smoke tests do not calculate subset metrics'
        result_files, temporary = dataset.format_results(outputs, jsonfile_prefix=str(out / 'nuscenes_predictions'))
        assert temporary is None
        result_path = result_files['pts_bbox'] if isinstance(result_files, dict) else result_files
        evaluator = SubsetDetectionEval(NuScenes(version=dataset.version, dataroot=dataset.data_root, verbose=False),
                                       dataset.eval_detection_configs, result_path, out / 'nuscenes_metrics', tokens)
        metrics = evaluator.main(render_curves=False)
        assert math.isfinite(metrics['mean_ap']) and math.isfinite(metrics['nd_score'])
        summary.update(mAP=float(metrics['mean_ap']), NDS=float(metrics['nd_score']),
                       prediction_json=str(result_path))
    summary['runtime_seconds'] = time.time() - started
    write_json(out / 'export_summary.json', summary)
    write_json(feature_root / 'export_summary.json', summary)
    print('STAGE2_EXPORT_PASS', json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--checkpoint')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--output-dir')
    parser.add_argument('--feature-dir')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--skip-metrics', action='store_true')
    parser.add_argument('--seed', type=int, default=920)
    parser.add_argument('--temporal-mode', choices=['scene-stream', 'reset-each-frame'], default='scene-stream')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        assert all((args.config, args.checkpoint, args.output_dir, args.feature_dir))
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        main(args)
