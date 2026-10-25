"""Export decoded valid detections and aligned classifier-input features.

No prototype projection is used. All original predictions still go to evaluation;
NPY rows retain only valid Queries among the actual decoded detections.
"""
import argparse
import functools
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
import evaluate_online_ema_splits_1011 as evaluation


def export_test(model, loader, root):
    root = Path(root)
    directory = root / 'npy'
    directory.mkdir()
    detector = model.module
    head = detector.fusion_bbox_head
    inputs, heads = [], []
    pre = head.cls_branches[-1][-1].register_forward_pre_hook(
        lambda module, args: inputs.append(args[0].detach().clone()))
    post = head.register_forward_hook(lambda module, args, result: heads.append(result))
    predictions, manifest = [], []
    try:
        for frame, batch in enumerate(loader):
            inputs.clear(); heads.clear()
            with torch.no_grad():
                output = model(return_loss=False, rescale=True, **batch)
            assert len(output) == 1 and len(heads) == 1 and inputs
            info = loader.dataset.data_infos[frame]
            token = info['token']
            result = output[0]['pts_bbox']
            cls = heads[0]['all_cls_scores'][-1, 0].detach()
            boxes = heads[0]['all_bbox_preds'][-1, 0].detach()
            feature = inputs[-1][0]
            assert len(cls) == len(boxes) == len(feature)
            assert feature.shape[1] == 256 and cls.shape[1] == 10
            # Mirror the exact coder top-k and range/score masks, preserving indices.
            coder = head.bbox_coder
            assert type(coder).__name__ == 'NMSFreeCoder'
            scores, flat = cls.sigmoid().flatten().topk(min(coder.max_num, cls.numel()))
            labels = flat % coder.num_classes
            query = torch.div(flat, coder.num_classes, rounding_mode='floor')
            decoded = denormalize_bbox(boxes[query], coder.pc_range)
            limits = decoded.new_tensor(coder.post_center_range)
            mask = ((decoded[:, :3] >= limits[:3]) & (decoded[:, :3] <= limits[3:])).all(1)
            if coder.score_threshold:
                mask &= scores >= coder.score_threshold
            scores, labels, query, decoded = scores[mask], labels[mask], query[mask], decoded[mask]
            actual = result['boxes_3d'].tensor.to(decoded.device)
            expected = decoded.clone(); expected[:, 2] -= expected[:, 5] * 0.5
            assert torch.allclose(scores, result['scores_3d'].to(scores.device), atol=1e-6)
            assert torch.equal(labels, result['labels_3d'].to(labels.device))
            assert torch.allclose(expected, actual, atol=1e-4, rtol=1e-4)
            valid_query = ~(cls <= -39.5).all(1)  # head's padding + NMS rejection sentinel
            if valid_query.any():
                with torch.no_grad():
                    reconstructed = head.cls_branches[-1][-1](inputs[-1])[0]
                assert torch.allclose(reconstructed[valid_query], cls[valid_query], atol=1e-3, rtol=1e-3)
            keep = valid_query[query] & torch.isfinite(decoded).all(1) & torch.isfinite(scores)
            keep &= (decoded[:, 3:6] > 0).all(1)
            prediction_indices = torch.arange(len(scores), device=scores.device)[keep]
            prefix = directory / ('sample_%04d' % frame)
            arrays = dict(features=feature[query[keep]], boxes=decoded[keep],
                          scores=scores[keep], labels=labels[keep],
                          query_indices=query[keep], prediction_indices=prediction_indices,
                          class_logits=cls[query[keep]])
            for key, array in arrays.items():
                values = array.detach().cpu().numpy()
                assert np.isfinite(values).all()
                np.save(str(prefix)+'_'+key+'.npy', values)
            names = np.asarray(info['gt_names'])
            gt_valid = np.isin(names, loader.dataset.CLASSES)
            gt_boxes = np.asarray(info['gt_boxes'])[gt_valid]
            gt_labels = np.asarray([loader.dataset.CLASSES.index(str(name)) for name in names[gt_valid]], dtype=np.int64)
            np.save(str(prefix)+'_gt_boxes.npy', gt_boxes)
            np.save(str(prefix)+'_gt_labels.npy', gt_labels)
            np.save(str(prefix)+'_token.npy', np.asarray([token]))
            manifest.append(dict(frame=frame, token=token, query_count=len(cls),
                                 valid_queries=int(valid_query.sum()), decoded_predictions=len(scores),
                                 exported_predictions=int(keep.sum()),
                                 temporal_predictions=int((query[keep] >= head.num_query).sum()),
                                 lidar_path=info['lidar_path']))
            predictions.extend(output)
            if (frame+1) % 100 == 0 or frame+1 == len(loader.dataset):
                print('NPY_EXPORT_PROGRESS {}/{}'.format(frame+1, len(loader.dataset)), flush=True)
    finally:
        pre.remove(); post.remove()
    (root/'npy_manifest.json').write_text(json.dumps(manifest, indent=2))
    summary = dict(frames=len(manifest), exported_predictions=sum(x['exported_predictions'] for x in manifest),
                   decoded_predictions=sum(x['decoded_predictions'] for x in manifest),
                   temporal_mode='reset_each_frame', feature_space='final_classifier_input_256_no_prototype',
                   npy_files=len(list(directory.glob('*.npy'))),
                   token_sha256=hashlib.sha256('\n'.join(sorted(x['token'] for x in manifest)).encode()).hexdigest())
    (root/'npy_export_summary.json').write_text(json.dumps(summary, indent=2))
    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output-dir', required=True)
    args, _ = parser.parse_known_args()
    evaluation.single_gpu_test = functools.partial(export_test, root=args.output_dir)
    evaluation.main()
