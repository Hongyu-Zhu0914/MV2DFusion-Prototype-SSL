"""Independent CPU-only plots for the aligned Stage2 export; never modifies models.

Candidates are used for prototype distributions, final decoded detections for BEV.
All matching indices are consumed as exported, not silently recomputed.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

CLASSES = ('car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
           'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone')
TITLE = 'Stage2 +6000 steps | cumulative 12000'
BINS = np.linspace(-1, 1, 41)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def json_out(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding='utf-8')


def save(fig, path):
    # Reserve an explicit title band: tight_layout otherwise overlaps the
    # two-line figure title and the first pair of subplot titles.
    fig.tight_layout(rect=(0, 0, 1, .955) if fig._suptitle is not None else (0, 0, 1, 1))
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    with Image.open(path) as image:
        image.verify()


def array(root, group, index, name):
    return np.load(str(root / group / ('sample_%d_%s.npy' % (index, name))),
                   allow_pickle=False)


def checkpoint_values(path):
    import torch
    checkpoint = torch.load(path, map_location='cpu')
    state = checkpoint['state_dict']
    def find(suffix):
        keys = [k for k in state if k.endswith('.' + suffix)]
        assert len(keys) == 1, (suffix, keys)
        return state[keys[0]].detach().float().cpu().numpy().copy()
    prototypes = find('prototypes')
    counts = find('proto_update_count')
    del state, checkpoint
    assert prototypes.shape == (10, 10, 256) and counts.shape == (10, 10)
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=-1, keepdims=True), 1e-8)
    return prototypes, counts


def heatmap(matrix, title, path, labels=None, limits=None, column_labels=None):
    fig, ax = plt.subplots(figsize=(12, 10))
    options = dict(cmap='coolwarm') if limits else dict(cmap='viridis')
    if limits:
        options.update(vmin=limits[0], vmax=limits[1])
    image = ax.imshow(matrix, **options)
    if labels:
        ax.set_xticks(range(len(labels)), column_labels or labels, rotation=60, ha='right')
        ax.set_yticks(range(len(labels)), labels)
    else:
        ax.set_xlabel('Class-major prototype index (10 prototypes per class)')
        ax.set_ylabel('Class-major prototype index')
        for edge in np.arange(9.5, 100, 10):
            ax.axhline(edge, color='black', lw=.4)
            ax.axvline(edge, color='black', lw=.4)
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    save(fig, path)


def geometry(run, first_path, out):
    initial, before_counts = checkpoint_values(first_path)
    final, after_counts = checkpoint_values(run / 'checkpoints/iter_6000.pth')
    increment = after_counts - before_counts
    assert (increment >= 0).all() and (increment.sum(axis=1) > 0).all()
    pca = PCA(n_components=2, random_state=920)
    points = pca.fit_transform(np.concatenate((initial.reshape(100, 256), final.reshape(100, 256))))
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    for ax, coords, title in zip(axes, (points[:100], points[100:]),
                                ('Stage1 6000 (initializer)', TITLE)):
        for c in range(10):
            ax.scatter(*coords[c*10:(c+1)*10].T, color=plt.cm.tab10(c), label=CLASSES[c], s=28)
        ax.set_title(title)
        ax.set_xlabel('PC1 (%.1f%%)' % (100*pca.explained_variance_ratio_[0]))
        ax.set_ylabel('PC2 (%.1f%%)' % (100*pca.explained_variance_ratio_[1]))
    axes[1].legend(bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.suptitle('Shared PCA of prototype coordinates; projection spaces may rotate')
    save(fig, out / 'prototype_pca_stage1_6000_vs_stage2_local12000.png')
    full = final.reshape(100, 256) @ final.reshape(100, 256).T
    heatmap(full, TITLE + '\n100x100 prototype cosine', out / 'prototype_cosine_100x100.png', limits=(-1, 1))
    aggregate = np.empty((10, 10))
    rows = []
    for c in range(10):
        for d in range(10):
            sims = final[c] @ final[d].T
            aggregate[c, d] = (sims[np.triu_indices(10, 1)] if c == d else sims).mean()
        within = np.linalg.norm(final[c][:, None] - final[c][None, :], axis=-1)[np.triu_indices(10, 1)]
        other = final[np.arange(10) != c].reshape(90, 256)
        nearest = np.linalg.norm(final[c][:, None] - other[None], axis=-1).min(axis=1)
        rows.append(dict(class_name=CLASSES[c], stage2_updates=int(increment[c].sum()),
                         within_mean_cosine=float(aggregate[c, c]),
                         within_mean_distance=float(within.mean()),
                         nearest_other_mean_distance=float(nearest.mean())))
    heatmap(aggregate, TITLE + '\nMean class cosine (diagonal excludes self)',
            out / 'prototype_class_cosine_10x10.png', CLASSES, (-1, 1))
    heatmap(increment, TITLE + '\nNew per-prototype update counts, not lifetime counts',
            out / 'prototype_update_increments.png', CLASSES, column_labels=['P%d'%i for i in range(10)])
    with (out / 'prototype_distance_attributes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(10)
    ax.bar(x-.2, [r['within_mean_distance'] for r in rows], .4, label='Within-class pair distance')
    ax.bar(x+.2, [r['nearest_other_mean_distance'] for r in rows], .4, label='Nearest other-class distance')
    ax.set_xticks(x, CLASSES, rotation=45, ha='right'); ax.legend()
    ax.set_title(TITLE + '\nEuclidean distance between L2-normalized prototypes')
    save(fig, out / 'prototype_distance_attributes.png')
    return final


def distributions(root, manifest, out, prototypes):
    modes = ('global_all_scores', 'conf02_08_after_global_match', 'conf02_08_band_rematch')
    hist = np.zeros((3, 10, 2, 40), dtype=np.int64)
    sums = np.zeros((3, 10, 2)); counts = np.zeros((3, 10, 2), dtype=np.int64)
    sampled = [np.empty((0, 256), np.float32) for _ in range(10)]
    rng = np.random.RandomState(920)
    for entry in manifest:
        i = entry['index']
        labels = array(root, 'candidates', i, 'labels')
        conf = array(root, 'candidates', i, 'confidence')
        sim = array(root, 'candidates', i, 'similarity')
        matched = array(root, 'candidates', i, 'matched_gt_indices')
        rematch = array(root, 'candidates', i, 'matched_gt_indices_band_rematch')
        distance = array(root, 'candidates', i, 'nearest_gt_distance')
        features = array(root, 'candidates', i, 'features')
        band = (conf > .2) & (conf < .8)
        masks = ((matched >= 0, (matched < 0) & (distance > 2)),
                 ((matched >= 0) & band, (matched < 0) & (distance > 2) & band),
                 ((rematch >= 0) & band, (rematch < 0) & (distance > 2) & band))
        for c in range(10):
            cls = labels == c
            for m, pair in enumerate(masks):
                for pn, mask in enumerate(pair):
                    values = sim[cls & mask]
                    hist[m, c, pn] += np.histogram(np.clip(values,-1,1), BINS)[0]
                    counts[m, c, pn] += len(values); sums[m, c, pn] += values.sum(dtype=np.float64)
            selected = features[cls & (conf > .2)]
            combined = np.concatenate((sampled[c], selected))
            if len(combined) > 1000:
                combined = combined[rng.choice(len(combined), 1000, replace=False)]
            sampled[c] = combined
    rows = []
    for m, mode in enumerate(modes):
        for per_frame in (False, True):
            fig, axes = plt.subplots(5, 2, figsize=(17, 23))
            width = (BINS[1]-BINS[0])*.43
            centers = (BINS[:-1]+BINS[1:])/2
            for c, ax in enumerate(axes.flat):
                means = [float(sums[m,c,j]/counts[m,c,j]) if counts[m,c,j] else None for j in range(2)]
                for j, (color, label) in enumerate((('#3676b5','Strict positives'),('#ef5350','Clear negatives'))):
                    values = hist[m,c,j] / (len(manifest) if per_frame else 1)
                    ax.bar(centers+(j-.5)*width, values, width=width, color=color, label=label)
                ax.axvline(.94 if c == 7 else .90, color='black', ls='--', label='Legacy reference only')
                ax.set_xlim(-1, 1); ax.grid(axis='y', alpha=.2)
                ax.set_title('%s | positive n=%d mean=%s; negative n=%d mean=%s' %
                             (CLASSES[c], counts[m,c,0], 'NA' if means[0] is None else '%.3f'%means[0],
                              counts[m,c,1], 'NA' if means[1] is None else '%.3f'%means[1]), fontsize=9)
                ax.set_xlabel('Maximum cosine to 10 prototypes of predicted class')
                ax.set_ylabel('Samples / frame / bin' if per_frame else 'Samples / bin')
                ax.legend(fontsize=7)
                if not per_frame:
                    rows.append(dict(mode=mode, class_name=CLASSES[c], positives=int(counts[m,c,0]),
                                     negatives=int(counts[m,c,1]), positive_mean=means[0], negative_mean=means[1],
                                     gap=None if None in means else means[0]-means[1]))
            fig.suptitle(TITLE+' | '+mode+'\nCandidates only; raw PKL GT; negatives nearest ANY GT >2m; ambiguous excluded', fontsize=11)
            fig.subplots_adjust(top=.96)
            save(fig, out / ('similarity_%s_%s.png' % (mode, 'per_frame' if per_frame else 'counts')))
    json_out(out / 'similarity_statistics.json', dict(frames=len(manifest), rows=rows,
             sampling='PCA capped streaming sample (not an unbiased population estimate); histograms use every candidate'))
    nonempty = [(c, a) for c, a in enumerate(sampled) if len(a)]
    features = np.concatenate([a for _, a in nonempty])
    ids = np.concatenate([np.full(len(a), c) for c, a in nonempty])
    if len(features) >= 2:
        pca = PCA(n_components=2, random_state=920)
        coords = pca.fit_transform(features)
        pp = pca.transform(prototypes.reshape(100,256))
        fig, ax = plt.subplots(figsize=(12, 9))
        for c, _ in nonempty:
            ax.scatter(*coords[ids==c].T, s=3, color=plt.cm.tab10(c), alpha=.25, label=CLASSES[c])
            ax.scatter(*pp[c*10:(c+1)*10].T, s=60, marker='X', color=plt.cm.tab10(c), edgecolor='black', lw=.5)
        ax.legend(bbox_to_anchor=(1.02,1), fontsize=8)
        ax.set_title(TITLE+'\nProjected candidate features score>0.2 (sampled), X=current prototypes')
        ax.set_xlabel('PC1 %.1f%%' % (pca.explained_variance_ratio_[0]*100))
        ax.set_ylabel('PC2 %.1f%%' % (pca.explained_variance_ratio_[1]*100))
        save(fig, out / 'feature_and_current_prototype_pca.png')


def corners(box):
    x, y, _, w, length, _, yaw = box[:7]
    rect = np.array([[w/2,length/2],[w/2,-length/2],[-w/2,-length/2],[-w/2,length/2],[w/2,length/2]])
    rot = np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    return rect @ rot.T + [x,y]


def bev(root, manifest, annotation, out):
    with open(annotation,'rb') as stream:
        data = pickle.load(stream)
    infos = data['infos'] if isinstance(data,dict) else data
    by_token = {x['token']:x for x in infos}
    indices = sorted(set([0, min(22,len(manifest)-1), min(100,len(manifest)-1), len(manifest)-1]))
    for i in indices:
        token = manifest[i]['token']; info = by_token[token]
        points_path = Path(info['lidar_path'])
        if not points_path.is_file():
            raise FileNotFoundError(points_path)
        points = np.fromfile(points_path,dtype=np.float32).reshape(-1,5)
        boxes = array(root,'detections',i,'boxes'); labels = array(root,'detections',i,'labels')
        conf = array(root,'detections',i,'confidence'); sim = array(root,'detections',i,'similarity')
        gt = array(root,'gt',i,'boxes')
        fig,ax=plt.subplots(figsize=(14,14)); fig.patch.set_facecolor('black'); ax.set_facecolor('black')
        mask=(np.abs(points[:,0])<60)&(np.abs(points[:,1])<60)
        ax.scatter(points[mask,0],points[mask,1],s=.1,c='gray',alpha=.3,lw=0)
        for box in gt:
            xy=corners(box); ax.plot(*xy.T,c='#24ff38',lw=1.2)
        selected=np.where(conf>.2)[0]
        for j in selected[np.argsort(conf[selected])]:
            xy=corners(boxes[j]); color=(.35+.65*float(conf[j]),0,0)
            ax.plot(*xy.T,c=color,lw=1)
            ax.text(*xy[0],'%s %.2f/%.2f'%(CLASSES[labels[j]],conf[j],sim[j]),fontsize=4.2,c=color,clip_on=True)
        ax.set(xlim=(-60,60),ylim=(-60,60),aspect='equal'); ax.tick_params(colors='white')
        ax.set_title(TITLE+' | sample %d\nGreen=raw GT; red=final detection confidence>0.2, class confidence/similarity'%i,color='white')
        save(fig,out/('bev_sample_%04d.png'%i))


def training_monitor(run, out):
    records=[json.loads(x) for x in (run/'logs/iterations.jsonl').read_text().splitlines() if x]
    steps=np.array([x['step'] for x in records]); assert len(steps)==6000 and steps[-1]==6000
    weights=np.array([x['negative_weights'] for x in records])
    fig,axes=plt.subplots(2,1,figsize=(15,9))
    for c in range(10): axes[0].plot(steps,weights[:,c],label=CLASSES[c],lw=1)
    axes[0].legend(ncol=5,fontsize=8); axes[0].set_ylabel('Per-class negative loss weight')
    for key in ('frame_0_loss_proto','frame_0_loss_neg_proto','frame_0_loss_proto_intra_band','frame_0_loss_proto_inter_upper'):
        values=np.array([r['losses'][key] for r in records])
        smooth=np.convolve(values,np.ones(100)/100,mode='valid')
        axes[1].plot(steps[99:],smooth,label=key, lw=1)
    axes[1].set_yscale('symlog',linthresh=1e-6); axes[1].legend(fontsize=8)
    axes[1].set_xlabel('Stage2 new step (add 6000 for cumulative local_iter)')
    axes[0].set_title(TITLE+' | 100-step moving-average prototype losses')
    save(fig,out/'stage2_weights_and_prototype_losses.png')
    windows=[json.loads(x) for x in (run/'analysis/similarity_windows.jsonl').read_text().splitlines() if x]
    fig,axes=plt.subplots(5,2,figsize=(15,18))
    for c,ax in enumerate(axes.flat):
        for pkey,nkey,label in (('positive','clear_negative','All-score gap'),
                                 ('positive_conf02_08','clear_negative_conf02_08','0.2<score<0.8 gap')):
            pairs=[(r['stage2_step'],r['per_class'][c][pkey]['mean'],r['per_class'][c][nkey]['mean']) for r in windows]
            valid=[(s,p-n) for s,p,n in pairs if p is not None and n is not None]
            if valid: ax.plot(*np.array(valid).T,label=label)
        ax.set_title(CLASSES[c]); ax.legend(fontsize=7); ax.grid(alpha=.2)
    fig.suptitle('Training windows: positive-minus-clear-negative mean similarity; not fixed validation samples')
    save(fig,out/'stage2_training_similarity_gaps.png')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True); parser.add_argument('--features',required=True)
    parser.add_argument('--initial-checkpoint',required=True); parser.add_argument('--out',required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args(); run=Path(args.run); root=Path(args.features); out=Path(args.out)
    if out.exists() and any(out.iterdir()): raise RuntimeError('Refusing nonempty plot output: '+str(out))
    summary=read_json(root/'export_summary.json'); assert summary['status']=='PASS'
    if not args.smoke: assert summary['frames'] in (3014,3005)
    out.mkdir(parents=True,exist_ok=True)
    manifest=read_json(root/'feature_manifest.json'); assert len(manifest)==summary['frames']
    prototypes=geometry(run,args.initial_checkpoint,out)
    assert np.allclose(prototypes,np.load(root/'prototypes.npy'),atol=1e-6)
    distributions(root,manifest,out,prototypes); bev(root,manifest,summary['annotation'],out)
    training_monitor(run,out)
    files=list(out.glob('*.png')); assert len(files)>=(16 if args.smoke else 18)
    json_out(out/'plot_summary.json',dict(status='PASS',frames=len(manifest),png_count=len(files),
        source_summary=summary,files=[p.name for p in files],
        limitations=['Raw ten-class GT differs from official evaluation filtering',
                     'PCA is descriptive, not evidence of detector improvement',
                     'No paired fixed-sample Stage1 feature export was rerun; training gaps are not a controlled stage comparison']))
    print('STAGE2_PLOT_PASS',len(files),flush=True)


if __name__=='__main__': main()
