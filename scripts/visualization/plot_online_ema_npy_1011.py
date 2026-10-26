"""Scientific diagnostics for valid final detections; no learned prototypes."""

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
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from sklearn.decomposition import PCA

CLASSES = ['car','truck','construction_vehicle','bus','trailer','barrier','motorcycle','bicycle','pedestrian','traffic_cone']
DISTANCE = np.asarray([1.,1.2,1.5,1.5,1.5,.5,.5,.4,.35,.25])
DIRECTIONAL = {0,1,2,3,4,6,7}

def match(boxes, scores, labels, gt, gl, threshold):
    n = len(boxes)
    selected = scores > threshold
    distances = np.linalg.norm(boxes[:,None,:2]-gt[None,:,:2],axis=2)
    eligible = (labels[:,None] == gl[None,:]) & (distances < DISTANCE[labels,None])
    angles = np.abs(np.arctan2(np.sin(boxes[:,None,6]-gt[None,:,6]),np.cos(boxes[:,None,6]-gt[None,:,6])))
    eligible &= (~np.isin(labels,list(DIRECTIONAL)))[:,None] | (angles <= np.pi/4)
    result = np.full(n,-1,dtype=np.int64)
    used = set()
    for i in np.argsort(-scores,kind='stable'):
        if not selected[i]:
            continue
        ids = [j for j in np.flatnonzero(eligible[i]) if j not in used]
        if ids:
            j = min(ids,key=lambda j:distances[i,j]); result[i]=j; used.add(j)
    support = eligible.any(1)
    clear = distances.min(1)>2 if len(gt) else np.ones(n,dtype=bool)
    return selected, result, support, clear

def rectangle(box):
    x,y,_,width,length,_,yaw = box[:7]
    corners=np.asarray([[-length/2,-width/2],[length/2,-width/2],[length/2,width/2],[-length/2,width/2]])
    rotation=np.asarray([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    return corners@rotation.T+np.asarray([x,y])

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--export-dir',required=True); parser.add_argument('--out-dir',required=True)
    args=parser.parse_args(); source=Path(args.export_dir); out=Path(args.out_dir)
    if out.exists() and any(out.iterdir()): raise RuntimeError('Refusing overwrite: '+str(out))
    out.mkdir(parents=True); (out/'matches').mkdir(); (out/'bev').mkdir()
    manifest=json.loads((source/'npy_manifest.json').read_text())
    summary=json.loads((source/'npy_export_summary.json').read_text())
    assert len(manifest)==summary['frames']==3014
    thresholds=np.arange(0,1.001,.05)
    totals=np.zeros((len(thresholds),10,5),dtype=np.int64) # pred, TP, GT, supported, absent
    hist=np.zeros((10,2,20),dtype=np.int64)
    counts=np.zeros((10,4),dtype=np.int64) # conf>.2 pred, strict positives, clear neg, ambiguous
    samples=[[] for _ in CLASSES]; samples_count=np.zeros(10,dtype=int)
    per_frame=[]; rng=np.random.default_rng(1011)
    example_frames=set(np.linspace(0,len(manifest)-1,12,dtype=int).tolist()+[22])
    for row in manifest:
        i=row['frame']; prefix=source/'npy'/('sample_%04d'%i)
        def load(key): return np.load(str(prefix)+'_'+key+'.npy',allow_pickle=False)
        boxes,scores,labels,features=load('boxes'),load('scores'),load('labels'),load('features')
        gt,gl=load('gt_boxes'),load('gt_labels')
        assert len(boxes)==len(scores)==len(labels)==len(features)==row['exported_predictions']
        assert load('token')[0]==row['token']
        selected,matched,support,clear=match(boxes,scores,labels,gt,gl,.2)
        np.save(out/'matches'/('sample_%04d_gt_indices.npy'%i),matched)
        for t,threshold in enumerate(thresholds):
            sel,m,s,c=match(boxes,scores,labels,gt,gl,float(threshold))
            for cls in range(10):
                k=sel & (labels==cls)
                totals[t,cls]+= [k.sum(),(k & (m>=0)).sum(),(gl==cls).sum(),(k&s).sum(),(k&~s).sum()]
        pos=selected & (matched>=0); neg=selected & clear; ambiguous=selected & ~pos & ~neg
        per_frame.append(dict(frame=i,token=row['token'],predictions=int(selected.sum()),
                              positive=int(pos.sum()),clear_negative=int(neg.sum()),ambiguous=int(ambiguous.sum())))
        for cls in range(10):
            k=labels==cls; counts[cls]+=[(selected&k).sum(),(pos&k).sum(),(neg&k).sum(),(ambiguous&k).sum()]
            for group,mask in enumerate([pos,neg]):
                chosen=scores[mask&k&(scores>.2)&(scores<.8)]
                hist[cls,group]+=np.histogram(chosen,bins=np.linspace(.2,.8,21))[0]
            ids=np.flatnonzero(selected&k)
            room=5000-samples_count[cls]
            if room>0 and len(ids):
                ids=rng.choice(ids,min(room,len(ids),50),replace=False)
                samples[cls].append(features[ids]); samples_count[cls]+=len(ids)
        if i in example_frames:
            fig,ax=plt.subplots(figsize=(12,12),facecolor='black'); ax.set_facecolor('black')
            path=Path(row['lidar_path'])
            if not path.exists(): path=Path(str(REPO_ROOT))/path
            if path.exists():
                points=np.fromfile(path,dtype=np.float32).reshape(-1,5)
                points=points[(np.abs(points[:,:2])<55).all(1)][::3]
                ax.scatter(points[:,0],points[:,1],s=.08,c='gray',alpha=.35)
            for b in gt: ax.add_patch(Polygon(rectangle(b),fill=False,edgecolor='lime',lw=.8,zorder=2))
            for b,s,l in zip(boxes,scores,labels):
                if s<=.2: continue
                color=plt.cm.Reds(.25+.7*float(s))
                ax.add_patch(Polygon(rectangle(b),fill=False,edgecolor=color,lw=.8,zorder=3))
                ax.text(b[0],b[1],'%s %.2f'%(CLASSES[l],s),color=color,fontsize=3,zorder=4)
            ax.set(xlim=(-55,55),ylim=(-55,55),aspect='equal'); ax.axis('off')
            ax.set_title('sample %d | green GT first; red predictions >0.2 | no prototype similarity'%i,color='white',fontsize=9)
            fig.savefig(out/'bev'/('sample_%04d.png'%i),dpi=160,facecolor='black'); plt.close(fig)
        if (i+1)%200==0: print('PLOT_PROGRESS',i+1,len(manifest),flush=True)
    np.save(out/'threshold_counts.npy',totals); np.save(out/'confidence_thresholds.npy',thresholds)
    (out/'per_frame_counts.json').write_text(json.dumps(per_frame,indent=2))
    precision=np.divide(totals[:,:,1],totals[:,:,0],out=np.full(totals[:,:,0].shape,np.nan),where=totals[:,:,0]>0)
    recall=np.divide(totals[:,:,1],totals[:,:,2],out=np.full(totals[:,:,0].shape,np.nan),where=totals[:,:,2]>0)
    agg=totals.sum(1); p=np.divide(agg[:,1],agg[:,0],out=np.full(len(thresholds),np.nan),where=agg[:,0]>0); r=agg[:,1]/agg[:,2]
    for name,values in [('precision',p),('recall',r),('f1',2*p*r/(p+r))]:
        fig,ax=plt.subplots(); ax.plot(thresholds,values,'o-'); ax.set(xlabel='Confidence threshold (strict >)',ylabel=name,ylim=(0,1)); ax.grid(alpha=.3)
        ax.set_title('Class-distance + directional 45 deg; one-to-one matching')
        fig.savefig(out/(name+'_curve.png'),dpi=180,bbox_inches='tight'); plt.close(fig)
    fig,axes=plt.subplots(5,2,figsize=(14,18))
    for c,ax in enumerate(axes.flat):
        ax.plot(thresholds,precision[:,c],label='Precision'); ax.plot(thresholds,recall[:,c],label='Recall'); ax.set(title=CLASSES[c],ylim=(0,1)); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(out/'class_precision_recall.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(5,2,figsize=(14,18)); centers=(np.linspace(.2,.8,21)[:-1]+np.linspace(.2,.8,21)[1:])/2
    for c,ax in enumerate(axes.flat):
        ax.bar(centers-.006,hist[c,0]/len(manifest),width=.012,color='steelblue',label='Strict positives')
        ax.bar(centers+.006,hist[c,1]/len(manifest),width=.012,color='salmon',label='Clear negatives: nearest any GT >2m')
        ax.set(title=CLASSES[c],xlabel='Confidence: 0.2 < score < 0.8',ylabel='Samples per frame / bin'); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out/'positive_clear_negative_confidence_hist.png',dpi=160); plt.close(fig)
    matrices=[np.concatenate(x) if x else np.empty((0,256)) for x in samples]
    data=np.concatenate(matrices); lab=np.concatenate([np.full(len(x),c) for c,x in enumerate(matrices)])
    norm=data/(np.linalg.norm(data,axis=1,keepdims=True)+1e-8)
    if len(norm)>=2:
        pca=PCA(n_components=2,random_state=1011); projected=pca.fit_transform(norm)
        fig,ax=plt.subplots(figsize=(10,8))
        for c in range(10):
            k=lab==c; ax.scatter(projected[k,0],projected[k,1],s=2,alpha=.3,label=CLASSES[c])
        ax.legend(markerscale=4,fontsize=7); ax.set_title('L2-normalized detector features PCA (not prototype space)')
        fig.savefig(out/'detector_feature_pca.png',dpi=180,bbox_inches='tight'); plt.close(fig)
        np.save(out/'pca_features.npy',projected); np.save(out/'pca_labels.npy',lab)
        means=np.asarray([norm[lab==c].mean(0) if (lab==c).any() else np.zeros(256) for c in range(10)])
        means/=np.linalg.norm(means,axis=1,keepdims=True)+1e-8
        similarity=means@means.T; missing=~np.asarray([(lab==c).any() for c in range(10)])
        similarity[missing,:]=np.nan; similarity[:,missing]=np.nan
        fig,ax=plt.subplots(figsize=(10,9)); image=ax.imshow(similarity,vmin=-1,vmax=1,cmap='coolwarm'); fig.colorbar(image,ax=ax)
        ax.set_xticks(range(10)); ax.set_xticklabels(CLASSES,rotation=60,ha='right'); ax.set_yticks(range(10)); ax.set_yticklabels(CLASSES)
        ax.set_title('Sample feature-centroid cosine similarity — NOT learned prototypes')
        fig.savefig(out/'feature_centroid_heatmap.png',dpi=180,bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,5)); ax.bar(np.arange(10)-.2,counts[:,1]/len(manifest),.4,label='Strict positives/frame'); ax.bar(np.arange(10)+.2,counts[:,2]/len(manifest),.4,label='Clear negatives/frame'); ax.set_xticks(range(10)); ax.set_xticklabels(CLASSES,rotation=45,ha='right'); ax.legend(); fig.savefig(out/'class_sample_counts_per_frame.png',dpi=180,bbox_inches='tight'); plt.close(fig)
    report=dict(frames=len(manifest),png_files=len(list(out.rglob('*.png'))),distance_thresholds=dict(zip(CLASSES,DISTANCE.tolist())),directional_classes=sorted(DIRECTIONAL),
                counts=counts.tolist(),columns=['predictions_conf_gt_0.2','strict_positive','clear_negative','ambiguous'],
                feature_space=summary['feature_space'],prototype_similarity_available=False,
                limitation='No learned prototypes in EMA model. Centroid heatmap is a descriptive sampled-feature statistic, not prototype analysis. Diagnostic GT uses exported PKL boxes; metrics here are not official nuScenes AP.',
                matching='Confidence-descending same-class one-to-one; class distance strict <; directional angle <=45deg; nearest any GT >2m clear negatives; duplicates/other unmatched are ambiguous.',
                aggregate_curves=[dict(threshold=float(t),precision=float(pp),recall=float(rr)) for t,pp,rr in zip(thresholds,p,r)])
    (out/'visualisation_summary.json').write_text(json.dumps(report,indent=2)); print('VISUALISATION_PASS',report['png_files'],flush=True)

if __name__=='__main__': main()
