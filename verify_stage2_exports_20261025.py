"""Independent read-only per-array and per-frame Stage2 export audit."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
from export_stage2_aligned_20261025 import CLASSES, match

FIELDS=('features','boxes','labels','scores','confidence','query_indices','result_indices',
        'token','matched_gt_indices','matched_gt_indices_band_rematch',
        'nearest_gt_distance','clear_negative','similarity')


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def arrays(root,group,i,fields):
    return {k:np.load(str(root/group/('sample_%d_%s.npy'%(i,k))),allow_pickle=False) for k in fields}


def audit(root,smoke=False):
    summary=read(root/'export_summary.json'); assert summary['status']=='PASS'
    count=summary['frames']; assert smoke or count in (3014,3005)
    manifest=read(root/'feature_manifest.json'); assert len(manifest)==count
    tokens=[m['token'] for m in manifest]; assert len(set(tokens))==count
    with open(summary['annotation'],'rb') as stream: raw=pickle.load(stream)
    infos=raw['infos'] if isinstance(raw,dict) else raw
    assert tokens==[x['token'] for x in infos][:count]
    assert summary['token_sha256']==hashlib.sha256('\n'.join(sorted(tokens)).encode()).hexdigest()
    proto=np.load(root/'prototypes.npy',allow_pickle=False)
    assert proto.shape==(10,10,256) and np.isfinite(proto).all()
    assert np.allclose(np.linalg.norm(proto,axis=-1),1,atol=1e-5)
    total_c=total_d=0
    for i,m in enumerate(manifest):
        assert m['index']==i
        gt=arrays(root,'gt',i,('boxes','labels','raw_indices','valid_flag','token'))
        ng=len(gt['boxes']); assert gt['boxes'].shape[1]>=7
        assert gt['labels'].shape==gt['raw_indices'].shape==gt['valid_flag'].shape==(ng,)
        assert gt['token'].tolist()==[tokens[i]]
        assert np.isfinite(gt['boxes']).all() and ((gt['labels']>=0)&(gt['labels']<10)).all()
        groups={}
        for group in ('candidates','detections'):
            a=arrays(root,group,i,FIELDS); groups[group]=a; n=len(a['boxes'])
            assert a['token'].tolist()==[tokens[i]]
            assert a['features'].shape==(n,256) and a['scores'].shape==(n,10)
            assert a['boxes'].shape==(n,9)
            for key in FIELDS:
                if key=='token': continue
                assert len(a[key])==n,(i,group,key)
                if key!='nearest_gt_distance': assert np.isfinite(a[key]).all(),(i,group,key)
            assert np.allclose(np.linalg.norm(a['features'],axis=1),1,atol=2e-5)
            assert ((a['labels']>=0)&(a['labels']<10)).all()
            prob=1/(1+np.exp(-np.clip(a['scores'][np.arange(n),a['labels']],-50,50)))
            assert np.allclose(prob,a['confidence'],atol=1e-6)
            assert ((a['query_indices']>=0)&(a['query_indices']<m['all_queries'])).all()
            assert ((a['matched_gt_indices']>=-1)&(a['matched_gt_indices']<ng)).all()
            full,nearest=match(a['boxes'],a['labels'],a['confidence'],gt['boxes'],gt['labels'])
            band,_=match(a['boxes'],a['labels'],a['confidence'],gt['boxes'],gt['labels'],upper=.8)
            assert np.array_equal(full,a['matched_gt_indices'])
            assert np.array_equal(band,a['matched_gt_indices_band_rematch'])
            assert np.allclose(nearest,a['nearest_gt_distance'],atol=1e-6)
            assert np.array_equal((full<0)&(nearest>2),a['clear_negative'])
            expected=np.empty(n,np.float32)
            for c in range(10):
                mask=a['labels']==c
                if mask.any(): expected[mask]=(a['features'][mask]@proto[c].T).max(axis=1)
            assert np.allclose(expected,a['similarity'],atol=2e-5)
            selected=full[full>=0]; assert len(selected)==len(set(selected.tolist()))
        c,d=groups['candidates'],groups['detections']
        assert len(c['boxes'])==m['valid_candidates'] and len(d['boxes'])==m['final_detections']
        assert len(set(c['query_indices'].tolist()))==len(c['boxes'])
        assert np.array_equal(c['labels'],c['scores'].argmax(axis=1))
        lookup={int(q):j for j,q in enumerate(c['query_indices'])}
        indices=np.array([lookup[int(q)] for q in d['query_indices']],dtype=np.int64)
        assert np.allclose(c['features'][indices],d['features'],atol=1e-6)
        assert np.allclose(c['boxes'][indices],d['boxes'],atol=1e-5)
        assert np.allclose(c['scores'][indices],d['scores'],atol=1e-6)
        assert len(set(d['result_indices'].tolist()))==len(d['boxes'])
        total_c+=len(c['boxes']); total_d+=len(d['boxes'])
        if (i+1)%100==0: print('AUDIT_PROGRESS',i+1,count,flush=True)
    file_count=sum(1 for _ in root.rglob('*.npy'))
    assert file_count==count*31+1==summary['npy_files']
    assert total_c==summary['valid_candidates'] and total_d==summary['final_detections']
    if not smoke: assert all(np.isfinite(summary[k]) for k in ('mAP','NDS'))
    return dict(status='PASS',frames=count,npy_files=file_count,valid_candidates=total_c,
                final_detections=total_d,checkpoint_sha256=summary['checkpoint_sha256'],
                scope='Every NPY read; token, lengths, finite/unit features, class logits/confidence, '
                      'query mapping, current-space similarity, greedy matching, nearest GT, exact counts')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--features',required=True)
    parser.add_argument('--out',required=True); parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args(); destination=Path(args.out)
    if destination.exists(): raise RuntimeError('Refusing existing audit output')
    result=audit(Path(args.features),args.smoke)
    destination.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print('EXPORT_AUDIT_PASS',json.dumps(result),flush=True)
