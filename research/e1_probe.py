"""Matched train-only scaled logistic probes; val-only selection; locked test pass."""
import argparse,csv,json,pickle,warnings
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score,f1_score,precision_score,recall_score
from research.common import ROOT,RESEARCH,sha,write_csv,write_json,now
warnings.filterwarnings('ignore',message='No positive class found')
TITLE='NIH exploratory; potential CXR encoder pretraining overlap; not independent external test'
ORIGINAL=['Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion','Emphysema','Fibrosis','Hernia','Infiltration','Mass','Nodule','Pleural Thickening','Pneumonia','Pneumothorax']
def values(y,s,th):
    pred=s>=th
    tp=int(np.sum((y==1)&pred));fp=int(np.sum((y==0)&pred));fn=int(np.sum((y==1)&~pred))
    return {'AUROC':roc_auc_score(y,s) if len(np.unique(y))==2 else np.nan,'AUPRC':average_precision_score(y,s) if len(np.unique(y))==2 else np.nan,'F1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan,'precision':tp/(tp+fp) if tp+fp else np.nan,'recall':tp/(tp+fn) if tp+fn else np.nan}
def run(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    protocol=RESEARCH/'protocols/E1_nih_exploratory_protocol.yaml'; frozen=sha(protocol)
    rows=list(csv.DictReader((RESEARCH/'inventory/e1_nih/cohort_full.csv').open()));supports=list(csv.DictReader((RESEARCH/'inventory/e1_nih/class_support.csv').open()))
    support={(r['finding'],r['split']):r for r in supports}
    included=[];excluded=[]
    for name in ORIGINAL:
        ok=all(int(support[name,s]['positive'])>= (20 if s=='train' else 5) and int(support[name,s]['negative'])>= (20 if s=='train' else 5) and (s=='train' or int(support[name,s]['positive_patients'])>=3) for s in ('train','val','test'))
        (included if ok else excluded).append(name)
    write_json(out/'frozen_selection.json',dict(timestamp=now(),protocol_sha256=frozen,included=included,excluded=excluded))
    masks={s:np.array([r['split']==s for r in rows]) for s in ('train','val','test')};testrows=[r for r in rows if r['split']=='test']
    feature_paths={'cxr':Path(a.cxr),'clear':Path(a.clear)}
    cache_hashes={}
    cohort_hash=sha(RESEARCH/'inventory/e1_nih/cohort_full.csv')
    for encoder,path in feature_paths.items():
        manifest=json.loads((path/'run_manifest.json').read_text())
        assert manifest['status']=='COMPLETE' and manifest['encoder']==encoder
        assert manifest['protocol_sha256']==frozen and manifest['cohort_sha256']==cohort_hash
        assert manifest['checkpoint_integrity']=='PASS'
        cache_hashes[encoder]=sha(path/'features.npz')
        assert cache_hashes[encoder]==manifest['cache_sha256'], 'Feature cache integrity mismatch'
    features={k:np.load(p/'features.npz') for k,p in feature_paths.items()}
    for f in features.values():assert f['complete'].all() and list(f['image_ids'])==[r['image_id'] for r in rows]
    models={};selection=[];convergence=[]
    for encoder,f in features.items():
        for representation in ('cls','patchmean'):
            X=f[representation];assert np.isfinite(X).all();scaler=StandardScaler().fit(X[masks['train']]);train=scaler.transform(X[masks['train']]);val=scaler.transform(X[masks['val']])
            for name in included:
                y=np.array([int(r[name]) for r in rows]);best=None
                for c in [.001,.01,.1,1.]:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        model=LogisticRegression(C=c,solver='liblinear',penalty='l2',class_weight=None,max_iter=1000,random_state=42).fit(train,y[masks['train']])
                    score=model.predict_proba(val)[:,1];ap=average_precision_score(y[masks['val']],score)
                    convergence.append(dict(encoder=encoder,representation=representation,finding=name,C=c,n_iter=int(model.n_iter_.max()),warnings='; '.join(str(w.message) for w in caught)))
                    if best is None or ap>best[0]+1e-12:best=(ap,c,model,score)
                ap,c,model,score=best
                grid=np.round(np.arange(.01,1,.01),2);rank=sorted([(f1_score(y[masks['val']],score>=t,zero_division=0),-abs(t-.5),t) for t in grid],reverse=True);threshold=float(rank[0][2])
                key=(encoder,representation,name);models[key]=(scaler,model,threshold);selection.append(dict(encoder=encoder,representation=representation,finding=name,C=c,threshold=threshold,val_AUPRC=ap,val_F1=rank[0][0]))
    write_csv(out/'validation_selection.csv',selection)
    write_csv(out/'convergence_log.csv',convergence)
    with (out/'frozen_probes.pkl').open('wb') as fp:pickle.dump(models,fp)
    assert sha(protocol)==frozen
    write_json(out/'test_unlock_manifest.json',dict(timestamp=now(),protocol_sha256=frozen,runner_sha256=sha(__file__),cohort_sha256=cohort_hash,feature_cache_sha256=cache_hashes,selection_sha256=sha(out/'validation_selection.csv'),models_sha256=sha(out/'frozen_probes.pkl'),rule='all choices frozen; test scores computed once now'))
    results=[]; predictions=[];scores={};truth={}
    for (encoder,rep,name),(scaler,model,th) in models.items():
        s=model.predict_proba(scaler.transform(features[encoder][rep][masks['test']]))[:,1];y=np.array([int(r[name]) for r in testrows]);truth[name]=y;scores[encoder,rep,name]=(s,th);v=values(y,s,th)
        results.append(dict(header=TITLE,encoder=encoder,representation=rep,finding=name,positive=int(y.sum()),negative=int(len(y)-y.sum()),test_patients=56,threshold=th,**v))
        for row,score in zip(testrows,s):predictions.append(dict(image_id=row['image_id'],patient_id=row['patient_id'],finding=name,encoder=encoder,representation=rep,GT=int(row[name]),score=float(score),threshold=th,prediction=int(score>=th)))
    write_csv(out/'test_predictions.csv',predictions)
    # Shared cluster resamples preserve pairing and all follow-up exams within sampled patients.
    patient=np.array([r['patient_id'] for r in testrows]);unique=np.unique(patient);groups={p:np.flatnonzero(patient==p) for p in unique};rng=np.random.default_rng(20261013)
    draws=[np.concatenate([groups[p] for p in rng.choice(unique,len(unique),replace=True)]) for _ in range(1000)]
    intervals={};boot={}
    for key,(s,th) in scores.items():
        y=truth[key[2]];bs={m:[] for m in ('AUROC','AUPRC','F1','precision','recall')}
        for idx in draws:
            v=values(y[idx],s[idx],th)
            for m,z in v.items():bs[m].append(z)
        boot[key]=bs
        for r in results:
            if (r['encoder'],r['representation'],r['finding'])==key:
                for m,b in bs.items():
                    arr=np.array(b);arr=arr[np.isfinite(arr)];r[m+'_CI_low'],r[m+'_CI_high']=map(float,np.percentile(arr,[2.5,97.5])) if len(arr) else (float('nan'),float('nan'));r[m+'_bootstrap_valid']=len(arr)
    write_csv(out/'e1_metrics.csv',results)
    differences=[]
    for rep in ('cls','patchmean'):
        for name in included:
            for metric in ('AUROC','AUPRC','F1'):
                delta=np.asarray(boot['clear',rep,name][metric])-np.asarray(boot['cxr',rep,name][metric]);delta=delta[np.isfinite(delta)];lo,hi=np.percentile(delta,[2.5,97.5]) if len(delta) else (np.nan,np.nan);differences.append(dict(header=TITLE,representation=rep,finding=name,metric=metric,delta_CLEAR_minus_CXR=values(truth[name],*scores['clear',rep,name])[metric]-values(truth[name],*scores['cxr',rep,name])[metric],CI_low=float(lo),CI_high=float(hi),bootstrap_valid=len(delta)))
    write_csv(out/'paired_cluster_differences.csv',differences)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(12,9),sharex=True)
    for ax,rep in zip(axes,('cls','patchmean')):
        for encoder,offset,color in [('cxr',-.15,'#4477AA'),('clear',.15,'#CC6677')]:
            rr=[next(r for r in results if r['encoder']==encoder and r['representation']==rep and r['finding']==name) for name in included];yv=np.array([r['AUPRC'] for r in rr]);low=np.array([r['AUPRC_CI_low'] for r in rr]);high=np.array([r['AUPRC_CI_high'] for r in rr]);ax.errorbar(np.arange(len(rr))+offset,yv,yerr=[np.maximum(0,yv-low),np.maximum(0,high-yv)],fmt='o',label=encoder,color=color,capsize=3)
        ax.set_ylabel('Test AUPRC / patient95%CI');ax.set_title(rep);ax.legend();ax.grid(alpha=.2)
    axes[-1].set_xticks(np.arange(len(included)),included,rotation=45,ha='right');fig.suptitle(TITLE,fontsize=10);fig.tight_layout();fig.savefig(out/'e1_per_class_comparison.png',dpi=150);plt.close(fig)
    summary={}
    for encoder in features:
        for rep in ('cls','patchmean'):
            rr=[r for r in results if r['encoder']==encoder and r['representation']==rep];summary[encoder+'_'+rep]={m:float(np.mean([r[m] for r in rr])) for m in ('AUROC','AUPRC','F1')}
    write_json(out/'run_manifest.json',dict(status='COMPLETE',end_time=now(),protocol_sha256=frozen,runner_sha256=sha(__file__),feature_cache_sha256={k:sha(Path(p)/'features.npz') for k,p in [('cxr',a.cxr),('clear',a.clear)]},included_classes=included,excluded_classes=excluded,summary=summary,bootstrap_unit='patient',bootstrap_repeats=1000,clinical_review=False))
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cxr',required=True);p.add_argument('--clear',required=True);p.add_argument('--output',required=True);run(p.parse_args())
