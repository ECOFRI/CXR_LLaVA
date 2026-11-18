"""Bounded NIH provenance audit and immutable local cohort; no encoder execution."""
import csv, hashlib, itertools, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from research.common import ROOT, RESEARCH, sha, pseudonym, write_csv, write_json, now

def run():
    data=ROOT.parent/'datasets/CXR8'; out=RESEARCH/'inventory/e1_nih'; out.mkdir(exist_ok=True)
    local={p.name:p for p in (data/'images').glob('*.png')}
    meta={r['Image Index']:r for r in csv.DictReader((data/'Data_Entry_2017_v2020.csv').open())}
    records=[]; supports=[]; sources={}; patients={}; ids={}
    for split in ['train','val','test']:
        p=data/'PruneCXR'/f'miccai2023_nih-cxr-lt_labels_{split}.csv'; sources[str(p)]=sha(p)
        rows=[r for r in csv.DictReader(p.open()) if r['id'] in local]
        labels=[k for k in rows[0] if k not in ('id','subj_id')]
        assert len(rows)=={'train':3783,'val':344,'test':872}[split]
        assert all(r[k] in ('0','1') for r in rows for k in labels)
        assert all(int(r['subj_id'])==int(meta[r['id']]['Patient ID'])==int(r['id'].split('_')[0]) for r in rows)
        patients[split]={r['subj_id'] for r in rows}; ids[split]={r['id'] for r in rows}
        for label in labels:
            pos=sum(int(r[label]) for r in rows)
            supports.append(dict(split=split,finding=label,positive=pos,negative=len(rows)-pos,patients=len(patients[split]),positive_patients=len({r['subj_id'] for r in rows if r[label]=='1'})))
        for r in sorted(rows,key=lambda x:x['id']):
            records.append(dict(image_id=pseudonym('nih:'+r['id']),patient_id=pseudonym('nih-patient:'+r['subj_id']),source_id=r['id'],image_path=str(local[r['id']]),split=split,view=meta[r['id']]['View Position'],**{k:int(r[k]) for k in labels}))
    overlap=[]
    for a,b in itertools.combinations(patients,2):
        overlap.append(dict(split_a=a,split_b=b,image_id_overlap=len(ids[a]&ids[b]),patient_overlap=len(patients[a]&patients[b])))
    assert all(not r['image_id_overlap'] and not r['patient_overlap'] for r in overlap)
    # Bounded deterministic 100 images/split; exact decoded-pixel plus low-resolution near-dup screen.
    sample=[]; exact=defaultdict(list); thumbs=[]
    for split in patients:
        chosen=sorted([r for r in records if r['split']==split],key=lambda r:hashlib.sha256(('42:'+r['image_id']).encode()).hexdigest())[:100]
        for r in chosen:
            p=Path(r['image_path']); binary=sha(p)
            with Image.open(p) as im:
                arr=np.asarray(im.convert('L')); px=hashlib.sha256(str(arr.shape).encode()+arr.tobytes()).hexdigest()
                thumb=np.asarray(im.convert('L').resize((32,32)),dtype=np.float32); thumb=(thumb-thumb.mean())/(thumb.std()+1e-8)
            sample.append(dict(image_id=r['image_id'],split=split,source_sha256=binary,decoded_pixel_sha256=px))
            exact[px].append((split,r['image_id'])); thumbs.append((split,r['image_id'],thumb.ravel()))
    near=[]
    for i,(s,id1,a) in enumerate(thumbs):
        for t,id2,b in thumbs[i+1:]:
            if s!=t and float(np.mean((a-b)**2))<0.001:
                near.append(dict(split_a=s,image_a=id1,split_b=t,image_b=id2,normalized_thumbnail_mse=float(np.mean((a-b)**2))))
    dup=[v for v in exact.values() if len({x[0] for x in v})>1]
    write_csv(out/'cohort_full.csv',records);write_csv(out/'class_support.csv',supports);write_csv(out/'split_overlap.csv',overlap);write_csv(out/'bounded_pixel_sample.csv',sample)
    write_json(out/'audit_manifest.json',dict(timestamp=now(),source_sha256=sources,metadata_sha256=sha(data/'Data_Entry_2017_v2020.csv'),runner_sha256=sha(__file__),n=len(records),split_counts=dict(Counter(r['split'] for r in records)),bounded_pixel_images=len(sample),cross_split_exact_pixel_groups=dup,near_duplicate_candidates=near,label_values=[0,1],status='E1_NIH_EXPLORATORY_GO' if not dup else 'REQUIRES_DUPLICATE_REVIEW',pretraining_membership='UNVERIFIABLE_FROM_AVAILABLE_MANIFESTS',provenance=['Local PruneCXR README.txt points arXiv:2308.09180','https://github.com/VITA-Group/PruneCXR Dataset Access links official NIH label folder','https://arxiv.org/html/2308.09180v1 section2.1: original14 +5 report-parsed rare findings +No Finding; patient split'],limitations=['No Finding is bookkeeping not pathology; excluded from primary probe','Bounded 300-image check does not establish all4999 pixel independence','NLP-derived finding labels are imperfect, not expert adjudication','NIH potentially exposed in CXR encoder pretraining; not independent external test']))
    print(json.dumps({'cohort':len(records),'patients':{k:len(v) for k,v in patients.items()},'pixel_sample':len(sample),'exact_duplicate_groups':len(dup),'near_candidates':len(near)}))
if __name__=='__main__':run()
