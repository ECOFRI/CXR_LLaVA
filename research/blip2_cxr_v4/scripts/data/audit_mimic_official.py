"""Read-only complete official MIMIC validate/test linkage audit."""
import argparse
import csv
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

VR=Path(__file__).resolve().parents[2]


def rows(path):
    with gzip.open(path,'rt',newline='') as f:
        yield from csv.DictReader(f)


def aggregate(config, out, resume):
    jpg=Path(config['jpg_root']);alt=Path(config['jpg_alt_files']);reports=Path(config['report_files'])
    metadata={r['dicom_id']:r for r in rows(jpg/'mimic-cxr-2.0.0-metadata.csv.gz')}
    patients=defaultdict(set);studies=defaultdict(set); images=defaultdict(list)
    split_counts=Counter();missing_meta=Counter();views=defaultdict(Counter)
    for r in rows(jpg/'mimic-cxr-2.0.0-split.csv.gz'):
        split=r['split'];split_counts[split]+=1;patients[split].add(r['subject_id']);studies[split].add(r['study_id'])
        if split not in ('validate','test'):continue
        m=metadata.get(r['dicom_id'])
        if not m or m['subject_id']!=r['subject_id'] or m['study_id']!=r['study_id']:
            missing_meta[split]+=1;continue
        views[split][m.get('ViewPosition') or 'BLANK']+=1
        images[(split,r['subject_id'],r['study_id'])].append(r['dicom_id'])
    overlap={a+'_'+b:len(patients[a]&patients[b]) for a,b in [('train','validate'),('train','test'),('validate','test')]}
    keys=sorted(images)
    state_path=out/'mimic_official_state.json'
    if resume and state_path.exists():state=json.loads(state_path.read_text())
    else:state={'cursor':0,'jpg_found':{},'jpg_primary':{},'jpg_alt':{},'report_found':{},'section_counts':{},'study_counts':{}}
    for ix in range(state['cursor'],len(keys)):
        split,subject,study=keys[ix];prefix='p'+subject[:2]
        for dicom in images[keys[ix]]:
            primary=jpg/'files'/prefix/('p'+subject)/('s'+study)/(dicom+'.jpg')
            alternate=alt/prefix/('p'+subject)/('s'+study)/(dicom+'.jpg')
            p=primary.is_file();a=alternate.is_file()
            for field,yes in [('jpg_primary',p),('jpg_alt',a),('jpg_found',p or a)]:
                if yes:state[field][split]=state[field].get(split,0)+1
        report=reports/prefix/('p'+subject)/('s'+study+'.txt')
        if report.is_file():
            state['report_found'][split]=state['report_found'].get(split,0)+1
            content=report.read_text(errors='replace')
            finding=bool(re.search(r'(?im)^\s*FINDINGS\s*:',content))
            impression=bool(re.search(r'(?im)^\s*IMPRESSION\s*:',content))
            for name,yes in [('findings',finding),('impression',impression),('both',finding and impression),('neither',not finding and not impression)]:
                if yes:
                    key=split+'_'+name;state['section_counts'][key]=state['section_counts'].get(key,0)+1
        state['study_counts'][split]=state['study_counts'].get(split,0)+1
        state['cursor']=ix+1
        if state['cursor']%25==0:state_path.write_text(json.dumps(state)+'\n')
    state_path.write_text(json.dumps(state)+'\n')
    result={'status':'PASS','method':'official split; every validate/test image path and unique study report checked; both JPG roots','split_image_rows':dict(split_counts),'split_studies':{k:len(v) for k,v in studies.items()},'split_patients':{k:len(v) for k,v in patients.items()},'patient_overlap':overlap,'missing_metadata':dict(missing_meta),'view_position':{k:dict(v) for k,v in views.items()},'coverage':{k:state[k] for k in ('jpg_found','jpg_primary','jpg_alt','report_found','section_counts','study_counts')},'pretraining_contamination':'CXR-LLaVA trained on MIMIC; CLEAR exposure UNKNOWN; this audit is not an independent performance result'}
    (out/'mimic_official_coverage.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=VR/'configs/eval/mimic_p1.yaml')
    p.add_argument('--output-dir',type=Path,default=VR/'manifests')
    p.add_argument('--dry-run',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args();c=json.loads(a.config.read_text())
    if a.dry_run:print(json.dumps({'status':'DRY_RUN','split':'official'}));return
    a.output_dir.mkdir(parents=True,exist_ok=True)
    if (a.output_dir/'mimic_official_state.json').exists() and not a.resume:raise FileExistsError('Use --resume for existing state')
    result=aggregate(c,a.output_dir,a.resume)
    print(json.dumps({'status':'PASS','split_image_rows':result['split_image_rows'],'coverage':result['coverage'],'patient_overlap':result['patient_overlap']}))

if __name__=='__main__':main()
