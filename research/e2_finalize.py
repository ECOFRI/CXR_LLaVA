"""Offline evidence and matched paired-CI additions; never rerun model inference."""
import csv,json,shutil,sys,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from research.common import ROOT,BASELINE,sha,write_csv,write_json
from research.e_audit import statistics
run=Path(sys.argv[1]).resolve();m=json.loads((run/'run_manifest.json').read_text());cohort=json.loads((BASELINE/'cohort.json').read_text());alignment=list(csv.DictReader((run/'alignment_manifest.csv').open()));rows=list(csv.DictReader((run/'e2_clear_scores.csv').open()));labels=['Atelectasis','Cardiomegaly','Consolidation','Edema','Pleural Effusion'];lookup={(r['image_id'],r['finding']):r for r in rows};used=set();selected=[]
for l in labels:
 candidates=[(i,a) for i,a in enumerate(alignment) if a['image_id'] not in used and lookup[a['image_id'],l]['cxr_prediction']!='' and lookup[a['image_id'],l]['cxr_prediction']!=lookup[a['image_id'],l]['clear_prediction']]
 for i,a in candidates[:2]:selected.append((i,l));used.add(a['image_id'])
canvas=Image.new('RGB',(1500,550*((len(selected)+2)//3)),'white');draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15);evidence=[]
for k,(i,l) in enumerate(selected):
 a=alignment[i];r=lookup[a['image_id'],l];im=Image.open(a['source_path']).convert('RGB');w,h=im.size;box=(int(w*.1),int(h*.1),int(w*.9),int(h*.9));im=im.crop(box);im.thumbnail((480,410));x=k%3*500;y=k//3*550;canvas.paste(im,(x+(500-im.width)//2,y));draw.multiline_text((x+8,y+415),f"{a['image_id']} / {l}\nGT={r['gt']} CXR={r['cxr_prediction']} CLEAR={r['clear_prediction']} score={float(r['clear_score']):.4f}\nTechnical disagreement; no clinical adjudication\n10%margin crop; original pixels linked locally",font=font,fill='black');evidence.append(dict(image_id=a['image_id'],finding=l,source_path=a['source_path'],source_sha256=a['source_sha256'],crop_box=json.dumps(box),gt=r['gt'],cxr=r['cxr_prediction'],clear_score=r['clear_score'],note='local only;10%margin crop; no expert review'))
canvas.save(run/'e2_disagreements_contact_sheet.png');write_csv(run/'image_evidence.csv',evidence)
groups={}
for i,a in enumerate(alignment):groups.setdefault(a['patient_id'],[]).append(i)
group=list(groups.values());rng=np.random.default_rng(42);samples=[np.concatenate([group[k] for k in rng.integers(0,len(group),len(group))]) for _ in range(1000)];deltas=[]
for l in labels:
 y=np.array([int(lookup[a['image_id'],l]['gt']) for a in alignment]);cx=np.array([float(lookup[a['image_id'],l]['cxr_prediction']) if lookup[a['image_id'],l]['cxr_prediction'] else np.nan for a in alignment]);cl=np.array([int(lookup[a['image_id'],l]['clear_prediction']) if np.isfinite(cx[i]) else np.nan for i,a in enumerate(alignment)]);delta=statistics(y,cl)['f1']-statistics(y,cx)['f1'];bs=[statistics(y[b],cl[b])['f1']-statistics(y[b],cx[b])['f1'] for b in samples];lo,hi=np.nanquantile(bs,[.025,.975]);deltas.append(dict(finding=l,clear_minus_cxr_F1_same_parsed_mask=delta,ci95_low=lo,ci95_high=hi,covered=int(np.isfinite(cx).sum()),bootstrap='patientcluster1000 seed42'))
write_csv(run/'paired_f1_differences.csv',deltas)
shutil.copyfile(ROOT/'research/e2_execution.log',run/'execution.log');env=json.loads((run/'environment_manifest.json').read_text());env.update(CLEAR_commit=subprocess.check_output(['git','-C',str(ROOT.parent/'CLEAR'),'rev-parse','HEAD'],text=True).strip(),DINO_commit=subprocess.check_output(['git','-C',str(ROOT/'research/upstream/dinov2'),'rev-parse','HEAD'],text=True).strip(),pip_freeze_sha256=sha(run/'pip_freeze.txt'));write_json(run/'environment_manifest.json',env)
report=ROOT/'reports'/f'{m["start_time"][:10]}_E2_direct_contrastive.md';s=report.read_text().replace('Contact sheets mask margins/highlights','Contact sheets crop10% margins and preserve interior pixel intensities');s+='\nOffline evidence refinement:10 unique disagreement images (two per finding), crop boxes recorded in image_evidence.csv. Same-mask paired F1 differences with1000patientcluster CIs in paired_f1_differences.csv; interpret as pipeline contrasts under fixed exploratory threshold, not clinical superiority.\n';report.write_text(s)
write_json(run/'offline_finalization.json',dict(script_sha256=sha(__file__),unique_images=len(used),no_inference=True,no_prompt_or_threshold_changes=True,paired_difference_metrics='paired_f1_differences.csv',evidence_policy='10%border crop; interior intensities unchanged; sourcehash/cropbox linked'))
m['output_sha256']={str(f.relative_to(ROOT)):sha(f) for f in run.rglob('*') if f.is_file() and f.name!='run_manifest.json'};write_json(run/'run_manifest.json',m)
print(json.dumps(deltas,indent=2))
