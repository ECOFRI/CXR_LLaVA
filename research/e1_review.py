"""Local linked model disagreement evidence; labels are dataset labels, not adjudication."""
import argparse,csv,json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from research.common import RESEARCH,write_csv,sha
TITLE='NIH exploratory; potential CXR encoder pretraining overlap; not independent external test'
def run(args):
    out=Path(args.output);rows=list(csv.DictReader((out/'test_predictions.csv').open()));inputs={r['image_id']:r for r in csv.DictReader((RESEARCH/'inventory/e1_nih/cohort_full.csv').open())}
    grouped={}
    for r in rows:
        if r['representation']=='cls':grouped.setdefault((r['image_id'],r['finding']),{})[r['encoder']]=r
    candidates=[]
    for key,v in grouped.items():
        if v['clear']['prediction']!=v['cxr']['prediction']:
            candidates.append((abs(float(v['clear']['score'])-float(v['cxr']['score'])),key,v))
    candidates.sort(key=lambda x:(-x[0],x[1]));selected=[];seen=set()
    for _,key,v in candidates:
        if key[0] not in seen:selected.append((key,v));seen.add(key[0])
        if len(selected)>=12:break
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    sheet=Image.new('RGB',(1320,120+len(selected[:12])//3*430+(430 if len(selected)%3 else 0)),'white');d=ImageDraw.Draw(sheet)
    d.text((12,10),TITLE,fill='black',font=font);d.text((12,38),'CLS paired disagreement review; dataset GT, no expert clinical adjudication. Scores are probe outputs.',fill='black',font=font)
    evidence=[]
    for i,(key,v) in enumerate(selected):
        source=inputs[key[0]];x=(i%3)*440;y=90+(i//3)*430
        with Image.open(source['image_path']) as im:
            im=im.convert('RGB');im.thumbnail((410,330));sheet.paste(im,(x+(440-im.width)//2,y))
        lines=[f'Case {key[0][:12]} / {source["view"]}',f'{key[1]}: dataset GT={v["clear"]["GT"]}',f'CXR {float(v["cxr"]["score"]):.3f} => {v["cxr"]["prediction"]} (thr {float(v["cxr"]["threshold"]):.2f})',f'CLEAR {float(v["clear"]["score"]):.3f} => {v["clear"]["prediction"]} (thr {float(v["clear"]["threshold"]):.2f})']
        for j,line in enumerate(lines):d.text((x+10,y+335+j*18),line,fill='black',font=small)
        evidence.append(dict(image_id=key[0],finding=key[1],source_path=source['image_path'],source_sha256=sha(source['image_path']),GT=v['clear']['GT'],CXR_score=v['cxr']['score'],CLEAR_score=v['clear']['score'],CXR_threshold=v['cxr']['threshold'],CLEAR_threshold=v['clear']['threshold'],CXR_prediction=v['cxr']['prediction'],CLEAR_prediction=v['clear']['prediction'],CXR_case_type=('TP' if v['cxr']['prediction']=='1' else 'FN') if v['cxr']['GT']=='1' else ('FP' if v['cxr']['prediction']=='1' else 'TN'),CLEAR_case_type=('TP' if v['clear']['prediction']=='1' else 'FN') if v['clear']['GT']=='1' else ('FP' if v['clear']['prediction']=='1' else 'TN'),CXR_native_CLS_dimension=1024,CLEAR_native_CLS_dimension=768,run_manifest=str(out/'run_manifest.json'),test_predictions=str(out/'test_predictions.csv'),clinical_review='NOT_PERFORMED'))
    sheet.save(out/'e1_disagreements_contact_sheet.png');write_csv(out/'image_evidence_manifest.csv',evidence,fields=['image_id','finding','source_path','source_sha256','GT','CXR_score','CLEAR_score','CXR_threshold','CLEAR_threshold','CXR_prediction','CLEAR_prediction','CXR_case_type','CLEAR_case_type','CXR_native_CLS_dimension','CLEAR_native_CLS_dimension','run_manifest','test_predictions','clinical_review'])
    print('review cases',len(selected),'disagreements',len(candidates))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args())
