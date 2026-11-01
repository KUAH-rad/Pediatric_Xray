"""Audit source annotations and build a conservative exploratory patient split.
9 is never treated as a confirmed negative without a codebook.
"""
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv, json, hashlib, random, time, shutil
import numpy as np
from PIL import Image, ImageOps
import config

ROOT=config.DATA_ROOT
WORK=config.COHORT_DIR; WORK.mkdir(parents=True,exist_ok=True)
CACHE=config.CACHE224
LABELS=['정상','기복증','종괴','장중첩증','공기-액체음영','이물','변비','석회화병변','비장비대','간비대','전내장역위증']

def write_csv(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def collect_item(f):
    parts=f.stem.split('_'); batch=parts[0]; label=parts[1]; candidate=parts[2]
    jf=config.annotation_dir(config.group_number(batch))/(f.stem+'.json')
    row={'image_id':f.stem,'image_path':str(f),'batch':batch,'folder_label':label,'candidate_patient_id':candidate,
         'patient_id':candidate,'patient_id_source':'filename_unverified','gender':'','body_weight':'','acquisition_date':'',
         'annotation_labels':'','json_found':False,'json_error':'','code_positive_count':0,'label_conflict':False}
    for name in LABELS:row['code_'+name]=''
    try:
        d=json.loads(jf.read_text(encoding='utf-8-sig'));row['json_found']=True
        ci=d.get('Clinic_info',{});pid=ci.get('PatientID')
        if pid:row['patient_id']=str(pid);row['patient_id_source']='Clinic_info.PatientID'
        row['gender']=ci.get('Gender') or '';row['body_weight']=ci.get('Body_weight') if ci.get('Body_weight') is not None else ''
        row['acquisition_date']=ci.get('Acquisition_date') or '';row['metadata_type']=ci.get('Type') or ''
        info=d.get('dataInfo',[]);info=info[0] if isinstance(info,list) and info else info if isinstance(info,dict) else {}
        for name in LABELS:row['code_'+name]=str(info.get(name,''))
        row['code_positive_count']=sum(row['code_'+name]=='1' for name in LABELS)
        row['annotation_labels']='|'.join(sorted(set(str(a.get('label','')) for a in d.get('annotation',[]) if isinstance(a,dict))))
        idx=int(label[1:]);row['label_conflict']=row['code_'+LABELS[idx]] not in ('','1')
    except Exception as e:row['json_error']=type(e).__name__
    return row

def cache_one(row):
    target=CACHE/(row['image_id']+'.png')
    if target.exists():
        try:
            with Image.open(target) as im:im.verify()
            return None
        except Exception:pass
    try:
        with Image.open(row['image_path']) as im:
            if im.mode in ('I','I;16','I;16B','I;16L','F'):
                a=np.asarray(im,dtype=np.float32);lo=float(a.min());hi=float(a.max())
                a=np.clip((a-lo)/max(hi-lo,1e-8)*255,0,255).astype(np.uint8);im=Image.fromarray(a)
            im=ImageOps.pad(im.convert('L'),(224,224),method=Image.Resampling.BILINEAR,color=0,centering=(.5,.5))
            im.save(target,compress_level=1)
        return None
    except Exception as e:return {'image_id':row['image_id'],'error':str(e)}

def main():
    if (WORK/'manifest.csv').exists() and (WORK/'split_audit.json').exists():
        print('Existing frozen manifest found; reuse it without repartitioning.',flush=True);return
    CACHE.mkdir(parents=True,exist_ok=True)
    files=[]
    for n in range(1,config.N_GROUPS+1):files.extend(sorted(config.image_dir(n).glob('*.png')))
    print('images',len(files),flush=True)
    raw=WORK/'metadata_manifest.csv'
    if raw.exists():
        with raw.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        print('loaded metadata cache',len(rows),flush=True)
    else:
        rows=[]
        with ThreadPoolExecutor(max_workers=12) as pool:
            for i,row in enumerate(pool.map(collect_item,files)):
                rows.append(row)
                if (i+1)%5000==0:print('metadata',i+1,flush=True)
        # All items share metadata_type even when JSON is missing.
        for row in rows:row.setdefault('metadata_type','')
        write_csv(raw,rows)
    counts={name:dict(Counter(str(r['code_'+name]) for r in rows)) for name in LABELS}
    # Union images sharing identical decoded 224x224 pixels with patient IDs before splitting.
    # Hashing after cache creation below avoids treating byte-only differences as distinct.
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r['patient_id']].append(i)
    audit={'n_images':len(rows),'patient_ids':len(groups),'codes_by_finding':counts,
           'code_positive_count_distribution':dict(Counter(int(r['code_positive_count']) for r in rows)),
           'annotation_label_combinations':dict(Counter(r['annotation_labels'] for r in rows)),
           'json_errors':dict(Counter(r['json_error'] for r in rows if r['json_error'])),
           'patients_with_multiple_folder_labels':sum(len(set(rows[i]['folder_label'] for i in ids))>1 for ids in groups.values()),
           'patient_identity_warning':'Conservative grouping uses exact Clinic_info.PatientID across all folders; global identifier semantics remain unverified.',
           'task':'Exploratory representative-folder-label classification; not definitive multilabel disease detection. Codes 9 remain unknown.'}
    (WORK/'annotation_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print('annotation audit',json.dumps(audit,ensure_ascii=False)[:3500],flush=True)
    errors=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i,error in enumerate(pool.map(cache_one,rows)):
            if error:errors.append(error)
            if (i+1)%2500==0:print('cached',i+1,flush=True)
    (WORK/'image_errors.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2),encoding='utf-8')
    bad={r['image_id'] for r in errors};rows=[r for r in rows if r['image_id'] not in bad]
    parent={r['patient_id']:r['patient_id'] for r in rows}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    hashes={};duplicates=[]
    for i,r in enumerate(rows):
        with Image.open(CACHE/(r['image_id']+'.png')) as im:h=hashlib.sha256(im.tobytes()).hexdigest()
        r['pixel_sha256_224']=h
        if h in hashes:
            prev=hashes[h];a=find(r['patient_id']);b=find(prev['patient_id']);parent[a]=b
            duplicates.append({'image_id':r['image_id'],'duplicate_of':prev['image_id'],'label_conflict':r['folder_label']!=prev['folder_label']})
        else:hashes[h]=r
        if (i+1)%10000==0:print('hashes',i+1,flush=True)
    write_csv(WORK/'exact_pixel_duplicates.csv',duplicates)
    # One copy of identical cached pixels; exclude all cross-label duplicate conflicts.
    labels_by_hash=defaultdict(set)
    for r in rows:labels_by_hash[r['pixel_sha256_224']].add(r['folder_label'])
    conflict_hashes={h for h,ls in labels_by_hash.items() if len(ls)>1}
    seen=set();clean=[]
    for r in rows:
        h=r['pixel_sha256_224']
        if h in seen or h in conflict_hashes:continue
        seen.add(h);r['split_group']=find(r['patient_id']);clean.append(r)
    # Hash-based split independent of labels, batch and acquisition chronology.
    # Reader-set images (one image ID or file name per line); their patient-image groups are reserved from model development.
    reader_stems={Path(s.strip()).stem for s in config.READER_IMAGE_LIST.read_text(encoding='utf-8').splitlines() if s.strip()}
    reserved_groups={find(r['patient_id']) for r in rows if r['image_id'] in reader_stems}
    for r in clean:
        u=int(hashlib.sha256(('radiology-20261022-v1|'+r['split_group']).encode()).hexdigest()[:8],16)/2**32
        r['split']='train' if u<.70 else 'validation' if u<.85 else 'test'
        if r['split_group'] in reserved_groups:r['split']='reader_reserved'
        r['cache_path']=str(CACHE/(r['image_id']+'.png'))
    write_csv(WORK/'manifest.csv',clean)
    split_audit={s:{'images':sum(r['split']==s for r in clean),'groups':len({r['split_group'] for r in clean if r['split']==s}),
                  'classes':dict(Counter(r['folder_label'] for r in clean if r['split']==s))} for s in ['train','validation','test','reader_reserved']}
    split_audit['reader_image_stems']=len(reader_stems)
    split_audit['reader_reserved_groups']=len(reserved_groups)
    (WORK/'reader_case_stems.json').write_text(json.dumps(sorted(reader_stems)),encoding='utf-8')
    split_audit['excluded_cross_label_pixel_hashes']=len(conflict_hashes);split_audit['exact_duplicate_extra_images']=len(duplicates)
    split_audit['manifest_sha256']=hashlib.sha256((WORK/'manifest.csv').read_bytes()).hexdigest()
    split_audit['split_group_overlap']=0
    (WORK/'split_audit.json').write_text(json.dumps(split_audit,indent=2),encoding='utf-8')
    print('READY',json.dumps(split_audit),flush=True)

if __name__=='__main__':
    main()
