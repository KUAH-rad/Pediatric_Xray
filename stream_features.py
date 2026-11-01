"""Pretrained fixed features only: no labels, fitting, or test-driven adaptation."""
from pathlib import Path
import csv,json,time,os,traceback
import numpy as np
import torch,timm
from PIL import Image
from torchvision import transforms as T

import config
W=config.COHORT_DIR;CACHE=config.CACHE224
MODEL='vit_small_patch16_dinov3.lvd1689m'
def status(state,**kw):
    p=W/'stream_features_state.json';t=p.with_suffix('.tmp');t.write_text(json.dumps({'pid':os.getpid(),'status':state,**kw},indent=2),encoding='utf-8');t.replace(p)
def main():
    torch.set_num_threads(4);status('loading')
    with (W/'metadata_manifest.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    m=timm.create_model(MODEL,pretrained=True,num_classes=0,img_size=224).cuda().eval()
    transform=T.Compose([T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))])
    path=CACHE.parent/'fixed_dinov3_all_features.npy';progress=W/'stream_features_progress.json';start=0;dim=m.num_features*3
    if path.exists() and progress.exists():
        start=json.loads(progress.read_text())['completed'];features=np.lib.format.open_memmap(path,mode='r+')
    else:features=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(len(rows),dim))
    images=[];indices=[]
    def flush_batch():
        if not images:return
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            x=torch.stack(images).cuda();tokens=m.forward_features(x);g=m.forward_head(tokens,pre_logits=True);p=tokens[:,m.num_prefix_tokens:];z=torch.cat([g,p.mean(1),p.amax(1)],1)
        features[indices]=z.float().cpu().numpy();images.clear();indices.clear()
    for i in range(start,len(rows)):
        f=CACHE/(rows[i]['image_id']+'.png');t0=time.time()
        while True:
            try:
                with Image.open(f) as im:x=transform(im.convert('RGB'))
                break
            except (FileNotFoundError,OSError):
                if (W/'image_errors.json').exists():
                    bad={r['image_id'] for r in json.loads((W/'image_errors.json').read_text(encoding='utf-8'))}
                    if rows[i]['image_id'] in bad:x=None;features[i]=np.nan;break
                status('waiting_for_cache',completed=i,total=len(rows));time.sleep(1)
        if x is not None:images.append(x);indices.append(i)
        if len(images)==16 or i==len(rows)-1:flush_batch()
        if (i+1)%480==0 or i==len(rows)-1:
            flush_batch();features.flush();progress.write_text(json.dumps({'completed':i+1,'total':len(rows),'model':MODEL}),encoding='utf-8');status('running',completed=i+1,total=len(rows));print('fixed features',i+1,'/',len(rows),flush=True)
    features.flush();(W/'stream_features_complete.json').write_text(json.dumps({'model':MODEL,'path':str(path),'n':len(rows),'dim':dim,'global_dim':m.num_features,'metadata_manifest_sha256':__import__('hashlib').sha256((W/'metadata_manifest.csv').read_bytes()).hexdigest()}),encoding='utf-8');status('complete',completed=len(rows),total=len(rows))
if __name__=='__main__':
    try:main()
    except Exception:status('failed',error=traceback.format_exc());traceback.print_exc();raise
