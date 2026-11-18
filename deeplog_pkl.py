"""PKL benchmark adapter for LogADEmpirical master DeepLog (MIT upstream).
Place in repository root. The upstream model is imported, not rewritten.
Only load trusted PKLs/checkpoints. Row order must be chronological per block.
"""
import argparse, csv, json, random, math
from pathlib import Path
from bisect import bisect_right
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def label(x, raw):
    if pd.isna(x): raise ValueError('Missing label')
    s=str(x).strip().lower()
    if s in ('-', 'normal', '0', '0.0', 'false', 'benign'): return 0
    if s in ('anomaly','anomalous','abnormal','1','1.0','true'): return 1
    if raw and s: return 1
    raise ValueError(f'Unknown binary label: {x!r}')


def read(path,a):
    df=pd.read_pickle(path)
    cols=[a.id_column,a.event_column,a.label_column]
    if a.order_column: cols.append(a.order_column)
    if any(c not in df for c in cols): raise ValueError(f'{path}: missing columns {set(cols)-set(df.columns)}')
    if df[cols].isna().any().any(): raise ValueError(f'{path}: missing required values')
    result=[]
    for key,g in df.groupby(a.id_column,sort=False):
        if a.order_column: g=g.sort_values(a.order_column,kind='stable')
        events=[]; ys=[]
        for e,y in zip(g[a.event_column],g[a.label_column]):
            es=list(e) if isinstance(e,(list,tuple,np.ndarray)) else [e]
            if not es or any(pd.isna(v) for v in es): raise ValueError(f'Empty/missing event for {key}')
            events.extend(str(v) for v in es)
            if isinstance(y,(list,tuple,np.ndarray)):
                if len(y)!=len(es): raise ValueError('Event and label length mismatch')
                ys.extend(label(v,a.label_column=='Original_Label') for v in y)
            else: ys.append(label(y,a.label_column=='Original_Label'))
        result.append((str(key),events,int(any(ys))))
    if not result: raise ValueError(f'Empty split: {path}')
    return result


def disjoint(splits):
    seen=set()
    for name,s in splits.items():
        ids={k for k,_,_ in s}
        if len(ids)!=len(s) or ids & seen: raise ValueError(f'Duplicate or overlapping block IDs in {name}')
        seen.update(ids)


class Windows(Dataset):
    def __init__(self,seqs,h):
        self.seqs=[s for s in seqs if len(s)>h]; self.h=h
        self.ends=np.cumsum([len(s)-h for s in self.seqs]).tolist()
    def __len__(self): return self.ends[-1] if self.ends else 0
    def __getitem__(self,i):
        k=bisect_right(self.ends,i); j=i-(self.ends[k-1] if k else 0); s=self.seqs[k]
        return torch.tensor(s[j:j+self.h]),torch.tensor(s[j+self.h])


def encode(s,v): return [v.get(e,len(v)+1) for e in s]


@torch.inference_mode()
def scores(model,seqs,v,a,device):
    model.eval(); out=[[] for _ in seqs]; xs=[]; targets=[]; loc=[]
    def flush():
        if not xs: return
        x=torch.tensor(xs,device=device); y=torch.tensor(targets,device=device)
        logits,_=model([x],device)
        # Only real training events are valid predictions. Padding class 0 excluded.
        logits=logits[:,1:]; valid=(y>=1)&(y<=len(v)); ix=(y-1).clamp(0,len(v)-1)
        # Stable rank: ties resolved by lower class index.
        actual=logits.gather(1,ix[:,None]); ranks=1+(logits>actual).sum(1)+((logits==actual)&(torch.arange(len(v),device=device)[None,:]<ix[:,None])).sum(1)
        ranks=torch.where(valid,ranks,torch.full_like(ranks,len(v)+1)).cpu().tolist()
        for (i,t),r in zip(loc,ranks): out[i].append((t,r))
        xs.clear(); targets.clear(); loc.clear()
    for i,(_,events,_) in enumerate(tqdm(seqs,desc='Causal sequence scoring')):
        ids=encode(events,v)
        for t in range(a.history_size,len(ids)):
            xs.append(ids[t-a.history_size:t]); targets.append(ids[t]); loc.append((i,t+1))
            if len(xs)>=a.batch_size: flush()
    flush(); return out


def alerts(ranks,k,p):
    result=[]
    for stream in ranks:
        run=0; first=None
        for t,r in stream:
            run=run+1 if r>k else 0
            if run>=p: first=t; break
        result.append(first)
    return result


def metrics(seqs,first,a):
    tp=tn=fp=fn=0; ratios=[]
    for (_,s,y),t in zip(seqs,first):
        if y and t is not None: tp+=1; ratios.append(t/len(s))
        elif y: fn+=1
        elif t is not None: fp+=1
        else: tn+=1
    div=lambda x,y:x/y if y else 0.
    p=div(tp,tp+fp); r=div(tp,tp+fn)
    d={'Test sequences':len(seqs),'Truly anomalous sequences':tp+fn,'TP':tp,'TN':tn,'FP':fp,'FN':fn,'Precision':p,'Recall':r,'F1':div(2*p*r,p+r),'FPR':div(fp,fp+tn),'AvgRatio':div(sum(ratios),tp) if tp else None}
    for q in (25,50,75): d[f'EDR@{q}']=div(sum(t<=q/100 for t in ratios),tp+fn)
    d.update({'FP cost':a.fp_weight*fp,'FN cost':a.fn_weight*fn,'Delay cost':a.delay_weight*sum(ratios)})
    d['Total cost']=sum(d[k] for k in ('FP cost','FN cost','Delay cost'))
    return d


def select(seqs,ranks,a,ks,ps):
    best=None
    for k in ks:
        for p in ps:
            m=metrics(seqs,alerts(ranks,k,p),a)
            key=(m['F1'],-m['FPR'],m['Recall'],-p,-k)
            if best is None or key>best[0]: best=(key,k,p,m)
    return {'top_k':best[1],'persistence':best[2],'validation':best[3]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset_name',required=True); p.add_argument('--data_root',default='../LWADLS/datasets')
    p.add_argument('--id-column',default='Node_block_id'); p.add_argument('--event-column',default='EventId'); p.add_argument('--label-column',default='Original_Label'); p.add_argument('--order-column',default=None)
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto'); p.add_argument('--history_size',type=int,default=10); p.add_argument('--hidden_size',type=int,default=128); p.add_argument('--num_layers',type=int,default=2); p.add_argument('--embedding_dim',type=int,default=50)
    p.add_argument('--max_epoch',type=int,default=100); p.add_argument('--batch_size',type=int,default=1024); p.add_argument('--lr',type=float,default=.001); p.add_argument('--patience',type=int,default=10); p.add_argument('--num_candidates',type=int,default=9); p.add_argument('--persistence',default='1,2,3,4,5'); p.add_argument('--seed',type=int,default=123)
    for name in ('fp','fn','delay'): p.add_argument(f'--{name}-weight',type=float,default=1.)
    p.add_argument('--output_dir',default='deeplog_runs')
    p.add_argument('--eval-only',action='store_true',help='Reuse existing checkpoint; verify saved configuration and vocabulary')
    a=p.parse_args(); ps=sorted(set(map(int,a.persistence.split(','))))
    if min(a.history_size,a.max_epoch,a.batch_size,a.num_candidates,a.patience,*ps)<1: p.error('positive hyperparameters required')
    from logadempirical.logdeep.models.lstm import deeplog
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device=torch.device('cuda' if a.device!='cpu' and torch.cuda.is_available() else 'cpu')
    if a.device=='cuda' and device.type!='cuda': raise RuntimeError('CUDA requested but unavailable')
    print('Selected device:',device,flush=True)
    root=Path(a.data_root)/a.dataset_name/f'1_{a.dataset_name}_Splitted_Datasets'
    train=read(root/'train_df.pkl',a); val=read(root/'val_df.pkl',a); disjoint({'train':train,'val':val})
    normal=[s for _,s,y in train if not y]; v={e:i+1 for i,e in enumerate(dict.fromkeys(e for s in normal for e in s))}
    if not v: raise ValueError('No normal training events')
    ds=Windows([encode(s,v) for s in normal],a.history_size)
    if not len(ds): raise ValueError('No normal training windows; check history size')
    print('Train/validation blocks:',len(train),len(val),'normal training windows:',len(ds),'vocabulary:',len(v))
    if not any(y for _,_,y in val) or all(y for _,_,y in val): raise ValueError('Validation must contain both classes')
    loader=DataLoader(ds,batch_size=a.batch_size,shuffle=True)
    model=deeplog(1,a.hidden_size,a.num_layers,len(v)+1,a.embedding_dim).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=a.lr); lossfn=torch.nn.CrossEntropyLoss()
    # Select checkpoint by normal validation next-event loss, never test F1.
    vd=Windows([encode(s,v) for _,s,y in val if not y],a.history_size)
    if not len(vd): raise ValueError('No normal validation windows')
    vl=DataLoader(vd,batch_size=a.batch_size); out=Path(a.output_dir)/a.dataset_name; out.mkdir(parents=True,exist_ok=True)
    if a.eval_only:
        saved=json.loads((out/'config.json').read_text())
        for key in ('dataset_name','data_root','id_column','event_column','label_column','order_column','history_size','hidden_size','num_layers','embedding_dim','seed'):
            if saved.get(key)!=getattr(a,key): raise ValueError(f'Checkpoint configuration mismatch: {key}')
        if json.loads((out/'vocabulary.json').read_text())!=v:
            raise ValueError('Checkpoint vocabulary mismatch')
        if not (out/'model_best.pt').is_file(): raise FileNotFoundError(out/'model_best.pt')
    else:
        best=float('inf'); stale=0
        for epoch in range(a.max_epoch):
            model.train()
            for x,y in tqdm(loader,desc=f'Train {epoch+1}/{a.max_epoch}'):
                opt.zero_grad(); logits,_=model([x.to(device)],device); loss=lossfn(logits,y.to(device)); loss.backward(); opt.step()
            model.eval(); total=0.; count=0
            with torch.inference_mode():
                for x,y in vl:
                    x=x.to(device); y=y.to(device); valid=y<len(v)+1
                    if valid.any():
                        logits,_=model([x[valid]],device); total+=torch.nn.functional.cross_entropy(logits,y[valid],reduction='sum').item(); count+=int(valid.sum())
            if not count: raise ValueError('No in-vocabulary normal validation targets')
            score=total/count; print(f'Epoch {epoch+1}: validation loss {score:.6f}',flush=True)
            if score<best:
                best=score; stale=0; torch.save(model.state_dict(),out/'model_best.pt')
            else: stale+=1
            if stale>=a.patience: break
    model.load_state_dict(torch.load(out/'model_best.pt',map_location=device)); model.eval(); model.requires_grad_(False)
    if not a.eval_only:
        (out/'vocabulary.json').write_text(json.dumps(v)); (out/'config.json').write_text(json.dumps(vars(a),indent=2))
    vr=scores(model,val,v,a,device); ks=range(1,min(a.num_candidates,len(v))+1)
    policies={'DeepLog-Standard':select(val,vr,a,ks,[1]),'DeepLog-Online':select(val,vr,a,ks,ps)}
    policies['DeepLog-Standard']['decision_timing']='sequence_completion'
    policies['DeepLog-Online']['decision_timing']='first_qualifying_alert'
    # Freeze policies before loading test data.
    (out/'policies.json').write_text(json.dumps(policies,indent=2))
    test=read(root/'test_df.pkl',a); disjoint({'train':train,'val':val,'test':test}); print('Test blocks:',len(test),'without full history:',sum(len(s)<=a.history_size for _,s,_ in test))
    tr=scores(model,test,v,a,device); results={}
    for name,policy in policies.items():
        first=alerts(tr,policy['top_k'],policy['persistence'])
        if name=='DeepLog-Standard':
            first=[len(events) if t is not None else None for (_,events,_),t in zip(test,first)]
        results[name]=metrics(test,first,a)
    (out/'metrics.json').write_text(json.dumps({'policies':policies,'test':results},indent=2))
    names=list(results); print(f'{"Metric":30} {names[0]:>20} {names[1]:>20}')
    with open(out/'comparison.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['Metric',*names])
        for k in results[names[0]]:
            vals=[results[n][k] for n in names]; w.writerow([k,*vals]); print(f'{k:30} '+ ' '.join(f'{v:20.6f}' if isinstance(v,float) else f'{str(v):>20}' for v in vals))

if __name__=='__main__': main()
