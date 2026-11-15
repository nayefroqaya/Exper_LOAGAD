#!/usr/bin/env python3
"""Batched causal DeepLog Standard/Online evaluation using EXISTING checkpoint.

No training, no prefix augmentation, no modification to the model or threshold.
Standard = any next-event alert, declared at session completion.
Online = same alerts, first detection step. This matches the previous FAIR_MINCHANGE runner.
"""
import argparse
import csv
import json
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from logadempirical.data.vocab import Vocab
from logadempirical.models import ModelConfig, get_model


def safe_div(a, b):
    return float(a) / b if b else 0.0


def read_config(path):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def resolve_dir(args, cfg):
    if args.run_dir:
        return Path(args.run_dir).expanduser().resolve()
    root = Path(cfg['output_dir']).expanduser()
    grouping = 'block' if args.dataset.upper() == 'HDFS' else 'sliding'
    h = int(cfg['history_size'])
    if grouping == 'block':
        return root / 'DeepLog-Standard-Online' / args.dataset / f'block/H{h}'
    win = int(cfg['window_size'])
    step = int(cfg['step_size'])
    chrono = cfg.get('is_chronological', False)
    return root / 'DeepLog-Standard-Online' / args.dataset / f'sliding/W{win}_S{step}_H{h}_C{chrono}'


def build_model(cfg, vocabulary_size):
    model_cfg = ModelConfig(
        num_layers=int(cfg['num_layers']),
        hidden_size=int(cfg['hidden_size']),
        vocab_size=int(vocabulary_size),
        embedding_dim=int(cfg['embedding_dim']),
        dropout=float(cfg['dropout']),
        criterion=torch.nn.CrossEntropyLoss(ignore_index=0),
    )
    return get_model('DeepLog', model_cfg)


def get_event(vocab, template):
    x = vocab.get_event(template)
    if not isinstance(x, (int, np.integer)):
        raise TypeError(f'vocab.get_event() returned non-integer index: {x!r}')
    return int(x)


def infer_first_alerts(model, sessions, vocab, h, topk, batch_size, device, progress_every):
    """Process all causal H-history real-target examples in GPU batches.

    Generates windows in session order. Each window receives only H past events;
    the just-observed target is used solely in the top-k membership decision.
    No test-label values are sent to the predictor.
    """
    n = len(sessions)
    first = np.full(n, -1, dtype=np.int32)
    inspected = np.zeros(n, dtype=np.int32)
    total_scored = 0
    oov_context_windows = 0
    oov_target_windows = 0
    oov_examples = []
    xs, ys, session_ids, steps = [], [], [], []
    classes = len(vocab)
    if topk < 1 or topk > classes:
        raise ValueError(f'Invalid selected topk={topk}; vocabulary size={classes}')

    def flush():
        nonlocal total_scored, oov_context_windows, oov_target_windows
        if not xs:
            return
        # Original DeepLog trainer.predict_unsupervised_helper:
        # anomaly = (target not in predicted_top_k) OR any(context_id >= classes)
        # We retain it verbatim, including OOV treatment.
        context_oov = np.asarray([max(ctx) >= classes or min(ctx) < 0 for ctx in xs], dtype=bool)
        target_oov = np.asarray([target >= classes or target < 0 for target in ys], dtype=bool)
        if len(oov_examples) < 10:
            for j in np.flatnonzero(context_oov | target_oov)[:10-len(oov_examples)]:
                oov_examples.append({'session_id': int(session_ids[j]), 'step': int(steps[j]),
                                     'context_oov': bool(context_oov[j]), 'target_oov': bool(target_oov[j])})
        oov_context_windows += int(context_oov.sum())
        oov_target_windows += int(target_oov.sum())
        # Do not feed OOV context IDs to embedding lookups. The original trainer
        # unconditionally alerts on OOV contexts, independent of model scores.
        need_model = ~(context_oov | target_oov)
        anomalous = context_oov | target_oov
        if need_model.any():
            positions = np.flatnonzero(need_model)
            features = torch.tensor([xs[j] for j in positions], dtype=torch.long, device=device)
            # label is a real observed event, never a ground-truth anomaly label.
            targets = torch.tensor([ys[j] for j in positions], dtype=torch.long, device=device)
            with torch.inference_mode():
                # Exact native model top-k function used by original Trainer.
                predictions = model.predict_class(
                    {'sequential': features, 'label': targets},
                    top_k=topk, device=device)
            if isinstance(predictions, torch.Tensor):
                preds = predictions.detach().cpu().numpy()
            else:
                preds = np.asarray(predictions)
            if preds.ndim != 2 or preds.shape[0] != len(positions):
                raise RuntimeError(f'Unexpected model.predict_class output shape {preds.shape}; expected (batch, topk).')
            anomalous[positions] = ~(preds == np.asarray(ys, dtype=np.int64)[positions, None]).any(axis=1)
        for j, alerted in enumerate(anomalous):
            sid = session_ids[j]
            if first[sid] < 0:
                inspected[sid] += 1
                if alerted:
                    first[sid] = int(steps[j])
        total_scored += len(xs)
        xs.clear(); ys.clear(); session_ids.clear(); steps.clear()

    for sid, session in enumerate(sessions):
        events = session['EventTemplate']
        if len(events) <= h:
            raise ValueError(f'Session {sid} length {len(events)} <= history_size={h}. '
                             'Cannot evaluate real next-event prediction without losing a session.')
        encoded = [get_event(vocab, e) for e in events]
        # Session-level first-alert stopping is logically equivalent to scoring all
        # windows, then taking the earliest positive. Batch processing may compute
        # extra windows, but does not change the decision or use future context.
        for p in range(h, len(encoded)):
            xs.append(encoded[p-h:p])
            ys.append(encoded[p])
            session_ids.append(sid)
            steps.append(p + 1)
            if len(xs) >= batch_size:
                flush()
        if progress_every and (sid+1) % progress_every == 0:
            flush()
            print(f'[BATCHED EVAL] sessions={sid+1}/{n}, windows={total_scored}', flush=True)
    flush()
    return first, inspected, {
        'windows_evaluated': int(total_scored),
        'oov_context_windows': oov_context_windows,
        'oov_target_windows': oov_target_windows,
        'oov_examples': oov_examples,
    }


def assess(sessions, first, inspected, cfg, topk, diagnostics, duration, out_dir, dataset, h):
    labels = []
    rows_online = []
    rows_standard = []
    true_positives_ratios = []
    onset = {'anomalous_sequences': 0, 'alerts_before_first_anomalous_event': 0,
             'alerts_at_or_after_first_anomalous_event': 0, 'no_alert': 0,
             'onset_valid_edr25': 0, 'onset_valid_edr50': 0, 'onset_valid_edr75': 0}
    for i, session in enumerate(sessions):
        event_labels = [int(x) for x in session['Label']]
        if len(event_labels) != len(session['EventTemplate']):
            raise AssertionError(f'Label/template length differs in session {i}')
        if not set(event_labels).issubset({0,1}):
            raise ValueError(f'Nonbinary event labels in session {i}')
        truth = int(any(event_labels))
        labels.append(truth)
        t = len(event_labels)
        step = int(first[i]) if first[i] >= 0 else None
        pred = int(step is not None)
        onset_step = next((j for j, lab in enumerate(event_labels, 1) if lab == 1), None)
        premature = bool(step is not None and onset_step is not None and step < onset_step)
        ratio = step / t if step is not None else None
        base = {'sequence_id': i, 'true_label': truth, 'predicted_label': pred, 'sequence_length': t,
                'first_anomaly_step': onset_step if onset_step is not None else '',
                'premature_alert': int(premature)}
        rows_online.append({**base, 'first_alert_step': step if step is not None else '',
                            'detection_ratio': ratio if ratio is not None else '',
                            'prefixes_evaluated': int(inspected[i])})
        rows_standard.append({**base, 'first_alert_step': t if pred else '',
                              'detection_ratio': 1.0 if pred else '',
                              'prefixes_evaluated': t-h})
        if truth and pred:
            true_positives_ratios.append(ratio)
        if truth:
            onset['anomalous_sequences'] += 1
            if step is None:
                onset['no_alert'] += 1
            elif premature:
                onset['alerts_before_first_anomalous_event'] += 1
            else:
                onset['alerts_at_or_after_first_anomalous_event'] += 1
                for pct in (25,50,75):
                    if ratio <= pct / 100:
                        onset[f'onset_valid_edr{pct}'] += 1
    yt = np.asarray(labels, dtype=np.int8)
    yp = (first >= 0).astype(np.int8)
    tp = int(((yt == 1) & (yp == 1)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    p = safe_div(tp,tp+fp); r = safe_div(tp,tp+fn)
    f1 = safe_div(2*p*r,p+r)
    fpr = safe_div(fp,fp+tn)
    avg_ratio = float(np.mean(true_positives_ratios)) if true_positives_ratios else None
    for pct in (25,50,75):
        onset[f'onset_valid_edr{pct}'] = safe_div(onset[f'onset_valid_edr{pct}'], onset['anomalous_sequences'])
    n_anom = tp+fn
    online_edr = {f'EDR@{pct}': safe_div(sum(x <= pct/100 for x in true_positives_ratios), n_anom)
                  for pct in (25,50,75)}
    fp_cost = fp*float(cfg.get('fp_unit_cost',10))
    fn_cost = fn*float(cfg.get('fn_unit_cost',20))
    delay_unit = float(cfg.get('delay_unit_cost',5))
    standard_delay = tp * delay_unit
    online_delay = float(sum(true_positives_ratios)) * delay_unit
    common = {'Precision': p, 'Recall': r, 'F1': f1, 'FPR': fpr,
              'FP Cost': fp_cost, 'FN Cost': fn_cost}
    std = {**common, 'AvgRatio': 1.0 if tp else None,
           'EDR@25': 0.0, 'EDR@50': 0.0, 'EDR@75': 0.0,
           'Delay Cost': standard_delay, 'Total Cost': fp_cost+fn_cost+standard_delay}
    online = {**common, 'AvgRatio': avg_ratio, **online_edr,
              'Delay Cost': online_delay, 'Total Cost': fp_cost+fn_cost+online_delay}
    out_dir.mkdir(parents=True, exist_ok=True)
    for variant, rows in [('DeepLog-Standard', rows_standard), ('DeepLog-Online', rows_online)]:
        with (out_dir/f'{variant}_{dataset}_predictions.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    result = {'dataset': dataset, 'history_size': h, 'topk': topk, 'n_test_sequences': len(sessions),
              'tp':tp,'tn':tn,'fp':fp,'fn':fn, 'standard':std,'online':online,
              'onset_audit':onset, 'diagnostics':diagnostics, 'evaluation_seconds':duration,
              'protocol': 'Same model, same top-k and same event decisions; Standard reported at end, Online at first alert',
              'early_detection_caveat': 'EDR based on sequence-level labels; onset-valid EDR reported separately'}
    (out_dir/f'{dataset}_DeepLog_BATCHED_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines = ['='*76, f'DeepLog {dataset} — BATCHED STANDARD vs ONLINE (H={h}, topk={topk})','='*76,
             f'{"Metric":<25} {"Standard":>18} {"Online":>18}', '-'*76]
    for name in std:
        a=std[name];b=online[name]
        lines.append(f'{name:<25} {a if a is not None else float("nan"):>18.6f} {b if b is not None else float("nan"):>18.6f}')
    lines += ['='*76, f'TP={tp} TN={tn} FP={fp} FN={fn}', f'First-alert onset audit: {onset}',
              f'Windows scored: {diagnostics["windows_evaluated"]:,}',
              f'Elapsed inference: {duration:.2f} seconds',
              'Note: Standard and Online classification match by design; timing differs.']
    report='\n'.join(lines)+'\n'
    print(report, flush=True)
    (out_dir/f'{dataset}_DeepLog_Standard_vs_Online_BATCHED.txt').write_text(report,encoding='utf-8')
    print('Saved reports to:',out_dir,flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config_file',required=True)
    ap.add_argument('--dataset',required=True)
    ap.add_argument('--run-dir',help='Exact directory containing checkpoint, vocab, topk metadata and test.pkl')
    ap.add_argument('--batch-size',type=int,default=4096)
    ap.add_argument('--progress-every',type=int,default=500)
    ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=ap.parse_args()
    if args.batch_size<1:ap.error('--batch-size must be positive')
    cfg=read_config(args.config_file)
    h=int(cfg['history_size'])
    if not cfg.get('sequential',True) or cfg.get('quantitative',False) or cfg.get('semantic',False):
        raise ValueError('This evaluation-only implementation requires sequential=True, quantitative=False, semantic=False')
    directory=resolve_dir(args,cfg)
    needed=[directory/'models/DeepLog.pt',directory/'vocabs/DeepLog.pkl',directory/'deeplog_selected_topk.json',directory/'test.pkl']
    missing=[str(p) for p in needed if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing already-trained artifacts (DO NOT RETRAIN yet):\n'+'\n'.join(missing)+'\nPass --run-dir if the saved output differs.')
    print('Using EXISTING trained model from:',directory,flush=True)
    vocab=Vocab.load_vocab(str(needed[1]))
    with needed[2].open(encoding='utf-8') as f:topk=int(json.load(f)['topk'])
    with needed[3].open('rb') as f:sessions=pickle.load(f)
    if not sessions:raise ValueError('Empty saved test.pkl')
    device=torch.device('cuda' if (args.device=='auto' and torch.cuda.is_available()) or args.device=='cuda' else 'cpu')
    model=build_model(cfg,len(vocab))
    checkpoint=torch.load(str(needed[0]),map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model=model.to(device).eval()
    print(f'Dataset={args.dataset} sequences={len(sessions)} H={h} topk={topk} classes={len(vocab)} device={device} batch={args.batch_size}',flush=True)
    start=time.perf_counter()
    first, inspected, diagnostics=infer_first_alerts(model,sessions,vocab,h,topk,args.batch_size,device,args.progress_every)
    elapsed=time.perf_counter()-start
    assess(sessions,first,inspected,cfg,topk,diagnostics,elapsed,directory/'batched_results',args.dataset,h)

if __name__=='__main__':
    main()
