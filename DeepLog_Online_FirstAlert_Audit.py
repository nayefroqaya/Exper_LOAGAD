#!/usr/bin/env python3
"""Audit DeepLog Standard / Online saved first-alert decisions; no model changes.

Uses only saved *_predictions.csv outputs. This does NOT confirm event-level
anomaly onset, which is absent from those CSV files.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_predictions(path):
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    need = {'sequence_id','true_label','predicted_label','sequence_length','first_alert_step'}
    if not rows or not need.issubset(rows[0]):
        raise ValueError(f'{path}: missing required columns {sorted(need)} or no rows')
    result = {}
    for r in rows:
        key = r['sequence_id']
        if key in result:
            raise ValueError(f'{path}: duplicate sequence ID {key}')
        y = int(r['true_label'])
        pred = int(r['predicted_label'])
        length = int(r['sequence_length'])
        step = None if not r['first_alert_step'].strip() else int(r['first_alert_step'])
        if y not in (0, 1) or pred not in (0, 1) or length < 1:
            raise ValueError(f'{path}: invalid prediction row for sequence {key}')
        if (step is None) != (pred == 0):
            raise ValueError(f'{path}: prediction / alert mismatch for sequence {key}')
        if step is not None and not (1 <= step <= length):
            raise ValueError(f'{path}: invalid alert step for sequence {key}')
        result[key] = {'label': y, 'pred': pred, 'length': length, 'step': step}
    return result


def compute(rows):
    vals = list(rows.values())
    tp=sum(x['label']==1 and x['pred']==1 for x in vals)
    fp=sum(x['label']==0 and x['pred']==1 for x in vals)
    fn=sum(x['label']==1 and x['pred']==0 for x in vals)
    tn=sum(x['label']==0 and x['pred']==0 for x in vals)
    alerts=[x for x in vals if x['step'] is not None]
    out={'total':len(vals),'TP':tp,'TN':tn,'FP':fp,'FN':fn,
         'precision':tp/(tp+fp) if tp+fp else 0,
         'recall':tp/(tp+fn) if tp+fn else 0,
         'FPR':fp/(fp+tn) if fp+tn else 0,
         'alert_count':len(alerts),
         'alert_steps':dict(sorted(Counter(x['step'] for x in alerts).items())[:15]),
         'alert_at_step_2':sum(x['step']==2 for x in alerts),
         'alert_at_step_3_or_less':sum(x['step']<=3 for x in alerts),
         'alert_at_step_5_or_less':sum(x['step']<=5 for x in alerts),
         'alert_at_step_10_or_less':sum(x['step']<=10 for x in alerts),
         'alerts_before_120_context_events':sum(x['step']<=120 for x in alerts),
         'normal_early_false_alerts_step_2':sum(x['label']==0 and x['step']==2 for x in vals),
         'anomaly_early_alerts_step_2':sum(x['label']==1 and x['step']==2 for x in vals),
         'length_distribution':dict(Counter(x['length'] for x in vals).most_common(10)),
         'mean_TP_first_alert_ratio':(sum(x['step']/x['length'] for x in vals if x['label']==1 and x['pred']==1)/tp) if tp else None}
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--online',type=Path,required=True,help='DeepLog-Online_BGL_predictions.csv')
    p.add_argument('--standard',type=Path,required=True,help='DeepLog-Standard_BGL_predictions.csv')
    p.add_argument('--output',type=Path,default=Path('DeepLog_Online_FirstAlert_Audit.json'))
    a=p.parse_args()
    on=read_predictions(a.online); std=read_predictions(a.standard)
    if set(on)!=set(std):
        raise ValueError('Standard and Online have different sequence_id sets')
    mismatched_labels=[k for k in on if on[k]['label']!=std[k]['label'] or on[k]['length']!=std[k]['length']]
    if mismatched_labels:
        raise ValueError(f'Different ground truth labels or sequence lengths for {len(mismatched_labels)} sequences')
    report={'online':compute(on),'standard':compute(std),
            'different_sequence_predictions':sum(on[k]['pred']!=std[k]['pred'] for k in on),
            'online_alert_step_2_normal_share': None,
            'notes':['Early decision at step 2 uses only 1 real prior event and 119 pad tokens with history_size=120.',
                     'The alert-step CSV does not include true anomaly onset. Early-alert validity relative to first anomaly cannot be proven from these outputs.',
                     'A 100% sequence FPR can follow from OR aggregation of per-event alerts, even without any software bug.']}
    fp=report['online']['FP']
    report['online_alert_step_2_normal_share']=(report['online']['normal_early_false_alerts_step_2']/fp) if fp else None
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('ONLINE:', {k:report['online'][k] for k in ('total','TP','TN','FP','FN','FPR','alert_at_step_2','alert_at_step_3_or_less','normal_early_false_alerts_step_2','mean_TP_first_alert_ratio')})
    print('STANDARD:', {k:report['standard'][k] for k in ('total','TP','TN','FP','FN','FPR')})
    print('Different per-sequence predictions:',report['different_sequence_predictions'])
    print('Saved:',a.output.resolve())

if __name__=='__main__':
    main()
