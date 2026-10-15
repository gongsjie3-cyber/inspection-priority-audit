"""Monthly, history-only retrospective inspection ranking. Run from any directory."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import platform
import sqlite3
import sys
from fractions import Fraction
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['history_log','fail_rate','recent_fail_rate','conditions_rate',
            'last_fail','last_conditions','days_since_last_log','days_since_routine_log',
            'recent_count_log','complaint_count_log','complaint_fail_log',
            'risk_high','risk_medium','risk_low','no_history','month_sin','month_cos']
VALID_RESULTS = ['Pass','Pass w/ Conditions','Fail']

def clean(raw):
    d = raw.copy()
    audit = {'raw_rows': len(d)}
    d['inspection_id'] = pd.to_numeric(d['inspection_id'], errors='coerce')
    d['license_'] = pd.to_numeric(d['license_'], errors='coerce')
    d['date'] = pd.to_datetime(d['inspection_date'], errors='coerce')
    d['address'] = d['address'].fillna('').str.upper().str.replace(r'\s+', ' ', regex=True).str.strip()
    d = d.drop_duplicates('inspection_id')
    audit['duplicate_id_removed'] = len(raw) - len(d)
    usable = d['date'].notna() & d['license_'].gt(0) & d['address'].ne('')
    audit['invalid_identity_or_date_removed'] = int((~usable).sum())
    d = d[usable].copy()
    d['venue'] = d['license_'].astype('int64').astype(str) + '|' + d['address']
    # Two outcomes at a venue on one date have no reliable within-day ordering.
    group = d.groupby(['venue','date'])['results'].transform('nunique')
    audit['conflicting_same_day_rows_removed'] = int((group > 1).sum())
    d = d[group <= 1].copy()
    before = len(d)
    d = d.sort_values(['date','inspection_id']).drop_duplicates(['venue','date'])
    audit['same_day_same_result_duplicates_removed'] = before - len(d)
    audit['valid_identity_dedup_rows'] = len(d)
    history = d[d['results'].isin(VALID_RESULTS)].copy()
    audit['non_evaluable_results_removed_from_history'] = len(d) - len(history)
    history['y'] = history['results'].eq('Fail').astype(int)
    history['month'] = history['date'].dt.to_period('M').astype(str)
    target = history[history['inspection_type'].eq('Canvass') &
                     history['facility_type'].fillna('').str.lower().eq('restaurant') &
                     history['date'].ge('2020-01-01')].copy()
    audit['target_before_month_dedup'] = len(target)
    target = target.sort_values(['date','inspection_id']).drop_duplicates(['venue','month'])
    audit['repeated_routine_same_venue_month_removed'] = audit['target_before_month_dedup'] - len(target)
    audit['target_rows'] = len(target)
    audit['history_rows'] = len(history)
    audit['target_venues'] = int(target['venue'].nunique())
    audit['raw_result_counts'] = raw['results'].value_counts().to_dict()
    return history, target, audit

def build_features(history, target, lag_days=14):
    """A target month's first day is the decision time, never its inspection day.

    Historical event date + assumed publication lag must be strictly before
    decision time. No current risk, ZIP, result or inspection description is used.
    """
    records = []
    groups = {v: g.sort_values(['date','inspection_id']).reset_index(drop=True)
              for v,g in history.groupby('venue', sort=False)}
    for venue, t in target.groupby('venue', sort=False):
        g = groups[venue]
        dates = g['date'].to_numpy(dtype='datetime64[ns]')
        for row in t.itertuples():
            decision = pd.Timestamp(row.month + '-01')
            cutoff = decision - np.timedelta64(lag_days, 'D')
            n = int(np.searchsorted(dates, np.datetime64(cutoff), side='left'))
            h = g.iloc[:n]
            recent = h[h['date'] >= cutoff - np.timedelta64(365, 'D')]
            routine = h[h['inspection_type'].eq('Canvass')]
            complaint = recent[recent['inspection_type'].fillna('').str.startswith('Complaint')]
            last = h.iloc[-1] if n else None
            last_date = last['date'] if n else pd.NaT
            risk = str(last['risk']) if n else ''
            days = (decision-last_date).days if n else 3650
            days_routine = (decision-routine['date'].max()).days if len(routine) else 3650
            rec = {'inspection_id': int(row.inspection_id),
                   'venue_id': hashlib.sha256(venue.encode()).hexdigest()[:16],
                   'month': row.month, 'date': row.date.strftime('%Y-%m-%d'),
                   'decision_date': decision.strftime('%Y-%m-%d'),
                   'history_cutoff_exclusive': cutoff.strftime('%Y-%m-%d'),
                   'max_history_date': last_date.strftime('%Y-%m-%d') if n else '',
                   'zip_group': str(last['zip']) if n and pd.notna(last['zip']) else 'UNKNOWN',
                   'y': int(row.y), 'n_history': n, 'days_since_last': days,
                   'days_since_routine': days_routine,
                   'history_log': np.log1p(n),
                   'fail_rate': (h['y'].sum()+1)/(n+5),
                   'recent_fail_rate': (recent['y'].sum()+1)/(len(recent)+5),
                   'conditions_rate': (h['results'].eq('Pass w/ Conditions').sum()+1)/(n+5),
                   'last_fail': int(n > 0 and last['results'] == 'Fail'),
                   'last_conditions': int(n > 0 and last['results'] == 'Pass w/ Conditions'),
                   'days_since_last_log': np.log1p(days),
                   'days_since_routine_log': np.log1p(days_routine),
                   'recent_count_log': np.log1p(len(recent)),
                   'complaint_count_log': np.log1p(len(complaint)),
                   'complaint_fail_log': np.log1p(complaint['y'].sum()),
                   'risk_high': int(risk == 'Risk 1 (High)'),
                   'risk_medium': int(risk == 'Risk 2 (Medium)'),
                   'risk_low': int(risk == 'Risk 3 (Low)'),
                   'no_history': int(n == 0),
                   'month_sin': np.sin(2*np.pi*decision.month/12),
                   'month_cos': np.cos(2*np.pi*decision.month/12)}
            records.append(rec)
    return pd.DataFrame(records).sort_values(['month','venue_id']).reset_index(drop=True)

def monthly_budget(n, budget):
    """Interpret a decimal budget exactly before rounding up."""
    rate = Fraction(str(budget))
    if not 0 < rate <= 1:
        raise ValueError('budget must be in (0, 1].')
    return math.ceil(n * rate)

def select_indices(g, budget, method='model', reserve=0.):
    """Same monthly K for each rule; all ties break on hashed venue, never label."""
    k = monthly_budget(len(g),budget)
    if method == 'high_risk':
        ordered = g.sort_values(['risk_high','risk_medium','venue_id'], ascending=[False,False,True])
    elif method == 'longest_wait':
        ordered = g.sort_values(['days_since_routine','venue_id'], ascending=[False,True])
    elif method == 'history_rule':
        ordered = g.sort_values(['last_fail','fail_rate','days_since_routine','venue_id'],
                                ascending=[False,False,False,True])
    else:
        ordered = g.sort_values(['score','venue_id'], ascending=[False,True])
    if reserve == 0:
        return ordered.index[:k].tolist()
    # Hamilton largest-remainder allocation of reserved slots, then fill globally.
    slots = math.floor(k * Fraction(str(reserve)))
    size = g.groupby('zip_group').size().sort_index()
    # A common denominator makes integer remainders directly comparable.
    # Floating subtraction can otherwise break mathematically equal ties.
    numerator = size * slots
    quota = numerator // len(g)
    leftovers = slots - int(quota.sum())
    priority = pd.DataFrame({'remainder': numerator % len(g), 'zip':size.index}).sort_values(
        ['remainder','zip'], ascending=[False,True])
    for z in priority['zip'].iloc[:leftovers]:
        quota.loc[z] += 1
    picked = []
    for z, q in quota.items():
        picked.extend(ordered[ordered['zip_group'].eq(z)].index[:q].tolist())
    used = set(picked)
    picked.extend([i for i in ordered.index if i not in used][:k-len(picked)])
    assert len(picked) == k and len(set(picked)) == k
    return picked

POLICIES = [('model',0.),('history_rule',0.),('high_risk',0.),('longest_wait',0.),
            ('model',.5),('model',1.)]
def policy_name(method,reserve):
    return f'model_quota{int(reserve*100)}' if reserve else method

def evaluate(d, budget=.2, policies=POLICIES):
    rows = []
    for month, g in d.groupby('month'):
        k = monthly_budget(len(g),budget)
        base = dict(month=month, candidates=len(g), positives=int(g.y.sum()), selected=k, budget=budget)
        rows.append(dict(base, policy='random_expected', hits=float(k*g.y.mean()),
                         zip_tv=np.nan, zip_coverage=np.nan))
        population = g.zip_group.value_counts(normalize=True)
        for method,reserve in policies:
            s = g.loc[select_indices(g,budget,method,reserve)]
            shares = s.zip_group.value_counts(normalize=True).reindex(population.index,fill_value=0)
            rows.append(dict(base, policy=policy_name(method,reserve), hits=int(s.y.sum()),
                             zip_tv=float(np.abs(shares-population).sum()/2),
                             zip_coverage=float(s.zip_group.nunique()/g.zip_group.nunique())))
    monthly = pd.DataFrame(rows)
    return monthly, summarize(monthly)

def summarize(monthly):
    rows=[]
    for policy,g in monthly.groupby('policy',sort=False):
        total, positives, selected, hits = [float(g[c].sum()) for c in ['candidates','positives','selected','hits']]
        random_hits=float((g.selected*g.positives/g.candidates).sum())
        rows.append({'policy':policy,'candidates':int(total),'positives':int(positives),
                     'selected':int(selected),'hits':hits,'precision':hits/selected,
                     'recall':hits/positives,'lift_vs_random':hits/random_hits,
                     'zip_tv':float(g.zip_tv.mean()),'zip_coverage':float(g.zip_coverage.mean())})
    return pd.DataFrame(rows)

def block_interval(monthly, p='model', q='history_rule', repetitions=2000):
    a=monthly[monthly.policy.eq(p)].set_index('month')
    b=monthly[monthly.policy.eq(q)].set_index('month').reindex(a.index)
    rng=np.random.default_rng(20261010)
    idx=rng.integers(0,len(a),size=(repetitions,len(a)))
    diff=(a.hits.to_numpy()-b.hits.to_numpy())[idx].sum(axis=1)/a.selected.to_numpy()[idx].sum(axis=1)
    return {'comparison':f'{p} minus {q}', 'unit':'absolute precision difference',
            'estimate':float((a.hits.sum()-b.hits.sum())/a.selected.sum()),
            'lower95':float(np.quantile(diff,.025)), 'upper95':float(np.quantile(diff,.975)),
            'bootstrap':'paired whole-month resampling; does not account for cross-month venue dependence',
            'repetitions':repetitions}

def metric_scores(y,score):
    return {'roc_auc':float(roc_auc_score(y,score)),
            'average_precision':float(average_precision_score(y,score)),
            'brier':float(brier_score_loss(y,score))}

def split_features(f, lag_days):
    """Keep model selection and fitting inside their information boundaries.

    Validate on complete January-November 2023 pools. December is excluded
    because its last labels are not available on 2024-01-01 under either lag;
    filtering individual December outcomes would change that month's pool.
    December labels that are already available may still enter the final fit.
    """
    if lag_days not in (14, 30):
        raise ValueError('Supported information-lag scenarios are 14 and 30 days.')
    dates = pd.to_datetime(f.date)
    train = f[dates < pd.Timestamp('2023-01-01') - np.timedelta64(lag_days, 'D')]
    valid = f[dates.ge('2023-01-01') & dates.lt('2023-12-01')]
    fit = f[dates < pd.Timestamp('2024-01-01') - np.timedelta64(lag_days, 'D')]
    test = f[dates.ge('2024-01-01') & dates.lt('2026-01-01')].copy()
    return train, valid, fit, test

def main(lag=14):
    out=ROOT/'results'/f'lag{lag}'
    out.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(ROOT/'data/inspections_2019_2025.csv.gz',dtype=str)
    history,target,audit=clean(raw)
    f=build_features(history,target,lag)
    f.to_csv(out/'features.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    print('Features ready:',len(f),flush=True)
    train,valid,fit,test=split_features(f,lag)
    candidates={
        'logistic':make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000,random_state=42)),
        'hist_gradient_boosting':HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=7,
              learning_rate=.05,l2_regularization=10,min_samples_leaf=50,early_stopping=False,random_state=42)}
    val_rows=[]
    for name,model in candidates.items():
        model.fit(train[FEATURES],train.y)
        v=valid.copy()
        v['score']=model.predict_proba(v[FEATURES])[:,1]
        _,summary=evaluate(v)
        record=summary[summary.policy.eq('model')].iloc[0].to_dict()
        record.update(model=name,**metric_scores(v.y,v.score))
        val_rows.append(record)
    selection=pd.DataFrame(val_rows).sort_values(['precision','model'],ascending=[False,True])
    if lag == 14:
        winner=selection.iloc[0]['model']
    else:
        winner=json.loads((ROOT/'results/lag14/summary.json').read_text())['chosen_model']
    model=candidates[winner].fit(fit[FEATURES],fit.y)
    test['score']=model.predict_proba(test[FEATURES])[:,1]
    monthly,summary=evaluate(test)
    for method,reserve in POLICIES:
        name=policy_name(method,reserve)
        chosen=[]
        for _,g in test.groupby('month'):
            chosen.extend(select_indices(g,.2,method,reserve))
        test[name+'_selected']=test.index.isin(chosen).astype(int)
    curves=[]
    for budget in [.05,.1,.15,.2,.25,.3,.4,.5]:
        _,s=evaluate(test,budget)
        s['budget']=budget
        curves.append(s)
    curve=pd.concat(curves,ignore_index=True)
    yearly=[]
    for year in ['2024','2025']:
        _,s=evaluate(test[test.month.str.startswith(year)])
        s['year']=year
        yearly.append(s)
    segment=[]
    for group,g in test.groupby('no_history'):
        for policy in ['model','history_rule','model_quota50']:
            chosen=g[g[policy+'_selected'].eq(1)]
            segment.append({'segment':'no_history' if group else 'has_history','policy':policy,
                            'candidates':len(g),'positives':int(g.y.sum()),'selected':len(chosen),
                            'hits':int(chosen.y.sum()),'precision':float(chosen.y.mean()) if len(chosen) else None,
                            'recall':float(chosen.y.sum()/g.y.sum()) if g.y.sum() else None})
    # Reliability is descriptive; no test-time calibration or threshold tuning.
    test['score_band']=pd.cut(test.score,bins=np.linspace(0,1,11),include_lowest=True).astype(str)
    calibration=test.groupby('score_band').agg(n=('y','size'),score=('score','mean'),observed=('y','mean')).reset_index()
    baseline_names=['history_rule','high_risk','longest_wait']
    v=valid.copy()
    v['score']=0.  # Baselines do not use model scores; no refit-model validation scoring.
    _,valid_rules=evaluate(v,policies=[(x,0) for x in baseline_names])
    best_rule=valid_rules[valid_rules.policy.isin(baseline_names)].sort_values('precision',ascending=False).iloc[0]['policy']
    metadata={'lag_days':lag,'features':FEATURES,'chosen_model':winner,
              'selection_rule':('highest pooled monthly Precision@20% in January-November 2023; no test tuning' if lag == 14 else 'model family fixed from lag14 January-November 2023 validation; refit with 30-day availability'),
              'best_rule_selected_on_validation':best_rule,
              'audit':audit,'splits':{'train':len(train),'validation':len(valid),'refit':len(fit),'test':len(test)},
              'split_dates':{
                  'initial_fit_as_of':'2023-01-01',
                  'final_fit_and_selection_as_of':'2024-01-01',
                  'validation_start':'2023-01-01',
                  'validation_end_exclusive':'2023-12-01',
                  'validation_months':int(valid.month.nunique()),
                  'train_latest_label':str(train.date.max()),
                  'validation_latest_label':str(valid.date.max()),
                  'refit_latest_label':str(fit.date.max())},
              'test_scores':metric_scores(test.y,test.score),
              'precision_difference_ci':block_interval(monthly,'model',best_rule),
              'random_difference_ci':block_interval(monthly,'model','random_expected'),
              'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__},
              'limitations':['Retrospective candidate pools restricted to observed routine restaurant inspections.',
                  'Fail is an administrative inspection label, not illness or a critical-violation label.',
                  '14/30-day publication lag is assumed; historical version timestamps are unavailable.',
                  'Historical exports can contain retrospective corrections.',
                  'ZIP proportionality is a coverage constraint, not proof of demographic fairness.',
                  'No staffing cost, causal effect or operational deployment is estimated.']}
    (out/'summary.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
    for file,frame in [('validation_models',selection),('validation_rules',valid_rules),('policies',summary),
                       ('monthly',monthly),('budget_curve',curve),('yearly',pd.concat(yearly)),
                       ('segments',pd.DataFrame(segment)),('calibration',calibration)]:
        frame.to_csv(out/(file+'.csv'),index=False)
    test.to_csv(out/'scored_test.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    joblib.dump(model,out/'model.joblib')
    if winner == 'logistic':
        coefs=pd.DataFrame({'feature':FEATURES,'standardized_log_odds':model[-1].coef_[0]}).sort_values('standardized_log_odds',ascending=False)
        coefs.to_csv(out/'coefficients.csv',index=False)
    if lag==14:
        history['venue_id']=history['venue'].map(lambda x: hashlib.sha256(x.encode()).hexdigest()[:16])
        with sqlite3.connect(ROOT/'data/audit.sqlite') as con:
            history[['inspection_id','venue_id','date','month','inspection_type','facility_type','results','y']].to_sql('inspections',con,if_exists='replace',index=False)
            con.execute('CREATE INDEX idx_history_venue_date ON inspections(venue_id,date)')
            f.to_sql('monthly_features',con,if_exists='replace',index=False)
            test.to_sql('scored_test',con,if_exists='replace',index=False)
    print(selection[['model','precision','roc_auc']].to_string(index=False),flush=True)
    print(summary[['policy','selected','hits','precision','recall','zip_tv']].to_string(index=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--lag',type=int,choices=[14,30],default=14)
    main(p.parse_args().lag)
