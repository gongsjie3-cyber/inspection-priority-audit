"""Tests target temporal leakage, identities, budgets and cross-engine agreement."""
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
import unittest
import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pipeline import FEATURES, clean, build_features, select_indices, split_features, monthly_budget, POLICIES

class IntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=pd.read_csv(ROOT/'data/inspections_2019_2025.csv.gz',dtype=str)
        cls.f=pd.read_csv(ROOT/'results/lag14/features.csv.gz')
        cls.test=pd.read_csv(ROOT/'results/lag14/scored_test.csv.gz')

    def test_snapshot_checksum(self):
        m=json.loads((ROOT/'data/manifest.json').read_text())
        self.assertEqual(hashlib.sha256((ROOT/'data/inspections_2019_2025.csv.gz').read_bytes()).hexdigest(),m['snapshot_sha256'])
        self.assertEqual(sum(s['rows'] for s in m['sources']),len(self.raw))

    def test_strict_information_cutoff(self):
        d=self.f.dropna(subset=['max_history_date'])
        self.assertTrue((pd.to_datetime(d.max_history_date)<pd.to_datetime(d.history_cutoff_exclusive)).all())
        self.assertTrue((pd.to_datetime(self.f.decision_date)<=pd.to_datetime(self.f.date)).all())

    def test_target_outcome_mutation_does_not_change_features(self):
        base=dict(license_='1',facility_type='Restaurant',risk='Risk 1 (High)',address='ONE ROAD',zip='60601',inspection_type='Canvass')
        rows=[dict(base,inspection_id='1',inspection_date='2022-11-01',results='Fail'),
              dict(base,inspection_id='2',inspection_date='2023-01-17',results='Pass'),
              dict(base,inspection_id='3',inspection_date='2023-01-25',results='Pass')]
        h,t,_=clean(pd.DataFrame(rows))
        a=build_features(h,t)
        rows[1]['results']='Fail'
        rows[2]['results']='Fail'
        h,t,_=clean(pd.DataFrame(rows))
        b=build_features(h,t)
        np.testing.assert_allclose(a[FEATURES],b[FEATURES])

    def test_cutoff_day_is_not_available(self):
        base=dict(license_='1',facility_type='Restaurant',risk='Risk 1 (High)',address='ONE ROAD',zip='60601',inspection_type='Canvass')
        rows=[dict(base,inspection_id='1',inspection_date='2022-12-17',results='Pass'),
              dict(base,inspection_id='2',inspection_date='2022-12-18',results='Fail'),
              dict(base,inspection_id='3',inspection_date='2023-01-20',results='Pass')]
        h,t,_=clean(pd.DataFrame(rows))
        f=build_features(h,t,14)
        jan=f[f.month.eq('2023-01')].iloc[0]
        self.assertEqual(jan.n_history,1)
        self.assertEqual(jan.last_fail,0)

    def test_no_current_outcome_features_or_identity_features(self):
        self.assertFalse(set(FEATURES)&{'y','results','inspection_id','venue_id','violations','zip_group','date'})
        self.assertEqual(self.f[FEATURES].isna().sum().sum(),0)
        self.assertEqual(self.f.duplicated(['venue_id','month']).sum(),0)

    def test_budget_and_no_duplicate_selections(self):
        for _,g in self.test.groupby('month'):
            for budget in [.05,.2,.5]:
                for method,reserve in POLICIES:
                    ids=select_indices(g,budget,method,reserve)
                    self.assertEqual(len(ids),math.ceil(len(g)*budget))
                    self.assertEqual(len(ids),len(set(ids)))

    def test_label_blind_selection(self):
        g=self.test[self.test.month.eq('2024-01')].copy()
        changed=g.copy(); changed['y']=1-changed['y']
        for method,reserve in POLICIES:
            self.assertEqual(select_indices(g,.2,method,reserve),select_indices(changed,.2,method,reserve))

    def test_same_day_conflicts_removed(self):
        d=self.raw.iloc[:1].copy()
        d['license_']='123'; d['address']='TEST'; d['results']='Pass'
        other=d.copy(); other['inspection_id']='999999999'; other['results']='Fail'
        h,_,audit=clean(pd.concat([d,other]))
        self.assertEqual(len(h),0)
        self.assertEqual(audit['conflicting_same_day_rows_removed'],2)

    def test_sql_point_in_time_join(self):
        with sqlite3.connect(ROOT/'data/audit.sqlite') as con:
            mismatch=pd.read_sql_query((ROOT/'sql/audit.sql').read_text(),con)
        self.assertTrue(mismatch.empty)

    def test_sql_ranking_reproduces_python(self):
        with sqlite3.connect(ROOT/'data/audit.sqlite') as con:
            sql=pd.read_sql_query((ROOT/'sql/monthly_decisions.sql').read_text(),con)
        py=pd.read_csv(ROOT/'results/lag14/monthly.csv')
        py=py[py.policy.eq('model')]
        joined=sql.merge(py,on='month',suffixes=('_sql','_py'))
        self.assertEqual(len(joined),24)
        self.assertTrue((joined.hits_sql==joined.hits_py).all())
        self.assertTrue((joined.selected_sql==joined.selected_py).all())

    def test_validation_labels_available_at_selection_time(self):
        # Complete months only: never choose a model using late December labels.
        for lag in (14,30):
            f=pd.read_csv(ROOT/f'results/lag{lag}/features.csv.gz')
            train,valid,fit,test=split_features(f,lag)
            self.assertEqual(sorted(valid.month.unique().tolist()),
                             [f'2023-{m:02d}' for m in range(1,12)])
            for frame,as_of in [(train,'2023-01-01'),(valid,'2024-01-01'),(fit,'2024-01-01')]:
                available=pd.to_datetime(frame.date)+np.timedelta64(lag, 'D')
                self.assertTrue((available<pd.Timestamp(as_of)).all())
            self.assertTrue(set(fit.inspection_id).isdisjoint(test.inspection_id))

    def test_split_year_boundaries(self):
        f=pd.DataFrame({'date':['2022-12-01','2022-12-02','2022-12-17','2022-12-18',
                                '2023-11-30','2023-12-01','2023-12-02','2023-12-17',
                                '2023-12-18','2024-01-01','2025-12-31','2026-01-01']})
        train,valid,fit,test=split_features(f,14)
        self.assertEqual(train.date.tolist(),f.date.iloc[:3].tolist())
        self.assertEqual(valid.date.tolist(),['2023-11-30'])
        self.assertEqual(fit.date.max(),'2023-12-17')
        self.assertEqual(test.date.tolist(),['2024-01-01','2025-12-31'])
        train,valid,fit,test=split_features(f,30)
        self.assertEqual(train.date.tolist(),['2022-12-01'])
        self.assertEqual(fit.date.max(),'2023-12-01')
        self.assertEqual(valid.date.tolist(),['2023-11-30'])

    def test_current_risk_and_zip_do_not_enter_decision_inputs(self):
        base=dict(license_='1',facility_type='Restaurant',risk='Risk 1 (High)',
                  address='ONE ROAD',zip='60601',inspection_type='Canvass')
        rows=[dict(base,inspection_id='1',inspection_date='2022-11-01',results='Pass'),
              dict(base,inspection_id='2',inspection_date='2023-01-20',results='Fail')]
        h,t,_=clean(pd.DataFrame(rows)); a=build_features(h,t)
        rows[1].update(risk='Risk 3 (Low)',zip='99999',results='Pass')
        h,t,_=clean(pd.DataFrame(rows)); b=build_features(h,t)
        pd.testing.assert_frame_equal(a[FEATURES+['zip_group']],b[FEATURES+['zip_group']])

    def test_published_policy_metrics_match_row_level_outcomes(self):
        for lag in (14,30):
            d=pd.read_csv(ROOT/f'results/lag{lag}/scored_test.csv.gz')
            p=pd.read_csv(ROOT/f'results/lag{lag}/policies.csv').set_index('policy')
            self.assertTrue(d.score.between(0,1).all())
            random_hits=sum(math.ceil(len(g)*.2)*g.y.mean() for _,g in d.groupby('month'))
            self.assertAlmostEqual(p.loc['random_expected','hits'],random_hits)
            for name,row in p.drop(index='random_expected').iterrows():
                selected=d[d[name+'_selected'].eq(1)]
                self.assertEqual(len(selected),row.selected)
                self.assertEqual(selected.y.sum(),row.hits)
                self.assertAlmostEqual(selected.y.mean(),row.precision)
                self.assertAlmostEqual(selected.y.sum()/d.y.sum(),row.recall)
                self.assertAlmostEqual(selected.y.sum()/random_hits,row.lift_vs_random)

    def test_saved_model_and_scaler_match_available_refit_data(self):
        for lag in (14,30):
            f=pd.read_csv(ROOT/f'results/lag{lag}/features.csv.gz')
            _,_,fit,test=split_features(f,lag)
            model=joblib.load(ROOT/f'results/lag{lag}/model.joblib')
            np.testing.assert_allclose(model[0].mean_,fit[FEATURES].mean(),rtol=1e-12,atol=1e-12)
            saved=pd.read_csv(ROOT/f'results/lag{lag}/scored_test.csv.gz')
            self.assertEqual(test.inspection_id.tolist(),saved.inspection_id.tolist())
            np.testing.assert_allclose(model.predict_proba(test[FEATURES])[:,1],saved.score,rtol=0,atol=1e-12)

    def test_quota_equal_remainders_use_declared_zip_tie_break(self):
        # Exact quotas are 0.6, 1.6, 7.8; a and b tie at remainder 0.6.
        # Binary floats can incorrectly make b's remainder larger than a's.
        g=pd.DataFrame({'zip_group':['a']*3+['b']*8+['c']*39,
                        'score':[.5]*50,'venue_id':[f'{i:03d}' for i in range(50)]})
        chosen=g.loc[select_indices(g,.2,'model',1.)]
        self.assertEqual(chosen.zip_group.value_counts().to_dict(),{'c':8,'a':1,'b':1})

    def test_budget_uses_exact_decimal_fraction(self):
        self.assertEqual(monthly_budget(100,.07),7)
        self.assertEqual(monthly_budget(100,.14),14)
        self.assertEqual(monthly_budget(101,.2),21)

if __name__=='__main__':
    unittest.main(verbosity=2)
