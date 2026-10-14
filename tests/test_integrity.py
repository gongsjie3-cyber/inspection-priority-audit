"""Tests target temporal leakage, identities, budgets and cross-engine agreement."""
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
import unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pipeline import FEATURES, clean, build_features, select_indices, POLICIES

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

if __name__=='__main__':
    unittest.main(verbosity=2)
