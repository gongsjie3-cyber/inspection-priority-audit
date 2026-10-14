-- SQLite. This query independently verifies the Python point-in-time join.
-- Same-month outcomes and events at the exact cutoff are excluded.
SELECT t.inspection_id, t.n_history AS python_count,
       COUNT(h.inspection_id) AS sql_count
FROM monthly_features AS t
LEFT JOIN inspections AS h
  ON h.venue_id = t.venue_id
 AND h.date < t.history_cutoff_exclusive
GROUP BY t.inspection_id, t.n_history
HAVING COUNT(h.inspection_id) <> t.n_history;
