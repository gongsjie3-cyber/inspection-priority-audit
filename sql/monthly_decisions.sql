-- Reproduce unconstrained top-20% selection with a SQL window function.
WITH ranked AS (
  SELECT *, ROW_NUMBER() OVER (
    PARTITION BY month ORDER BY score DESC, venue_id ASC
  ) AS rank_in_month,
  COUNT(*) OVER (PARTITION BY month) AS n_in_month
  FROM scored_test
)
SELECT month, COUNT(*) AS selected, SUM(y) AS hits,
       1.0 * SUM(y) / COUNT(*) AS precision
FROM ranked
WHERE rank_in_month <= (n_in_month + 4) / 5
GROUP BY month
ORDER BY month;
