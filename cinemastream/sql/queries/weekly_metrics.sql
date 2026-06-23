-- CinemaStream Weekly Business Review Metrics
-- Chapter 38 — Writing Queries for Real Datasets
-- Dialect: PostgreSQL (use strftime equivalents for SQLite)
-- Author: Data Team
-- Last updated: 2024-01-01

WITH

-- Subscriber health by plan
subscriber_health AS (
    SELECT
        plan,
        COUNT(*)                                        AS total,
        SUM(churned::int)                               AS churned,
        ROUND(100.0 * SUM(churned::int) / COUNT(*), 1) AS churn_rate_pct,
        COUNT(*) - SUM(churned::int)                    AS active
    FROM users
    GROUP BY plan
),

-- Monthly recurring revenue (active subscribers only)
monthly_revenue AS (
    SELECT
        ROUND(SUM(
            CASE plan
                WHEN 'Premium' THEN 12.90
                WHEN 'Basic'   THEN  8.90
                ELSE 0.0
            END
        ), 2)  AS total_mrr_sgd
    FROM users
    WHERE churned = FALSE
),

-- Watch engagement summary
watch_summary AS (
    SELECT
        COUNT(DISTINCT user_id)             AS active_viewers,
        COUNT(*)                            AS total_sessions,
        ROUND(AVG(watch_minutes), 1)        AS avg_session_min,
        ROUND(100.0 * SUM(completed::int) / COUNT(*), 1) AS completion_pct
    FROM watch_events
),

-- Top content this period
top_movie AS (
    SELECT
        m.title,
        COUNT(*) AS watches
    FROM   watch_events we
    JOIN   movies m ON we.movie_id = m.movie_id
    GROUP BY m.movie_id, m.title
    ORDER BY watches DESC
    LIMIT 1
),

-- Inactive paying users — revenue at risk
inactive_paid AS (
    SELECT COUNT(*) AS count
    FROM      users u
    LEFT JOIN watch_events we ON u.user_id = we.user_id
    WHERE     u.plan IN ('Basic', 'Premium')
      AND     u.churned = FALSE
      AND     we.event_id IS NULL
)

-- Final assembled metrics
SELECT 'Subscriber Health' AS metric_group, 'Active Premium' AS metric,
    (SELECT active FROM subscriber_health WHERE plan = 'Premium')::text AS value
UNION ALL SELECT 'Subscriber Health', 'Active Basic',
    (SELECT active FROM subscriber_health WHERE plan = 'Basic')::text
UNION ALL SELECT 'Subscriber Health', 'Active Free',
    (SELECT active FROM subscriber_health WHERE plan = 'Free')::text
UNION ALL SELECT 'Revenue', 'Monthly MRR (S$)',
    (SELECT total_mrr_sgd FROM monthly_revenue)::text
UNION ALL SELECT 'Engagement', 'Active Viewers',
    (SELECT active_viewers FROM watch_summary)::text
UNION ALL SELECT 'Engagement', 'Avg Session (min)',
    (SELECT avg_session_min FROM watch_summary)::text
UNION ALL SELECT 'Engagement', 'Completion Rate (%)',
    (SELECT completion_pct FROM watch_summary)::text
UNION ALL SELECT 'Content', 'Top Movie',
    (SELECT title FROM top_movie)
UNION ALL SELECT 'Risk', 'Inactive Paying Users',
    (SELECT count FROM inactive_paid)::text;
