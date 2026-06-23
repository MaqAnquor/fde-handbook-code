# CinemaStream Data Team Runbook

This runbook collects "if you get this alert, here's what to check first, second,
third" entries for the on-call data engineer. It grows as pipelines, alerts, and
incidents accumulate — every postmortem's action items should result in either a
new entry here or an update to an existing one.

**Severity scale (Chapter 57):**
- **SEV1** — data loss, or revenue-impacting with no workaround. Page immediately, 15-min response SLA.
- **SEV2** — >=10% of users affected, or revenue-impacting with a workaround. Page, 60-min response SLA.
- **SEV3** — <10% of users affected but no workaround, or 100% affected with a workaround. No page, 4-hour business-hours SLA.
- **SEV4** — cosmetic / minor. No page, next-business-day SLA.

Classify with `classify_severity()` (see Chapter 57, `cinemastream/docs/runbook.md`
links to the chapter for the full function).

---

## On-Call Quick Reference

- Rotation is weekly, tracked informally in `#data-team` pinned message.
- All schedules and SLAs in this runbook are stated in **UTC**. CinemaStream HQ is
  SGT (UTC+8) — convert before assuming "2am" means the same thing to everyone.
- When you acknowledge a page, post in `#data-team` immediately, even before you
  know the cause — this starts the incident timeline (Chapter 57, Section 2.2).
- After resolving a SEV1/SEV2 (and ideally any recurring SEV3/SEV4), write a
  blameless postmortem using the template in Chapter 57, Section 2.3, and link it
  from the relevant entry below.

---

## Alert: `watch_events_ingestion` — 0 rows loaded

**Introduced:** Chapter 52 (original threshold: 0 rows for 8h, daily schedule
assumption). **Updated:** Chapter 57 (threshold tightened to 0 rows for 2h,
pipeline now runs hourly).

1. Check the Chapter 52 monitoring dashboard — did the pipeline report "success"
   with 0 rows, or did it fail outright? "Success with 0 rows" is the dangerous
   case (see Chapter 52's "Success is not the same as Correct" pitfall).
2. If the file/API response is empty or missing, page backend on-call (Carlos) —
   likely an upstream service deploy issue.
3. Compare the upstream response shape against the Chapter 54 contract
   (`cinemastream/dbt_cinemastream/contracts/`). Look specifically for renamed or
   removed required fields — a missing required field is the most common cause
   (see the 2026-05-22 postmortem, linked below).
4. If a field was renamed: apply a mapping shim in the extract step, then backfill
   the affected window using the Chapter 55 incremental-load pattern
   (re-run extract for the specific `watch_started` range).
5. **Do not** treat "validator passed but field is missing" as safe — the contract
   validator should fail loudly on missing required fields. If it doesn't, that's
   itself a bug to fix (see action items below).

**Linked postmortems:**
- 2026-05-22 — `watch_events` 0 rows for 8h, caused by upstream rename
  `watch_minutes` -> `duration_seconds`. SEV2. See Chapter 57, Section 2.3 for the
  full postmortem document.

---

## Alert: Country-level metric anomaly (Ch 53 observability checks)

**Introduced:** Chapter 53.

1. Identify which country and which metric (null-rate spike, distribution shift,
   schema drift) triggered the check.
2. Page backend on-call (Carlos) with the affected country and the report JSON.
3. Cross-check whether the same upstream deploy also affected `watch_events_ingestion`
   (see entry above) — schema changes often affect multiple downstream checks at once.

---

## Cost alert: monthly budget threshold breached (Ch 56 FinOps)

**Introduced:** Chapter 56.

1. Run `compare_to_baseline()` (Chapter 56) to see which pipeline/query's cost
   changed and by how much.
2. Check recent postmortems and Chapter 55/56 baseline-update notes — a cost
   increase may already be a *known, deliberate* consequence of a prior fix
   (e.g., the Chapter 55 lookback-window change). If so, this alert is expected;
   update the budget baseline rather than treating it as an incident.
3. If the increase is unexplained, treat it as a SEV3/SEV4 (no page) and
   investigate during business hours using `diagnose_high_cost_query()`.

---

## Open Action Items (from postmortems)

- [x] Lower row-count alert threshold to "0 rows for 2h" for hourly pipelines —
      done, Chapter 57.
- [x] Make contract validator raise (not silently skip) when a required field is
      missing — done, Chapter 57.
- [ ] Add data team to `#backend-deploys` Slack channel for advance notice of API
      changes — owner: Carlos, due 2026-05-23 (see Chapter 57 postmortem).
