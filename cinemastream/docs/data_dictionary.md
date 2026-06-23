# CinemaStream Data Dictionary

Created Chapter 54 (Data Contracts & API-First Data Sharing). Grows as new
contracts and tables are introduced. Contracts themselves live in
`cinemastream/dbt_cinemastream/contracts/` as versioned JSON files.

---

## watch_events contract v2 (in migration, both versions valid until 2026-06-24)

Producer: backend team (owner: Carlos Reyes). Consumer: analytics warehouse
(dbt marts, owner: data team).

- `completed` (bool, **DEPRECATED** — remove after 2026-06-24): `True`/`False`, legacy field.
- `completed_status` (str, **NEW**): `not_started` | `in_progress` | `completed` — use this in all new models.
- `subtitle_lang` (str, nullable, **NEW**): ISO language code (`en`, `ms`, `id`, `tl`, `th`, `vi`, `hi`, `ta`) or `null` if no subtitles selected.
- All marts referencing `completed = TRUE` must migrate to `completed_status = 'completed'` before 2026-06-24.

Full machine-readable specs: `cinemastream/dbt_cinemastream/contracts/watch_events_v1.json` and `watch_events_v2.json`.

### Migration tracking

| Model / dashboard | Uses `completed = TRUE`? | Migrated to `completed_status`? |
|---|---|---|
| `mart_monthly_revenue` (Ch 49) | Yes | Pending |
| Executive completion-rate widget | Yes | Pending |

Run `grep -r "completed = " cinemastream/dbt_cinemastream/models/` before 2026-06-24 to confirm this table is complete.
