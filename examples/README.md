# Planner examples (phone-first)

**One story, two images** — not a metrics dump.

| # | File | Caption |
| --- | --- | --- |
| 1 | [01_meridian_annual_mix_story.png](01_meridian_annual_mix_story.png) | Cut Channel3 15% / scale Channel2 — lowest ROI (~53) vs highest (~104). That’s an annual-mix call. |
| 2 | [02_tabfm_kpi_stops_here.png](02_tabfm_kpi_stops_here.png) | Same data: KPI forecast only. No contribution, ROI, or cut/scale — planner stops here. |
| 3 | [00_one_planner_story.png](00_one_planner_story.png) | Optional side-by-side scroll strip |

Numbers: [`story_numbers.json`](story_numbers.json)

```bash
python scripts/generate_planner_examples.py
# Caption-only redraw from saved numbers (no Meridian refit):
python scripts/generate_planner_examples.py --from-saved
```

Story: [`../docs/examples_for_planners.md`](../docs/examples_for_planners.md)

> Official Meridian simulated/demo data only. Not real campaign performance.
