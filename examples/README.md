# Planner examples (visual story)

**One story, two images** — not a metrics dump.

| # | File | Caption |
| --- | --- | --- |
| 1 | [01_meridian_annual_mix_story.png](01_meridian_annual_mix_story.png) | Cut Channel3 15% / scale Channel2 — lowest ROI (~53) vs highest (~104). That’s an annual-mix call. |
| 2 | [02_tabfm_kpi_stops_here.png](02_tabfm_kpi_stops_here.png) | KPI forecast only — planner stops here. |
| 3 | [03_curiosity_kpi_overlay.png](03_curiosity_kpi_overlay.png) | Similar KPI fit ≠ same model for annual mix. Overlay is curiosity only. |
| 4 | [00_one_planner_story.png](00_one_planner_story.png) | Optional side-by-side scroll strip |

Numbers: [`story_numbers.json`](story_numbers.json) · overlay preds from [`../results/metrics.json`](../results/metrics.json)

```bash
python scripts/generate_planner_examples.py
python scripts/generate_planner_examples.py --from-saved
python scripts/generate_curiosity_overlay.py
```

Story: [`../docs/examples_for_planners.md`](../docs/examples_for_planners.md)

> Official Meridian simulated/demo data only. Not real campaign performance.
