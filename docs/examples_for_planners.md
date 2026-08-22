# For planners: can TabFM replace Meridian for annual budget & strategy?

**Short answer: No.** TabFM can forecast KPI. Meridian can recommend a mix.

Open these images on your phone (same simulated Meridian demo data):

1. [Meridian annual mix story](../examples/01_meridian_annual_mix_story.png)  
2. [TabFM KPI + planner stops here](../examples/02_tabfm_kpi_stops_here.png)  
3. *(Curiosity only)* [Actual vs Meridian vs TabFM KPI overlay](../examples/03_curiosity_kpi_overlay.png) — same holdout 2023-01-23 → 2024-01-15

Optional scroll strip: [one planner story](../examples/00_one_planner_story.png)

> Simulated / demo data only — not real campaigns. Tiny MCMC = directional, not decision-grade.
>
> **Similar KPI fit ≠ same model for annual mix. Overlay is curiosity only.** It does not change the annual-mix success bar.

---

## The planner question

> “For next year’s marketing budget, what should we **cut** and what should we **scale**?”

### Meridian’s example answer (this run)

**Cut Channel3 15% / scale Channel2 — lowest ROI (~53) vs highest (~104). That’s an annual-mix call.**

| Detail (this dummy run) | Number |
| --- | ---: |
| Channel3 ROI (lowest) | ~53 |
| Channel2 ROI (highest) | ~104 |
| Spend moved | ~$13.2M simulated |
| Illustrative incremental (ROI × spend scenario) | ~+0.68B |

Contribution is shown so you can see Channel3 is large but inefficient — the mix call is driven by ROI, not volume alone.

This is a **labeled scenario** using Meridian ROI on official demo data — not a full production optimizer, and **not** advice for a real brand.

### TabFM’s example answer (same data)

**Same data: KPI forecast only. No contribution, ROI, or cut/scale — planner stops here.**

TabFM draws a holdout KPI line (actual vs predicted). That is all.

---

## 3 more planner Qs (same story)

| You ask | Meridian (this PoC) | TabFM |
| --- | --- | --- |
| What drove outcomes? | Channel contribution chart | Can’t answer |
| Which channels are efficient? | ROI chart | Can’t answer |
| What might KPI be next? | Can forecast too | **Yes — this is all it does** |

---

## Blunt takeaway

- **Meridian** can feed **annual marketing budget + strategy recommendations** (contribution → ROI → cut/scale).
- **TabFM** is **not** an MMM alternative for that job unless you add another causal/MMM layer.
- We do **not** fake TabFM budget recs. We do **not** use SHAP/ablation as “contribution.”

Checklist one-pager: [`decisioning_checklist.md`](decisioning_checklist.md)
