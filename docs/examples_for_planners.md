# For planners: can TabFM replace Meridian for annual budget & strategy?

**Short answer: No.** TabFM can forecast KPI. Meridian can recommend a mix.

Open these two images on your phone (same simulated Meridian demo data):

1. [Meridian annual mix story](../examples/01_meridian_annual_mix_story.png) — cut / scale with numbers  
2. [TabFM KPI + “stops here”](../examples/02_tabfm_kpi_stops_here.png) — forecast only  

Optional scroll strip: [one planner story](../examples/00_one_planner_story.png)

> Simulated / demo data only — not real campaigns. Tiny MCMC = directional, not decision-grade.

---

## The planner question

> “For next year’s marketing budget, what should we **cut** and what should we **scale**?”

### Meridian’s example answer (this run)

**Cut Channel3 by 15% → scale Channel2.**

| Why (this dummy run) | Number |
| --- | ---: |
| Channel3 ROI (lowest) | ~52.9 |
| Channel2 ROI (highest) | ~104.3 |
| Spend moved | ~$13.2M simulated |
| Illustrative incremental (ROI × spend scenario) | ~+0.68B |

Plain language: **Meridian says cut Channel3 / scale Channel2 — here’s why** (lowest vs highest ROI; contribution shown so you see Channel3 is big but inefficient).

This is a **labeled scenario** using Meridian ROI on official demo data — not a full production optimizer, and **not** advice for a real brand.

### TabFM’s example answer (same data)

TabFM draws a holdout KPI line (actual vs predicted).

Plain language: **TabFM only says KPI might look like X. It stops there.**

No contribution. No ROI. No cut vs scale. No annual strategy plan.

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
