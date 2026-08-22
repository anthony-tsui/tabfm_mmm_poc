# Decisioning checklist (phone one-pager)

**Bar:** Can this support **annual marketing budget + strategy recommendations**?

**Lead:** TabFM predicts KPI; it does **not** replace Meridian for contribution / ROI / curves / budget opt.

Official Meridian **simulated/demo** data only — not real campaign numbers. Tiny MCMC in this PoC is directional, not decision-grade.

| Planner question | Meridian answer | TabFM answer | Notes |
| --- | --- | --- | --- |
| What will KPI look like next period? (forecast) | Yes — expected outcome / predictive fit | **Yes** — holdout KPI prediction | Only place TabFM competes. Fit ≠ decisioning. |
| Channel contribution — what drove the outcome? | Yes — incremental outcome | **No** | Needs a causal/MMM layer. Ablation hacks ≠ Meridian. |
| Channel ROI / efficiency? | Yes — ROI / summary metrics | **No** | Meridian-only product surface. |
| Response / diminishing returns? | Yes — response curves | **No** | No Hill/adstock response API in TabFM. |
| What to cut vs scale (floors/ceilings)? | Yes — optimizer / scenarios | **No** | Budget opt is Meridian (or another MMM). |
| Annual mix / next-year strategy recommendation? | **Yes — can feed the annual-plan path** (with calibrated priors + decision-grade chains) | **No** | TabFM stops at predictive KPI unless an extra causal/MMM layer exists. |
| Small-data MMM decisioning (priors, uncertainty)? | Designed for it (Bayesian priors, posteriors) | **No** | TabFM ICL ≠ calibrated MMM uncertainty for spend decisions. |

### Blunt takeaway

- **Meridian** can feed annual budget + strategy recommendations (contribution → ROI → curves → opt).
- **TabFM** stops at “what might KPI be?” — useful signal, not a media plan.
- Meridian-as-teacher distillation into TabFM: **out of scope** (Someday).

**See the pictures:** [`examples_for_planners.md`](examples_for_planners.md)

→ Full gap matrix: [`../results/deliverables_matrix.md`](../results/deliverables_matrix.md)
