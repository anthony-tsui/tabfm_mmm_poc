"""Honest Meridian vs TabFM deliverables matrix.

Predictive KPI is the only Yes. Contribution / ROI / curves / budget / geo
decisioning are Meridian-only product surface (hard No for TabFM). No proxy stretching.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

DELIVERABLES: list[dict[str, str]] = [
    {
        "meridian_deliverable": "Expected outcome / predictive KPI fit",
        "how_meridian_produces_it": (
            "ModelSpec.holdout_id + Analyzer.expected_outcome / predictive_accuracy — "
            "Bayesian posterior expected KPI; holdout KPI excluded from training."
        ),
        "tabfm_equivalent": "Yes",
        "tabfm_notes": (
            "TabFMRegressor can score holdout KPI from media/control features via ICL. "
            "Comparable only as predictive fit — not media decisioning."
        ),
    },
    {
        "meridian_deliverable": "Channel contribution (incremental outcome)",
        "how_meridian_produces_it": (
            "Analyzer.incremental_outcome — structural counterfactual under the fitted MMM."
        ),
        "tabfm_equivalent": "No",
        "tabfm_notes": (
            "Meridian-only. TabFM has no incremental_outcome. Optional --ablation-proxy is a "
            "sensitivity hack — NOT Meridian-equivalent, not causal, excluded from headline results."
        ),
    },
    {
        "meridian_deliverable": "ROI / effectiveness by channel",
        "how_meridian_produces_it": "Analyzer.roi / summary_metrics",
        "tabfm_equivalent": "No",
        "tabfm_notes": "Meridian-only product surface. No TabFM structural ROI.",
    },
    {
        "meridian_deliverable": "Response curves (diminishing returns)",
        "how_meridian_produces_it": "Analyzer.response_curves (Hill/saturation under flighting)",
        "tabfm_equivalent": "No",
        "tabfm_notes": "Meridian-only. TabFM has no adstock/Hill response API.",
    },
    {
        "meridian_deliverable": "Budget optimization / scenario planning",
        "how_meridian_produces_it": "meridian.analysis.optimizer",
        "tabfm_equivalent": "No",
        "tabfm_notes": "Meridian-only. No TabFM budget optimizer.",
    },
    {
        "meridian_deliverable": "Geo-level media decisioning",
        "how_meridian_produces_it": "Geo hierarchical Meridian + Analyzer geo views",
        "tabfm_equivalent": "No",
        "tabfm_notes": (
            "Meridian-only for geo ROI/contribution. TabFM may score geo-week rows as tabular "
            "prediction, but that is not hierarchical geo MMM decisioning."
        ),
    },
]


def deliverables_dataframe() -> pd.DataFrame:
    return pd.DataFrame(DELIVERABLES)[
        ["meridian_deliverable", "how_meridian_produces_it", "tabfm_equivalent", "tabfm_notes"]
    ]


def deliverables_markdown() -> str:
    lines = [
        "| Meridian deliverable | How Meridian produces it | TabFM equivalent? | Notes |",
        "| --- | --- | --- | --- |",
    ]
    for row in DELIVERABLES:
        lines.append(
            "| {meridian_deliverable} | {how_meridian_produces_it} | **{tabfm_equivalent}** | {tabfm_notes} |".format(
                **row
            )
        )
    return "\n".join(lines)


def capability_summary() -> dict[str, Any]:
    counts: dict[str, int] = {"Yes": 0, "Partial": 0, "No": 0}
    for row in DELIVERABLES:
        counts[row["tabfm_equivalent"]] = counts.get(row["tabfm_equivalent"], 0) + 1
    return {
        "counts": counts,
        "headline": (
            "Practitioner takeaway: predictive fit ≠ media decisioning. "
            "TabFM can match Meridian on holdout KPI prediction only. "
            "Contribution, ROI, response curves, budget optimization, and geo decisioning "
            "are Meridian-only (hard No for TabFM)."
        ),
        "rows": DELIVERABLES,
    }
