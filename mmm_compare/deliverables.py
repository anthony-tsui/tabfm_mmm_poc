"""Honest Meridian vs TabFM deliverables matrix.

TabFM is a tabular in-context learning regressor. It is not an MMM and does not
natively produce causal media mix deliverables. Approximations below are labeled
as Partial / No and must not be presented as Meridian-equivalent.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# Capability: Yes | Partial | No
DELIVERABLES: list[dict[str, str]] = [
    {
        "meridian_deliverable": "Expected outcome / predictive fit",
        "how_meridian_produces_it": (
            "Analyzer.expected_outcome / predictive_accuracy — Bayesian posterior "
            "expected KPI given media, controls, and model structure (adstock, Hill, etc.)."
        ),
        "tabfm_equivalent": "Yes",
        "tabfm_notes": (
            "TabFMRegressor can predict a continuous KPI from tabular media/control "
            "features via ICL. Comparable only as a predictive score (RMSE/MAE/R²), "
            "not as a structural MMM fit."
        ),
    },
    {
        "meridian_deliverable": "Channel contribution (incremental outcome)",
        "how_meridian_produces_it": (
            "Analyzer.incremental_outcome — counterfactual E(Y|treatment) − E(Y|no treatment) "
            "under the fitted Bayesian MMM."
        ),
        "tabfm_equivalent": "Partial",
        "tabfm_notes": (
            "No native incremental contribution. Optional leave-one-channel ablation "
            "on predictions is a hacky sensitivity proxy — NOT Meridian-equivalent "
            "and not causal."
        ),
    },
    {
        "meridian_deliverable": "ROI / effectiveness by channel",
        "how_meridian_produces_it": (
            "Analyzer.roi / summary_metrics — incremental outcome ÷ spend with posterior uncertainty."
        ),
        "tabfm_equivalent": "No",
        "tabfm_notes": (
            "TabFM has no spend→outcome structural ROI. Dividing ablation deltas by spend "
            "would be misleading and is not implemented as a Meridian-equivalent deliverable."
        ),
    },
    {
        "meridian_deliverable": "Response curves (diminishing returns)",
        "how_meridian_produces_it": (
            "Analyzer.response_curves — Hill/saturation response under historical flighting "
            "with spend multipliers."
        ),
        "tabfm_equivalent": "No",
        "tabfm_notes": (
            "Zero-shot tabular regression has no adstock/Hill media transform or native "
            "response-curve API."
        ),
    },
    {
        "meridian_deliverable": "Budget optimization / scenario planning",
        "how_meridian_produces_it": (
            "meridian.analysis.optimizer — allocate budget under response curves / ROI constraints."
        ),
        "tabfm_equivalent": "No",
        "tabfm_notes": (
            "No optimizer. Any grid-search over TabFM predictions would be an ad-hoc experiment, "
            "not Meridian budget optimization."
        ),
    },
    {
        "meridian_deliverable": "Geo-level insights",
        "how_meridian_produces_it": (
            "Geo hierarchical Meridian model + Analyzer with selected_geos / geo aggregation flags."
        ),
        "tabfm_equivalent": "Partial",
        "tabfm_notes": (
            "TabFM can take geo-week rows as tabular features and predict KPI per row, but it does "
            "not provide hierarchical geo shrinkage, geo ROI, or Meridian geo posteriors."
        ),
    },
]


def deliverables_dataframe() -> pd.DataFrame:
    return pd.DataFrame(DELIVERABLES)[
        [
            "meridian_deliverable",
            "how_meridian_produces_it",
            "tabfm_equivalent",
            "tabfm_notes",
        ]
    ]


def deliverables_markdown() -> str:
    lines = [
        "| Meridian deliverable | How Meridian produces it | TabFM equivalent? | If Partial / notes |",
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
            "TabFM can match Meridian on predictive KPI scoring only. "
            "Contribution/ROI/response curves/budget optimization are MMM-structural "
            "and are not native TabFM deliverables."
        ),
        "rows": DELIVERABLES,
    }
