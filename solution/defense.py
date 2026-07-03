"""
Your defense. Implement register(ctx) and a handler per event type.
See ../README.md for the full interface + toolkit reference, and
../RULES.md before you start.
"""
from api import Verdict


def register(ctx):
    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)


def _verdict(alert, pillar, reason="", confidence=1.0):
    return Verdict(alert=alert, pillar=pillar, reason=reason, confidence=confidence)


def _tool_error(result):
    return isinstance(result, dict) and "error" in result


def _truthy_collection(value):
    if value in (None, "", [], (), {}, set(), frozenset()):
        return False
    try:
        return len(value) > 0
    except TypeError:
        return bool(value)


def _collection_size(value):
    try:
        return len(value)
    except TypeError:
        return 1 if value else 0


def _lineage_profile(ctx):
    return ctx.state.setdefault(
        "lineage_profile",
        {"upstream_sizes": [], "downstream_counts": []},
    )


def _median(values):
    if not values:
        return 0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _severity_confidence(count):
    base = 0.55 + 0.1 * count
    return min(0.95, base)


def check_data_batch(payload, ctx):
    result = ctx.tools.batch_profile(payload["batch_id"])
    if _tool_error(result):
        return _verdict(False, "checks", result["error"], 0.0)

    baseline = ctx.baseline
    anomalies = []

    row_count = result.get("row_count")
    if row_count is not None and (
        row_count < baseline["row_count_min"] or row_count >= baseline["row_count_max"] * 0.985
    ):
        anomalies.append(f"row_count={row_count}")

    null_rate = result.get("null_rate", {}).get("customer_id")
    if null_rate is not None and null_rate >= baseline["null_rate_max"]:
        anomalies.append(f"null_rate={null_rate}")

    mean_amount = result.get("mean_amount")
    if mean_amount is not None and (
        mean_amount < baseline["mean_amount_min"]
        or mean_amount >= baseline["mean_amount_max"] * 0.98
    ):
        anomalies.append(f"mean_amount={mean_amount}")

    std_amount = result.get("std_amount")
    if std_amount is not None and mean_amount is not None and mean_amount:
        spread = std_amount / mean_amount
        if spread >= 0.75:
            anomalies.append(f"std_amount={std_amount}")

    staleness_min = result.get("staleness_min")
    if staleness_min is not None and staleness_min >= baseline["staleness_min_max"] * 0.65:
        anomalies.append(f"staleness_min={staleness_min}")

    if anomalies:
        return _verdict(
            True,
            "checks",
            "data_batch anomaly: " + ", ".join(anomalies),
            _severity_confidence(len(anomalies)),
        )
    return _verdict(False, "checks", "within baseline", 0.35)


def check_contract_checkpoint(payload, ctx):
    result = ctx.tools.contract_diff(payload["contract_id"], payload["checkpoint_batch_id"])
    if _tool_error(result):
        return _verdict(False, "contracts", result["error"], 0.0)

    baseline = ctx.baseline
    violations = list(result.get("violations", []))
    freshness_delay = result.get("freshness_delay_min")
    if freshness_delay is not None and freshness_delay >= baseline["freshness_delay_max_min"]:
        violations.append(f"freshness_delay={freshness_delay}")

    if violations:
        return _verdict(
            True,
            "contracts",
            "contract anomaly: " + ", ".join(violations),
            _severity_confidence(len(violations)),
        )
    return _verdict(False, "contracts", "within baseline", 0.35)


def check_lineage_run(payload, ctx):
    result = ctx.tools.lineage_graph_slice(payload["run_id"])
    if _tool_error(result):
        return _verdict(False, "lineage", result["error"], 0.0)

    baseline = ctx.baseline
    profile = _lineage_profile(ctx)
    anomalies = []

    upstream = result.get("actual_upstream")
    upstream_size = _collection_size(upstream)
    if not _truthy_collection(upstream):
        anomalies.append("missing_upstream")
    elif upstream_size < 2:
        anomalies.append(f"missing_upstream={upstream_size}<2")
    elif len(profile["upstream_sizes"]) >= 2:
        typical_upstream = _median(profile["upstream_sizes"])
        if typical_upstream >= 2 and upstream_size < typical_upstream:
            anomalies.append(f"missing_upstream={upstream_size}<typical{typical_upstream}")

    downstream_count = result.get("actual_downstream_count")
    if downstream_count == 0:
        anomalies.append("orphan_output")
    elif len(profile["downstream_counts"]) >= 2:
        typical_downstream = _median(profile["downstream_counts"])
        if typical_downstream >= 2 and downstream_count < typical_downstream:
            anomalies.append(
                f"orphan_output={downstream_count}<typical{typical_downstream}"
            )

    duration_ms = result.get("duration_ms")
    if duration_ms is not None and duration_ms >= baseline["lineage_duration_ms_max"] * 0.85:
        anomalies.append(f"runtime={duration_ms}")

    profile["upstream_sizes"].append(upstream_size)
    if downstream_count is not None:
        profile["downstream_counts"].append(downstream_count)

    if anomalies:
        return _verdict(
            True,
            "lineage",
            "lineage anomaly: " + ", ".join(anomalies),
            _severity_confidence(len(anomalies)),
        )
    return _verdict(False, "lineage", "within baseline", 0.35)


def check_feature_materialization(payload, ctx):
    result = ctx.tools.feature_drift(payload["feature_view"], payload["batch_id"])
    if _tool_error(result):
        return _verdict(False, "ai_infra", result["error"], 0.0)

    mean_shift_sigma = result.get("mean_shift_sigma")
    feature_threshold = ctx.baseline["feature_mean_shift_sigma_max"]
    if mean_shift_sigma is not None and mean_shift_sigma >= feature_threshold:
        return _verdict(
            True,
            "ai_infra",
            f"feature drift sigma={mean_shift_sigma}",
            min(0.95, 0.65 + 0.1 * (mean_shift_sigma / feature_threshold)),
        )
    return _verdict(False, "ai_infra", "within baseline", 0.35)


def check_embedding_batch(payload, ctx):
    result = ctx.tools.embedding_drift(payload["corpus"], payload["chunk_batch_id"])
    if _tool_error(result):
        return _verdict(False, "ai_infra", result["error"], 0.0)

    baseline = ctx.baseline
    anomalies = []

    centroid_shift = result.get("centroid_shift")
    if centroid_shift is not None and centroid_shift >= baseline["embedding_centroid_shift_max"] * 0.6:
        anomalies.append(f"centroid_shift={centroid_shift}")

    avg_doc_age_days = result.get("avg_doc_age_days")
    if avg_doc_age_days is not None and avg_doc_age_days >= baseline["corpus_avg_doc_age_days_max"] * 0.7:
        anomalies.append(f"avg_doc_age_days={avg_doc_age_days}")

    if anomalies:
        return _verdict(
            True,
            "ai_infra",
            "embedding anomaly: " + ", ".join(anomalies),
            _severity_confidence(len(anomalies)),
        )
    return _verdict(False, "ai_infra", "within baseline", 0.35)
