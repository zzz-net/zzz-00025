from typing import List, Dict, Any, Set, Optional
import pandas as pd


REQUIRED_COLUMNS: List[str] = ["title", "content", "channel", "tags"]

SUPPORTED_CHANNELS: Set[str] = {
    "email",
    "phone",
    "web",
    "app",
    "wechat",
    "weibo",
}

KNOWN_LABELS: Set[str] = {
    "billing",
    "technical_support",
    "account_management",
    "product_inquiry",
    "feature_request",
    "bug_report",
    "complaint",
    "praise",
    "general_question",
    "refund_request",
    "cancellation",
    "upgrade",
    "downgrade",
    "security",
    "accessibility",
}


def validate_csv_structure(df: pd.DataFrame) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        errors.append(f"缺少必要列: {', '.join(missing_columns)}")

    extra_columns = [col for col in df.columns if col not in REQUIRED_COLUMNS]
    if extra_columns:
        warnings.append(f"存在额外列: {', '.join(extra_columns)}")

    return {
        "errors": errors,
        "warnings": warnings,
    }


def count_empty_labels(df: pd.DataFrame) -> int:
    if "tags" not in df.columns:
        return 0

    empty_count = 0
    for tags_str in df["tags"]:
        if pd.isna(tags_str) or not str(tags_str).strip():
            empty_count += 1
        else:
            tags = [t.strip() for t in str(tags_str).split(",")]
            tags = [t for t in tags if t]
            if len(tags) == 0:
                empty_count += 1

    return empty_count


def count_unknown_labels(df: pd.DataFrame, known_labels: Optional[Set[str]] = None) -> int:
    if "tags" not in df.columns:
        return 0

    known = known_labels or KNOWN_LABELS
    unknown_count = 0

    for tags_str in df["tags"]:
        if pd.isna(tags_str) or not str(tags_str).strip():
            continue
        tags = [t.strip() for t in str(tags_str).split(",")]
        tags = [t for t in tags if t]
        for tag in tags:
            if tag not in known:
                unknown_count += 1

    return unknown_count


def validate_labels(df: pd.DataFrame, known_labels: Optional[Set[str]] = None) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    stats: Dict[str, Any] = {}

    if "tags" not in df.columns:
        errors.append("缺少 tags 列，无法验证标签")
        return {"errors": errors, "warnings": warnings, "stats": stats}

    known = known_labels or KNOWN_LABELS
    empty_count = 0
    unknown_count = 0
    unknown_labels_set: Set[str] = set()
    total_rows = len(df)

    for idx, tags_str in enumerate(df["tags"]):
        if pd.isna(tags_str) or not str(tags_str).strip():
            empty_count += 1
            errors.append(f"第 {idx + 2} 行: tags 为空")
            continue

        tags = [t.strip() for t in str(tags_str).split(",")]
        tags = [t for t in tags if t]

        if len(tags) == 0:
            empty_count += 1
            errors.append(f"第 {idx + 2} 行: tags 解析后为空")
            continue

        for tag in tags:
            if tag not in known:
                unknown_count += 1
                unknown_labels_set.add(tag)
                errors.append(f"第 {idx + 2} 行: 未知标签 '{tag}'")

    stats["total_rows"] = total_rows
    stats["empty_labels_count"] = empty_count
    stats["unknown_labels_count"] = unknown_count
    stats["unique_unknown_labels"] = sorted(list(unknown_labels_set))
    stats["empty_labels_ratio"] = empty_count / total_rows if total_rows > 0 else 0
    stats["unknown_labels_ratio"] = unknown_count / total_rows if total_rows > 0 else 0

    if empty_count > 0:
        warnings.append(f"发现 {empty_count} 条空标签记录，占比 {stats['empty_labels_ratio']:.2%}")

    if unknown_count > 0:
        warnings.append(
            f"发现 {unknown_count} 个未知标签（{len(unknown_labels_set)} 种），"
            f"占比 {stats['unknown_labels_ratio']:.2%}"
        )

    return {
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
    }


def validate_channels(df: pd.DataFrame, supported_channels: Optional[Set[str]] = None) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    stats: Dict[str, Any] = {}

    if "channel" not in df.columns:
        errors.append("缺少 channel 列，无法验证渠道")
        return {"errors": errors, "warnings": warnings, "stats": stats}

    supported = supported_channels or SUPPORTED_CHANNELS
    invalid_channels: Dict[str, List[int]] = {}
    channel_counts: Dict[str, int] = {}
    total_rows = len(df)

    for idx, channel in enumerate(df["channel"]):
        if pd.isna(channel):
            errors.append(f"第 {idx + 2} 行: channel 为空")
            invalid_channels.setdefault("NaN", []).append(idx + 2)
            continue

        channel_str = str(channel).strip()
        channel_counts[channel_str] = channel_counts.get(channel_str, 0) + 1

        if channel_str not in supported:
            errors.append(f"第 {idx + 2} 行: 不支持的渠道 '{channel_str}'")
            invalid_channels.setdefault(channel_str, []).append(idx + 2)

    stats["total_rows"] = total_rows
    stats["unique_channels"] = sorted(list(channel_counts.keys()))
    stats["channel_distribution"] = channel_counts
    stats["invalid_channel_count"] = sum(len(rows) for rows in invalid_channels.values())
    stats["invalid_channel_types"] = sorted(list(invalid_channels.keys()))

    if invalid_channels:
        warnings.append(
            f"发现 {stats['invalid_channel_count']} 条无效渠道记录，"
            f"涉及 {len(invalid_channels)} 种渠道类型"
        )

    return {
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
    }


def validate_data(
    df: pd.DataFrame,
    known_labels: Optional[Set[str]] = None,
    supported_channels: Optional[Set[str]] = None,
    fail_on_empty_labels: bool = True,
    fail_on_unknown_labels: bool = True,
    fail_on_invalid_channels: bool = True,
) -> Dict[str, Any]:
    all_errors: List[str] = []
    all_warnings: List[str] = []
    all_stats: Dict[str, Any] = {}

    structure_result = validate_csv_structure(df)
    all_errors.extend(structure_result["errors"])
    all_warnings.extend(structure_result["warnings"])

    if structure_result["errors"]:
        return {
            "is_valid": False,
            "errors": all_errors,
            "warnings": all_warnings,
            "stats": all_stats,
        }

    labels_result = validate_labels(df, known_labels)
    all_errors.extend(labels_result["errors"])
    all_warnings.extend(labels_result["warnings"])
    all_stats["labels"] = labels_result["stats"]

    channels_result = validate_channels(df, supported_channels)
    all_errors.extend(channels_result["errors"])
    all_warnings.extend(channels_result["warnings"])
    all_stats["channels"] = channels_result["stats"]

    all_stats["total_rows"] = len(df)
    all_stats["total_errors"] = len(all_errors)
    all_stats["total_warnings"] = len(all_warnings)

    is_valid = True

    if fail_on_empty_labels and labels_result["stats"].get("empty_labels_count", 0) > 0:
        is_valid = False

    if fail_on_unknown_labels and labels_result["stats"].get("unknown_labels_count", 0) > 0:
        is_valid = False

    if fail_on_invalid_channels and channels_result["stats"].get("invalid_channel_count", 0) > 0:
        is_valid = False

    if not all_errors:
        is_valid = is_valid and True
    else:
        is_valid = False

    return {
        "is_valid": is_valid,
        "errors": all_errors,
        "warnings": all_warnings,
        "stats": all_stats,
    }
