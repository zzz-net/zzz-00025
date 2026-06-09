import os
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Set

import pandas as pd

from config import DATASET_DIR, SUPPORTED_CHANNELS
from .database import db
from .models import Dataset
from .data_validator import validate_data


os.makedirs(DATASET_DIR, exist_ok=True)


def generate_timestamp_filename(prefix: str = "dataset", extension: str = "csv") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    random_suffix = uuid.uuid4().hex[:6]
    return f"{prefix}_{timestamp}_{random_suffix}.{extension}"


def save_dataframe_to_csv(df: pd.DataFrame, filename: Optional[str] = None) -> str:
    if filename is None:
        filename = generate_timestamp_filename()
    file_path = os.path.join(DATASET_DIR, filename)
    df.to_csv(file_path, index=False, encoding="utf-8-sig")
    return file_path


def import_csv(
    csv_path: str,
    app,
    known_labels: Optional[Set[str]] = None,
    supported_channels: Optional[Set[str]] = None,
    fail_on_empty_labels: bool = True,
    fail_on_unknown_labels: bool = True,
    fail_on_invalid_channels: bool = True,
    save_invalid: bool = True,
) -> Dict[str, Any]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    if supported_channels is None:
        supported_channels = set(SUPPORTED_CHANNELS)

    validation_result = validate_data(
        df,
        known_labels=known_labels,
        supported_channels=supported_channels,
        fail_on_empty_labels=fail_on_empty_labels,
        fail_on_unknown_labels=fail_on_unknown_labels,
        fail_on_invalid_channels=fail_on_invalid_channels,
    )

    is_valid = validation_result["is_valid"]
    error_count = validation_result["stats"].get("total_errors", 0)
    warning_count = validation_result["stats"].get("total_warnings", 0)
    row_count = len(df)

    status = 'completed' if is_valid else 'failed'
    error_msg = None
    if not is_valid:
        error_msg = f"Errors: {error_count}, Warnings: {warning_count}"

    with app.app_context():
        if not is_valid and not save_invalid:
            dataset = Dataset(
                name=os.path.splitext(os.path.basename(csv_path))[0],
                file_path="",
                row_count=row_count,
                status=status,
                error_message=error_msg
            )
            db.session.add(dataset)
            db.session.commit()
            return {
                "dataset_id": dataset.id,
                "is_valid": False,
                "saved": False,
                "validation_result": validation_result,
            }

        filename = generate_timestamp_filename()
        file_path = save_dataframe_to_csv(df, filename)

        dataset = Dataset(
            name=os.path.splitext(os.path.basename(csv_path))[0],
            file_path=file_path,
            row_count=row_count,
            status=status,
            error_message=error_msg
        )
        db.session.add(dataset)
        db.session.commit()

        return {
            "dataset_id": dataset.id,
            "is_valid": is_valid,
            "saved": True,
            "file_path": file_path,
            "row_count": row_count,
            "validation_result": validation_result,
        }


def import_csv_from_dataframe(
    df: pd.DataFrame,
    dataset_name: str,
    app,
    known_labels: Optional[Set[str]] = None,
    supported_channels: Optional[Set[str]] = None,
    fail_on_empty_labels: bool = True,
    fail_on_unknown_labels: bool = True,
    fail_on_invalid_channels: bool = True,
    save_invalid: bool = True,
) -> Dict[str, Any]:
    if supported_channels is None:
        supported_channels = set(SUPPORTED_CHANNELS)

    validation_result = validate_data(
        df,
        known_labels=known_labels,
        supported_channels=supported_channels,
        fail_on_empty_labels=fail_on_empty_labels,
        fail_on_unknown_labels=fail_on_unknown_labels,
        fail_on_invalid_channels=fail_on_invalid_channels,
    )

    is_valid = validation_result["is_valid"]
    error_count = validation_result["stats"].get("total_errors", 0)
    warning_count = validation_result["stats"].get("total_warnings", 0)
    row_count = len(df)

    status = 'completed' if is_valid else 'failed'
    error_msg = None
    if not is_valid:
        error_msg = f"Errors: {error_count}, Warnings: {warning_count}"

    with app.app_context():
        if not is_valid and not save_invalid:
            dataset = Dataset(
                name=dataset_name,
                file_path="",
                row_count=row_count,
                status=status,
                error_message=error_msg
            )
            db.session.add(dataset)
            db.session.commit()
            return {
                "dataset_id": dataset.id,
                "is_valid": False,
                "saved": False,
                "validation_result": validation_result,
            }

        filename = generate_timestamp_filename(prefix=dataset_name)
        file_path = save_dataframe_to_csv(df, filename)

        dataset = Dataset(
            name=dataset_name,
            file_path=file_path,
            row_count=row_count,
            status=status,
            error_message=error_msg
        )
        db.session.add(dataset)
        db.session.commit()

        return {
            "dataset_id": dataset.id,
            "is_valid": is_valid,
            "saved": True,
            "file_path": file_path,
            "row_count": row_count,
            "validation_result": validation_result,
        }


def list_datasets(app, limit: int = 100) -> list:
    with app.app_context():
        datasets = Dataset.query.order_by(Dataset.created_at.desc()).limit(limit).all()
        return [dataset.to_dict() for dataset in datasets]


def get_dataset(dataset_id: int, app) -> Optional[Dict[str, Any]]:
    with app.app_context():
        dataset = Dataset.query.get(dataset_id)
        return dataset.to_dict() if dataset else None
