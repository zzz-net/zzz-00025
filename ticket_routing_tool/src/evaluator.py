from typing import Dict, Any, List, Optional
import json
import numpy as np
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
)

from .database import db
from .models import EvaluationReport
from config import METRIC_EXPLANATIONS


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str]
) -> Dict[str, Any]:
    accuracy = accuracy_score(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='macro', zero_division=0
    )

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='weighted', zero_division=0
    )

    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    per_class_metrics = {}
    for i, label in enumerate(labels):
        per_class_metrics[label] = {
            'precision': float(per_class_precision[i]),
            'recall': float(per_class_recall[i]),
            'f1': float(per_class_f1[i]),
            'support': int(per_class_support[i])
        }

    return {
        'overall': {
            'accuracy': float(accuracy),
            'macro_precision': float(precision_macro),
            'macro_recall': float(recall_macro),
            'macro_f1': float(f1_macro),
            'weighted_precision': float(precision_weighted),
            'weighted_recall': float(recall_weighted),
            'weighted_f1': float(f1_weighted)
        },
        'per_class': per_class_metrics,
        'labels': labels
    }


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str]
) -> str:
    return classification_report(
        y_true, y_pred, labels=labels, zero_division=0
    )


def evaluate_model(
    model: Any,
    vectorizer: Any,
    X_test: List[str],
    y_test: np.ndarray,
    labels: List[str]
) -> Dict[str, Any]:
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)

    metrics = compute_metrics(y_test, y_pred, labels)
    report_text = generate_classification_report(y_test, y_pred, labels)

    return {
        'metrics': metrics,
        'classification_report_text': report_text,
        'metric_explanations': METRIC_EXPLANATIONS
    }


def save_evaluation_report(
    model_version_id: int,
    metrics: Dict[str, Any],
    app: Any
) -> Optional[EvaluationReport]:
    try:
        with app.app_context():
            metrics_json = json.dumps(metrics, ensure_ascii=False, indent=2)

            report = EvaluationReport(
                model_version_id=model_version_id,
                metrics_json=metrics
            )

            db.session.add(report)
            db.session.commit()

            return report
    except Exception as e:
        with app.app_context():
            db.session.rollback()
        raise e
