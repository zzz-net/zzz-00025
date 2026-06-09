from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import os
import uuid
import pickle

import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .database import db
from .models import Dataset, ModelVersion
from .evaluator import evaluate_model, save_evaluation_report
from .data_validator import KNOWN_LABELS
from config import MODEL_DIR, DATASET_DIR


os.makedirs(MODEL_DIR, exist_ok=True)


def _validate_training_labels(
    df: pd.DataFrame,
    known_labels: List[str]
) -> Tuple[bool, List[str]]:
    errors = []
    known_set = set(known_labels)

    if "tags" not in df.columns:
        errors.append("缺少 tags 列")
        return False, errors

    for idx, tags_str in enumerate(df["tags"]):
        if pd.isna(tags_str) or not str(tags_str).strip():
            errors.append(f"第 {idx + 2} 行: 标签为空")
            continue

        tags = [t.strip() for t in str(tags_str).split(",")]
        tags = [t for t in tags if t]

        if len(tags) == 0:
            errors.append(f"第 {idx + 2} 行: 标签解析后为空")
            continue

        for tag in tags:
            if tag not in known_set:
                errors.append(f"第 {idx + 2} 行: 未知标签 '{tag}'")

    is_valid = len(errors) == 0
    return is_valid, errors


def _extract_features(texts: List[str]) -> Tuple[TfidfVectorizer, np.ndarray]:
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )
    features = vectorizer.fit_transform(texts)
    return vectorizer, features


def _save_model(
    model: LogisticRegression,
    vectorizer: TfidfVectorizer,
    version: str
) -> Tuple[str, str]:
    model_filename = f"model_{version}.pkl"
    vectorizer_filename = f"model_{version}_vectorizer.pkl"

    model_path = os.path.join(MODEL_DIR, model_filename)
    vectorizer_path = os.path.join(MODEL_DIR, vectorizer_filename)

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)

    return model_path, vectorizer_path


def train_model(dataset_id: int, app: Any) -> Dict[str, Any]:
    with app.app_context():
        dataset = Dataset.query.get(dataset_id)
        if not dataset:
            return {
                "success": False,
                "error": f"数据集 {dataset_id} 不存在"
            }

        version = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{uuid.uuid4().hex[:8]}"

        model_version = ModelVersion(
            version=version,
            dataset_id=dataset_id,
            model_path="",
            status="training"
        )
        db.session.add(model_version)
        db.session.commit()

        try:
            df = pd.read_csv(dataset.file_path, encoding="utf-8-sig")

            known_labels = sorted(list(KNOWN_LABELS))

            is_valid, label_errors = _validate_training_labels(df, known_labels)
            if not is_valid:
                model_version.status = "failed"
                model_version.error_message = "标签验证失败: " + "; ".join(label_errors)
                db.session.commit()
                return {
                    "success": False,
                    "error": model_version.error_message,
                    "model_version_id": model_version.id
                }

            df["combined_text"] = df["title"].fillna("") + " " + df["content"].fillna("")
            texts = df["combined_text"].tolist()
            
            def extract_first_tag(tags_str):
                if pd.isna(tags_str) or not str(tags_str).strip():
                    return ""
                tags = [t.strip() for t in str(tags_str).split(",")]
                tags = [t for t in tags if t]
                return tags[0] if tags else ""
            
            labels = df["tags"].apply(extract_first_tag).tolist()

            vectorizer, X = _extract_features(texts)
            y = np.array(labels)

            X_train, X_test, y_train, y_test = train_test_split(
                texts, y, test_size=0.2, random_state=42, stratify=y
            )

            X_train_vec = vectorizer.transform(X_train)

            model = LogisticRegression(
                max_iter=1000,
                C=1.0,
                solver="lbfgs",
                class_weight="balanced",
                random_state=42
            )
            model.fit(X_train_vec, y_train)

            eval_result = evaluate_model(
                model=model,
                vectorizer=vectorizer,
                X_test=X_test,
                y_test=y_test,
                labels=known_labels
            )

            model_path, _ = _save_model(model, vectorizer, version)

            model_version.model_path = model_path
            model_version.metrics = eval_result["metrics"]
            model_version.status = "completed"
            model_version.trained_at = datetime.utcnow()

            existing_active = ModelVersion.query.filter_by(is_active=True).all()
            for active in existing_active:
                active.is_active = False
                active.status = "rolled_back"
                db.session.add(active)

            model_version.is_active = True
            model_version.status = "active"
            db.session.add(model_version)
            db.session.commit()

            save_evaluation_report(
                model_version_id=model_version.id,
                metrics=eval_result,
                app=app
            )

            return {
                "success": True,
                "model_version_id": model_version.id,
                "version": version,
                "metrics": eval_result["metrics"],
                "classification_report": eval_result["classification_report_text"]
            }

        except Exception as e:
            model_version.status = "failed"
            model_version.error_message = f"训练失败: {str(e)}"
            db.session.commit()
            return {
                "success": False,
                "error": model_version.error_message,
                "model_version_id": model_version.id
            }
