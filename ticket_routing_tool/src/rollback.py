import os
import threading
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from .models import ModelVersion, ModelActivationLog, Ticket, Dataset
from .database import db

_activation_lock = threading.Lock()


def _get_model_usage_count(model_version_id: int, app) -> int:
    with app.app_context():
        return Ticket.query.filter_by(model_version_id=model_version_id).count()


def _get_model_details(model_version: ModelVersion, app) -> Dict[str, Any]:
    with app.app_context():
        data = model_version.to_dict()
        dataset = Dataset.query.get(model_version.dataset_id)
        data['dataset_name'] = dataset.name if dataset else '未知'
        data['dataset_row_count'] = dataset.row_count if dataset else 0
        data['usage_count'] = _get_model_usage_count(model_version.id, app)
        data['model_file_exists'] = os.path.exists(model_version.model_path)
        
        vectorizer_path = os.path.splitext(model_version.model_path)[0] + '_vectorizer.pkl'
        data['vectorizer_file_exists'] = os.path.exists(vectorizer_path)
        
        return data


def list_rollback_candidates(app) -> List[Dict[str, Any]]:
    with app.app_context():
        candidates = ModelVersion.query.filter(
            ModelVersion.status != 'failed',
            ModelVersion.is_active == False
        ).order_by(ModelVersion.trained_at.desc()).all()
        
        return [_get_model_details(candidate, app) for candidate in candidates]


def get_activation_history(app, limit: int = 20) -> List[Dict[str, Any]]:
    with app.app_context():
        logs = ModelActivationLog.query.order_by(
            ModelActivationLog.created_at.desc()
        ).limit(limit).all()
        
        result = []
        for log in logs:
            log_dict = log.to_dict()
            model = ModelVersion.query.get(log.model_version_id)
            if model:
                log_dict['model_version'] = model.version
            if log.previous_version_id:
                prev_model = ModelVersion.query.get(log.previous_version_id)
                if prev_model:
                    log_dict['previous_version'] = prev_model.version
            result.append(log_dict)
        
        return result


def get_version_with_details(model_version_id: int, app) -> Optional[Dict[str, Any]]:
    with app.app_context():
        model_version = ModelVersion.query.get(model_version_id)
        if not model_version:
            return None
        return _get_model_details(model_version, app)


def get_all_versions_with_details(app) -> List[Dict[str, Any]]:
    with app.app_context():
        versions = ModelVersion.query.order_by(ModelVersion.trained_at.desc()).all()
        return [_get_model_details(v, app) for v in versions]


def compare_versions(version_a_id: int, version_b_id: int, app) -> Dict[str, Any]:
    with app.app_context():
        version_a = ModelVersion.query.get(version_a_id)
        version_b = ModelVersion.query.get(version_b_id)
        
        if not version_a or not version_b:
            missing = []
            if not version_a:
                missing.append(f"版本 {version_a_id}")
            if not version_b:
                missing.append(f"版本 {version_b_id}")
            return {
                'success': False,
                'error': f"以下版本不存在: {', '.join(missing)}"
            }
        
        details_a = _get_model_details(version_a, app)
        details_b = _get_model_details(version_b, app)
        
        metrics_diff = {}
        if details_a.get('metrics') and details_b.get('metrics'):
            overall_a = details_a['metrics'].get('overall', {})
            overall_b = details_b['metrics'].get('overall', {})
            
            for key in ['accuracy', 'precision', 'recall', 'f1']:
                val_a = overall_a.get(key, 0) or 0
                val_b = overall_b.get(key, 0) or 0
                diff = val_b - val_a
                metrics_diff[key] = {
                    'version_a': val_a,
                    'version_b': val_b,
                    'difference': diff,
                    'difference_percent': f"{diff * 100:+.2f}%" if val_a else 'N/A'
                }
        
        return {
            'success': True,
            'version_a': details_a,
            'version_b': details_b,
            'metrics_diff': metrics_diff,
            'comparison': {
                'training_time_diff': (version_b.trained_at - version_a.trained_at).total_seconds() if version_a.trained_at and version_b.trained_at else None,
                'dataset_diff': details_a['dataset_name'] != details_b['dataset_name'],
                'status_diff': details_a['status'] != details_b['status'],
                'accuracy_diff': metrics_diff.get('accuracy', {}).get('difference', 0)
            }
        }


def _log_activation(
    model_version_id: int,
    previous_version_id: Optional[int],
    operator: str,
    action: str,
    status: str,
    error_message: Optional[str] = None
) -> None:
    log = ModelActivationLog(
        model_version_id=model_version_id,
        previous_version_id=previous_version_id,
        operator=operator,
        action=action,
        status=status,
        error_message=error_message
    )
    db.session.add(log)


def activate_model(
    model_version_id: int,
    app,
    operator: str = 'system'
) -> Dict[str, Any]:
    with _activation_lock:
        with app.app_context():
            target_version = ModelVersion.query.get(model_version_id)
            
            if target_version is None:
                _log_activation(
                    model_version_id=model_version_id,
                    previous_version_id=None,
                    operator=operator,
                    action='activate',
                    status='failed',
                    error_message=f"模型版本 ID {model_version_id} 不存在"
                )
                db.session.commit()
                return {
                    'success': False,
                    'error': f"模型版本 ID {model_version_id} 不存在",
                    'error_code': 'NOT_FOUND'
                }
            
            if target_version.status == 'failed':
                _log_activation(
                    model_version_id=model_version_id,
                    previous_version_id=None,
                    operator=operator,
                    action='activate',
                    status='failed',
                    error_message="目标版本状态为失败，不能激活失败的版本"
                )
                db.session.commit()
                return {
                    'success': False,
                    'error': "目标版本状态为失败，不能激活失败的版本",
                    'error_code': 'FAILED_VERSION'
                }
            
            if not os.path.exists(target_version.model_path):
                _log_activation(
                    model_version_id=model_version_id,
                    previous_version_id=None,
                    operator=operator,
                    action='activate',
                    status='failed',
                    error_message=f"模型文件不存在: {target_version.model_path}"
                )
                db.session.commit()
                return {
                    'success': False,
                    'error': f"模型文件不存在，无法激活该版本",
                    'error_code': 'FILE_MISSING'
                }
            
            vectorizer_path = os.path.splitext(target_version.model_path)[0] + '_vectorizer.pkl'
            if not os.path.exists(vectorizer_path):
                _log_activation(
                    model_version_id=model_version_id,
                    previous_version_id=None,
                    operator=operator,
                    action='activate',
                    status='failed',
                    error_message=f"向量化器文件不存在: {vectorizer_path}"
                )
                db.session.commit()
                return {
                    'success': False,
                    'error': f"向量化器文件不存在，无法激活该版本",
                    'error_code': 'VECTORIZER_MISSING'
                }
            
            current_actives = ModelVersion.query.filter_by(is_active=True).all()
            current_active = current_actives[0] if current_actives else None
            
            if current_active and current_active.id == target_version.id:
                _log_activation(
                    model_version_id=model_version_id,
                    previous_version_id=current_active.id,
                    operator=operator,
                    action='activate',
                    status='skipped',
                    error_message="目标版本已经是当前激活版本"
                )
                db.session.commit()
                return {
                    'success': False,
                    'error': "目标版本已经是当前激活版本",
                    'error_code': 'ALREADY_ACTIVE',
                    'current_version': target_version.to_dict()
                }
            
            previous_version_id = current_active.id if current_active else None
            
            try:
                for active in current_actives:
                    active.is_active = False
                    if active.status == 'active':
                        active.status = 'rolled_back'
                    db.session.add(active)
                
                target_version.is_active = True
                target_version.status = 'active'
                db.session.add(target_version)
                
                _log_activation(
                    model_version_id=model_version_id,
                    previous_version_id=previous_version_id,
                    operator=operator,
                    action='activate',
                    status='success'
                )
                
                db.session.commit()
                
                return {
                    'success': True,
                    'message': f"已成功激活版本 {target_version.version}",
                    'previous_version': current_active.to_dict() if current_active else None,
                    'current_version': target_version.to_dict()
                }
                
            except Exception as e:
                db.session.rollback()
                _log_activation(
                    model_version_id=model_version_id,
                    previous_version_id=previous_version_id,
                    operator=operator,
                    action='activate',
                    status='failed',
                    error_message=f"激活失败: {str(e)}"
                )
                db.session.commit()
                return {
                    'success': False,
                    'error': f"激活失败: {str(e)}",
                    'error_code': 'INTERNAL_ERROR'
                }


def rollback_to_version(model_version_id: int, app) -> Dict[str, Any]:
    return activate_model(model_version_id, app, operator='rollback')


def get_active_version(app) -> Optional[Dict[str, Any]]:
    with app.app_context():
        active_version = ModelVersion.query.filter_by(is_active=True).first()
        if active_version:
            return _get_model_details(active_version, app)
        return None


def get_version_history(app) -> List[Dict[str, Any]]:
    return get_all_versions_with_details(app)
