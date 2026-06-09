import os
import csv
import json
import threading
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

from .models import ModelVersion, ModelActivationLog, Ticket, Dataset, OperationLog
from .database import db
import config

_activation_lock = threading.Lock()


def _get_model_usage_count(model_version_id: int, app) -> int:
    with app.app_context():
        return Ticket.query.filter_by(model_version_id=model_version_id).count()


def _get_model_recent_usage_count(model_version_id: int, app, window_days: int = None) -> int:
    if window_days is None:
        window_days = getattr(config, 'USAGE_WINDOW_DAYS', 7)
    cutoff_date = datetime.utcnow() - timedelta(days=window_days)
    with app.app_context():
        return Ticket.query.filter(
            Ticket.model_version_id == model_version_id,
            Ticket.predicted_at >= cutoff_date
        ).count()


def _get_available_versions(app) -> List[ModelVersion]:
    with app.app_context():
        versions = ModelVersion.query.filter(
            ModelVersion.status.in_(['completed', 'active', 'rolled_back'])
        ).order_by(ModelVersion.trained_at.desc()).all()
        
        available = []
        for v in versions:
            model_path = v.model_path
            vectorizer_path = os.path.splitext(model_path)[0] + '_vectorizer.pkl'
            if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                available.append(v)
        return available


def _get_suggestion_message(version_id: int, app, reason: str) -> str:
    available = _get_available_versions(app)
    if not available:
        return f"{reason}。当前没有可用的版本，请先训练一个完整的模型。"
    
    suggestions = []
    for v in available[:3]:
        suggestions.append(f"v{v.version} (ID: {v.id})")
    
    return f"{reason}。请选择以下可用版本之一：{', '.join(suggestions)}"


def _log_operation(
    operation_type: str,
    status: str,
    operator: str = None,
    details: Dict[str, Any] = None,
    error_message: str = None
) -> None:
    log = OperationLog(
        operation_type=operation_type,
        status=status,
        operator=operator,
        details=details or {},
        error_message=error_message
    )
    db.session.add(log)


def _validate_version_for_comparison(version: ModelVersion, app) -> Tuple[bool, str, str]:
    if not version:
        return False, 'NOT_FOUND', '版本不存在'
    
    if version.status == 'failed':
        return False, 'FAILED_VERSION', f"版本 {version.id} 状态为 failed"
    
    if version.status == 'training':
        return False, 'TRAINING_VERSION', f"版本 {version.id} 正在训练中"
    
    if version.status == 'pending':
        return False, 'PENDING_VERSION', f"版本 {version.id} 等待训练中"
    
    if not os.path.exists(version.model_path):
        return False, 'FILE_MISSING', f"版本 {version.id} 模型文件不存在"
    
    vectorizer_path = os.path.splitext(version.model_path)[0] + '_vectorizer.pkl'
    if not os.path.exists(vectorizer_path):
        return False, 'VECTORIZER_MISSING', f"版本 {version.id} 向量化器文件不存在"
    
    return True, None, None


def _get_model_details(model_version: ModelVersion, app) -> Dict[str, Any]:
    with app.app_context():
        data = model_version.to_dict()
        dataset = Dataset.query.get(model_version.dataset_id)
        data['dataset_name'] = dataset.name if dataset else '未知'
        data['dataset_row_count'] = dataset.row_count if dataset else 0
        data['usage_count'] = _get_model_usage_count(model_version.id, app)
        data['recent_usage_count'] = _get_model_recent_usage_count(model_version.id, app)
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


def get_operation_history(app, operation_type: str = None, limit: int = 50) -> List[Dict[str, Any]]:
    with app.app_context():
        query = OperationLog.query.order_by(OperationLog.created_at.desc())
        if operation_type:
            query = query.filter_by(operation_type=operation_type)
        logs = query.limit(limit).all()
        return [log.to_dict() for log in logs]


def get_usage_window_config(app) -> Dict[str, Any]:
    window_days = getattr(config, 'USAGE_WINDOW_DAYS', 7)
    with app.app_context():
        _log_operation(
            operation_type='config_read',
            status='success',
            operator='system',
            details={'config_key': 'USAGE_WINDOW_DAYS', 'config_value': window_days}
        )
        db.session.commit()
    return {
        'window_days': window_days,
        'description': f"最近 {window_days} 天的预测使用量统计窗口"
    }


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


def get_available_versions_for_comparison(app) -> List[Dict[str, Any]]:
    versions = _get_available_versions(app)
    return [_get_model_details(v, app) for v in versions]


def compare_versions(
    version_a_id: int, 
    version_b_id: int, 
    app, 
    operator: str = None,
    window_days: int = None
) -> Dict[str, Any]:
    with app.app_context():
        version_a = ModelVersion.query.get(version_a_id)
        version_b = ModelVersion.query.get(version_b_id)
        
        comparison_details = {
            'version_a_id': version_a_id,
            'version_b_id': version_b_id,
            'window_days': window_days or getattr(config, 'USAGE_WINDOW_DAYS', 7)
        }
        
        if not version_a or not version_b:
            missing = []
            if not version_a:
                missing.append(f"版本 {version_a_id}")
            if not version_b:
                missing.append(f"版本 {version_b_id}")
            error_msg = f"以下版本不存在: {', '.join(missing)}"
            
            suggestion = _get_suggestion_message(version_a_id, app, error_msg)
            
            _log_operation(
                operation_type='compare_versions',
                status='rejected',
                operator=operator,
                details=comparison_details,
                error_message=error_msg
            )
            db.session.commit()
            
            return {
                'success': False,
                'error': suggestion,
                'error_code': 'NOT_FOUND',
                'available_versions': [
                    {'id': v.id, 'version': v.version} 
                    for v in _get_available_versions(app)
                ]
            }
        
        is_valid_a, error_code_a, error_msg_a = _validate_version_for_comparison(version_a, app)
        is_valid_b, error_code_b, error_msg_b = _validate_version_for_comparison(version_b, app)
        
        if not is_valid_a or not is_valid_b:
            errors = []
            error_codes = []
            if not is_valid_a:
                errors.append(error_msg_a)
                error_codes.append(error_code_a)
            if not is_valid_b:
                errors.append(error_msg_b)
                error_codes.append(error_code_b)
            
            combined_error = '; '.join(errors)
            suggestion = _get_suggestion_message(
                version_a_id if not is_valid_a else version_b_id, 
                app, 
                combined_error
            )
            
            comparison_details['validation_errors'] = errors
            comparison_details['error_codes'] = error_codes
            
            _log_operation(
                operation_type='compare_versions',
                status='rejected',
                operator=operator,
                details=comparison_details,
                error_message=combined_error
            )
            db.session.commit()
            
            return {
                'success': False,
                'error': suggestion,
                'error_code': error_codes[0],
                'available_versions': [
                    {'id': v.id, 'version': v.version} 
                    for v in _get_available_versions(app)
                ]
            }
        
        if version_a_id == version_b_id:
            error_msg = "请选择两个不同的版本进行对比"
            _log_operation(
                operation_type='compare_versions',
                status='rejected',
                operator=operator,
                details=comparison_details,
                error_message=error_msg
            )
            db.session.commit()
            
            return {
                'success': False,
                'error': error_msg,
                'error_code': 'SAME_VERSION'
            }
        
        details_a = _get_model_details(version_a, app)
        details_b = _get_model_details(version_b, app)
        
        if window_days is not None:
            details_a['recent_usage_count'] = _get_model_recent_usage_count(version_a.id, app, window_days)
            details_b['recent_usage_count'] = _get_model_recent_usage_count(version_b.id, app, window_days)
        
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
        
        _log_operation(
            operation_type='compare_versions',
            status='success',
            operator=operator,
            details={
                **comparison_details,
                'version_a_version': version_a.version,
                'version_b_version': version_b.version
            }
        )
        db.session.commit()
        
        return {
            'success': True,
            'version_a': details_a,
            'version_b': details_b,
            'metrics_diff': metrics_diff,
            'window_days': window_days or getattr(config, 'USAGE_WINDOW_DAYS', 7),
            'comparison': {
                'training_time_diff': (version_b.trained_at - version_a.trained_at).total_seconds() if version_a.trained_at and version_b.trained_at else None,
                'dataset_diff': details_a['dataset_name'] != details_b['dataset_name'],
                'status_diff': details_a['status'] != details_b['status'],
                'accuracy_diff': metrics_diff.get('accuracy', {}).get('difference', 0)
            }
        }


def export_comparison_result(
    comparison_data: Dict[str, Any],
    format_type: str,
    app,
    operator: str = None
) -> Dict[str, Any]:
    if format_type not in getattr(config, 'COMPARISON_EXPORT_FORMATS', ['csv', 'json']):
        error_msg = f"不支持的导出格式: {format_type}"
        with app.app_context():
            _log_operation(
                operation_type='export_comparison',
                status='rejected',
                operator=operator,
                details={'format': format_type},
                error_message=error_msg
            )
            db.session.commit()
        return {
            'success': False,
            'error': error_msg,
            'error_code': 'INVALID_FORMAT'
        }
    
    if not comparison_data.get('success'):
        error_msg = "对比数据无效，无法导出"
        with app.app_context():
            _log_operation(
                operation_type='export_comparison',
                status='rejected',
                operator=operator,
                details={'format': format_type},
                error_message=error_msg
            )
            db.session.commit()
        return {
            'success': False,
            'error': error_msg,
            'error_code': 'INVALID_DATA'
        }
    
    va = comparison_data['version_a']
    vb = comparison_data['version_b']
    metrics_diff = comparison_data.get('metrics_diff', {})
    window_days = comparison_data.get('window_days', getattr(config, 'USAGE_WINDOW_DAYS', 7))
    
    export_data = {
        'exported_at': datetime.utcnow().isoformat(),
        'window_days': window_days,
        'version_a': {
            'id': va['id'],
            'version': va['version'],
            'dataset_name': va.get('dataset_name', ''),
            'trained_at': va.get('trained_at', ''),
            'is_active': va.get('is_active', False),
            'status': va.get('status', ''),
            'total_usage': va.get('usage_count', 0),
            'recent_usage': va.get('recent_usage_count', 0),
            'metrics': {
                'accuracy': va.get('metrics', {}).get('overall', {}).get('accuracy', 0),
                'precision': va.get('metrics', {}).get('overall', {}).get('precision', 0),
                'recall': va.get('metrics', {}).get('overall', {}).get('recall', 0),
                'f1': va.get('metrics', {}).get('overall', {}).get('f1', 0)
            }
        },
        'version_b': {
            'id': vb['id'],
            'version': vb['version'],
            'dataset_name': vb.get('dataset_name', ''),
            'trained_at': vb.get('trained_at', ''),
            'is_active': vb.get('is_active', False),
            'status': vb.get('status', ''),
            'total_usage': vb.get('usage_count', 0),
            'recent_usage': vb.get('recent_usage_count', 0),
            'metrics': {
                'accuracy': vb.get('metrics', {}).get('overall', {}).get('accuracy', 0),
                'precision': vb.get('metrics', {}).get('overall', {}).get('precision', 0),
                'recall': vb.get('metrics', {}).get('overall', {}).get('recall', 0),
                'f1': vb.get('metrics', {}).get('overall', {}).get('f1', 0)
            }
        },
        'metrics_diff': {
            'accuracy': metrics_diff.get('accuracy', {}),
            'precision': metrics_diff.get('precision', {}),
            'recall': metrics_diff.get('recall', {}),
            'f1': metrics_diff.get('f1', {})
        }
    }
    
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"comparison_export_v{va['version']}_vs_v{vb['version']}_{timestamp}.{format_type}"
    output_path = os.path.join(config.REPORT_DIR, filename)
    
    try:
        if format_type == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
        elif format_type == 'csv':
            with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(['模型对比结果导出', ''])
                writer.writerow(['导出时间', export_data['exported_at']])
                writer.writerow(['统计窗口天数', f"{window_days} 天"])
                writer.writerow([])
                
                writer.writerow(['指标', '版本 A', '版本 B', '差异'])
                writer.writerow(['版本号', f"v{va['version']}", f"v{vb['version']}", ''])
                writer.writerow(['模型ID', va['id'], vb['id'], ''])
                writer.writerow(['数据集', va.get('dataset_name', ''), vb.get('dataset_name', ''), ''])
                writer.writerow(['训练时间', va.get('trained_at', ''), vb.get('trained_at', ''), ''])
                writer.writerow(['激活状态', '是' if va.get('is_active') else '否', '是' if vb.get('is_active') else '否', ''])
                writer.writerow(['模型状态', va.get('status', ''), vb.get('status', ''), ''])
                writer.writerow([])
                
                writer.writerow(['使用量统计', '', '', ''])
                writer.writerow(['历史总使用量', va.get('usage_count', 0), vb.get('usage_count', 0), 
                                (vb.get('usage_count', 0) - va.get('usage_count', 0))])
                writer.writerow([f'最近{window_days}天使用量', va.get('recent_usage_count', 0), vb.get('recent_usage_count', 0),
                                (vb.get('recent_usage_count', 0) - va.get('recent_usage_count', 0))])
                writer.writerow([])
                
                writer.writerow(['评估指标', '', '', ''])
                for metric_key, metric_label in [('accuracy', '准确率'), ('precision', '精确率'), ('recall', '召回率'), ('f1', 'F1值')]:
                    a_val = export_data['version_a']['metrics'][metric_key]
                    b_val = export_data['version_b']['metrics'][metric_key]
                    diff = metrics_diff.get(metric_key, {}).get('difference', 0)
                    writer.writerow([
                        metric_label,
                        f"{a_val * 100:.2f}%" if a_val else '-',
                        f"{b_val * 100:.2f}%" if b_val else '-',
                        f"{diff * 100:+.2f}%" if diff else '0.00%'
                    ])
        
        with app.app_context():
            _log_operation(
                operation_type='export_comparison',
                status='success',
                operator=operator,
                details={
                    'format': format_type,
                    'output_path': output_path,
                    'version_a_id': va['id'],
                    'version_b_id': vb['id']
                }
            )
            db.session.commit()
        
        return {
            'success': True,
            'output_path': output_path,
            'filename': filename,
            'format': format_type,
            'export_data': export_data
        }
        
    except Exception as e:
        with app.app_context():
            _log_operation(
                operation_type='export_comparison',
                status='failed',
                operator=operator,
                details={'format': format_type},
                error_message=str(e)
            )
            db.session.commit()
        
        return {
            'success': False,
            'error': f"导出失败: {str(e)}",
            'error_code': 'EXPORT_FAILED'
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
                _log_operation(
                    operation_type='activate_model',
                    status='rejected',
                    operator=operator,
                    details={'model_version_id': model_version_id},
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
                _log_operation(
                    operation_type='activate_model',
                    status='rejected',
                    operator=operator,
                    details={'model_version_id': model_version_id, 'version_status': target_version.status},
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
                _log_operation(
                    operation_type='activate_model',
                    status='rejected',
                    operator=operator,
                    details={'model_version_id': model_version_id, 'missing_file': target_version.model_path},
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
                _log_operation(
                    operation_type='activate_model',
                    status='rejected',
                    operator=operator,
                    details={'model_version_id': model_version_id, 'missing_file': vectorizer_path},
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
                _log_operation(
                    operation_type='activate_model',
                    status='rejected',
                    operator=operator,
                    details={'model_version_id': model_version_id},
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
                
                _log_operation(
                    operation_type='activate_model',
                    status='success',
                    operator=operator,
                    details={
                        'model_version_id': model_version_id,
                        'previous_version_id': previous_version_id,
                        'version': target_version.version
                    }
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
                _log_operation(
                    operation_type='activate_model',
                    status='failed',
                    operator=operator,
                    details={'model_version_id': model_version_id},
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
