from typing import Dict, Any, List, Optional
from datetime import datetime

from .models import ModelVersion
from .database import db


def list_rollback_candidates(app) -> List[Dict[str, Any]]:
    with app.app_context():
        candidates = ModelVersion.query.filter(
            ModelVersion.status == 'completed',
            ModelVersion.status != 'failed',
            ModelVersion.is_active == False
        ).order_by(ModelVersion.trained_at.desc()).all()
        
        return [candidate.to_dict() for candidate in candidates]


def rollback_to_version(model_version_id: int, app) -> Dict[str, Any]:
    with app.app_context():
        target_version = ModelVersion.query.get(model_version_id)
        if target_version is None:
            raise Exception(f"模型版本 ID {model_version_id} 不存在")
        
        if target_version.status == 'failed':
            raise Exception("目标版本状态无效，不能回滚到失败的版本")
        
        current_actives = ModelVersion.query.filter_by(is_active=True).all()
        current_active = current_actives[0] if current_actives else None
        
        if current_active and current_active.id == target_version.id:
            raise Exception("目标版本已经是当前激活版本")
        
        for active in current_actives:
            active.is_active = False
            active.status = 'rolled_back'
            db.session.add(active)
        
        target_version.is_active = True
        target_version.status = 'active'
        db.session.add(target_version)
        
        db.session.commit()
        
        return {
            'success': True,
            'message': f"已成功回滚到版本 {target_version.version}",
            'previous_version': current_active.to_dict() if current_active else None,
            'current_version': target_version.to_dict()
        }


def get_active_version(app) -> Optional[Dict[str, Any]]:
    with app.app_context():
        active_version = ModelVersion.query.filter_by(is_active=True).first()
        return active_version.to_dict() if active_version else None


def get_version_history(app) -> List[Dict[str, Any]]:
    with app.app_context():
        versions = ModelVersion.query.order_by(ModelVersion.trained_at.desc()).all()
        return [version.to_dict() for version in versions]
