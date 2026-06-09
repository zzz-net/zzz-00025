from typing import Dict, Any, List, Optional
from datetime import datetime

from .models import Ticket, ModelVersion, HumanOverride
from .database import db
from .batch_manager import update_override_flag


def create_override(
    ticket_id: int,
    corrected_queue: str,
    operator: str,
    reason: str,
    app
) -> Dict[str, Any]:
    with app.app_context():
        ticket = Ticket.query.get(ticket_id)
        if ticket is None:
            raise Exception(f"工单 ID {ticket_id} 不存在")
        
        original_prediction = ticket.predicted_queue
        
        active_model = ModelVersion.query.filter_by(is_active=True).first()
        model_version_id = active_model.id if active_model else None
        
        override = HumanOverride(
            ticket_id=ticket_id,
            original_prediction=original_prediction,
            corrected_queue=corrected_queue,
            operator=operator,
            reason=reason,
            model_version_id=model_version_id,
            created_at=datetime.utcnow()
        )
        
        ticket.actual_queue = corrected_queue
        
        db.session.add(override)
        db.session.commit()

        update_override_flag(ticket_id, app)

        return override.to_dict()


def list_overrides(
    ticket_id: Optional[int] = None,
    operator: Optional[str] = None,
    app = None
) -> List[Dict[str, Any]]:
    with app.app_context():
        query = HumanOverride.query
        
        if ticket_id is not None:
            query = query.filter_by(ticket_id=ticket_id)
        
        if operator is not None:
            query = query.filter_by(operator=operator)
        
        overrides = query.order_by(HumanOverride.created_at.desc()).all()
        
        return [override.to_dict() for override in overrides]


def get_override_stats(app) -> Dict[str, Any]:
    with app.app_context():
        total_overrides = HumanOverride.query.count()
        total_tickets = Ticket.query.count()
        
        override_rate = (total_overrides / total_tickets) if total_tickets > 0 else 0
        
        queue_stats = db.session.query(
            HumanOverride.corrected_queue,
            db.func.count(HumanOverride.id).label('count')
        ).group_by(HumanOverride.corrected_queue).all()
        
        queue_override_counts = {}
        for queue, count in queue_stats:
            queue_override_counts[queue] = count
        
        original_queue_stats = db.session.query(
            HumanOverride.original_prediction,
            db.func.count(HumanOverride.id).label('count')
        ).group_by(HumanOverride.original_prediction).all()
        
        original_queue_counts = {}
        for queue, count in original_queue_stats:
            original_queue_counts[queue] = count
        
        operator_stats = db.session.query(
            HumanOverride.operator,
            db.func.count(HumanOverride.id).label('count')
        ).group_by(HumanOverride.operator).all()
        
        operator_counts = {}
        for op, count in operator_stats:
            operator_counts[op] = count
        
        return {
            'total_overrides': total_overrides,
            'total_tickets': total_tickets,
            'override_rate': override_rate,
            'override_rate_percentage': f"{override_rate * 100:.2f}%",
            'queue_override_counts': queue_override_counts,
            'original_prediction_counts': original_queue_counts,
            'operator_counts': operator_counts
        }
