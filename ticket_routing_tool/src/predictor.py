import os
import pickle
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from config import QUEUE_MAPPING
from .models import Ticket, ModelVersion
from .database import db


def get_active_model(app) -> Optional[ModelVersion]:
    with app.app_context():
        return ModelVersion.query.filter_by(is_active=True).first()


def load_model(model_version: ModelVersion, app) -> Tuple[Any, Any]:
    model_path = model_version.model_path
    vectorizer_path = os.path.splitext(model_path)[0] + '_vectorizer.pkl'
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    return model, vectorizer


def predict_ticket(title: str, content: str, channel: str, app) -> Dict[str, Any]:
    active_model = get_active_model(app)
    if active_model is None:
        raise Exception("尚未训练可用模型，请先训练并激活一个模型")
    
    model, vectorizer = load_model(active_model, app)
    
    text = f"{title} {content}"
    X = vectorizer.transform([text])
    
    prediction = model.predict(X)[0]
    confidence = float(model.predict_proba(X).max())
    
    predicted_queue = QUEUE_MAPPING.get(prediction, prediction)
    
    with app.app_context():
        ticket = Ticket(
            title=title,
            content=content,
            channel=channel,
            predicted_queue=predicted_queue,
            confidence=confidence,
            predicted_at=datetime.utcnow(),
            model_version_id=active_model.id
        )
        db.session.add(ticket)
        db.session.commit()
    
    return {
        'predicted_queue': predicted_queue,
        'confidence': confidence,
        'model_version_id': active_model.id
    }


def predict_batch(tickets: List[Dict[str, str]], app) -> List[Dict[str, Any]]:
    active_model = get_active_model(app)
    if active_model is None:
        raise Exception("尚未训练可用模型，请先训练并激活一个模型")
    
    model, vectorizer = load_model(active_model, app)
    
    results = []
    with app.app_context():
        for ticket_data in tickets:
            title = ticket_data.get('title', '')
            content = ticket_data.get('content', '')
            channel = ticket_data.get('channel', '')
            
            text = f"{title} {content}"
            X = vectorizer.transform([text])
            
            prediction = model.predict(X)[0]
            confidence = float(model.predict_proba(X).max())
            
            predicted_queue = QUEUE_MAPPING.get(prediction, prediction)
            
            ticket = Ticket(
                title=title,
                content=content,
                channel=channel,
                predicted_queue=predicted_queue,
                confidence=confidence,
                predicted_at=datetime.utcnow(),
                model_version_id=active_model.id
            )
            db.session.add(ticket)
            
            results.append({
                'predicted_queue': predicted_queue,
                'confidence': confidence,
                'model_version_id': active_model.id
            })
        
        db.session.commit()
    
    return results


def get_queue_suggestion(predicted_queue: str, confidence: float) -> Dict[str, Any]:
    if confidence >= 0.8:
        return {
            'suggested_queue': predicted_queue,
            'priority': 'high',
            'action': 'auto_assign',
            'message': '置信度高，建议直接分配到对应队列'
        }
    elif 0.5 <= confidence < 0.8:
        return {
            'suggested_queue': predicted_queue,
            'priority': 'medium',
            'action': 'assign_with_review',
            'message': '置信度中等，建议分配到对应队列，可进行二次确认'
        }
    else:
        return {
            'suggested_queue': None,
            'priority': 'low',
            'action': 'manual_review',
            'message': '置信度低，建议人工审核后分配'
        }
