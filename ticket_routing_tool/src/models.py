from datetime import datetime
from sqlalchemy import JSON
from .database import db

STATUS_CHOICES = ['pending', 'training', 'completed', 'failed', 'active', 'rolled_back']


class Dataset(db.Model):
    __tablename__ = 'datasets'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    row_count = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(50), default='pending', nullable=False)
    error_message = db.Column(db.Text, nullable=True)

    model_versions = db.relationship('ModelVersion', backref='dataset', lazy=True)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'file_path': self.file_path,
            'row_count': self.row_count,
            'created_at': self.created_at,
            'status': self.status,
            'error_message': self.error_message
        }


class ModelVersion(db.Model):
    __tablename__ = 'model_versions'

    id = db.Column(db.Integer, primary_key=True)
    version = db.Column(db.String(50), nullable=False)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=False)
    model_path = db.Column(db.String(500), nullable=False)
    metrics = db.Column(JSON, nullable=True)
    is_active = db.Column(db.Boolean, default=False)
    trained_at = db.Column(db.DateTime, nullable=True)
    status = db.Column(db.String(50), default='pending', nullable=False)
    error_message = db.Column(db.Text, nullable=True)

    human_overrides = db.relationship('HumanOverride', backref='model_version', lazy=True)
    evaluation_reports = db.relationship('EvaluationReport', backref='model_version', lazy=True)

    def to_dict(self):
        return {
            'id': self.id,
            'version': self.version,
            'dataset_id': self.dataset_id,
            'model_path': self.model_path,
            'metrics': self.metrics,
            'is_active': self.is_active,
            'trained_at': self.trained_at.isoformat() if self.trained_at else None,
            'status': self.status,
            'error_message': self.error_message
        }


class Ticket(db.Model):
    __tablename__ = 'tickets'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(500), nullable=False)
    content = db.Column(db.Text, nullable=False)
    channel = db.Column(db.String(100), nullable=False)
    tags = db.Column(JSON, nullable=True)
    predicted_queue = db.Column(db.String(100), nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    actual_queue = db.Column(db.String(100), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predicted_at = db.Column(db.DateTime, nullable=True)
    model_version_id = db.Column(db.Integer, db.ForeignKey('model_versions.id'), nullable=True)

    human_overrides = db.relationship('HumanOverride', backref='ticket', lazy=True)
    model_version = db.relationship('ModelVersion', backref='tickets', lazy=True)

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'channel': self.channel,
            'tags': self.tags,
            'predicted_queue': self.predicted_queue,
            'confidence': self.confidence,
            'actual_queue': self.actual_queue,
            'created_at': self.created_at,
            'predicted_at': self.predicted_at,
            'model_version_id': self.model_version_id
        }


class HumanOverride(db.Model):
    __tablename__ = 'human_overrides'

    id = db.Column(db.Integer, primary_key=True)
    ticket_id = db.Column(db.Integer, db.ForeignKey('tickets.id'), nullable=False)
    original_prediction = db.Column(db.String(100), nullable=False)
    corrected_queue = db.Column(db.String(100), nullable=False)
    operator = db.Column(db.String(100), nullable=False)
    reason = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    model_version_id = db.Column(db.Integer, db.ForeignKey('model_versions.id'), nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'ticket_id': self.ticket_id,
            'original_prediction': self.original_prediction,
            'corrected_queue': self.corrected_queue,
            'operator': self.operator,
            'reason': self.reason,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'model_version_id': self.model_version_id
        }


class EvaluationReport(db.Model):
    __tablename__ = 'evaluation_reports'

    id = db.Column(db.Integer, primary_key=True)
    model_version_id = db.Column(db.Integer, db.ForeignKey('model_versions.id'), nullable=False)
    metrics_json = db.Column(JSON, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'model_version_id': self.model_version_id,
            'metrics_json': self.metrics_json,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
