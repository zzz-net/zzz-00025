from datetime import datetime
from sqlalchemy import JSON
from .database import db

STATUS_CHOICES = ['pending', 'training', 'completed', 'failed', 'active', 'rolled_back']
BATCH_STATUS_CHOICES = ['pending', 'processing', 'completed', 'partial_failed', 'failed']
BATCH_TICKET_STATUS_CHOICES = ['success', 'failed']


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


class PredictionBatch(db.Model):
    __tablename__ = 'prediction_batches'

    id = db.Column(db.Integer, primary_key=True)
    batch_uid = db.Column(db.String(100), unique=True, nullable=False, index=True)
    original_filename = db.Column(db.String(255), nullable=False)
    operator = db.Column(db.String(100), nullable=False, index=True)
    config_snapshot = db.Column(JSON, nullable=True)
    model_version_id = db.Column(db.Integer, db.ForeignKey('model_versions.id'), nullable=True)
    model_version_snapshot = db.Column(JSON, nullable=True)
    file_checksum = db.Column(db.String(64), nullable=True, index=True)
    total_count = db.Column(db.Integer, default=0)
    success_count = db.Column(db.Integer, default=0)
    failed_count = db.Column(db.Integer, default=0)
    status = db.Column(db.String(50), default='pending', nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    completed_at = db.Column(db.DateTime, nullable=True)
    error_message = db.Column(db.Text, nullable=True)

    model_version = db.relationship('ModelVersion', backref='prediction_batches', lazy=True)
    batch_tickets = db.relationship('BatchTicket', backref='prediction_batch', lazy=True, cascade='all, delete-orphan')

    def to_dict(self, include_details=False):
        data = {
            'id': self.id,
            'batch_uid': self.batch_uid,
            'original_filename': self.original_filename,
            'operator': self.operator,
            'model_version_id': self.model_version_id,
            'model_version_snapshot': self.model_version_snapshot,
            'total_count': self.total_count,
            'success_count': self.success_count,
            'failed_count': self.failed_count,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message
        }
        if include_details:
            data['config_snapshot'] = self.config_snapshot
            data['file_checksum'] = self.file_checksum
        return data


class BatchTicket(db.Model):
    __tablename__ = 'batch_tickets'

    id = db.Column(db.Integer, primary_key=True)
    batch_id = db.Column(db.Integer, db.ForeignKey('prediction_batches.id'), nullable=False, index=True)
    ticket_id = db.Column(db.Integer, db.ForeignKey('tickets.id'), nullable=True, index=True)
    row_index = db.Column(db.Integer, nullable=False)
    title = db.Column(db.String(500), nullable=False)
    content = db.Column(db.Text, nullable=False)
    channel = db.Column(db.String(100), nullable=False)
    predicted_queue = db.Column(db.String(100), nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    status = db.Column(db.String(20), nullable=False, index=True)
    error_message = db.Column(db.Text, nullable=True)
    has_override = db.Column(db.Boolean, default=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    ticket = db.relationship('Ticket', backref='batch_tickets', lazy=True)

    def to_dict(self):
        return {
            'id': self.id,
            'batch_id': self.batch_id,
            'ticket_id': self.ticket_id,
            'row_index': self.row_index,
            'title': self.title,
            'content': self.content,
            'channel': self.channel,
            'predicted_queue': self.predicted_queue,
            'confidence': self.confidence,
            'status': self.status,
            'error_message': self.error_message,
            'has_override': self.has_override,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class ModelActivationLog(db.Model):
    __tablename__ = 'model_activation_logs'

    id = db.Column(db.Integer, primary_key=True)
    model_version_id = db.Column(db.Integer, db.ForeignKey('model_versions.id'), nullable=False, index=True)
    previous_version_id = db.Column(db.Integer, db.ForeignKey('model_versions.id'), nullable=True)
    operator = db.Column(db.String(100), nullable=False)
    action = db.Column(db.String(50), nullable=False)
    status = db.Column(db.String(50), nullable=False)
    error_message = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    model_version = db.relationship('ModelVersion', foreign_keys=[model_version_id], backref='activation_logs', lazy=True)
    previous_version = db.relationship('ModelVersion', foreign_keys=[previous_version_id], lazy=True)

    def to_dict(self):
        return {
            'id': self.id,
            'model_version_id': self.model_version_id,
            'previous_version_id': self.previous_version_id,
            'operator': self.operator,
            'action': self.action,
            'status': self.status,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
