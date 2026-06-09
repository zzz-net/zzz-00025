import os
import uuid
import hashlib
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from io import BytesIO

from config import BATCH_RESULT_DIR, LOW_CONFIDENCE_THRESHOLD, SUPPORTED_CHANNELS
from .database import db
from .models import PredictionBatch, BatchTicket, ModelVersion, Ticket, HumanOverride
from .predictor import get_active_model, load_model, get_queue_suggestion
from .csv_importer import generate_timestamp_filename

logger = logging.getLogger(__name__)

os.makedirs(BATCH_RESULT_DIR, exist_ok=True)


def calculate_file_checksum(file_path: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def generate_batch_uid() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = uuid.uuid4().hex[:8]
    return f"BATCH_{timestamp}_{random_suffix}"


def get_config_snapshot(app) -> Dict[str, Any]:
    return {
        'supported_channels': list(SUPPORTED_CHANNELS),
        'low_confidence_threshold': LOW_CONFIDENCE_THRESHOLD,
        'max_batch_size': 1000,
        'timestamp': datetime.utcnow().isoformat()
    }


def create_batch(
    original_filename: str,
    operator: str,
    file_path: str,
    app
) -> Tuple[bool, Dict[str, Any]]:
    try:
        if not operator or not operator.strip():
            return False, {'error': '操作者不能为空'}

        file_checksum = calculate_file_checksum(file_path)

        with app.app_context():
            existing_batch = PredictionBatch.query.filter_by(
                file_checksum=file_checksum,
                operator=operator.strip()
            ).first()

            if existing_batch:
                return False, {
                    'error': '该文件已由此操作者提交过',
                    'is_duplicate': True,
                    'existing_batch': existing_batch.to_dict()
                }

        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
        except Exception as e:
            return False, {'error': f'CSV 文件解析失败: {str(e)}'}

        required_columns = ['title', 'content', 'channel']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, {'error': f'缺少必要列: {", ".join(missing_columns)}'}

        if len(df) > 1000:
            return False, {'error': '批量预测最多支持 1000 条工单'}

        if len(df) == 0:
            return False, {'error': 'CSV 文件为空'}

        active_model = get_active_model(app)
        if active_model is None:
            return False, {'error': '尚未训练可用模型，请先训练并激活一个模型'}

        model_version_snapshot = active_model.to_dict()
        with app.app_context():
            from .models import Dataset
            dataset = Dataset.query.get(active_model.dataset_id)
            if dataset:
                model_version_snapshot['dataset_name'] = dataset.name

        config_snapshot = get_config_snapshot(app)
        batch_uid = generate_batch_uid()

        with app.app_context():
            batch = PredictionBatch(
                batch_uid=batch_uid,
                original_filename=original_filename,
                operator=operator.strip(),
                config_snapshot=config_snapshot,
                model_version_id=active_model.id,
                model_version_snapshot=model_version_snapshot,
                file_checksum=file_checksum,
                total_count=len(df),
                success_count=0,
                failed_count=0,
                status='processing',
                created_at=datetime.utcnow()
            )
            db.session.add(batch)
            db.session.flush()
            batch_id = batch.id
            db.session.commit()

        return True, {
            'batch_id': batch_id,
            'batch_uid': batch_uid,
            'total_count': len(df),
            'model_version': model_version_snapshot
        }

    except Exception as e:
        logger.error(f"创建批次失败: {str(e)}", exc_info=True)
        return False, {'error': f'创建批次失败: {str(e)}'}


def process_batch(batch_id: int, file_path: str, app) -> Dict[str, Any]:
    success_count = 0
    failed_count = 0
    results = []

    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')

        with app.app_context():
            batch = PredictionBatch.query.get(batch_id)
            if batch is None:
                return {'success': False, 'error': '批次不存在'}

            active_model = ModelVersion.query.get(batch.model_version_id)
            if active_model is None:
                batch.status = 'failed'
                batch.error_message = '关联的模型版本不存在'
                batch.completed_at = datetime.utcnow()
                db.session.commit()
                return {'success': False, 'error': '关联的模型版本不存在'}

            try:
                model, vectorizer = load_model(active_model, app)
            except Exception as e:
                batch.status = 'failed'
                batch.error_message = f'加载模型失败: {str(e)}'
                batch.completed_at = datetime.utcnow()
                db.session.commit()
                return {'success': False, 'error': f'加载模型失败: {str(e)}'}

            for idx, row in df.iterrows():
                row_index = idx + 1
                try:
                    title_val = row.get('title', '')
                    content_val = row.get('content', '')
                    channel_val = row.get('channel', '')

                    title = '' if pd.isna(title_val) else str(title_val).strip()
                    content = '' if pd.isna(content_val) else str(content_val).strip()
                    channel = '' if pd.isna(channel_val) else str(channel_val).strip()

                    if not title or not content or not channel:
                        raise ValueError('缺少必填字段: title, content, channel')

                    if channel not in SUPPORTED_CHANNELS:
                        raise ValueError(f'不支持的渠道: {channel}')

                    text = f"{title} {content}"
                    X = vectorizer.transform([text])

                    prediction = model.predict(X)[0]
                    confidence = float(model.predict_proba(X).max())

                    from config import QUEUE_MAPPING
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
                    db.session.flush()

                    batch_ticket = BatchTicket(
                        batch_id=batch_id,
                        ticket_id=ticket.id,
                        row_index=row_index,
                        title=title,
                        content=content,
                        channel=channel,
                        predicted_queue=predicted_queue,
                        confidence=confidence,
                        status='success',
                        error_message=None,
                        has_override=False
                    )
                    db.session.add(batch_ticket)

                    success_count += 1
                    results.append({
                        'row_index': row_index,
                        'ticket_id': ticket.id,
                        'predicted_queue': predicted_queue,
                        'confidence': confidence,
                        'status': 'success'
                    })

                except Exception as e:
                    error_msg = str(e)
                    logger.warning(f"批次 {batch_id} 第 {row_index} 行处理失败: {error_msg}")

                    title_val = row.get('title', '')
                    content_val = row.get('content', '')
                    channel_val = row.get('channel', '')
                    title = ('(空)' if pd.isna(title_val) else str(title_val).strip()) or '(空)'
                    content = ('(空)' if pd.isna(content_val) else str(content_val).strip()) or '(空)'
                    channel = ('(空)' if pd.isna(channel_val) else str(channel_val).strip()) or '(空)'

                    batch_ticket = BatchTicket(
                        batch_id=batch_id,
                        ticket_id=None,
                        row_index=row_index,
                        title=title,
                        content=content,
                        channel=channel,
                        predicted_queue=None,
                        confidence=None,
                        status='failed',
                        error_message=error_msg,
                        has_override=False
                    )
                    db.session.add(batch_ticket)

                    failed_count += 1
                    results.append({
                        'row_index': row_index,
                        'status': 'failed',
                        'error_message': error_msg
                    })

            batch.success_count = success_count
            batch.failed_count = failed_count

            if failed_count == 0:
                batch.status = 'completed'
            elif success_count > 0:
                batch.status = 'partial_failed'
            else:
                batch.status = 'failed'

            batch.completed_at = datetime.utcnow()
            db.session.commit()

        return {
            'success': True,
            'batch_id': batch_id,
            'success_count': success_count,
            'failed_count': failed_count,
            'total_count': success_count + failed_count,
            'results': results
        }

    except Exception as e:
        logger.error(f"处理批次 {batch_id} 失败: {str(e)}", exc_info=True)
        with app.app_context():
            batch = PredictionBatch.query.get(batch_id)
            if batch:
                batch.status = 'failed'
                batch.error_message = str(e)
                batch.completed_at = datetime.utcnow()
                db.session.commit()
        return {'success': False, 'error': f'处理批次失败: {str(e)}'}


def get_batch(batch_id: int, app, include_details: bool = False) -> Optional[Dict[str, Any]]:
    with app.app_context():
        batch = PredictionBatch.query.get(batch_id)
        if batch is None:
            return None

        batch_dict = batch.to_dict(include_details=include_details)

        override_count = db.session.query(HumanOverride).join(
            BatchTicket, HumanOverride.ticket_id == BatchTicket.ticket_id
        ).filter(BatchTicket.batch_id == batch_id).count()

        batch_dict['override_count'] = override_count
        return batch_dict


def list_batches(
    app,
    operator: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> Tuple[List[Dict[str, Any]], int]:
    with app.app_context():
        query = PredictionBatch.query

        if operator:
            query = query.filter(PredictionBatch.operator == operator)

        if status:
            query = query.filter(PredictionBatch.status == status)

        total = query.count()

        batches = query.order_by(PredictionBatch.created_at.desc()).limit(limit).offset(offset).all()

        result = []
        for batch in batches:
            batch_dict = batch.to_dict()

            override_count = db.session.query(HumanOverride).join(
                BatchTicket, HumanOverride.ticket_id == BatchTicket.ticket_id
            ).filter(BatchTicket.batch_id == batch.id).count()

            batch_dict['override_count'] = override_count
            result.append(batch_dict)

        return result, total


def get_batch_tickets(
    batch_id: int,
    app,
    status: Optional[str] = None,
    low_confidence_only: bool = False,
    overridden_only: bool = False,
    limit: int = 1000,
    offset: int = 0
) -> Tuple[List[Dict[str, Any]], int]:
    with app.app_context():
        batch = PredictionBatch.query.get(batch_id)
        if batch is None:
            return [], 0

        query = BatchTicket.query.filter_by(batch_id=batch_id)

        if status:
            query = query.filter(BatchTicket.status == status)

        if low_confidence_only:
            query = query.filter(
                BatchTicket.confidence < LOW_CONFIDENCE_THRESHOLD,
                BatchTicket.status == 'success'
            )

        if overridden_only:
            query = query.filter(BatchTicket.has_override == True)

        total = query.count()

        batch_tickets = query.order_by(BatchTicket.row_index).limit(limit).offset(offset).all()

        result = []
        for bt in batch_tickets:
            bt_dict = bt.to_dict()

            if bt.status == 'success' and bt.predicted_queue:
                bt_dict['queue_suggestion'] = get_queue_suggestion(bt.predicted_queue, bt.confidence or 0)

            if bt.ticket_id:
                override = HumanOverride.query.filter_by(ticket_id=bt.ticket_id).first()
                if override:
                    bt_dict['override'] = override.to_dict()
                    bt_dict['has_override'] = True

            result.append(bt_dict)

        return result, total


def export_batch_results(
    batch_id: int,
    app,
    format_type: str = 'csv',
    include_failed: bool = True,
    operator: Optional[str] = None
) -> Tuple[bool, Dict[str, Any]]:
    try:
        if not operator or not operator.strip():
            return False, {'error': '操作者不能为空，无权下载'}

        with app.app_context():
            batch = PredictionBatch.query.get(batch_id)
            if batch is None:
                return False, {'error': '批次不存在'}

            query = BatchTicket.query.filter_by(batch_id=batch_id)
            if not include_failed:
                query = query.filter_by(status='success')

            batch_tickets = query.order_by(BatchTicket.row_index).all()

            rows = []
            for bt in batch_tickets:
                row = {
                    '行号': bt.row_index,
                    '工单ID': bt.ticket_id or '',
                    '标题': bt.title,
                    '内容': bt.content,
                    '渠道': bt.channel,
                    '状态': '成功' if bt.status == 'success' else '失败',
                    '预测队列': bt.predicted_queue or '',
                    '置信度': f"{(bt.confidence * 100):.2f}%" if bt.confidence is not None else '',
                    '是否已改判': '是' if bt.has_override else '否',
                    '错误信息': bt.error_message or ''
                }

                if bt.ticket_id:
                    override = HumanOverride.query.filter_by(ticket_id=bt.ticket_id).first()
                    if override:
                        row['改判后队列'] = override.corrected_queue
                        row['改判操作者'] = override.operator
                        row['改判原因'] = override.reason or ''

                rows.append(row)

            df = pd.DataFrame(rows)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"batch_{batch.batch_uid}_results_{timestamp}.{format_type}"
            file_path = os.path.join(BATCH_RESULT_DIR, filename)

            if format_type == 'csv':
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
            elif format_type == 'xlsx':
                df.to_excel(file_path, index=False, engine='openpyxl')
            else:
                return False, {'error': f'不支持的格式: {format_type}'}

            return True, {
                'file_path': file_path,
                'filename': filename,
                'row_count': len(rows),
                'format': format_type
            }

    except Exception as e:
        logger.error(f"导出批次 {batch_id} 结果失败: {str(e)}", exc_info=True)
        return False, {'error': f'导出失败: {str(e)}'}


def update_override_flag(ticket_id: int, app):
    with app.app_context():
        batch_tickets = BatchTicket.query.filter_by(ticket_id=ticket_id).all()
        for bt in batch_tickets:
            bt.has_override = True
        db.session.commit()
