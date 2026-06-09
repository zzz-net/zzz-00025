import os
from datetime import datetime, date

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename

from src.database import db, init_db
from src.models import *
from src.csv_importer import import_csv, list_datasets, get_dataset
from src.classifier import train_model
from src.predictor import predict_ticket, get_active_model, get_queue_suggestion, predict_batch
from src.human_override import create_override, list_overrides, get_override_stats
from src.rollback import list_rollback_candidates, rollback_to_version, get_version_history, get_active_version
from src.report_exporter import export_audit_report, export_model_comparison_report
from src.batch_manager import (
    create_batch, process_batch, get_batch, list_batches,
    get_batch_tickets, export_batch_results
)
import config


def create_app():
    app = Flask(__name__)

    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{config.DB_PATH}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['UPLOAD_FOLDER'] = config.DATASET_DIR

    init_db(app)

    os.makedirs(config.DATASET_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    os.makedirs(config.BATCH_RESULT_DIR, exist_ok=True)

    def json_response(success, message, data=None):
        return jsonify({
            'success': success,
            'message': message,
            'data': data or {}
        })

    @app.route('/')
    def index():
        active_model_obj = get_active_model(app)
        datasets = list_datasets(app)
        version_history = get_version_history(app)
        today = date.today()

        with app.app_context():
            today_tickets = Ticket.query.filter(
                db.func.date(Ticket.predicted_at) == today
            ).count()

            total_tickets = Ticket.query.count()

            recent_models = ModelVersion.query.order_by(ModelVersion.trained_at.desc()).limit(5).all()
            recent_models_data = []
            for mv in recent_models:
                mv_dict = mv.to_dict()
                dataset = Dataset.query.get(mv.dataset_id)
                mv_dict['dataset_name'] = dataset.name if dataset else '未知'
                mv_dict['accuracy'] = mv.metrics.get('overall', {}).get('accuracy') if mv.metrics else None
                mv_dict['created_at'] = mv.trained_at or mv.dataset.created_at
                recent_models_data.append(mv_dict)

            recent_tickets = Ticket.query.order_by(Ticket.created_at.desc()).limit(5).all()
            recent_tickets_data = [t.to_dict() for t in recent_tickets]

        has_active_model = active_model_obj is not None
        
        active_model_dict = None
        if active_model_obj:
            active_model_dict = active_model_obj.to_dict()
            with app.app_context():
                dataset = Dataset.query.get(active_model_obj.dataset_id)
                if dataset:
                    active_model_dict['dataset_name'] = dataset.name
            active_model_dict['accuracy'] = active_model_obj.metrics.get('overall', {}).get('accuracy') if active_model_obj.metrics else None

        stats = {
            'has_active_model': has_active_model,
            'active_model': active_model_dict,
            'dataset_count': len(datasets),
            'model_version_count': len(version_history),
            'today_predictions': today_tickets,
            'total_predictions': total_tickets
        }

        return render_template('index.html',
                               stats=stats,
                               recent_models=recent_models_data,
                               recent_tickets=recent_tickets_data)

    @app.route('/import')
    def import_page():
        datasets = list_datasets(app)
        return render_template('import.html', datasets=datasets)

    @app.route('/train')
    def train_page():
        datasets = list_datasets(app)
        with app.app_context():
            version_objs = ModelVersion.query.order_by(ModelVersion.trained_at.desc()).all()
        
        versions_data = []
        for v in version_objs:
            v_dict = v.to_dict()
            with app.app_context():
                dataset = Dataset.query.get(v.dataset_id)
                if dataset:
                    v_dict['dataset_name'] = dataset.name
            v_dict['accuracy'] = v.metrics.get('overall', {}).get('accuracy') if v.metrics else None
            versions_data.append(v_dict)
        
        return render_template('train.html', datasets=datasets, versions=versions_data)

    @app.route('/evaluate')
    def evaluate_page():
        metric_explanations = config.METRIC_EXPLANATIONS
        
        with app.app_context():
            active_version_obj = ModelVersion.query.filter_by(is_active=True).first()
            version_objs = ModelVersion.query.order_by(ModelVersion.trained_at.desc()).all()
        
        active_model = None
        if active_version_obj:
            active_model = active_version_obj.to_dict()
            with app.app_context():
                dataset = Dataset.query.get(active_version_obj.dataset_id)
                if dataset:
                    active_model['dataset_name'] = dataset.name
            active_model['accuracy'] = active_version_obj.metrics.get('overall', {}).get('accuracy') if active_version_obj.metrics else None
        
        versions_data = []
        for v in version_objs:
            v_dict = v.to_dict()
            with app.app_context():
                dataset = Dataset.query.get(v.dataset_id)
                if dataset:
                    v_dict['dataset_name'] = dataset.name
            v_dict['accuracy'] = v.metrics.get('overall', {}).get('accuracy') if v.metrics else None
            versions_data.append(v_dict)
        
        return render_template('evaluate.html',
                               active_model=active_model,
                               versions=versions_data,
                               metric_explanations=metric_explanations)

    @app.route('/predict')
    def predict_page():
        with app.app_context():
            active_version_obj = ModelVersion.query.filter_by(is_active=True).first()
        active_model = None
        if active_version_obj:
            active_model = active_version_obj.to_dict()
            active_model['accuracy'] = active_version_obj.metrics.get('overall', {}).get('accuracy') if active_version_obj.metrics else None
        return render_template('predict.html', active_model=active_model)

    @app.route('/tickets')
    def tickets_page():
        with app.app_context():
            tickets = Ticket.query.order_by(Ticket.created_at.desc()).all()
            tickets_data = []
            for ticket in tickets:
                suggestion = get_queue_suggestion(ticket.predicted_queue, ticket.confidence)
                ticket_dict = ticket.to_dict()
                ticket_dict['suggestion'] = suggestion
                tickets_data.append(ticket_dict)
        return render_template('tickets.html', tickets=tickets_data)

    @app.route('/override/<int:ticket_id>')
    def override_page(ticket_id):
        with app.app_context():
            ticket = Ticket.query.get(ticket_id)
            if ticket is None:
                flash(f'工单 ID {ticket_id} 不存在', 'error')
                return redirect(url_for('tickets_page'))
        queues = list(config.SUPPORTED_QUEUES)
        return render_template('override.html', ticket=ticket.to_dict(), queues=queues)

    @app.route('/rollback')
    def rollback_page():
        candidates = list_rollback_candidates(app)
        
        with app.app_context():
            active_version_obj = ModelVersion.query.filter_by(is_active=True).first()
            version_objs = ModelVersion.query.order_by(ModelVersion.trained_at.desc()).all()
        
        active_model = None
        if active_version_obj:
            active_model = active_version_obj.to_dict()
            with app.app_context():
                dataset = Dataset.query.get(active_version_obj.dataset_id)
                if dataset:
                    active_model['dataset_name'] = dataset.name
            active_model['accuracy'] = active_version_obj.metrics.get('overall', {}).get('accuracy') if active_version_obj.metrics else None
        
        versions_data = []
        for v in version_objs:
            v_dict = v.to_dict()
            with app.app_context():
                dataset = Dataset.query.get(v.dataset_id)
                if dataset:
                    v_dict['dataset_name'] = dataset.name
            v_dict['accuracy'] = v.metrics.get('overall', {}).get('accuracy') if v.metrics else None
            versions_data.append(v_dict)
        
        return render_template('rollback.html',
                               candidates=candidates,
                               versions=versions_data,
                               active_model=active_model)

    @app.route('/reports')
    def reports_page():
        today = date.today().isoformat()
        return render_template('reports.html', today=today)

    @app.route('/batches')
    def batches_page():
        return render_template('batches.html')

    @app.route('/batches/<int:batch_id>')
    def batch_detail_page(batch_id):
        return render_template('batch_detail.html', batch_id=batch_id)

    @app.route('/api/import', methods=['POST'])
    def api_import():
        try:
            if 'file' not in request.files:
                return json_response(False, '未找到上传文件')

            file = request.files['file']
            if file.filename == '':
                return json_response(False, '未选择文件')

            if not file.filename.endswith('.csv'):
                return json_response(False, '仅支持 CSV 格式文件')

            filename = secure_filename(file.filename)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_{filename}')
            file.save(temp_path)

            result = import_csv(temp_path, app)

            os.remove(temp_path)

            if result.get('is_valid'):
                return json_response(True, '导入成功', {
                    'dataset_id': result.get('dataset_id'),
                    'row_count': result.get('row_count'),
                    'dataset': get_dataset(result.get('dataset_id'), app)
                })
            else:
                return json_response(False, '数据验证失败', {
                    'dataset_id': result.get('dataset_id'),
                    'validation_result': result.get('validation_result')
                })

        except Exception as e:
            return json_response(False, f'导入失败: {str(e)}')

    @app.route('/api/train/<int:dataset_id>', methods=['POST'])
    def api_train(dataset_id):
        try:
            dataset = get_dataset(dataset_id, app)
            if dataset is None:
                return json_response(False, f'数据集 ID {dataset_id} 不存在')

            if dataset.get('status') != 'completed':
                return json_response(False, '数据集状态无效，仅可使用状态为 completed 的数据集进行训练')

            result = train_model(dataset_id, app)

            if result.get('success'):
                return json_response(True, '模型训练成功', {
                    'model_version_id': result.get('model_version_id'),
                    'version': result.get('version'),
                    'metrics': result.get('metrics'),
                    'classification_report': result.get('classification_report')
                })
            else:
                return json_response(False, result.get('error', '训练失败'), {
                    'model_version_id': result.get('model_version_id')
                })

        except Exception as e:
            return json_response(False, f'训练失败: {str(e)}')

    @app.route('/api/predict', methods=['POST'])
    def api_predict():
        try:
            data = request.get_json() or request.form

            title = data.get('title', '').strip()
            content = data.get('content', '').strip()
            channel = data.get('channel', '').strip()

            if not title or not content or not channel:
                return json_response(False, '缺少必填字段: title, content, channel')

            if channel not in config.SUPPORTED_CHANNELS:
                return json_response(False, f'不支持的渠道: {channel}。支持的渠道: {", ".join(config.SUPPORTED_CHANNELS)}')

            result = predict_ticket(title, content, channel, app)

            suggestion = get_queue_suggestion(result['predicted_queue'], result['confidence'])

            with app.app_context():
                ticket = Ticket.query.order_by(Ticket.id.desc()).first()
                ticket_id = ticket.id if ticket else None

            return json_response(True, '预测成功', {
                'ticket_id': ticket_id,
                'predicted_queue': result['predicted_queue'],
                'confidence': result['confidence'],
                'model_version_id': result['model_version_id'],
                'queue_suggestion': suggestion
            })

        except Exception as e:
            return json_response(False, f'预测失败: {str(e)}')

    @app.route('/api/predict/batch', methods=['POST'])
    def api_predict_batch():
        try:
            data = request.get_json()
            tickets = data.get('tickets', [])

            if not tickets:
                return json_response(False, '未提供工单数据')

            if len(tickets) > 1000:
                return json_response(False, '批量预测最多支持 1000 条工单')

            results = predict_batch(tickets, app)

            with app.app_context():
                latest_tickets = Ticket.query.order_by(Ticket.id.desc()).limit(len(results)).all()
                ticket_ids = [t.id for t in reversed(latest_tickets)]

            response_data = []
            for i, result in enumerate(results):
                suggestion = get_queue_suggestion(result['predicted_queue'], result['confidence'])
                response_data.append({
                    'ticket_id': ticket_ids[i] if i < len(ticket_ids) else None,
                    'predicted_queue': result['predicted_queue'],
                    'confidence': result['confidence'],
                    'model_version_id': result['model_version_id'],
                    'queue_suggestion': suggestion
                })

            return json_response(True, f'批量预测完成，共 {len(results)} 条', {
                'results': response_data,
                'total': len(results)
            })

        except Exception as e:
            return json_response(False, f'批量预测失败: {str(e)}')

    @app.route('/api/override/<int:ticket_id>', methods=['POST'])
    def api_override(ticket_id):
        try:
            data = request.get_json() or request.form

            corrected_queue = data.get('corrected_queue', '').strip()
            operator = data.get('operator', '').strip()
            reason = data.get('reason', '').strip()

            if not corrected_queue or not operator:
                return json_response(False, '缺少必填字段: corrected_queue, operator')

            if corrected_queue not in config.SUPPORTED_QUEUES:
                return json_response(False, f'不支持的队列: {corrected_queue}。支持的队列: {", ".join(config.SUPPORTED_QUEUES)}')

            corrected_queue_internal = config.QUEUE_NAME_TO_TAG.get(corrected_queue, corrected_queue)
            corrected_queue_id = config.QUEUE_MAPPING.get(corrected_queue_internal, corrected_queue_internal)

            result = create_override(ticket_id, corrected_queue_id, operator, reason, app)

            return json_response(True, '人工改判成功', {
                'override': result
            })

        except Exception as e:
            return json_response(False, f'人工改判失败: {str(e)}')

    @app.route('/api/rollback/<int:model_version_id>', methods=['POST'])
    def api_rollback(model_version_id):
        try:
            result = rollback_to_version(model_version_id, app)

            if result.get('success'):
                return json_response(True, result.get('message', '回滚成功'), {
                    'previous_version': result.get('previous_version'),
                    'current_version': result.get('current_version')
                })
            else:
                return json_response(False, result.get('message', '回滚失败'))

        except Exception as e:
            return json_response(False, f'回滚失败: {str(e)}')

    @app.route('/api/report/audit', methods=['POST'])
    def api_report_audit():
        try:
            data = request.get_json() or request.form

            start_date = data.get('start_date', '').strip()
            end_date = data.get('end_date', '').strip()
            format_type = data.get('format', 'xlsx').strip()

            if not start_date or not end_date:
                return json_response(False, '缺少必填字段: start_date, end_date')

            if format_type not in ['xlsx', 'csv']:
                return json_response(False, '不支持的格式，仅支持 xlsx 或 csv')

            try:
                datetime.fromisoformat(start_date)
                datetime.fromisoformat(end_date)
            except ValueError:
                return json_response(False, '日期格式错误，请使用 ISO 格式 (YYYY-MM-DD)')

            result = export_audit_report(start_date, end_date, app, format_type)

            return json_response(True, '复核报告导出成功', {
                'output_path': result.get('output_path'),
                'format': result.get('format'),
                'tickets_count': result.get('tickets_count'),
                'overrides_count': result.get('overrides_count')
            })

        except Exception as e:
            return json_response(False, f'导出复核报告失败: {str(e)}')

    @app.route('/api/report/comparison', methods=['POST'])
    def api_report_comparison():
        try:
            data = request.get_json() or request.form
            format_type = data.get('format', 'xlsx').strip()

            if format_type not in ['xlsx', 'csv']:
                return json_response(False, '不支持的格式，仅支持 xlsx 或 csv')

            result = export_model_comparison_report(app, format_type)

            return json_response(True, '模型对比报告导出成功', {
                'output_path': result.get('output_path'),
                'format': result.get('format'),
                'models_count': result.get('models_count')
            })

        except Exception as e:
            return json_response(False, f'导出模型对比报告失败: {str(e)}')

    @app.route('/api/batch/upload', methods=['POST'])
    def api_batch_upload():
        try:
            operator = request.form.get('operator', '').strip()
            if not operator:
                return json_response(False, '操作者不能为空，请填写 operator 字段'), 403

            if 'file' not in request.files:
                return json_response(False, '未找到上传文件')

            file = request.files['file']
            if file.filename == '':
                return json_response(False, '未选择文件')

            if not file.filename.endswith('.csv'):
                return json_response(False, '仅支持 CSV 格式文件')

            import tempfile
            temp_dir = tempfile.gettempdir()
            temp_filename = f"temp_batch_{datetime.now().strftime('%Y%m%d%H%M%S')}_{os.urandom(4).hex()}.csv"
            temp_path = os.path.join(temp_dir, temp_filename)
            file.save(temp_path)

            original_filename = secure_filename(file.filename)

            success, result = create_batch(original_filename, operator, temp_path, app)

            if not success:
                os.remove(temp_path)
                if 'existing_batch' in result:
                    return json_response(False, result['error'], {
                        'existing_batch': result['existing_batch'],
                        'is_duplicate': True
                    }), 409
                return json_response(False, result['error']), 400

            batch_id = result['batch_id']

            process_result = process_batch(batch_id, temp_path, app)

            os.remove(temp_path)

            if not process_result['success']:
                return json_response(False, process_result.get('error', '批次处理失败'), {
                    'batch_id': batch_id
                })

            batch_info = get_batch(batch_id, app, include_details=True)

            return json_response(True, f'批次预测完成，成功 {process_result["success_count"]} 条，失败 {process_result["failed_count"]} 条', {
                'batch_id': batch_id,
                'batch_uid': result['batch_uid'],
                'batch': batch_info,
                'success_count': process_result['success_count'],
                'failed_count': process_result['failed_count'],
                'total_count': process_result['total_count']
            })

        except Exception as e:
            return json_response(False, f'批量预测失败: {str(e)}'), 500

    @app.route('/api/batches', methods=['GET'])
    def api_list_batches():
        try:
            operator = request.args.get('operator', '').strip()
            status = request.args.get('status', '').strip() or None
            limit = int(request.args.get('limit', 100))
            offset = int(request.args.get('offset', 0))

            batches, total = list_batches(app, operator=operator or None, status=status, limit=limit, offset=offset)

            return json_response(True, '查询成功', {
                'batches': batches,
                'total': total,
                'limit': limit,
                'offset': offset
            })

        except Exception as e:
            return json_response(False, f'查询批次列表失败: {str(e)}'), 500

    @app.route('/api/batch/<int:batch_id>', methods=['GET'])
    def api_get_batch(batch_id):
        try:
            batch = get_batch(batch_id, app, include_details=True)
            if batch is None:
                return json_response(False, '批次不存在'), 404

            return json_response(True, '查询成功', {'batch': batch})

        except Exception as e:
            return json_response(False, f'查询批次详情失败: {str(e)}'), 500

    @app.route('/api/batch/<int:batch_id>/tickets', methods=['GET'])
    def api_get_batch_tickets(batch_id):
        try:
            status = request.args.get('status', '').strip() or None
            low_confidence_only = request.args.get('low_confidence_only', 'false').lower() == 'true'
            overridden_only = request.args.get('overridden_only', 'false').lower() == 'true'
            limit = int(request.args.get('limit', 1000))
            offset = int(request.args.get('offset', 0))

            tickets, total = get_batch_tickets(
                batch_id, app,
                status=status,
                low_confidence_only=low_confidence_only,
                overridden_only=overridden_only,
                limit=limit,
                offset=offset
            )

            return json_response(True, '查询成功', {
                'tickets': tickets,
                'total': total,
                'limit': limit,
                'offset': offset
            })

        except Exception as e:
            return json_response(False, f'查询批次工单失败: {str(e)}'), 500

    @app.route('/api/batch/<int:batch_id>/export', methods=['POST'])
    def api_export_batch(batch_id):
        try:
            data = request.get_json() or request.form
            operator = data.get('operator', '').strip()
            format_type = data.get('format', 'csv').strip().lower()
            include_failed = data.get('include_failed', 'true').lower() != 'false'

            if not operator:
                return json_response(False, '操作者不能为空，无权下载'), 403

            if format_type not in ['csv', 'xlsx']:
                return json_response(False, '不支持的格式，仅支持 csv 或 xlsx')

            success, result = export_batch_results(
                batch_id, app,
                format_type=format_type,
                include_failed=include_failed,
                operator=operator
            )

            if not success:
                return json_response(False, result['error']), 400

            return json_response(True, '导出成功', {
                'download_url': f'/api/batch/{batch_id}/download?filename={result["filename"]}',
                'filename': result['filename'],
                'row_count': result['row_count'],
                'format': result['format']
            })

        except Exception as e:
            return json_response(False, f'导出批次结果失败: {str(e)}'), 500

    @app.route('/api/batch/<int:batch_id>/download', methods=['GET'])
    def api_download_batch(batch_id):
        try:
            from flask import send_from_directory

            operator = request.args.get('operator', '').strip()
            if not operator:
                return json_response(False, '操作者不能为空，无权下载'), 403

            filename = request.args.get('filename', '')
            if not filename:
                return json_response(False, '文件名不能为空'), 400

            if '..' in filename or filename.startswith('/'):
                return json_response(False, '无效的文件名'), 400

            file_path = os.path.join(config.BATCH_RESULT_DIR, filename)
            if not os.path.exists(file_path):
                return json_response(False, '文件不存在'), 404

            return send_from_directory(
                config.BATCH_RESULT_DIR,
                filename,
                as_attachment=True,
                download_name=filename
            )

        except Exception as e:
            return json_response(False, f'下载失败: {str(e)}'), 500

    @app.errorhandler(404)
    def not_found(error):
        if request.path.startswith('/api/'):
            return json_response(False, 'API 端点不存在'), 404
        return render_template('404.html'), 404

    @app.errorhandler(500)
    def internal_error(error):
        if request.path.startswith('/api/'):
            return json_response(False, f'服务器内部错误: {str(error)}'), 500
        return render_template('500.html'), 500

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
