import os
from datetime import datetime
from typing import Dict, List, Any

import pandas as pd

from config import REPORT_DIR
from .database import db
from .models import Ticket, HumanOverride, ModelVersion


os.makedirs(REPORT_DIR, exist_ok=True)


def export_audit_report(start_date: str, end_date: str, app, format: str = 'xlsx') -> Dict[str, Any]:
    start_dt = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)

    with app.app_context():
        tickets_query = db.session.query(Ticket).filter(
            Ticket.predicted_at >= start_dt,
            Ticket.predicted_at <= end_dt
        )
        tickets = tickets_query.all()

        tickets_data = []
        for ticket in tickets:
            is_overridden = db.session.query(HumanOverride).filter_by(ticket_id=ticket.id).first() is not None
            tickets_data.append({
                '工单ID': ticket.id,
                '标题': ticket.title,
                '渠道': ticket.channel,
                '预测队列': ticket.predicted_queue,
                '置信度': ticket.confidence,
                '实际队列': ticket.actual_queue,
                '是否被改判': '是' if is_overridden else '否'
            })
        tickets_df = pd.DataFrame(tickets_data)

        overrides_query = db.session.query(HumanOverride).filter(
            HumanOverride.created_at >= start_dt,
            HumanOverride.created_at <= end_dt
        )
        overrides = overrides_query.all()

        overrides_data = []
        for override in overrides:
            overrides_data.append({
                '工单ID': override.ticket_id,
                '原预测': override.original_prediction,
                '改判后队列': override.corrected_queue,
                '操作者': override.operator,
                '原因': override.reason,
                '时间': override.created_at.isoformat() if override.created_at else None
            })
        overrides_df = pd.DataFrame(overrides_data)

        stats_data = []
        if not tickets_df.empty:
            high_confidence_threshold = 0.8
            queues = tickets_df['预测队列'].dropna().unique()

            for queue in queues:
                queue_tickets = tickets_df[tickets_df['预测队列'] == queue]
                total = len(queue_tickets)
                if total == 0:
                    continue

                correct = len(queue_tickets[queue_tickets['预测队列'] == queue_tickets['实际队列']])
                accuracy = correct / total if total > 0 else 0

                overridden_count = len(queue_tickets[queue_tickets['是否被改判'] == '是'])
                override_rate = overridden_count / total if total > 0 else 0

                high_conf = len(queue_tickets[queue_tickets['置信度'] >= high_confidence_threshold])
                high_conf_ratio = high_conf / total if total > 0 else 0

                stats_data.append({
                    '队列': queue,
                    '预测总数': total,
                    '正确数': correct,
                    '准确率': f'{accuracy:.2%}',
                    '改判数': overridden_count,
                    '改判率': f'{override_rate:.2%}',
                    '高置信度预测比例': f'{high_conf_ratio:.2%}'
                })
        stats_df = pd.DataFrame(stats_data)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'audit_report_{timestamp}'

        dataframes = {
            '工单预测记录': tickets_df,
            '人工改判记录': overrides_df,
            '统计汇总': stats_df
        }

        if format == 'xlsx':
            output_path = os.path.join(REPORT_DIR, f'{filename}.xlsx')
            _export_to_excel(dataframes, output_path)
        elif format == 'csv':
            output_path = os.path.join(REPORT_DIR, filename)
            _export_to_csv(dataframes, output_path)
        else:
            raise ValueError(f"不支持的导出格式: {format}")

        return {
            'success': True,
            'format': format,
            'output_path': output_path,
            'tickets_count': len(tickets),
            'overrides_count': len(overrides)
        }


def export_model_comparison_report(app, format: str = 'xlsx') -> Dict[str, Any]:
    with app.app_context():
        models_query = db.session.query(ModelVersion).filter(
            ModelVersion.status.in_(['completed', 'active', 'rolled_back'])
        )
        models = models_query.all()

        comparison_data = []
        for model in models:
            metrics = model.metrics or {}
            comparison_data.append({
                '模型ID': model.id,
                '版本号': model.version,
                '数据集ID': model.dataset_id,
                '训练时间': model.trained_at.isoformat() if model.trained_at else None,
                '是否激活': '是' if model.is_active else '否',
                **metrics
            })
        comparison_df = pd.DataFrame(comparison_data)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'model_comparison_{timestamp}'

        dataframes = {
            '模型对比': comparison_df
        }

        if format == 'xlsx':
            output_path = os.path.join(REPORT_DIR, f'{filename}.xlsx')
            _export_to_excel(dataframes, output_path)
        elif format == 'csv':
            output_path = os.path.join(REPORT_DIR, filename)
            _export_to_csv(dataframes, output_path)
        else:
            raise ValueError(f"不支持的导出格式: {format}")

        return {
            'success': True,
            'format': format,
            'output_path': output_path,
            'models_count': len(models)
        }


def _export_to_excel(dataframes: Dict[str, pd.DataFrame], output_path: str) -> None:
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet_name, df in dataframes.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def _export_to_csv(dataframes: Dict[str, pd.DataFrame], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for filename, df in dataframes.items():
        file_path = os.path.join(output_dir, f'{filename}.csv')
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
