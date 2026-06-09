import os
import sys
import time
import uuid
import json
import csv
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from src.database import db, init_db
from src.csv_importer import import_csv, get_dataset
from src.classifier import train_model
from src.predictor import predict_ticket, predict_batch
from src.rollback import (
    activate_model, compare_versions, get_active_version,
    get_all_versions_with_details, get_available_versions_for_comparison,
    export_comparison_result, get_usage_window_config,
    get_operation_history
)
from src.models import ModelVersion, Ticket, OperationLog
import config

SAMPLE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'samples')
TRAINING_DATA = os.path.join(SAMPLE_DATA_DIR, 'sample_training_data.csv')
BAD_DATA = os.path.join(SAMPLE_DATA_DIR, 'sample_bad_data.csv')

RUN_ID = uuid.uuid4().hex[:8]


def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ticket_routing.db")}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    init_db(app)
    return app


def print_header(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def print_step(step_num, description):
    print(f"\n{'-'*80}")
    print(f"  步骤 {step_num}: {description}")
    print(f"{'-'*80}")


def print_result(success, message):
    status = "[OK]" if success else "[FAIL]"
    print(f"  {status} {message}")
    return success


def get_unique_title(base_name):
    return f"{base_name}_{RUN_ID}_{int(time.time() * 1000)}"


def cleanup_test_data(app, run_id=None):
    with app.app_context():
        if run_id:
            Ticket.query.filter(Ticket.title.contains(run_id)).delete()
        OperationLog.query.delete()
        db.session.commit()


def create_failed_version(app):
    """创建一个失败的模型版本"""
    import_result = import_csv(BAD_DATA, app)
    dataset_id = import_result['dataset_id']
    train_result = train_model(dataset_id, app)
    assert not train_result.get('success'), "坏数据训练应该失败"
    return train_result.get('model_version_id')


def create_training_version(app, dataset_id):
    """创建一个状态为 training 的模型版本"""
    with app.app_context():
        model_version = ModelVersion(
            version=f"test_training_{RUN_ID}",
            dataset_id=dataset_id,
            model_path="/nonexistent/path.pkl",
            status='training',
            trained_at=datetime.utcnow()
        )
        db.session.add(model_version)
        db.session.commit()
        return model_version.id


def create_missing_file_version(app, dataset_id):
    """创建一个文件缺失的模型版本"""
    with app.app_context():
        model_version = ModelVersion(
            version=f"test_missing_{RUN_ID}",
            dataset_id=dataset_id,
            model_path="/nonexistent/path.pkl",
            status='completed',
            trained_at=datetime.utcnow(),
            metrics={'overall': {'accuracy': 0.9, 'precision': 0.9, 'recall': 0.9, 'f1': 0.9}}
        )
        db.session.add(model_version)
        db.session.commit()
        return model_version.id


def test_scenario_11_comparison_validation():
    """
    场景11: 版本对比校验 - 非法版本拦截
    测试 failed、training、缺文件的版本不能被对比
    """
    print_header("场景11: 版本对比校验 - 非法版本拦截")
    
    app = create_app()
    cleanup_test_data(app, RUN_ID)
    
    try:
        print_step("11.1", "准备正常版本和非法版本")
        import_result_good = import_csv(TRAINING_DATA, app)
        dataset_id_good = import_result_good['dataset_id']
        train_result_good1 = train_model(dataset_id_good, app)
        model_id_good1 = train_result_good1['model_version_id']
        
        time.sleep(1)
        import_result_good2 = import_csv(TRAINING_DATA, app)
        dataset_id_good2 = import_result_good2['dataset_id']
        train_result_good2 = train_model(dataset_id_good2, app)
        model_id_good2 = train_result_good2['model_version_id']
        
        failed_model_id = create_failed_version(app)
        training_model_id = create_training_version(app, dataset_id_good)
        missing_file_model_id = create_missing_file_version(app, dataset_id_good)
        
        print_result(True, f"准备完成: 正常版本 {model_id_good1}, {model_id_good2}; 失败版本 {failed_model_id}; 训练中 {training_model_id}; 缺文件 {missing_file_model_id}")
        
        print_step("11.2", "测试对比 failed 版本")
        compare_result = compare_versions(model_id_good1, failed_model_id, app, operator='test_user')
        assert not compare_result.get('success'), "对比 failed 版本应该失败"
        assert compare_result.get('error_code') == 'FAILED_VERSION', f"错误码应为 FAILED_VERSION，实际: {compare_result.get('error_code')}"
        assert '可用版本' in compare_result.get('error', ''), "错误信息应包含可用版本建议"
        assert 'available_versions' in compare_result, "应返回可用版本列表"
        
        with app.app_context():
            logs = OperationLog.query.filter_by(operation_type='compare_versions', status='rejected').all()
            assert len(logs) >= 1, "应记录拒绝的对比操作日志"
        
        print_result(True, "对比 failed 版本正确拒绝，错误信息包含建议，日志已记录")
        
        print_step("11.3", "测试对比 training 版本")
        compare_result = compare_versions(model_id_good1, training_model_id, app, operator='test_user')
        assert not compare_result.get('success'), "对比 training 版本应该失败"
        assert compare_result.get('error_code') == 'TRAINING_VERSION', f"错误码应为 TRAINING_VERSION，实际: {compare_result.get('error_code')}"
        assert '可用版本' in compare_result.get('error', ''), "错误信息应包含可用版本建议"
        
        print_result(True, "对比 training 版本正确拒绝，错误信息包含建议")
        
        print_step("11.4", "测试对比缺文件版本")
        compare_result = compare_versions(model_id_good1, missing_file_model_id, app, operator='test_user')
        assert not compare_result.get('success'), "对比缺文件版本应该失败"
        assert compare_result.get('error_code') in ['FILE_MISSING', 'VECTORIZER_MISSING'], f"错误码应为 FILE_MISSING 或 VECTORIZER_MISSING，实际: {compare_result.get('error_code')}"
        assert '可用版本' in compare_result.get('error', ''), "错误信息应包含可用版本建议"
        
        print_result(True, "对比缺文件版本正确拒绝，错误信息包含建议")
        
        print_step("11.5", "测试两个都是非法版本")
        compare_result = compare_versions(failed_model_id, training_model_id, app, operator='test_user')
        assert not compare_result.get('success'), "两个非法版本对比应该失败"
        assert 'available_versions' in compare_result, "应返回可用版本列表"
        assert len(compare_result['available_versions']) >= 2, f"至少应有2个可用版本，实际: {len(compare_result['available_versions'])}"
        
        print_result(True, "两个非法版本对比正确拒绝，返回所有可用版本")
        
        print_step("11.6", "测试对比不存在的版本")
        compare_result = compare_versions(99999, model_id_good1, app, operator='test_user')
        assert not compare_result.get('success'), "对比不存在的版本应该失败"
        assert compare_result.get('error_code') == 'NOT_FOUND', f"错误码应为 NOT_FOUND，实际: {compare_result.get('error_code')}"
        
        print_result(True, "对比不存在的版本正确拒绝")
        
        print_step("11.7", "测试对比相同版本")
        compare_result = compare_versions(model_id_good1, model_id_good1, app, operator='test_user')
        assert not compare_result.get('success'), "对比相同版本应该失败"
        assert compare_result.get('error_code') == 'SAME_VERSION', f"错误码应为 SAME_VERSION，实际: {compare_result.get('error_code')}"
        
        print_result(True, "对比相同版本正确拒绝")
        
        print_step("11.8", "测试正常版本对比成功")
        compare_result = compare_versions(model_id_good1, model_id_good2, app, operator='test_user')
        assert compare_result.get('success'), "正常版本对比应该成功"
        assert 'version_a' in compare_result
        assert 'version_b' in compare_result
        assert 'metrics_diff' in compare_result
        assert 'window_days' in compare_result
        
        with app.app_context():
            success_logs = OperationLog.query.filter_by(operation_type='compare_versions', status='success').all()
            assert len(success_logs) >= 1, "应记录成功的对比操作日志"
        
        print_result(True, "正常版本对比成功，日志已记录")
        
        cleanup_test_data(app, RUN_ID)
        print_header("场景11: 全部通过 [OK]")
        return True
        
    except Exception as e:
        print(f"  [FAIL] 场景11异常: {e}")
        import traceback
        traceback.print_exc()
        cleanup_test_data(app, RUN_ID)
        return False


def test_scenario_12_recent_usage_window():
    """
    场景12: 时间窗口统计的最近使用量
    测试最近预测使用量的时间窗口统计功能
    """
    print_header("场景12: 时间窗口统计的最近使用量")
    
    app = create_app()
    cleanup_test_data(app, RUN_ID)
    
    try:
        print_step("12.1", "训练模型并创建不同时间的预测记录")
        import_result = import_csv(TRAINING_DATA, app)
        dataset_id = import_result['dataset_id']
        train_result = train_model(dataset_id, app)
        model_id = train_result['model_version_id']
        
        print_result(True, f"模型训练成功，ID: {model_id}")
        
        print_step("12.2", "创建不同时间的预测记录")
        with app.app_context():
            now = datetime.utcnow()
            
            for i in range(5):
                ticket = Ticket(
                    title=get_unique_title(f"场景12_最近7天_{i}"),
                    content=f"测试内容 {i}",
                    channel="web",
                    predicted_queue="tech_support_queue",
                    confidence=0.9,
                    model_version_id=model_id,
                    predicted_at=now - timedelta(days=i)
                )
                db.session.add(ticket)
            
            for i in range(3):
                ticket = Ticket(
                    title=get_unique_title(f"场景12_8到14天_{i}"),
                    content=f"测试内容 {i}",
                    channel="web",
                    predicted_queue="tech_support_queue",
                    confidence=0.9,
                    model_version_id=model_id,
                    predicted_at=now - timedelta(days=8 + i)
                )
                db.session.add(ticket)
            
            for i in range(2):
                ticket = Ticket(
                    title=get_unique_title(f"场景12_30天以上_{i}"),
                    content=f"测试内容 {i}",
                    channel="web",
                    predicted_queue="tech_support_queue",
                    confidence=0.9,
                    model_version_id=model_id,
                    predicted_at=now - timedelta(days=30 + i)
                )
                db.session.add(ticket)
            
            db.session.commit()
        
        print_result(True, "创建了10条预测记录: 5条最近7天, 3条8-14天, 2条30天以上")
        
        print_step("12.3", "验证默认7天窗口的最近使用量")
        versions = get_all_versions_with_details(app)
        target_version = None
        for v in versions:
            if v['id'] == model_id:
                target_version = v
                break
        
        assert target_version is not None, "找不到目标版本"
        assert target_version['usage_count'] == 10, f"历史总使用量应为10，实际: {target_version['usage_count']}"
        assert target_version['recent_usage_count'] == 5, f"最近7天使用量应为5，实际: {target_version['recent_usage_count']}"
        
        print_result(True, f"历史总使用量: {target_version['usage_count']}, 最近7天: {target_version['recent_usage_count']}")
        
        print_step("12.4", "测试自定义窗口天数（14天）")
        compare_result = compare_versions(model_id, model_id + 1000, app, operator='test_user')
        
        with app.app_context():
            from src.rollback import _get_model_recent_usage_count
            usage_14_days = _get_model_recent_usage_count(model_id, app, window_days=14)
            assert usage_14_days == 8, f"最近14天使用量应为8，实际: {usage_14_days}"
        
        print_result(True, f"最近14天使用量: {usage_14_days}")
        
        print_step("12.5", "测试自定义窗口天数（30天）")
        with app.app_context():
            usage_30_days = _get_model_recent_usage_count(model_id, app, window_days=30)
            assert usage_30_days == 8, f"最近30天使用量应为8，实际: {usage_30_days}"
        
        print_result(True, f"最近30天使用量: {usage_30_days}")
        
        print_step("12.6", "验证配置读取功能")
        config_data = get_usage_window_config(app)
        assert 'window_days' in config_data, "配置缺少 window_days"
        assert config_data['window_days'] == getattr(config, 'USAGE_WINDOW_DAYS', 7), "配置值不正确"
        
        with app.app_context():
            config_logs = OperationLog.query.filter_by(operation_type='config_read', status='success').all()
            assert len(config_logs) >= 1, "应记录配置读取操作日志"
        
        print_result(True, f"配置读取成功，窗口天数: {config_data['window_days']}，日志已记录")
        
        print_step("12.7", "测试可用版本列表只包含合法版本")
        available_versions = get_available_versions_for_comparison(app)
        assert len(available_versions) >= 1, "可用版本列表不应为空"
        
        for v in available_versions:
            assert v['status'] in ['completed', 'active', 'rolled_back'], f"可用版本状态不正确: {v['status']}"
            assert v['model_file_exists'] == True, f"可用版本文件应存在: {v['id']}"
            assert v['vectorizer_file_exists'] == True, f"可用版本向量化器文件应存在: {v['id']}"
        
        print_result(True, f"可用版本列表正确，共 {len(available_versions)} 个版本")
        
        cleanup_test_data(app, RUN_ID)
        print_header("场景12: 全部通过 [OK]")
        return True
        
    except Exception as e:
        print(f"  [FAIL] 场景12异常: {e}")
        import traceback
        traceback.print_exc()
        cleanup_test_data(app, RUN_ID)
        return False


def test_scenario_13_export_comparison():
    """
    场景13: 对比结果导出功能
    测试 CSV 和 JSON 格式的导出功能
    """
    print_header("场景13: 对比结果导出功能")
    
    app = create_app()
    cleanup_test_data(app, RUN_ID)
    
    try:
        print_step("13.1", "准备两个模型版本并进行预测")
        import_result1 = import_csv(TRAINING_DATA, app)
        dataset_id1 = import_result1['dataset_id']
        train_result1 = train_model(dataset_id1, app)
        model_id1 = train_result1['model_version_id']
        
        time.sleep(1)
        import_result2 = import_csv(TRAINING_DATA, app)
        dataset_id2 = import_result2['dataset_id']
        train_result2 = train_model(dataset_id2, app)
        model_id2 = train_result2['model_version_id']
        
        for i in range(3):
            predict_ticket(
                title=get_unique_title(f"场景13_预测_{i}"),
                content=f"测试内容 {i}",
                channel="web",
                app=app
            )
        
        print_result(True, f"准备完成: 模型 {model_id1}, {model_id2}，进行了3次预测")
        
        print_step("13.2", "进行版本对比")
        compare_result = compare_versions(model_id1, model_id2, app, operator='test_user')
        assert compare_result.get('success'), "对比应该成功"
        
        print_result(True, "版本对比成功")
        
        print_step("13.3", "测试导出为 JSON 格式")
        export_result_json = export_comparison_result(compare_result, 'json', app, operator='test_user')
        assert export_result_json.get('success'), f"JSON 导出失败: {export_result_json.get('error')}"
        assert os.path.exists(export_result_json['output_path']), "JSON 文件不存在"
        
        with open(export_result_json['output_path'], 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        assert 'version_a' in json_data, "JSON 缺少 version_a"
        assert 'version_b' in json_data, "JSON 缺少 version_b"
        assert 'metrics_diff' in json_data, "JSON 缺少 metrics_diff"
        assert json_data['version_a']['total_usage'] >= 0, "JSON 缺少 total_usage"
        assert json_data['version_a']['recent_usage'] >= 0, "JSON 缺少 recent_usage"
        assert json_data['version_a']['is_active'] is not None, "JSON 缺少 is_active"
        assert 'trained_at' in json_data['version_a'], "JSON 缺少 trained_at"
        
        with app.app_context():
            export_logs = OperationLog.query.filter_by(operation_type='export_comparison', status='success').all()
            assert len(export_logs) >= 1, "应记录导出操作日志"
        
        print_result(True, f"JSON 导出成功，文件: {export_result_json['filename']}")
        
        print_step("13.4", "测试导出为 CSV 格式")
        export_result_csv = export_comparison_result(compare_result, 'csv', app, operator='test_user')
        assert export_result_csv.get('success'), f"CSV 导出失败: {export_result_csv.get('error')}"
        assert os.path.exists(export_result_csv['output_path']), "CSV 文件不存在"
        
        with open(export_result_csv['output_path'], 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        assert len(rows) > 0, "CSV 文件为空"
        has_accuracy = any('准确率' in str(row) for row in rows)
        has_usage = any('使用量' in str(row) for row in rows)
        has_recent_usage = any('最近' in str(row) and '天使用量' in str(row) for row in rows)
        has_active = any('激活状态' in str(row) for row in rows)
        has_training_time = any('训练时间' in str(row) for row in rows)
        
        assert has_accuracy, "CSV 缺少准确率"
        assert has_usage, "CSV 缺少使用量"
        assert has_recent_usage, "CSV 缺少最近使用量"
        assert has_active, "CSV 缺少激活状态"
        assert has_training_time, "CSV 缺少训练时间"
        
        print_result(True, f"CSV 导出成功，文件: {export_result_csv['filename']}")
        
        print_step("13.5", "测试导出无效对比数据")
        invalid_data = {'success': False, 'error': 'test error'}
        export_result = export_comparison_result(invalid_data, 'json', app, operator='test_user')
        assert not export_result.get('success'), "无效数据导出应该失败"
        assert export_result.get('error_code') == 'INVALID_DATA', f"错误码应为 INVALID_DATA，实际: {export_result.get('error_code')}"
        
        with app.app_context():
            rejected_logs = OperationLog.query.filter_by(operation_type='export_comparison', status='rejected').all()
            assert len(rejected_logs) >= 1, "应记录拒绝的导出操作日志"
        
        print_result(True, "无效数据导出正确拒绝，日志已记录")
        
        print_step("13.6", "测试导出不支持的格式")
        export_result = export_comparison_result(compare_result, 'invalid_format', app, operator='test_user')
        assert not export_result.get('success'), "不支持的格式导出应该失败"
        assert export_result.get('error_code') == 'INVALID_FORMAT', f"错误码应为 INVALID_FORMAT，实际: {export_result.get('error_code')}"
        
        print_result(True, "不支持的格式导出正确拒绝")
        
        print_step("13.7", "清理导出的测试文件")
        os.remove(export_result_json['output_path'])
        os.remove(export_result_csv['output_path'])
        print_result(True, "测试文件已清理")
        
        cleanup_test_data(app, RUN_ID)
        print_header("场景13: 全部通过 [OK]")
        return True
        
    except Exception as e:
        print(f"  [FAIL] 场景13异常: {e}")
        import traceback
        traceback.print_exc()
        cleanup_test_data(app, RUN_ID)
        return False


def test_scenario_14_operation_log_persistence():
    """
    场景14: 操作日志持久化和重启追溯
    测试操作日志持久化、服务重启后仍能查到
    """
    print_header("场景14: 操作日志持久化和重启追溯")
    
    app1 = create_app()
    cleanup_test_data(app1, RUN_ID)
    
    try:
        print_step("14.1", "执行各种操作以生成日志")
        import_result1 = import_csv(TRAINING_DATA, app1)
        dataset_id1 = import_result1['dataset_id']
        train_result1 = train_model(dataset_id1, app1)
        model_id1 = train_result1['model_version_id']
        
        time.sleep(1)
        import_result2 = import_csv(TRAINING_DATA, app1)
        dataset_id2 = import_result2['dataset_id']
        train_result2 = train_model(dataset_id2, app1)
        model_id2 = train_result2['model_version_id']
        
        failed_model_id = create_failed_version(app1)
        
        compare_versions(model_id1, model_id2, app1, operator='user_a')
        compare_versions(model_id1, failed_model_id, app1, operator='user_b')
        activate_model(model_id1, app1, operator='user_c')
        get_usage_window_config(app1)
        
        compare_result = compare_versions(model_id1, model_id2, app1, operator='user_d')
        export_comparison_result(compare_result, 'json', app1, operator='user_e')
        
        print_result(True, "执行了多次对比、激活、配置读取、导出操作")
        
        print_step("14.2", "验证当前日志记录")
        with app1.app_context():
            all_logs = OperationLog.query.order_by(OperationLog.created_at).all()
            assert len(all_logs) >= 6, f"至少应有6条日志，实际: {len(all_logs)}"
            
            compare_logs = [l for l in all_logs if l.operation_type == 'compare_versions']
            assert len(compare_logs) >= 3, f"至少应有3条对比日志，实际: {len(compare_logs)}"
            
            success_compare = [l for l in compare_logs if l.status == 'success']
            rejected_compare = [l for l in compare_logs if l.status == 'rejected']
            assert len(success_compare) >= 2, f"至少应有2条成功对比日志，实际: {len(success_compare)}"
            assert len(rejected_compare) >= 1, f"至少应有1条拒绝对比日志，实际: {len(rejected_compare)}"
            
            activate_logs = [l for l in all_logs if l.operation_type == 'activate_model']
            assert len(activate_logs) >= 1, f"至少应有1条激活日志，实际: {len(activate_logs)}"
            
            export_logs = [l for l in all_logs if l.operation_type == 'export_comparison']
            assert len(export_logs) >= 1, f"至少应有1条导出日志，实际: {len(export_logs)}"
            
            config_logs = [l for l in all_logs if l.operation_type == 'config_read']
            assert len(config_logs) >= 1, f"至少应有1条配置读取日志，实际: {len(config_logs)}"
            
            for log in all_logs:
                assert log.created_at is not None, "日志时间不应为空"
                assert log.operation_type is not None, "操作类型不应为空"
                assert log.status is not None, "状态不应为空"
            
            log_before_restart = [l.to_dict() for l in all_logs]
        
        print_result(True, f"日志记录正确，共 {len(all_logs)} 条: 对比{len(compare_logs)}, 激活{len(activate_logs)}, 导出{len(export_logs)}, 配置{len(config_logs)}")
        
        print_step("14.3", "模拟服务重启，验证日志持久化")
        logs_count_before = len(log_before_restart)
        
        del app1
        time.sleep(1)
        
        app2 = create_app()
        
        with app2.app_context():
            logs_after_restart = OperationLog.query.order_by(OperationLog.created_at).all()
            assert len(logs_after_restart) >= logs_count_before, f"重启后日志丢失，之前: {logs_count_before}, 之后: {len(logs_after_restart)}"
            
            compare_logs_after = [l for l in logs_after_restart if l.operation_type == 'compare_versions']
            activate_logs_after = [l for l in logs_after_restart if l.operation_type == 'activate_model']
            export_logs_after = [l for l in logs_after_restart if l.operation_type == 'export_comparison']
            config_logs_after = [l for l in logs_after_restart if l.operation_type == 'config_read']
            
            assert len(compare_logs_after) >= len(compare_logs), "对比日志丢失"
            assert len(activate_logs_after) >= len(activate_logs), "激活日志丢失"
            assert len(export_logs_after) >= len(export_logs), "导出日志丢失"
            assert len(config_logs_after) >= len(config_logs), "配置读取日志丢失"
            
            first_log = logs_after_restart[0]
            assert 'operator' in first_log.to_dict(), "日志缺少 operator 字段"
            assert 'details' in first_log.to_dict(), "日志缺少 details 字段"
        
        print_result(True, f"重启后日志可追溯，共 {len(logs_after_restart)} 条")
        
        print_step("14.4", "验证通过 API 查询操作日志")
        logs_from_api = get_operation_history(app2, limit=100)
        assert len(logs_from_api) >= logs_count_before, "API 查询日志数量不足"
        
        compare_logs_api = get_operation_history(app2, operation_type='compare_versions', limit=100)
        assert len(compare_logs_api) >= len(compare_logs), "按类型查询对比日志失败"
        
        for log in compare_logs_api:
            assert log['operation_type'] == 'compare_versions', "按类型查询结果不正确"
        
        print_result(True, "API 查询操作日志功能正常")
        
        print_step("14.5", "验证重启后可以继续记录新日志")
        compare_versions(model_id2, model_id1, app2, operator='after_restart_user')
        
        with app2.app_context():
            new_logs = OperationLog.query.order_by(OperationLog.created_at.desc()).limit(5).all()
            new_compare_log = None
            for log in new_logs:
                if log.operation_type == 'compare_versions' and log.operator == 'after_restart_user':
                    new_compare_log = log
                    break
            
            assert new_compare_log is not None, "重启后无法记录新日志"
            assert new_compare_log.status == 'success', "新日志状态不正确"
        
        print_result(True, "重启后可以继续记录新日志")
        
        cleanup_test_data(app2, RUN_ID)
        del app2
        
        print_header("场景14: 全部通过 [OK]")
        return True
        
    except Exception as e:
        print(f"  [FAIL] 场景14异常: {e}")
        import traceback
        traceback.print_exc()
        if 'app1' in locals():
            cleanup_test_data(app1, RUN_ID)
        if 'app2' in locals():
            cleanup_test_data(app2, RUN_ID)
        return False


def test_scenario_15_config_change_and_window_filter():
    """
    场景15: 配置变更和窗口过滤
    测试时间窗口配置变更、窗口过滤功能
    """
    print_header("场景15: 配置变更和窗口过滤")
    
    app = create_app()
    cleanup_test_data(app, RUN_ID)
    
    try:
        print_step("15.1", "训练模型并创建不同时间的预测记录")
        import_result = import_csv(TRAINING_DATA, app)
        dataset_id = import_result['dataset_id']
        train_result = train_model(dataset_id, app)
        model_id1 = train_result['model_version_id']
        
        time.sleep(1)
        import_result2 = import_csv(TRAINING_DATA, app)
        dataset_id2 = import_result2['dataset_id']
        train_result2 = train_model(dataset_id2, app)
        model_id2 = train_result2['model_version_id']
        
        with app.app_context():
            now = datetime.utcnow()
            
            for i in range(10):
                ticket = Ticket(
                    title=get_unique_title(f"场景15_模型1_{i}"),
                    content=f"测试内容 {i}",
                    channel="web",
                    predicted_queue="tech_support_queue",
                    confidence=0.9,
                    model_version_id=model_id1,
                    predicted_at=now - timedelta(days=i)
                )
                db.session.add(ticket)
            
            for i in range(5):
                ticket = Ticket(
                    title=get_unique_title(f"场景15_模型2_{i}"),
                    content=f"测试内容 {i}",
                    channel="web",
                    predicted_queue="tech_support_queue",
                    confidence=0.9,
                    model_version_id=model_id2,
                    predicted_at=now - timedelta(days=i * 2)
                )
                db.session.add(ticket)
            
            db.session.commit()
        
        print_result(True, "创建了预测记录: 模型1有10条（每天1条），模型2有5条（每2天1条）")
        
        print_step("15.2", "测试不同窗口天数的对比结果")
        for window_days in [1, 3, 7, 14, 30]:
            compare_result = compare_versions(model_id1, model_id2, app, operator='test_user', window_days=window_days)
            assert compare_result.get('success'), f"窗口 {window_days} 天对比失败"
            assert compare_result['window_days'] == window_days, f"窗口天数不正确，期望 {window_days}，实际 {compare_result['window_days']}"
            
            va = compare_result['version_a']
            vb = compare_result['version_b']
            
            expected_recent_a = min(window_days, 10)
            expected_recent_b = min((window_days + 1) // 2, 5)
            
            assert va['recent_usage_count'] == expected_recent_a, f"模型1最近{window_days}天使用量应为{expected_recent_a}，实际: {va['recent_usage_count']}"
            assert vb['recent_usage_count'] == expected_recent_b, f"模型2最近{window_days}天使用量应为{expected_recent_b}，实际: {vb['recent_usage_count']}"
            
            assert va['usage_count'] == 10, f"模型1历史总使用量应为10，实际: {va['usage_count']}"
            assert vb['usage_count'] == 5, f"模型2历史总使用量应为5，实际: {vb['usage_count']}"
            
            print_result(True, f"窗口 {window_days} 天: 模型1最近{va['recent_usage_count']}条/总{va['usage_count']}条, 模型2最近{vb['recent_usage_count']}条/总{vb['usage_count']}条")
        
        print_step("15.3", "测试窗口天数边界值")
        compare_result = compare_versions(model_id1, model_id2, app, operator='test_user', window_days=1)
        assert compare_result.get('success'), "窗口 1 天对比应该成功"
        assert compare_result['version_a']['recent_usage_count'] == 1, "窗口1天使用量应为1"
        
        compare_result = compare_versions(model_id1, model_id2, app, operator='test_user', window_days=365)
        assert compare_result.get('success'), "窗口 365 天对比应该成功"
        assert compare_result['version_a']['recent_usage_count'] == 10, "窗口365天使用量应为10"
        
        print_result(True, "窗口边界值（1天、365天）处理正确")
        
        print_step("15.4", "验证导出内容包含窗口信息")
        compare_result = compare_versions(model_id1, model_id2, app, operator='test_user', window_days=14)
        export_result = export_comparison_result(compare_result, 'json', app, operator='test_user')
        assert export_result.get('success'), "导出失败"
        
        with open(export_result['output_path'], 'r', encoding='utf-8') as f:
            export_data = json.load(f)
        
        assert export_data['window_days'] == 14, f"导出的窗口天数应为14，实际: {export_data['window_days']}"
        assert export_data['version_a']['recent_usage'] == compare_result['version_a']['recent_usage_count'], "导出的最近使用量不一致"
        assert export_data['version_a']['total_usage'] == compare_result['version_a']['usage_count'], "导出的总使用量不一致"
        
        os.remove(export_result['output_path'])
        print_result(True, "导出内容包含正确的窗口信息和使用量数据")
        
        print_step("15.5", "验证对比结果中的指标完整性")
        compare_result = compare_versions(model_id1, model_id2, app, operator='test_user', window_days=7)
        va = compare_result['version_a']
        vb = compare_result['version_b']
        metrics_diff = compare_result['metrics_diff']
        
        required_metrics = ['accuracy', 'precision', 'recall', 'f1']
        for metric in required_metrics:
            assert metric in metrics_diff, f"缺少 {metric} 指标对比"
            assert 'version_a' in metrics_diff[metric], f"{metric} 缺少 version_a"
            assert 'version_b' in metrics_diff[metric], f"{metric} 缺少 version_b"
            assert 'difference' in metrics_diff[metric], f"{metric} 缺少 difference"
        
        assert 'is_active' in va, "缺少 is_active 字段"
        assert 'status' in va, "缺少 status 字段"
        assert 'trained_at' in va, "缺少 trained_at 字段"
        assert 'dataset_name' in va, "缺少 dataset_name 字段"
        
        print_result(True, "对比结果包含完整的指标和元数据")
        
        cleanup_test_data(app, RUN_ID)
        print_header("场景15: 全部通过 [OK]")
        return True
        
    except Exception as e:
        print(f"  [FAIL] 场景15异常: {e}")
        import traceback
        traceback.print_exc()
        cleanup_test_data(app, RUN_ID)
        return False


def main():
    print(f"\n{'#'*80}")
    print(f"#{' '*78}#")
    print(f"#{' '*15}  工单路由系统 - 版本对比增强回归测试  {' '*24}#")
    print(f"#{' '*78}#")
    print(f"{'#'*80}")
    print(f"\n测试时间: {datetime.now().isoformat()}")
    print(f"运行ID: {RUN_ID}")
    print(f"测试场景: 5个新增关键场景")
    
    results = []
    
    new_tests = [
        ('场景11: 版本对比校验 - 非法版本拦截', test_scenario_11_comparison_validation),
        ('场景12: 时间窗口统计的最近使用量', test_scenario_12_recent_usage_window),
        ('场景13: 对比结果导出功能', test_scenario_13_export_comparison),
        ('场景14: 操作日志持久化和重启追溯', test_scenario_14_operation_log_persistence),
        ('场景15: 配置变更和窗口过滤', test_scenario_15_config_change_and_window_filter),
    ]
    
    for name, test_func in new_tests:
        try:
            results.append((name, test_func()))
        except Exception as e:
            print(f"  [FAIL] {name}异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print(f"\n{'#'*80}")
    print(f"#{' '*78}#")
    print(f"#{' '*25}  测试结果汇总  {' '*35}#")
    print(f"#{' '*78}#")
    print(f"{'#'*80}\n")
    
    passed = 0
    failed = 0
    for name, success in results:
        status = "[OK]" if success else "[FAIL]"
        print(f"  {status} {name}")
        if success:
            passed += 1
        else:
            failed += 1
    
    print(f"\n总计: {len(results)}个场景, 通过: {passed}, 失败: {failed}")
    
    if failed == 0:
        print(f"\n{'='*80}")
        print(f"  所有测试通过！[OK]")
        print(f"{'='*80}\n")
        return 0
    else:
        print(f"\n{'='*80}")
        print(f"  有 {failed} 个测试失败！[FAIL]")
        print(f"{'='*80}\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
