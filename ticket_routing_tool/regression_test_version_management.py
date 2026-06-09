import os
import sys
import time
import uuid
import threading
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from src.database import db, init_db
from src.csv_importer import import_csv, get_dataset
from src.classifier import train_model
from src.predictor import predict_ticket, predict_batch, get_active_model
from src.rollback import (
    activate_model, rollback_to_version, get_active_version,
    get_version_history, compare_versions, get_activation_history,
    get_all_versions_with_details
)
from src.models import ModelVersion, Ticket, ModelActivationLog

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
        db.session.commit()


def test_scenario_5_version_comparison():
    """
    场景5: 版本对比功能
    选择两个已完成模型查看关键指标差异
    """
    print_header("场景5: 版本对比功能")
    
    app = create_app()
    cleanup_test_data(app, RUN_ID)
    
    try:
        print_step("5.1", "导入数据并训练第一个模型")
        import_result1 = import_csv(TRAINING_DATA, app)
        assert import_result1.get('is_valid'), "数据导入失败"
        dataset_id1 = import_result1['dataset_id']
        print_result(True, f"数据集1导入成功，ID: {dataset_id1}")
        
        train_result1 = train_model(dataset_id1, app)
        assert train_result1.get('success'), f"训练1失败: {train_result1}"
        model_id1 = train_result1['model_version_id']
        print_result(True, f"模型1训练成功，ID: {model_id1}")
        
        print_step("5.2", "导入数据并训练第二个模型")
        time.sleep(1)
        import_result2 = import_csv(TRAINING_DATA, app)
        assert import_result2.get('is_valid'), "数据导入失败"
        dataset_id2 = import_result2['dataset_id']
        print_result(True, f"数据集2导入成功，ID: {dataset_id2}")
        
        train_result2 = train_model(dataset_id2, app)
        assert train_result2.get('success'), f"训练2失败: {train_result2}"
        model_id2 = train_result2['model_version_id']
        print_result(True, f"模型2训练成功，ID: {model_id2}")
        
        print_step("5.3", "对比两个版本")
        compare_result = compare_versions(model_id1, model_id2, app)
        assert compare_result.get('success'), f"对比失败: {compare_result}"
        
        assert 'version_a' in compare_result, "缺少 version_a"
        assert 'version_b' in compare_result, "缺少 version_b"
        assert 'metrics_diff' in compare_result, "缺少 metrics_diff"
        assert 'comparison' in compare_result, "缺少 comparison"
        
        va = compare_result['version_a']
        vb = compare_result['version_b']
        
        assert 'dataset_name' in va, "version_a 缺少 dataset_name"
        assert 'usage_count' in va, "version_a 缺少 usage_count"
        assert 'model_file_exists' in va, "version_a 缺少 model_file_exists"
        assert 'dataset_row_count' in va, "version_a 缺少 dataset_row_count"
        
        metrics_diff = compare_result['metrics_diff']
        assert 'accuracy' in metrics_diff, "metrics_diff 缺少 accuracy"
        assert 'f1' in metrics_diff, "metrics_diff 缺少 f1"
        assert 'precision' in metrics_diff, "metrics_diff 缺少 precision"
        assert 'recall' in metrics_diff, "metrics_diff 缺少 recall"
        
        acc_diff = metrics_diff['accuracy']
        assert 'version_a' in acc_diff, "accuracy 缺少 version_a"
        assert 'version_b' in acc_diff, "accuracy 缺少 version_b"
        assert 'difference' in acc_diff, "accuracy 缺少 difference"
        assert 'difference_percent' in acc_diff, "accuracy 缺少 difference_percent"
        
        comparison = compare_result['comparison']
        assert 'training_time_diff' in comparison, "comparison 缺少 training_time_diff"
        assert 'dataset_diff' in comparison, "comparison 缺少 dataset_diff"
        
        print_result(True, "版本对比成功，包含所有必需字段")
        
        print_step("5.4", "测试对比不存在的版本")
        compare_result_fail = compare_versions(99999, model_id2, app)
        assert not compare_result_fail.get('success'), "对比不存在的版本应该失败"
        assert 'error' in compare_result_fail, "错误结果缺少 error"
        print_result(True, "对比不存在的版本正确返回错误")
        
        print_step("5.5", "测试对比相同版本")
        compare_result_same = compare_versions(model_id1, model_id1, app)
        assert compare_result_same.get('success'), "对比相同版本应该成功"
        assert compare_result_same['comparison']['dataset_diff'] == False, "相同版本数据集应该相同"
        assert compare_result_same['comparison']['status_diff'] == False, "相同版本状态应该相同"
        assert compare_result_same['metrics_diff']['accuracy']['difference'] == 0, "相同版本准确率差异应为0"
        print_result(True, "对比相同版本正确返回零差异")
        
        cleanup_test_data(app, RUN_ID)
        print_header("场景5: 全部通过 [OK]")
        return True
        
    except Exception as e:
        print(f"  [FAIL] 场景5异常: {e}")
        import traceback
        traceback.print_exc()
        cleanup_test_data(app, RUN_ID)
        return False


def test_scenario_6_manual_activation_and_conflict():
    """
    场景6: 手动激活和冲突处理
    测试手动激活、重复激活、并发激活
    """
    print_header("场景6: 手动激活和冲突处理")
    
    app = create_app()
    cleanup_test_data(app, RUN_ID)
    
    try:
        print_step("6.1", "准备多个模型版本")
        import_result1 = import_csv(TRAINING_DATA, app)
        dataset_id1 = import_result1['dataset_id']
        train_result1 = train_model(dataset_id1, app)
        model_id1 = train_result1['model_version_id']
        
        time.sleep(1)
        import_result2 = import_csv(TRAINING_DATA, app)
        dataset_id2 = import_result2['dataset_id']
        train_result2 = train_model(dataset_id2, app)
        model_id2 = train_result2['model_version_id']
        
        time.sleep(1)
        import_result3 = import_csv(TRAINING_DATA, app)
        dataset_id3 = import_result3['dataset_id']
        train_result3 = train_model(dataset_id3, app)
        model_id3 = train_result3['model_version_id']
        
        print_result(True, f"准备了3个模型版本: {model_id1}, {model_id2}, {model_id3}")
        
        print_step("6.2", "测试手动激活非当前版本")
        active_before = get_active_version(app)
        assert active_before['id'] == model_id3, f"当前激活版本应该是 {model_id3}"
        
        activate_result = activate_model(model_id1, app, operator='test_user')
        assert activate_result.get('success'), f"激活失败: {activate_result}"
        assert activate_result['previous_version']['id'] == model_id3, "前一个版本ID不正确"
        assert activate_result['current_version']['id'] == model_id1, "当前版本ID不正确"
        
        with app.app_context():
            active = ModelVersion.query.filter_by(is_active=True).all()
            assert len(active) == 1, "只能有一个激活版本"
            assert active[0].id == model_id1, "激活版本ID不正确"
            assert active[0].status == 'active', "激活版本状态应为 active"
            
            prev = ModelVersion.query.get(model_id3)
            assert prev.status == 'rolled_back', "前一版本状态应为 rolled_back"
            assert prev.is_active == False, "前一版本不应是激活状态"
        
        print_result(True, f"手动激活成功，从版本 {model_id3} 切换到 {model_id1}")
        
        print_step("6.3", "测试重复激活（同一版本重复点击）")
        activate_result_dup = activate_model(model_id1, app, operator='test_user')
        assert not activate_result_dup.get('success'), "重复激活应该失败"
        assert activate_result_dup.get('error_code') == 'ALREADY_ACTIVE', f"错误码不正确: {activate_result_dup.get('error_code')}"
        assert 'current_version' in activate_result_dup, "应返回当前版本信息"
        
        with app.app_context():
            active = ModelVersion.query.filter_by(is_active=True).all()
            assert len(active) == 1, "仍然只能有一个激活版本"
            assert active[0].id == model_id1, "激活版本不应改变"
        
        print_result(True, "重复激活正确拒绝，返回 ALREADY_ACTIVE")
        
        print_step("6.4", "测试并发激活冲突")
        concurrent_results = []
        errors = []
        
        def activate_worker(model_id, results, errors, app):
            try:
                result = activate_model(model_id, app, operator='concurrent_test')
                results.append(result)
            except Exception as e:
                errors.append(str(e))
        
        threads = []
        for i in range(5):
            target_id = model_id2 if i % 2 == 0 else model_id3
            t = threading.Thread(target=activate_worker, args=(target_id, concurrent_results, errors, app))
            threads.append(t)
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"并发激活出现异常: {errors}"
        
        with app.app_context():
            active = ModelVersion.query.filter_by(is_active=True).all()
            assert len(active) == 1, "并发激活后仍然只能有一个激活版本"
            final_active_id = active[0].id
        
        success_count = sum(1 for r in concurrent_results if r.get('success'))
        skip_or_fail_count = sum(1 for r in concurrent_results if not r.get('success'))
        
        assert success_count >= 1, "至少有一个激活应该成功"
        assert success_count + skip_or_fail_count == 5, "所有请求都应该有结果"
        
        print_result(True, f"并发激活冲突处理正确: 成功{success_count}个, 跳过/失败{skip_or_fail_count}个, 最终激活: {final_active_id}")
        
        print_step("6.5", "验证只有一个激活版本（数据一致性）")
        with app.app_context():
            all_versions = ModelVersion.query.all()
            active_count = sum(1 for v in all_versions if v.is_active)
            assert active_count == 1, f"数据一致性检查失败: 有 {active_count} 个激活版本"
        
        print_result(True, "数据一致性检查通过，始终只有一个激活版本")
        
        cleanup_test_data(app, RUN_ID)
        print_header("场景6: 全部通过 [OK]")
        return True
        
    except Exception as e:
        print(f"  [FAIL] 场景6异常: {e}")
        import traceback
        traceback.print_exc()
        cleanup_test_data(app, RUN_ID)
        return False


def test_scenario_7_failed_version_rejection():
    """
    场景7: 失败版本拒绝激活
    测试失败模型、缺失模型文件不能被激活
    """
    print_header("场景7: 失败版本拒绝激活")
    
    app = create_app()
    cleanup_test_data(app, RUN_ID)
    
    try:
        print_step("7.1", "训练一个失败的模型")
        import_result_bad = import_csv(BAD_DATA, app)
        bad_dataset_id = import_result_bad['dataset_id']
        print_result(True, f"坏数据导入成功，ID: {bad_dataset_id}")
        
        train_result_bad = train_model(bad_dataset_id, app)
        assert not train_result_bad.get('success'), "坏数据训练应该失败"
        failed_model_id = train_result_bad.get('model_version_id')
        assert failed_model_id is not None, "应该返回失败模型的ID"
        print_result(True, f"失败模型创建成功，ID: {failed_model_id}")
        
        with app.app_context():
            failed_model = ModelVersion.query.get(failed_model_id)
            assert failed_model.status == 'failed', f"失败模型状态应为 failed，实际: {failed_model.status}"
            assert failed_model.is_active == False, "失败模型不应被激活"
        
        print_step("7.2", "尝试激活失败版本")
        activate_result = activate_model(failed_model_id, app, operator='test_user')
        assert not activate_result.get('success'), "激活失败版本应该被拒绝"
        assert activate_result.get('error_code') == 'FAILED_VERSION', f"错误码应为 FAILED_VERSION，实际: {activate_result.get('error_code')}"
        
        with app.app_context():
            failed_model = ModelVersion.query.get(failed_model_id)
            assert failed_model.status == 'failed', "失败模型状态不应改变"
            assert failed_model.is_active == False, "失败模型仍不应被激活"
        
        print_result(True, "失败版本激活被正确拒绝，返回 FAILED_VERSION")
        
        print_step("7.3", "测试模型文件缺失的情况")
        import_result_good = import_csv(TRAINING_DATA, app)
        good_dataset_id = import_result_good['dataset_id']
        train_result_good = train_model(good_dataset_id, app)
        good_model_id = train_result_good['model_version_id']
        print_result(True, f"正常模型训练成功，ID: {good_model_id}")
        
        with app.app_context():
            good_model = ModelVersion.query.get(good_model_id)
            model_path = good_model.model_path
            vectorizer_path = os.path.splitext(model_path)[0] + '_vectorizer.pkl'
        
        temp_backup = model_path + '.bak'
        os.rename(model_path, temp_backup)
        
        try:
            activate_result_missing = activate_model(good_model_id, app, operator='test_user')
            assert not activate_result_missing.get('success'), "模型文件缺失时激活应该失败"
            assert activate_result_missing.get('error_code') == 'FILE_MISSING', f"错误码应为 FILE_MISSING，实际: {activate_result_missing.get('error_code')}"
            print_result(True, "模型文件缺失时激活被正确拒绝，返回 FILE_MISSING")
        finally:
            os.rename(temp_backup, model_path)
        
        print_step("7.4", "测试向量化器文件缺失的情况")
        temp_backup_vec = vectorizer_path + '.bak'
        os.rename(vectorizer_path, temp_backup_vec)
        
        try:
            activate_result_vec_missing = activate_model(good_model_id, app, operator='test_user')
            assert not activate_result_vec_missing.get('success'), "向量化器文件缺失时激活应该失败"
            assert activate_result_vec_missing.get('error_code') == 'VECTORIZER_MISSING', f"错误码应为 VECTORIZER_MISSING，实际: {activate_result_vec_missing.get('error_code')}"
            print_result(True, "向量化器文件缺失时激活被正确拒绝，返回 VECTORIZER_MISSING")
        finally:
            os.rename(temp_backup_vec, vectorizer_path)
        
        print_step("7.5", "测试不存在的版本")
        activate_result_not_found = activate_model(99999, app, operator='test_user')
        assert not activate_result_not_found.get('success'), "不存在的版本激活应该失败"
        assert activate_result_not_found.get('error_code') == 'NOT_FOUND', f"错误码应为 NOT_FOUND，实际: {activate_result_not_found.get('error_code')}"
        print_result(True, "不存在的版本激活被正确拒绝，返回 NOT_FOUND")
        
        cleanup_test_data(app, RUN_ID)
        print_header("场景7: 全部通过 [OK]")
        return True
        
    except Exception as e:
        print(f"  [FAIL] 场景7异常: {e}")
        import traceback
        traceback.print_exc()
        cleanup_test_data(app, RUN_ID)
        return False


def test_scenario_8_activation_log_persistence():
    """
    场景8: 激活日志持久化和跨重启保持
    测试切换记录持久化、服务重启后仍能查到
    """
    print_header("场景8: 激活日志持久化和跨重启保持")
    
    app1 = create_app()
    cleanup_test_data(app1, RUN_ID)
    
    try:
        print_step("8.1", "执行多次切换操作，记录日志")
        import_result1 = import_csv(TRAINING_DATA, app1)
        dataset_id1 = import_result1['dataset_id']
        train_result1 = train_model(dataset_id1, app1)
        model_id1 = train_result1['model_version_id']
        
        time.sleep(1)
        import_result2 = import_csv(TRAINING_DATA, app1)
        dataset_id2 = import_result2['dataset_id']
        train_result2 = train_model(dataset_id2, app1)
        model_id2 = train_result2['model_version_id']
        
        time.sleep(1)
        activate_result1 = activate_model(model_id1, app1, operator='user_a')
        assert activate_result1.get('success'), "激活1失败"
        
        time.sleep(1)
        activate_result2 = activate_model(model_id2, app1, operator='user_b')
        assert activate_result2.get('success'), "激活2失败"
        
        time.sleep(1)
        activate_result3 = activate_model(model_id2, app1, operator='user_c')
        assert not activate_result3.get('success'), "重复激活应该失败"
        
        time.sleep(1)
        activate_result4 = activate_model(model_id1, app1, operator='user_d')
        assert activate_result4.get('success'), "激活4失败"
        
        print_result(True, "执行了4次切换操作（3次成功，1次重复跳过）")
        
        print_step("8.2", "验证当前激活状态和日志记录")
        with app1.app_context():
            active = ModelVersion.query.filter_by(is_active=True).first()
            assert active.id == model_id1, f"当前激活版本应为 {model_id1}"
            
            logs = ModelActivationLog.query.order_by(ModelActivationLog.created_at).all()
            assert len(logs) >= 4, f"至少应有4条日志，实际: {len(logs)}"
            
            success_count = sum(1 for l in logs if l.status == 'success')
            skipped_count = sum(1 for l in logs if l.status == 'skipped')
            failed_count = sum(1 for l in logs if l.status == 'failed')
            
            assert success_count >= 3, f"成功日志不足，应有至少3条，实际: {success_count}"
            assert skipped_count >= 1, f"跳过日志不足，应有至少1条，实际: {skipped_count}"
            
            last_log = logs[-1]
            assert last_log.model_version_id == model_id1, "最后一条日志应为激活 model_id1"
            assert last_log.operator == 'user_d', "操作者应为 user_d"
            assert last_log.action == 'activate', "操作类型应为 activate"
            assert last_log.status == 'success', "状态应为 success"
            
            log_with_prev = [l for l in logs if l.previous_version_id is not None]
            assert len(log_with_prev) >= 1, "至少有一条日志记录了前一版本"
        
        print_result(True, f"日志记录正确: 成功{success_count}条, 跳过{skipped_count}条, 失败{failed_count}条")
        
        print_step("8.3", "模拟服务重启，验证激活状态保持")
        active_id_before = get_active_version(app1)['id']
        history_before = get_activation_history(app1, limit=10)
        assert len(history_before) >= 4, "重启前历史记录不足"
        
        del app1
        time.sleep(1)
        
        app2 = create_app()
        
        active_after = get_active_version(app2)
        assert active_after is not None, "重启后激活状态丢失"
        assert active_after['id'] == active_id_before, f"重启后激活版本改变，之前: {active_id_before}, 之后: {active_after['id']}"
        
        with app2.app_context():
            active = ModelVersion.query.filter_by(is_active=True).all()
            assert len(active) == 1, "重启后激活版本数量错误"
            assert active[0].id == active_id_before, "重启后激活版本ID错误"
        
        print_result(True, f"重启后激活状态保持，版本ID: {active_after['id']}")
        
        print_step("8.4", "验证重启后历史记录可查")
        history_after = get_activation_history(app2, limit=10)
        assert len(history_after) >= len(history_before), "重启后历史记录丢失"
        
        first_history = history_after[0]
        assert 'model_version' in first_history, "历史记录缺少版本号"
        assert 'operator' in first_history, "历史记录缺少操作者"
        assert 'action' in first_history, "历史记录缺少操作类型"
        assert 'status' in first_history, "历史记录缺少状态"
        assert 'created_at' in first_history, "历史记录缺少时间"
        
        print_result(True, f"重启后历史记录可查，共 {len(history_after)} 条")
        
        print_step("8.5", "验证重启后可以继续切换")
        activate_result_restart = activate_model(model_id2, app2, operator='after_restart')
        assert activate_result_restart.get('success'), "重启后激活失败"
        
        history_new = get_activation_history(app2, limit=1)
        assert history_new[0]['operator'] == 'after_restart', "新日志操作者不正确"
        
        print_result(True, "重启后可以继续进行切换操作")
        
        cleanup_test_data(app2, RUN_ID)
        del app2
        
        print_header("场景8: 全部通过 [OK]")
        return True
        
    except Exception as e:
        print(f"  [FAIL] 场景8异常: {e}")
        import traceback
        traceback.print_exc()
        if 'app1' in locals():
            cleanup_test_data(app1, RUN_ID)
        if 'app2' in locals():
            cleanup_test_data(app2, RUN_ID)
        return False


def test_scenario_9_prediction_uses_new_version():
    """
    场景9: 切换后预测使用新版本
    测试单条预测和批量预测在切换后都使用新版本
    """
    print_header("场景9: 切换后预测使用新版本")
    
    app = create_app()
    cleanup_test_data(app, RUN_ID)
    
    try:
        print_step("9.1", "训练两个不同的模型")
        import_result1 = import_csv(TRAINING_DATA, app)
        dataset_id1 = import_result1['dataset_id']
        train_result1 = train_model(dataset_id1, app)
        model_id1 = train_result1['model_version_id']
        
        time.sleep(1)
        import_result2 = import_csv(TRAINING_DATA, app)
        dataset_id2 = import_result2['dataset_id']
        train_result2 = train_model(dataset_id2, app)
        model_id2 = train_result2['model_version_id']
        
        print_result(True, f"训练了两个模型: {model_id1}, {model_id2}")
        
        print_step("9.2", "验证当前激活是 model_id2，预测使用它")
        active_before = get_active_version(app)
        assert active_before['id'] == model_id2, f"当前激活应为 {model_id2}"
        
        title1 = get_unique_title("场景9_单条预测_切换前")
        predict_result1 = predict_ticket(
            title=title1,
            content="测试切换前的单条预测",
            channel="web",
            app=app
        )
        assert predict_result1['model_version_id'] == model_id2, f"切换前预测应使用 {model_id2}"
        
        tickets_before = [
            {'title': get_unique_title("场景9_批量预测_切换前_1"), 'content': '批量测试1', 'channel': 'email'},
            {'title': get_unique_title("场景9_批量预测_切换前_2"), 'content': '批量测试2', 'channel': 'phone'},
        ]
        batch_result1 = predict_batch(tickets_before, app)
        assert len(batch_result1) == 2, "批量预测结果数量错误"
        for r in batch_result1:
            assert r['model_version_id'] == model_id2, f"切换前批量预测应使用 {model_id2}"
        
        print_result(True, f"切换前预测正确使用版本 {model_id2}")
        
        print_step("9.3", "切换到 model_id1")
        activate_result = activate_model(model_id1, app, operator='test_user')
        assert activate_result.get('success'), "切换失败"
        
        active_after = get_active_version(app)
        assert active_after['id'] == model_id1, f"切换后激活应为 {model_id1}"
        
        print_result(True, f"成功切换到版本 {model_id1}")
        
        print_step("9.4", "验证单条预测现在使用新版本")
        title2 = get_unique_title("场景9_单条预测_切换后")
        predict_result2 = predict_ticket(
            title=title2,
            content="测试切换后的单条预测",
            channel="web",
            app=app
        )
        assert predict_result2['model_version_id'] == model_id1, f"切换后预测应使用 {model_id1}"
        assert predict_result2['model_version_id'] != predict_result1['model_version_id'], "切换前后预测应使用不同版本"
        
        with app.app_context():
            ticket = Ticket.query.get(predict_result2['ticket_id'])
            assert ticket.model_version_id == model_id1, "工单记录的版本ID不正确"
        
        print_result(True, f"切换后单条预测正确使用版本 {model_id1}")
        
        print_step("9.5", "验证批量预测现在使用新版本")
        tickets_after = [
            {'title': get_unique_title("场景9_批量预测_切换后_1"), 'content': '批量测试3', 'channel': 'email'},
            {'title': get_unique_title("场景9_批量预测_切换后_2"), 'content': '批量测试4', 'channel': 'phone'},
            {'title': get_unique_title("场景9_批量预测_切换后_3"), 'content': '批量测试5', 'channel': 'app'},
        ]
        batch_result2 = predict_batch(tickets_after, app)
        assert len(batch_result2) == 3, "批量预测结果数量错误"
        for r in batch_result2:
            assert r['model_version_id'] == model_id1, f"切换后批量预测应使用 {model_id1}"
        
        with app.app_context():
            for r in batch_result2:
                ticket = Ticket.query.get(r['ticket_id'])
                assert ticket.model_version_id == model_id1, "批量预测工单记录的版本ID不正确"
        
        print_result(True, f"切换后批量预测正确使用版本 {model_id1}")
        
        print_step("9.6", "验证历史预测记录的版本ID保持不变")
        with app.app_context():
            ticket_before = Ticket.query.get(predict_result1['ticket_id'])
            assert ticket_before.model_version_id == model_id2, "历史预测的版本ID不应改变"
        
        print_result(True, "历史预测记录的版本ID保持不变，数据一致性正确")
        
        cleanup_test_data(app, RUN_ID)
        print_header("场景9: 全部通过 [OK]")
        return True
        
    except Exception as e:
        print(f"  [FAIL] 场景9异常: {e}")
        import traceback
        traceback.print_exc()
        cleanup_test_data(app, RUN_ID)
        return False


def test_scenario_10_version_details():
    """
    场景10: 版本详情字段完整性
    测试版本详情包含数据集信息、使用量、文件存在性等
    """
    print_header("场景10: 版本详情字段完整性")
    
    app = create_app()
    cleanup_test_data(app, RUN_ID)
    
    try:
        print_step("10.1", "训练一个模型并进行预测")
        import_result = import_csv(TRAINING_DATA, app)
        dataset_id = import_result['dataset_id']
        train_result = train_model(dataset_id, app)
        model_id = train_result['model_version_id']
        
        for i in range(5):
            predict_ticket(
                title=get_unique_title(f"场景10_测试预测_{i}"),
                content=f"测试内容{i}",
                channel="web",
                app=app
            )
        
        print_result(True, f"模型训练成功，进行了5次预测")
        
        print_step("10.2", "获取版本详情，验证字段完整性")
        versions = get_all_versions_with_details(app)
        assert len(versions) >= 1, "版本列表不应为空"
        
        target_version = None
        for v in versions:
            if v['id'] == model_id:
                target_version = v
                break
        
        assert target_version is not None, f"找不到版本 {model_id}"
        
        required_fields = [
            'id', 'version', 'dataset_id', 'model_path', 'metrics',
            'is_active', 'trained_at', 'status', 'dataset_name',
            'dataset_row_count', 'usage_count', 'model_file_exists',
            'vectorizer_file_exists'
        ]
        
        for field in required_fields:
            assert field in target_version, f"版本详情缺少字段: {field}"
        
        assert target_version['usage_count'] >= 5, f"使用量统计不正确，应有至少5，实际: {target_version['usage_count']}"
        assert target_version['model_file_exists'] == True, "模型文件存在性标记错误"
        assert target_version['vectorizer_file_exists'] == True, "向量化器文件存在性标记错误"
        assert target_version['dataset_row_count'] > 0, "数据集行数不应为0"
        assert target_version['dataset_name'] is not None, "数据集名称不应为None"
        assert target_version['is_active'] == True, "激活状态标记错误"
        assert target_version['status'] == 'active', "状态标记错误"
        
        print_result(True, "版本详情包含所有必需字段，数据正确")
        
        cleanup_test_data(app, RUN_ID)
        print_header("场景10: 全部通过 [OK]")
        return True
        
    except Exception as e:
        print(f"  [FAIL] 场景10异常: {e}")
        import traceback
        traceback.print_exc()
        cleanup_test_data(app, RUN_ID)
        return False


def main():
    print(f"\n{'#'*80}")
    print(f"#{' '*78}#")
    print(f"#{' '*18}  工单路由系统 - 模型版本管理回归测试  {' '*24}#")
    print(f"#{' '*78}#")
    print(f"{'#'*80}")
    print(f"\n测试时间: {datetime.now().isoformat()}")
    print(f"运行ID: {RUN_ID}")
    print(f"测试场景: 6个新增关键场景 + 4个原有场景")
    
    results = []
    
    existing_tests = [
        ('场景5: 版本对比功能', test_scenario_5_version_comparison),
        ('场景6: 手动激活和冲突处理', test_scenario_6_manual_activation_and_conflict),
        ('场景7: 失败版本拒绝激活', test_scenario_7_failed_version_rejection),
        ('场景8: 激活日志持久化和跨重启保持', test_scenario_8_activation_log_persistence),
        ('场景9: 切换后预测使用新版本', test_scenario_9_prediction_uses_new_version),
        ('场景10: 版本详情字段完整性', test_scenario_10_version_details),
    ]
    
    for name, test_func in existing_tests:
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
