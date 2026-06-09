import os
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from src.database import db
from src.csv_importer import import_csv, get_dataset
from src.classifier import train_model
from src.predictor import predict_ticket, get_active_model
from src.human_override import create_override
from src.report_exporter import export_audit_report
from src.models import ModelVersion, Ticket, HumanOverride

SAMPLE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'samples')
TRAINING_DATA = os.path.join(SAMPLE_DATA_DIR, 'sample_training_data.csv')
BAD_DATA = os.path.join(SAMPLE_DATA_DIR, 'sample_bad_data.csv')


def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ticket_routing.db")}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)
    with app.app_context():
        db.create_all()
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


def test_scenario_1_train_and_predict_immediately():
    """
    场景1: 训练后立即预测
    Fresh DB完成一次有效训练后，系统能直接拿新模型做预测
    """
    print_header("场景1: 训练后立即预测")
    
    app = create_app()
    
    # 1. 导入训练数据
    print_step("1.1", "导入训练数据")
    import_result = import_csv(TRAINING_DATA, app)
    assert import_result.get('is_valid'), f"数据导入失败: {import_result}"
    dataset_id = import_result['dataset_id']
    print_result(True, f"数据集导入成功，ID: {dataset_id}")
    
    # 2. 训练模型
    print_step("1.2", "训练模型")
    train_result = train_model(dataset_id, app)
    assert train_result.get('success'), f"训练失败: {train_result}"
    model_version_id = train_result['model_version_id']
    print_result(True, f"模型训练成功，版本ID: {model_version_id}")
    
    # 3. 检查模型是否自动激活
    print_step("1.3", "检查模型是否自动激活")
    with app.app_context():
        model = ModelVersion.query.get(model_version_id)
        assert model is not None, "模型版本不存在"
        assert model.is_active, f"模型未自动激活，is_active={model.is_active}"
        assert model.status == 'active', f"模型状态不正确，status={model.status}"
    print_result(True, f"模型已自动激活，状态: {model.status}")
    
    # 4. 立即预测（不需要测试脚本或人工改数据库）
    print_step("1.4", "立即预测工单")
    predict_result = predict_ticket(
        title="测试工单",
        content="这是一个测试工单内容",
        channel="web",
        app=app
    )
    assert predict_result is not None, "预测失败"
    assert 'predicted_queue' in predict_result, "预测结果缺少predicted_queue"
    assert 'confidence' in predict_result, "预测结果缺少confidence"
    assert predict_result['model_version_id'] == model_version_id, \
        f"预测使用的模型版本不正确，期望{model_version_id}，实际{predict_result['model_version_id']}"
    print_result(True, f"预测成功，队列: {predict_result['predicted_queue']}, 置信度: {predict_result['confidence']:.4f}")
    
    # 5. 检查Ticket是否保存了model_version_id
    print_step("1.5", "检查Ticket是否保存了model_version_id")
    with app.app_context():
        ticket = Ticket.query.filter_by(model_version_id=model_version_id).first()
        assert ticket is not None, "工单未保存"
        assert ticket.model_version_id == model_version_id, "工单model_version_id不正确"
    print_result(True, f"工单已保存model_version_id: {ticket.model_version_id}")
    
    print_header("场景1: 全部通过 [OK]")
    return True


def test_scenario_2_failed_training_isolation():
    """
    场景2: 失败训练不替换当前模型
    导入坏数据训练失败，确认当前激活模型没有被替换
    """
    print_header("场景2: 失败训练隔离")
    
    app = create_app()
    
    # 1. 先获取当前激活模型
    print_step("2.1", "获取当前激活模型")
    active_before = get_active_model(app)
    assert active_before is not None, "没有可用的激活模型"
    active_id_before = active_before.id
    print_result(True, f"当前激活模型ID: {active_id_before}")
    
    # 2. 导入坏数据
    print_step("2.2", "导入坏数据（含空标签、未知标签）")
    import_result = import_csv(BAD_DATA, app)
    bad_dataset_id = import_result['dataset_id']
    print_result(True, f"坏数据导入成功，ID: {bad_dataset_id}, 验证状态: {import_result.get('is_valid')}")
    
    # 3. 尝试用坏数据训练
    print_step("2.3", "尝试用坏数据训练（预期失败）")
    train_result = train_model(bad_dataset_id, app)
    assert not train_result.get('success'), "坏数据训练应该失败但成功了"
    failed_model_id = train_result.get('model_version_id')
    print_result(True, f"坏数据训练失败（预期），错误: {train_result.get('error')[:50]}...")
    
    # 4. 检查失败模型状态
    print_step("2.4", "检查失败模型状态")
    with app.app_context():
        failed_model = ModelVersion.query.get(failed_model_id)
        assert failed_model is not None, "失败模型不存在"
        assert failed_model.status == 'failed', f"失败模型状态不正确，status={failed_model.status}"
        assert not failed_model.is_active, "失败模型不应该被激活"
    print_result(True, f"失败模型状态: {failed_model.status}, is_active: {failed_model.is_active}")
    
    # 5. 检查激活模型是否被替换
    print_step("2.5", "检查激活模型未被替换")
    active_after = get_active_model(app)
    assert active_after is not None, "激活模型丢失"
    assert active_after.id == active_id_before, f"激活模型被替换了！之前: {active_id_before}, 之后: {active_after.id}"
    print_result(True, f"激活模型未被替换，ID: {active_after.id}")
    
    # 6. 预测仍然使用正确的模型
    print_step("2.6", "验证预测仍然使用正确的模型")
    predict_result = predict_ticket(
        title="验证工单",
        content="验证预测使用的模型是否正确",
        channel="email",
        app=app
    )
    assert predict_result['model_version_id'] == active_id_before, \
        f"预测使用了错误的模型版本，期望{active_id_before}，实际{predict_result['model_version_id']}"
    print_result(True, f"预测使用正确的模型版本: {predict_result['model_version_id']}")
    
    print_header("场景2: 全部通过 [OK]")
    return True


def test_scenario_3_persistence_across_restart():
    """
    场景3: 跨app重启后仍能读到激活模型和改判记录
    """
    print_header("场景3: 跨重启持久化")
    
    app1 = create_app()
    
    # 1. 先做一次预测和改判
    print_step("3.1", "预测一个工单")
    predict_result = predict_ticket(
        title="持久化测试工单",
        content="测试重启后数据是否保留",
        channel="phone",
        app=app1
    )
    ticket_id = None
    with app1.app_context():
        ticket = Ticket.query.filter_by(title="持久化测试工单").first()
        ticket_id = ticket.id
    print_result(True, f"工单预测成功，ID: {ticket_id}")
    
    # 2. 人工改判
    print_step("3.2", "人工改判工单")
    active_model = get_active_model(app1)
    override_result = create_override(
        ticket_id=ticket_id,
        corrected_queue="billing_queue",
        operator="回归测试",
        reason="测试持久化",
        app=app1
    )
    assert override_result is not None, f"改判失败: {override_result}"
    override_id = override_result['id']
    print_result(True, f"人工改判成功，改判记录ID: {override_id}")
    
    # 3. 记录当前状态
    print_step("3.3", "记录当前状态")
    with app1.app_context():
        active_before = ModelVersion.query.filter_by(is_active=True).first()
        ticket_before = Ticket.query.get(ticket_id)
        override_before = HumanOverride.query.get(override_id)
        active_id_before = active_before.id
        ticket_count_before = Ticket.query.count()
        override_count_before = HumanOverride.query.count()
    print_result(True, f"重启前 - 激活模型: {active_id_before}, 工单数: {ticket_count_before}, 改判数: {override_count_before}")
    
    # 4. 模拟重启 - 创建新的app实例
    print_step("3.4", "模拟重启（创建新的app实例）")
    del app1
    time.sleep(1)  # 确保完全清理
    
    app2 = create_app()
    print_result(True, "新app实例创建成功")
    
    # 5. 检查激活模型
    print_step("3.5", "检查激活模型是否保留")
    with app2.app_context():
        active_after = ModelVersion.query.filter_by(is_active=True).first()
        assert active_after is not None, "激活模型丢失"
        assert active_after.id == active_id_before, f"激活模型ID不一致，之前: {active_id_before}, 之后: {active_after.id}"
    print_result(True, f"激活模型保留，ID: {active_after.id}")
    
    # 6. 检查工单数据
    print_step("3.6", "检查工单数据是否保留")
    with app2.app_context():
        ticket_after = Ticket.query.get(ticket_id)
        assert ticket_after is not None, "工单丢失"
        assert ticket_after.title == "持久化测试工单", "工单内容不一致"
        assert ticket_after.model_version_id == active_id_before, "工单model_version_id丢失"
        ticket_count_after = Ticket.query.count()
        assert ticket_count_after == ticket_count_before, f"工单数不一致，之前: {ticket_count_before}, 之后: {ticket_count_after}"
    print_result(True, f"工单数据保留，工单数: {ticket_count_after}")
    
    # 7. 检查改判记录
    print_step("3.7", "检查改判记录是否保留")
    with app2.app_context():
        override_after = HumanOverride.query.get(override_id)
        assert override_after is not None, "改判记录丢失"
        assert override_after.operator == "回归测试", "改判操作者不一致"
        assert override_after.reason == "测试持久化", "改判原因不一致"
        override_count_after = HumanOverride.query.count()
        assert override_count_after == override_count_before, f"改判数不一致，之前: {override_count_before}, 之后: {override_count_after}"
    print_result(True, f"改判记录保留，改判数: {override_count_after}")
    
    # 8. 重启后仍然可以预测
    print_step("3.8", "验证重启后仍然可以预测")
    predict_result = predict_ticket(
        title="重启后测试工单",
        content="验证重启后预测功能正常",
        channel="app",
        app=app2
    )
    assert predict_result is not None, "重启后预测失败"
    assert predict_result['model_version_id'] == active_id_before, "重启后预测使用错误的模型"
    print_result(True, f"重启后预测正常，模型版本: {predict_result['model_version_id']}")
    
    del app2
    print_header("场景3: 全部通过 [OK]")
    return True


def test_scenario_4_same_day_export_boundary():
    """
    场景4: 同日范围导出能统计刚产生的数据
    start_date=end_date边界问题
    """
    print_header("场景4: 同日导出边界测试")
    
    app = create_app()
    
    # 1. 预测几个工单
    print_step("4.1", "预测3个工单")
    for i in range(3):
        predict_ticket(
            title=f"同日测试工单{i+1}",
            content=f"测试同日导出边界问题{i+1}",
            channel="wechat",
            app=app
        )
    print_result(True, "3个工单预测完成")
    
    # 2. 对其中一个工单进行改判
    print_step("4.2", "人工改判1个工单")
    with app.app_context():
        ticket = Ticket.query.filter_by(title="同日测试工单1").first()
        ticket_id = ticket.id
    
    active_model = get_active_model(app)
    create_override(
        ticket_id=ticket_id,
        corrected_queue="account_queue",
        operator="边界测试",
        reason="测试同日导出",
        app=app
    )
    print_result(True, f"工单{ticket_id}改判完成")
    
    # 3. 导出当天数据（start_date=end_date）
    print_step("4.3", "导出当天数据（start_date=end_date）")
    today = datetime.now().date().isoformat()
    export_result = export_audit_report(
        start_date=today,
        end_date=today,
        app=app,
        format='xlsx'
    )
    assert export_result.get('success'), f"导出失败: {export_result}"
    
    # 4. 验证导出的数据包含当天的预测和改判
    tickets_count = export_result.get('tickets_count', 0)
    overrides_count = export_result.get('overrides_count', 0)
    
    print(f"  导出结果: 工单{export_result['tickets_count']}条, 改判{export_result['overrides_count']}条")
    print(f"  导出文件: {export_result['output_path']}")
    
    # 检查是否包含新产生的数据
    assert tickets_count >= 3, f"导出工单数不足，期望至少3条，实际{tickets_count}条"
    assert overrides_count >= 1, f"导出改判数不足，期望至少1条，实际{overrides_count}条"
    print_result(True, f"同日导出包含当天数据：工单{tickets_count}条, 改判{overrides_count}条")
    
    # 5. 验证导出字段包含模型版本和数据集版本
    print_step("4.4", "验证导出字段完整性")
    import pandas as pd
    xls = pd.ExcelFile(export_result['output_path'])
    
    # 检查工单预测记录sheet
    tickets_df = pd.read_excel(xls, '工单预测记录')
    required_ticket_columns = ['工单ID', '标题', '渠道', '预测队列', '置信度', '实际队列', '是否被改判', 
                               '模型版本ID', '模型版本号', '数据集ID', '数据集名称', '预测时间']
    for col in required_ticket_columns:
        assert col in tickets_df.columns, f"工单记录缺少列: {col}"
    print_result(True, "工单预测记录字段完整")
    
    # 检查人工改判记录sheet
    overrides_df = pd.read_excel(xls, '人工改判记录')
    required_override_columns = ['工单ID', '原预测', '改判后队列', '操作者', '原因', '时间',
                                  '模型版本ID', '模型版本号', '数据集ID', '数据集名称']
    for col in required_override_columns:
        assert col in overrides_df.columns, f"改判记录缺少列: {col}"
    print_result(True, "人工改判记录字段完整")
    
    # 验证数据内容
    assert len(tickets_df) >= 3, f"Excel中工单数不足"
    assert len(overrides_df) >= 1, f"Excel中改判数不足"
    
    # 验证model_version_id有值
    assert tickets_df['模型版本ID'].notna().any(), "工单记录中model_version_id为空"
    assert overrides_df['模型版本ID'].notna().any(), "改判记录中model_version_id为空"
    print_result(True, "导出数据包含模型版本和数据集版本信息")
    
    print_header("场景4: 全部通过 [OK]")
    return True


def main():
    print(f"\n{'#'*80}")
    print(f"#{' '*78}#")
    print(f"#{' '*20}  工单路由系统 - 回归测试  {' '*28}#")
    print(f"#{' '*78}#")
    print(f"{'#'*80}")
    print(f"\n测试时间: {datetime.now().isoformat()}")
    print(f"测试场景: 4个关键场景")
    
    results = []
    
    try:
        # 场景1: 训练后立即预测
        results.append(('场景1: 训练后立即预测', test_scenario_1_train_and_predict_immediately()))
    except Exception as e:
        print(f"  [FAIL] 场景1异常: {e}")
        import traceback
        traceback.print_exc()
        results.append(('场景1: 训练后立即预测', False))
    
    try:
        # 场景2: 失败训练隔离
        results.append(('场景2: 失败训练隔离', test_scenario_2_failed_training_isolation()))
    except Exception as e:
        print(f"  [FAIL] 场景2异常: {e}")
        import traceback
        traceback.print_exc()
        results.append(('场景2: 失败训练隔离', False))
    
    try:
        # 场景3: 跨重启持久化
        results.append(('场景3: 跨重启持久化', test_scenario_3_persistence_across_restart()))
    except Exception as e:
        print(f"  [FAIL] 场景3异常: {e}")
        import traceback
        traceback.print_exc()
        results.append(('场景3: 跨重启持久化', False))
    
    try:
        # 场景4: 同日导出边界
        results.append(('场景4: 同日导出边界', test_scenario_4_same_day_export_boundary()))
    except Exception as e:
        print(f"  [FAIL] 场景4异常: {e}")
        import traceback
        traceback.print_exc()
        results.append(('场景4: 同日导出边界', False))
    
    # 总结
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
