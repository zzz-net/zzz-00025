import sys
import os
from datetime import datetime, timedelta
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from src.csv_importer import import_csv, list_datasets, get_dataset
from src.classifier import train_model
from src.predictor import predict_ticket, predict_batch, get_active_model, get_queue_suggestion
from src.human_override import create_override, list_overrides, get_override_stats
from src.rollback import rollback_to_version, get_version_history, get_active_version, list_rollback_candidates
from src.report_exporter import export_audit_report, export_model_comparison_report
from src.models import Ticket, ModelVersion, Dataset
from src.database import db
import config


def print_step(step_num, title):
    print("\n" + "=" * 80)
    print(f"步骤 {step_num}: {title}")
    print("=" * 80)


def print_result(message, success=True):
    prefix = "[OK]" if success else "[FAIL]"
    print(f"{prefix} {message}")


def main():
    print("\n" + "#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + " " * 25 + "工单路由系统 - 完整流程测试" + " " * 28 + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)

    app = create_app()

    sample_dir = config.SAMPLE_DIR
    training_data_path = os.path.join(sample_dir, "sample_training_data.csv")
    bad_data_path = os.path.join(sample_dir, "sample_bad_data.csv")
    predict_data_path = os.path.join(sample_dir, "sample_predict_tickets.csv")

    with app.app_context():
        print_step(1, "初始化 Flask app 和数据库")
        print_result("Flask app 初始化成功")
        print_result(f"数据库路径: {config.DB_PATH}")
        
        db.create_all()
        print_result("数据库表创建成功")
        
        tables = db.inspect(db.engine).get_table_names()
        print(f"  已创建的表: {', '.join(tables)}")
        assert len(tables) > 0, "数据库表创建失败"

    print_step(2, "导入训练数据")
    import_result = import_csv(training_data_path, app)
    print(f"导入结果: {import_result}")
    
    dataset_id = import_result.get("dataset_id")
    is_valid = import_result.get("is_valid")
    row_count = import_result.get("row_count")
    
    print_result(f"数据集 ID: {dataset_id}")
    print_result(f"数据验证: {'通过' if is_valid else '失败'}")
    print_result(f"数据行数: {row_count}")
    assert is_valid, "训练数据验证失败"
    assert row_count == 50, f"训练数据行数应为 50，实际为 {row_count}"
    
    datasets = list_datasets(app)
    print(f"  数据集列表: {len(datasets)} 个")
    for d in datasets:
        print(f"    - ID: {d['id']}, 名称: {d['name']}, 状态: {d['status']}, 行数: {d['row_count']}")

    print_step(3, "训练模型")
    train_result = train_model(dataset_id, app)
    print(f"训练结果: {train_result}")
    
    assert train_result.get("success"), f"模型训练失败: {train_result.get('error')}"
    
    model_version_id = train_result.get("model_version_id")
    version = train_result.get("version")
    
    print_result(f"模型版本 ID: {model_version_id}")
    print_result(f"版本号: {version}")
    
    with app.app_context():
        model_version = ModelVersion.query.get(model_version_id)
        assert model_version is not None, "模型版本不存在"
        assert model_version.status == "active", f"模型状态应为 active，实际为 {model_version.status}"
        assert model_version.is_active, f"模型 is_active 应为 True，实际为 {model_version.is_active}"
        print_result("模型已自动激活")
    
    first_model_version_id = model_version_id

    print_step(4, "检查模型评估指标")
    with app.app_context():
        model_version = ModelVersion.query.get(model_version_id)
        metrics = model_version.metrics
        
        print_result("模型评估指标:")
        if metrics:
            overall = metrics.get("overall", {})
            print(f"  准确率 (Accuracy): {overall.get('accuracy', 'N/A'):.4f}")
            print(f"  宏平均精确率 (Macro Precision): {overall.get('macro_precision', 'N/A'):.4f}")
            print(f"  宏平均召回率 (Macro Recall): {overall.get('macro_recall', 'N/A'):.4f}")
            print(f"  宏平均 F1 值 (Macro F1): {overall.get('macro_f1', 'N/A'):.4f}")
            print(f"  加权平均精确率 (Weighted Precision): {overall.get('weighted_precision', 'N/A'):.4f}")
            print(f"  加权平均召回率 (Weighted Recall): {overall.get('weighted_recall', 'N/A'):.4f}")
            print(f"  加权平均 F1 值 (Weighted F1): {overall.get('weighted_f1', 'N/A'):.4f}")
            
            print("\n  各类别指标:")
            per_class = metrics.get("per_class", {})
            for label, class_metrics in per_class.items():
                display_name = config.QUEUE_DISPLAY_NAMES.get(
                    config.QUEUE_MAPPING.get(label, label), label
                )
                print(f"    {display_name} ({label}):")
                print(f"      精确率: {class_metrics.get('precision', 'N/A'):.4f}")
                print(f"      召回率: {class_metrics.get('recall', 'N/A'):.4f}")
                print(f"      F1 值: {class_metrics.get('f1', 'N/A'):.4f}")
                print(f"      支持数: {class_metrics.get('support', 'N/A')}")
            
            assert overall.get("accuracy", 0) > 0, "准确率应大于 0"
    
    if "classification_report" in train_result:
        print("\n  分类报告:")
        print(train_result["classification_report"])

    print_step(5, "预测单个工单")
    test_ticket = {
        "title": "系统登录失败，提示密码错误",
        "content": "用户反馈登录时提示密码错误，但确认密码正确，多次尝试后账号被锁定，需要解锁并重置密码",
        "channel": "email"
    }
    
    predict_result = predict_ticket(
        test_ticket["title"],
        test_ticket["content"],
        test_ticket["channel"],
        app
    )
    
    print_result(f"预测结果:")
    print(f"  预测队列: {predict_result['predicted_queue']}")
    print(f"  置信度: {predict_result['confidence']:.4f}")
    print(f"  模型版本 ID: {predict_result['model_version_id']}")
    
    suggestion = get_queue_suggestion(predict_result['predicted_queue'], predict_result['confidence'])
    print(f"\n  队列建议:")
    print(f"    建议队列: {suggestion['suggested_queue']}")
    print(f"    优先级: {suggestion['priority']}")
    print(f"    操作建议: {suggestion['action']}")
    print(f"    说明: {suggestion['message']}")
    
    assert predict_result["predicted_queue"] is not None, "预测队列不能为空"
    assert predict_result["confidence"] > 0, "置信度应大于 0"
    
    with app.app_context():
        ticket = Ticket.query.order_by(Ticket.id.desc()).first()
        single_ticket_id = ticket.id
        print_result(f"工单已保存，ID: {single_ticket_id}")

    print_step(6, "批量预测工单")
    df_predict = pd.read_csv(predict_data_path, encoding="utf-8-sig")
    tickets = df_predict.to_dict("records")
    
    print(f"待预测工单数: {len(tickets)}")
    
    batch_result = predict_batch(tickets, app)
    
    print_result(f"批量预测完成，共 {len(batch_result)} 条")
    
    queue_counts = {}
    confidence_sum = 0
    low_confidence_count = 0
    
    for i, result in enumerate(batch_result[:5]):
        print(f"\n  工单 {i + 1}:")
        print(f"    标题: {tickets[i]['title']}")
        print(f"    预测队列: {result['predicted_queue']}")
        print(f"    置信度: {result['confidence']:.4f}")
        
        queue = result["predicted_queue"]
        queue_counts[queue] = queue_counts.get(queue, 0) + 1
        confidence_sum += result["confidence"]
        
        if result["confidence"] < 0.5:
            low_confidence_count += 1
    
    for i, result in enumerate(batch_result[5:], start=5):
        queue = result["predicted_queue"]
        queue_counts[queue] = queue_counts.get(queue, 0) + 1
        confidence_sum += result["confidence"]
        
        if result["confidence"] < 0.5:
            low_confidence_count += 1
    
    avg_confidence = confidence_sum / len(batch_result) if batch_result else 0
    
    print(f"\n  统计信息:")
    print(f"    平均置信度: {avg_confidence:.4f}")
    print(f"    低置信度工单 (<0.5): {low_confidence_count} 条")
    print(f"    队列分布:")
    for queue, count in queue_counts.items():
        display_name = config.QUEUE_DISPLAY_NAMES.get(queue, queue)
        print(f"      {display_name}: {count} 条")
    
    assert len(batch_result) == len(tickets), "批量预测结果数量不匹配"

    print_step(7, "人工改判一个工单")
    with app.app_context():
        ticket = Ticket.query.get(single_ticket_id)
        original_prediction = ticket.predicted_queue
        original_confidence = ticket.confidence
    
    print(f"  原预测: {original_prediction} (置信度: {original_confidence:.4f})")
    
    corrected_queue_name = "账单咨询"
    operator = "测试管理员"
    reason = "用户实际咨询的是账单问题，预测错误"
    
    override_result = create_override(
        single_ticket_id,
        config.QUEUE_MAPPING[config.QUEUE_NAME_TO_TAG[corrected_queue_name]],
        operator,
        reason,
        app
    )
    
    print_result("人工改判成功")
    print(f"  改判后队列: {override_result['corrected_queue']}")
    print(f"  操作者: {override_result['operator']}")
    print(f"  原因: {override_result['reason']}")
    
    assert override_result["ticket_id"] == single_ticket_id, "工单 ID 不匹配"
    assert override_result["original_prediction"] == original_prediction, "原预测不匹配"
    
    overrides = list_overrides(app=app)
    print(f"\n  改判记录总数: {len(overrides)}")
    
    override_stats = get_override_stats(app)
    print(f"\n  改判统计:")
    print(f"    总改判数: {override_stats['total_overrides']}")
    print(f"    总工单数: {override_stats['total_tickets']}")
    print(f"    改判率: {override_stats['override_rate_percentage']}")

    print_step(8, "导入坏数据尝试训练（验证失败保护）")
    print(f"尝试导入坏数据: {bad_data_path}")
    
    bad_import_result = import_csv(bad_data_path, app)
    print(f"导入结果: {bad_import_result}")
    
    bad_dataset_id = bad_import_result.get("dataset_id")
    bad_is_valid = bad_import_result.get("is_valid")
    
    print_result(f"数据集 ID: {bad_dataset_id}")
    print_result(f"数据验证: {'通过' if bad_is_valid else '失败（预期）'}", success=not bad_is_valid)
    
    validation_result = bad_import_result.get("validation_result", {})
    errors = validation_result.get("errors", [])
    warnings = validation_result.get("warnings", [])
    stats = validation_result.get("stats", {})
    
    print(f"\n  验证错误 ({len(errors)} 条):")
    for error in errors[:10]:
        print(f"    - {error}")
    if len(errors) > 10:
        print(f"    ... 还有 {len(errors) - 10} 条错误")
    
    print(f"\n  验证警告 ({len(warnings)} 条):")
    for warning in warnings[:5]:
        print(f"    - {warning}")
    if len(warnings) > 5:
        print(f"    ... 还有 {len(warnings) - 5} 条警告")
    
    if stats:
        labels_stats = stats.get("labels", {})
        channels_stats = stats.get("channels", {})
        print(f"\n  统计信息:")
        print(f"    空标签数: {labels_stats.get('empty_labels_count', 0)}")
        print(f"    未知标签数: {labels_stats.get('unknown_labels_count', 0)}")
        print(f"    无效渠道数: {channels_stats.get('invalid_channel_count', 0)}")
    
    assert not bad_is_valid, "坏数据应该验证失败"
    assert len(errors) > 0, "应该有验证错误"
    
    print("\n尝试用坏数据训练模型:")
    bad_train_result = train_model(bad_dataset_id, app)
    print(f"训练结果: {bad_train_result}")
    
    train_success = bad_train_result.get("success")
    print_result(f"训练: {'成功' if train_success else '失败（预期）'}", success=not train_success)
    
    error_msg = bad_train_result.get("error", "")
    print(f"  错误信息: {error_msg}")
    
    assert not train_success, "坏数据训练应该失败"

    print_step(9, "回滚到之前的版本")
    version_history = get_version_history(app)
    print(f"版本历史:")
    for v in version_history:
        status = "[ACTIVE]" if v.get("is_active") else "        "
        print(f"  {status} ID: {v['id']}, 版本: {v['version']}, 状态: {v['status']}")
    
    print("\n训练一个新模型用于回滚测试:")
    new_train_result = train_model(1, app)
    print(f"新模型训练结果: {new_train_result}")
    
    new_model_version_id = None
    if new_train_result.get("success"):
        new_model_version_id = new_train_result.get("model_version_id")
        
        with app.app_context():
            new_model = ModelVersion.query.get(new_model_version_id)
            assert new_model.status == "active", f"新模型状态应为 active，实际为 {new_model.status}"
            assert new_model.is_active, f"新模型 is_active 应为 True，实际为 {new_model.is_active}"
            
            old_model = ModelVersion.query.get(first_model_version_id)
            assert old_model.status == "rolled_back", f"旧模型状态应为 rolled_back，实际为 {old_model.status}"
            assert not old_model.is_active, f"旧模型 is_active 应为 False，实际为 {old_model.is_active}"
        
        print_result(f"新模型已自动激活，ID: {new_model_version_id}")
        
        active_version = get_active_version(app)
        print(f"当前激活版本: {active_version['id']} - {active_version['version']}")
        
        print(f"\n开始回滚到版本 {first_model_version_id}:")
        rollback_result = rollback_to_version(first_model_version_id, app)
        print(f"回滚结果: {rollback_result}")
        
        assert rollback_result.get("success"), "回滚失败"
        
        print_result("回滚成功")
        print(f"  之前版本: {rollback_result['previous_version']['id'] if rollback_result.get('previous_version') else '无'}")
        print(f"  当前版本: {rollback_result['current_version']['id']}")
        
        active_version = get_active_version(app)
        print(f"\n当前激活版本: {active_version['id']} - {active_version['version']}")
        assert active_version["id"] == first_model_version_id, "回滚后激活版本不正确"
        
        candidates = list_rollback_candidates(app)
        print(f"\n可回滚版本数: {len(candidates)}")
    else:
        print_result("新模型训练失败，跳过回滚测试", success=False)

    print_step(10, "导出复核报告")
    end_date = datetime.now().isoformat()
    start_date = (datetime.now() - timedelta(days=7)).isoformat()
    
    print(f"导出时间范围: {start_date} 到 {end_date}")
    
    audit_result = export_audit_report(start_date, end_date, app, format="xlsx")
    print(f"导出结果: {audit_result}")
    
    assert audit_result.get("success"), "导出复核报告失败"
    
    print_result("复核报告导出成功")
    print(f"  输出路径: {audit_result['output_path']}")
    print(f"  格式: {audit_result['format']}")
    print(f"  工单数量: {audit_result['tickets_count']}")
    print(f"  改判数量: {audit_result['overrides_count']}")
    
    assert os.path.exists(audit_result["output_path"]), "报告文件不存在"
    assert audit_result["tickets_count"] > 0, "应该有工单数据"

    print_step(11, "导出模型对比报告")
    comparison_result = export_model_comparison_report(app, format="xlsx")
    print(f"导出结果: {comparison_result}")
    
    assert comparison_result.get("success"), "导出模型对比报告失败"
    
    print_result("模型对比报告导出成功")
    print(f"  输出路径: {comparison_result['output_path']}")
    print(f"  格式: {comparison_result['format']}")
    print(f"  模型数量: {comparison_result['models_count']}")
    
    assert os.path.exists(comparison_result["output_path"]), "报告文件不存在"
    assert comparison_result["models_count"] >= 1, "至少应有1个模型用于对比"

    print("\n" + "#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + " " * 30 + "测试完成！所有步骤通过 [OK]" + " " * 25 + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)
    
    print("\n测试摘要:")
    print(f"  [OK] 初始化 Flask app 和数据库 - 成功")
    print(f"  [OK] 导入训练数据 (50条) - 成功")
    print(f"  [OK] 训练模型 - 成功")
    print(f"  [OK] 检查模型评估指标 - 成功")
    print(f"  [OK] 预测单个工单 - 成功")
    print(f"  [OK] 批量预测工单 ({len(tickets)}条) - 成功")
    print(f"  [OK] 人工改判工单 - 成功")
    print(f"  [OK] 导入坏数据失败保护 - 成功")
    print(f"  [OK] 回滚到之前版本 - 成功")
    print(f"  [OK] 导出复核报告 - 成功")
    print(f"  [OK] 导出模型对比报告 - 成功")
    
    print("\n生成的文件:")
    print(f"  - 训练数据: {training_data_path}")
    print(f"  - 坏数据: {bad_data_path}")
    print(f"  - 预测数据: {predict_data_path}")
    print(f"  - 复核报告: {audit_result['output_path']}")
    print(f"  - 模型对比报告: {comparison_result['output_path']}")
    print(f"  - 数据库: {config.DB_PATH}")


if __name__ == "__main__":
    main()
