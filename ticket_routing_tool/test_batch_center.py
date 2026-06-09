import sys
import os
import time
import tempfile
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from src.csv_importer import import_csv
from src.classifier import train_model
from src.batch_manager import (
    create_batch, process_batch, get_batch, list_batches,
    get_batch_tickets, export_batch_results, calculate_file_checksum
)
from src.models import PredictionBatch, BatchTicket, ModelVersion
from src.database import db
import config


def print_step(step_num, title):
    print("\n" + "=" * 80)
    print(f"步骤 {step_num}: {title}")
    print("=" * 80)


def print_result(message, success=True):
    prefix = "[OK]" if success else "[FAIL]"
    print(f"{prefix} {message}")


def create_test_csv(file_path, row_count=10, include_bad_rows=False):
    data = []
    for i in range(row_count):
        row = {
            'title': f'测试工单标题 {i+1}',
            'content': f'这是第 {i+1} 条测试工单的内容，描述用户遇到的问题。',
            'channel': 'email'
        }
        if include_bad_rows and i == 3:
            row['channel'] = 'invalid_channel'
        if include_bad_rows and i == 7:
            row['title'] = ''
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False, encoding='utf-8-sig')
    return file_path


def create_temp_csv(row_count=10, include_bad_rows=False):
    temp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
    temp_file.close()
    create_test_csv(temp_file.name, row_count, include_bad_rows)
    return temp_file.name


def main():
    print("\n" + "#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + " " * 28 + "批量预测任务中心 - 完整测试" + " " * 25 + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)

    app = create_app()
    test_operator = "测试用户张三"
    bad_operator = ""

    with app.app_context():
        print_step(1, "初始化数据库和准备测试环境")
        db.create_all()
        print_result("数据库表创建成功")

        existing_tables = db.inspect(db.engine).get_table_names()
        required_tables = ['prediction_batches', 'batch_tickets']
        for table in required_tables:
            assert table in existing_tables, f"缺少表: {table}"
        print_result("批次相关表已创建: " + ", ".join(required_tables))

        print("  清理旧的批次测试数据...")
        db.session.query(BatchTicket).delete()
        db.session.query(PredictionBatch).delete()
        db.session.commit()
        print_result("旧批次数据已清理")

    print_step(2, "导入训练数据并训练模型")
    sample_dir = config.SAMPLE_DIR
    training_data_path = os.path.join(sample_dir, "sample_training_data.csv")

    import_result = import_csv(training_data_path, app)
    assert import_result.get("is_valid"), "训练数据验证失败"
    dataset_id = import_result.get("dataset_id")
    print_result(f"训练数据导入成功，数据集 ID: {dataset_id}")

    train_result = train_model(dataset_id, app)
    assert train_result.get("success"), "模型训练失败"
    model_version_id = train_result.get("model_version_id")
    print_result(f"模型训练成功，版本 ID: {model_version_id}")

    with app.app_context():
        active_model = ModelVersion.query.filter_by(is_active=True).first()
        assert active_model is not None, "没有激活的模型"
        print_result(f"激活模型版本: v{active_model.version}")

    print_step(3, "测试权限校验 - 无操作者提交（应失败）")
    temp_path = create_temp_csv(row_count=5)

    success, result = create_batch("test.csv", bad_operator, temp_path, app)
    assert not success, "无操作者时应该失败"
    assert "操作者不能为空" in result.get("error", ""), "错误信息不正确"
    print_result(f"权限校验通过 - 无操作者提交失败: {result['error']}")

    os.remove(temp_path)

    print_step(4, "测试批量预测 - 正常数据")
    temp_path1 = create_temp_csv(row_count=10)
    original_filename = "test_predict_10.csv"

    success, result = create_batch(original_filename, test_operator, temp_path1, app)
    assert success, f"创建批次失败: {result.get('error')}"
    batch_id1 = result['batch_id']
    batch_uid1 = result['batch_uid']
    print_result(f"批次创建成功，ID: {batch_id1}, UID: {batch_uid1}")

    process_result = process_batch(batch_id1, temp_path1, app)
    assert process_result['success'], f"处理批次失败: {process_result.get('error')}"
    print_result(f"批次处理完成，成功 {process_result['success_count']} 条，失败 {process_result['failed_count']} 条")

    assert process_result['success_count'] == 10, f"应该成功 10 条，实际 {process_result['success_count']}"
    assert process_result['failed_count'] == 0, f"应该失败 0 条，实际 {process_result['failed_count']}"

    os.remove(temp_path1)

    batch_info = get_batch(batch_id1, app, include_details=True)
    assert batch_info is not None, "批次信息查询失败"
    assert batch_info['operator'] == test_operator, "操作者不匹配"
    assert batch_info['original_filename'] == original_filename, "文件名不匹配"
    assert batch_info['model_version_snapshot'] is not None, "缺少模型版本快照"
    assert batch_info['config_snapshot'] is not None, "缺少配置快照"
    print_result("批次信息完整 - 包含文件名、操作者、配置快照、模型版本快照")

    tickets, total = get_batch_tickets(batch_id1, app)
    assert total == 10, f"工单数量不匹配，应该 10 条，实际 {total}"
    print_result(f"工单明细查询成功，共 {total} 条")

    print_step(5, "测试批量预测 - 包含错误数据（部分失败）")
    temp_path2 = create_temp_csv(row_count=10, include_bad_rows=True)
    original_filename2 = "test_predict_with_errors.csv"

    success, result = create_batch(original_filename2, test_operator, temp_path2, app)
    assert success, f"创建批次失败: {result.get('error')}"
    batch_id2 = result['batch_id']
    print_result(f"批次创建成功，ID: {batch_id2}")

    process_result = process_batch(batch_id2, temp_path2, app)
    assert process_result['success'], f"处理批次失败: {process_result.get('error')}"
    print_result(f"批次处理完成，成功 {process_result['success_count']} 条，失败 {process_result['failed_count']} 条")

    assert process_result['success_count'] == 8, f"应该成功 8 条，实际 {process_result['success_count']}"
    assert process_result['failed_count'] == 2, f"应该失败 2 条，实际 {process_result['failed_count']}"
    print_result("部分失败场景验证通过 - 失败行未中断整批处理")

    os.remove(temp_path2)

    tickets, total = get_batch_tickets(batch_id2, app, status='failed')
    assert total == 2, f"失败工单数量不匹配，应该 2 条，实际 {total}"
    for ticket in tickets:
        assert ticket['error_message'] is not None, "失败工单缺少错误信息"
        print_result(f"  第 {ticket['row_index']} 行错误: {ticket['error_message']}")
    print_result("失败工单错误信息已记录")

    print_step(6, "测试重复提交检测")
    temp_path3 = create_temp_csv(row_count=5)

    success, result = create_batch("duplicate_test.csv", test_operator, temp_path3, app)
    assert success, f"第一次提交失败: {result.get('error')}"
    batch_id3 = result['batch_id']
    process_result = process_batch(batch_id3, temp_path3, app)
    assert process_result['success'], "第一次处理失败"
    print_result(f"第一次提交成功，批次 ID: {batch_id3}")

    success, result = create_batch("duplicate_test.csv", test_operator, temp_path3, app)
    assert not success, "重复提交应该失败"
    assert result.get('is_duplicate') == True, "应该标记为重复"
    assert 'existing_batch' in result, "应该返回已有批次信息"
    print_result(f"重复提交检测通过 - 已存在批次 ID: {result['existing_batch']['id']}")

    different_operator = "另一个测试用户"
    success, result = create_batch("duplicate_test.csv", different_operator, temp_path3, app)
    assert success, "不同操作者提交相同文件应该允许"
    batch_id4 = result['batch_id']
    process_result = process_batch(batch_id4, temp_path3, app)
    assert process_result['success'], "处理失败"
    print_result(f"不同操作者提交相同文件允许，新批次 ID: {batch_id4}")

    os.remove(temp_path3)

    print_step(7, "测试模型版本追踪（回滚后旧批次仍指向原模型）")
    with app.app_context():
        old_active_model = ModelVersion.query.filter_by(is_active=True).first()
        old_model_version = old_active_model.version
        print_result(f"当前激活模型: v{old_model_version}")

    train_result2 = train_model(dataset_id, app)
    assert train_result2.get("success"), "第二次训练失败"
    new_model_version_id = train_result2.get("model_version_id")

    with app.app_context():
        new_active_model = ModelVersion.query.filter_by(is_active=True).first()
        new_model_version = new_active_model.version
        print_result(f"训练新模型并激活: v{new_model_version}")

        old_model = ModelVersion.query.get(model_version_id)
        print_result(f"旧模型状态: {old_model.status}, is_active: {old_model.is_active}")

    batch1_info = get_batch(batch_id1, app)
    assert batch1_info['model_version_snapshot']['version'] == old_model_version, \
        f"旧批次应该指向原模型 v{old_model_version}，实际指向 v{batch1_info['model_version_snapshot']['version']}"
    print_result(f"模型版本追踪验证通过 - 批次 1 仍指向原模型 v{old_model_version}")

    with app.app_context():
        from src.rollback import rollback_to_version
        rollback_result = rollback_to_version(model_version_id, app)
        assert rollback_result.get('success'), "回滚失败"
        print_result(f"已回滚到模型 v{old_model_version}")

    batch1_info_after = get_batch(batch_id1, app)
    assert batch1_info_after['model_version_snapshot']['version'] == old_model_version, \
        "回滚后批次的模型版本快照不应改变"
    print_result("回滚验证通过 - 批次的模型版本快照保持不变")

    print_step(8, "测试导出功能 - CSV 格式")
    export_success, export_result = export_batch_results(
        batch_id2, app, format_type='csv', include_failed=True, operator=test_operator
    )
    assert export_success, f"CSV 导出失败: {export_result.get('error')}"
    assert os.path.exists(export_result['file_path']), "导出文件不存在"
    assert export_result['format'] == 'csv', "格式不正确"
    assert export_result['row_count'] == 10, f"导出行数不匹配，应该 10 行，实际 {export_result['row_count']}"
    print_result(f"CSV 导出成功，文件: {export_result['filename']}，共 {export_result['row_count']} 行")

    exported_df = pd.read_csv(export_result['file_path'], encoding='utf-8-sig')
    assert '错误信息' in exported_df.columns, "导出文件缺少错误信息列"
    failed_rows = exported_df[exported_df['状态'] == '失败']
    assert len(failed_rows) == 2, "导出文件中失败行数量不匹配"
    print_result("导出文件包含错误信息列，失败行已正确导出")

    print_step(9, "测试导出功能 - Excel 格式")
    export_success, export_result = export_batch_results(
        batch_id2, app, format_type='xlsx', include_failed=True, operator=test_operator
    )
    assert export_success, f"Excel 导出失败: {export_result.get('error')}"
    assert os.path.exists(export_result['file_path']), "导出文件不存在"
    assert export_result['format'] == 'xlsx', "格式不正确"
    print_result(f"Excel 导出成功，文件: {export_result['filename']}")

    print_step(10, "测试导出权限校验 - 无操作者（应失败）")
    export_success, export_result = export_batch_results(
        batch_id1, app, format_type='csv', include_failed=True, operator=bad_operator
    )
    assert not export_success, "无操作者导出应该失败"
    assert "操作者不能为空" in export_result.get("error", ""), "错误信息不正确"
    print_result(f"导出权限校验通过 - 无操作者导出失败: {export_result['error']}")

    print_step(11, "测试批次列表查询和筛选")
    all_batches, all_batches_total = list_batches(app)
    print_result(f"查询所有批次，共 {all_batches_total} 条")
    assert all_batches_total >= 4, f"批次数量不足，应该至少 4 条，实际 {all_batches_total}"

    filtered_batches, filtered_total = list_batches(app, operator=test_operator)
    print_result(f"按操作者筛选，共 {filtered_total} 条")
    assert filtered_total >= 3, f"筛选结果数量不足"

    completed_batches, completed_total = list_batches(app, status='completed')
    print_result(f"按状态筛选（全部成功），共 {completed_total} 条")

    partial_batches, partial_total = list_batches(app, status='partial_failed')
    print_result(f"按状态筛选（部分失败），共 {partial_total} 条")
    assert partial_total >= 1, "应该有部分失败的批次"

    print_step(12, "测试工单筛选 - 低置信度和已改判")
    tickets, total = get_batch_tickets(batch_id1, app, low_confidence_only=True)
    print_result(f"低置信度工单筛选，共 {total} 条")

    with app.app_context():
        from src.human_override import create_override
        first_ticket = BatchTicket.query.filter_by(batch_id=batch_id1, status='success').first()
        if first_ticket and first_ticket.ticket_id:
            override_result = create_override(
                first_ticket.ticket_id,
                'tech_support_queue',
                test_operator,
                '测试人工改判',
                app
            )
            print_result(f"已对工单 {first_ticket.ticket_id} 进行人工改判")

            overridden_tickets, overridden_total = get_batch_tickets(batch_id1, app, overridden_only=True)
            assert overridden_total == 1, f"已改判工单筛选失败，应该 1 条，实际 {overridden_total}"
            print_result(f"已改判工单筛选通过，共 {overridden_total} 条")

    print_step(13, "测试服务重启后查询（模拟 - 直接查询数据库）")
    with app.app_context():
        db.session.remove()
        db.engine.dispose()

    with app.app_context():
        batch_after_restart = get_batch(batch_id1, app)
        assert batch_after_restart is not None, "重启后批次查询失败"
        assert batch_after_restart['batch_uid'] == batch_uid1, "批次 UID 不匹配"
        print_result(f"重启后查询验证通过 - 批次 {batch_uid1} 仍可查询")

        tickets_after_restart, total_after_restart = get_batch_tickets(batch_id1, app)
        assert total_after_restart == 10, "重启后工单查询失败"
        print_result(f"重启后工单明细查询通过 - 共 {total_after_restart} 条")

        all_batches_after, total_after = list_batches(app)
        assert total_after == all_batches_total, f"重启后批次列表不完整，应该 {all_batches_total} 条，实际 {total_after} 条"
        print_result(f"重启后批次列表完整 - 共 {total_after} 条")

    print_step(14, "测试同名文件不同内容（允许提交）")
    temp_path5 = create_temp_csv(row_count=3)
    checksum1 = calculate_file_checksum(temp_path5)

    temp_path6 = create_temp_csv(row_count=2)
    df = pd.DataFrame([
        {'title': '不同内容1', 'content': '内容1', 'channel': 'web'},
        {'title': '不同内容2', 'content': '内容2', 'channel': 'web'},
    ])
    df.to_csv(temp_path6, index=False, encoding='utf-8-sig')
    checksum2 = calculate_file_checksum(temp_path6)

    assert checksum1 != checksum2, "两个文件校验和应该不同"
    print_result(f"文件校验和不同: {checksum1[:16]}... vs {checksum2[:16]}...")

    success, result = create_batch("same_name.csv", test_operator, temp_path5, app)
    assert success, "第一个同名文件提交失败"
    process_result = process_batch(result['batch_id'], temp_path5, app)
    assert process_result['success'], "处理失败"
    print_result(f"同名文件（内容A）提交成功，批次 ID: {result['batch_id']}")

    success, result = create_batch("same_name.csv", test_operator, temp_path6, app)
    assert success, "同名不同内容文件应该允许提交"
    process_result = process_batch(result['batch_id'], temp_path6, app)
    assert process_result['success'], "处理失败"
    print_result(f"同名文件（内容B）提交成功，批次 ID: {result['batch_id']}")

    os.remove(temp_path5)
    os.remove(temp_path6)

    print("\n" + "#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + " " * 30 + "测试完成！所有步骤通过 [OK]" + " " * 25 + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)

    print("\n测试摘要:")
    print(f"  [OK] 初始化数据库和准备测试环境 - 成功")
    print(f"  [OK] 导入训练数据并训练模型 - 成功")
    print(f"  [OK] 权限校验 - 无操作者提交失败 - 成功")
    print(f"  [OK] 批量预测 - 正常数据 (10条) - 成功")
    print(f"  [OK] 批量预测 - 包含错误数据 (8成功/2失败) - 成功")
    print(f"  [OK] 重复提交检测 - 同操作者同文件拒绝 - 成功")
    print(f"  [OK] 模型版本追踪 - 回滚后旧批次指向原模型 - 成功")
    print(f"  [OK] 导出功能 - CSV 格式 (含错误信息) - 成功")
    print(f"  [OK] 导出功能 - Excel 格式 - 成功")
    print(f"  [OK] 导出权限校验 - 无操作者导出失败 - 成功")
    print(f"  [OK] 批次列表查询和筛选 - 成功")
    print(f"  [OK] 工单筛选 - 低置信度和已改判 - 成功")
    print(f"  [OK] 服务重启后查询 - 数据持久化 - 成功")
    print(f"  [OK] 同名文件不同内容 - 允许提交 - 成功")

    print("\n测试场景覆盖:")
    print(f"  [OK] 批量导入（文件上传 + 批次记录）")
    print(f"  [OK] 导出（CSV/Excel 格式）")
    print(f"  [OK] 重启后查询（数据持久化）")
    print(f"  [OK] 冲突提交（重复提交检测、同名文件）")
    print(f"  [OK] 权限失败（无操作者提交/下载）")
    print(f"  [OK] 失败行不中断整批（错误容忍）")
    print(f"  [OK] 错误信息记录和导出")
    print(f"  [OK] 模型版本快照（回滚不影响旧批次）")
    print(f"  [OK] 配置快照保存")
    print(f"  [OK] 低置信度筛选")
    print(f"  [OK] 已改判记录筛选")

    print("\n生成的文件:")
    print(f"  - 训练数据: {training_data_path}")
    print(f"  - 导出文件目录: {config.BATCH_RESULT_DIR}")
    print(f"  - 数据库: {config.DB_PATH}")


if __name__ == "__main__":
    main()
