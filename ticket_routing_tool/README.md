# 客服工单分流训练与回放工具

完全本地运行的客服工单分类训练、预测、人工改判与模型回滚系统。

## 功能特性

### 核心功能
- **CSV导入**：支持导入包含标题、正文、渠道、标签的训练数据
- **模型训练**：使用 TF-IDF + LogisticRegression 的轻量分类模型
- **模型评估**：展示精确率、召回率、F1值、准确率等指标及详细解释
- **工单预测**：对新工单给出队列建议和处理优先级
- **人工改判**：记录操作者、改判原因和原预测结果
- **模型回滚**：支持版本历史管理和一键回滚
- **报告导出**：导出复核报告和模型对比报告（Excel/CSV格式）

### 失败保护机制
- **空标签保护**：训练集出现空标签时，训练标记为失败，不替换当前可用模型
- **未知标签保护**：训练集出现未知标签时，训练标记为失败
- **无模型保护**：未训练模型前进行预测会给出明确错误提示
- **回滚保护**：失败训练的模型版本不能被选为回滚目标

## 技术栈

- **后端框架**：Flask 3.0
- **数据库**：SQLite (Flask-SQLAlchemy)
- **机器学习**：scikit-learn (TF-IDF + LogisticRegression)
- **数据处理**：pandas, numpy
- **模型持久化**：joblib
- **前端**：Bootstrap 5 + Jinja2 模板

## 快速开始

### 1. 安装依赖
```bash
cd ticket_routing_tool
pip install -r requirements.txt
```

### 2. 运行完整流程测试
```bash
python test_flow.py
```

### 3. 启动Web服务
```bash
python app.py
```

### 4. 访问页面
打开浏览器访问：http://localhost:5000/

## 页面路径

| 页面 | 路径 | 功能 |
|------|------|------|
| 首页仪表盘 | `/` | 系统状态总览、快速操作 |
| 数据导入 | `/import` | 上传CSV训练数据 |
| 模型训练 | `/train` | 选择数据集开始训练 |
| 模型评估 | `/evaluate` | 查看评估指标和解释 |
| 工单预测 | `/predict` | 输入工单进行预测 |
| 工单列表 | `/tickets` | 查看所有预测工单 |
| 人工改判 | `/override/<ticket_id>` | 修正预测结果 |
| 模型回滚 | `/rollback` | 版本历史和回滚操作 |
| 报告导出 | `/reports` | 导出复核报告和对比报告 |

## API 接口

所有接口返回统一格式：
```json
{
  "success": true/false,
  "message": "操作说明",
  "data": {}
}
```

### 数据导入
```bash
curl -X POST http://localhost:5000/api/import \
  -F "file=@data/samples/sample_training_data.csv"
```

### 模型训练
```bash
curl -X POST http://localhost:5000/api/train/<dataset_id>
```

### 工单预测
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "系统登录失败",
    "content": "用户反馈登录时提示密码错误，但确认输入正确",
    "channel": "email"
  }'
```

### 人工改判
```bash
curl -X POST http://localhost:5000/api/override/<ticket_id> \
  -H "Content-Type: application/json" \
  -d '{
    "corrected_queue": "账单咨询",
    "operator": "管理员",
    "reason": "用户实际咨询的是账单问题，预测错误"
  }'
```

### 模型回滚
```bash
curl -X POST http://localhost:5000/api/rollback/<model_version_id>
```

### 导出复核报告
```bash
curl -X POST http://localhost:5000/api/report/audit \
  -H "Content-Type: application/json" \
  -d '{
    "start_date": "2026-06-01",
    "end_date": "2026-06-30",
    "format": "xlsx"
  }'
```

## 样本数据

项目自带3份样本数据，位于 `data/samples/` 目录：

### 1. sample_training_data.csv（50条）
用于训练的标准数据，包含：
- 5个分类标签各10条
- 渠道分布：email、phone、web、app、wechat各10条
- tags列格式：`主标签,副标签`

### 2. sample_bad_data.csv（10条）
用于测试失败保护机制，包含：
- 3条空标签记录
- 未知标签：unknown_tag、invalid_label等
- 无效渠道：invalid_channel等

### 3. sample_predict_tickets.csv（20条）
待预测工单数据，包含：
- 各种类型的工单
- 部分内容模糊的工单（测试低置信度场景）

## 队列映射

| 中文队列 | 英文标签 | 处理队列 |
|----------|----------|----------|
| 技术支持 | technical_support, bug_report | tech_support_queue |
| 账单咨询 | billing, refund_request | billing_queue |
| 账户问题 | account_management, security, cancellation | account_queue |
| 产品建议 | product_inquiry, feature_request, upgrade, downgrade, accessibility | product_suggestion_queue |
| 投诉建议 | complaint, praise, general_question | complaint_queue |

## 队列建议规则

根据预测置信度给出不同的处理建议：

| 置信度 | 建议 | 优先级 |
|--------|------|--------|
| >= 0.8 | 直接分配到对应队列 | 高 |
| 0.5 - 0.8 | 分配到对应队列，需关注 | 中 |
| < 0.5 | 建议人工审核 | 低 |

## 评估指标解释

- **精确率 (Precision)**：预测为正例中实际为正例的比例，衡量模型预测的准确性
- **召回率 (Recall)**：实际为正例中被预测为正例的比例，衡量模型的查全能力
- **F1值 (F1 Score)**：精确率和召回率的调和平均值，综合评价模型性能
- **准确率 (Accuracy)**：预测正确的样本占总样本的比例
- **支持数 (Support)**：该类别在数据集中的样本数量

## 数据持久化

所有数据均保存在本地，重启服务后可继续使用：

- **数据库**：`data/ticket_routing.db` - SQLite数据库，包含所有元数据
- **数据集**：`data/datasets/` - 导入的CSV文件版本
- **模型文件**：`data/models/` - 训练好的模型和向量化器
- **评估报告**：`data/reports/` - 导出的Excel/CSV报告

## 测试流程复现

运行 `test_flow.py` 可复现完整流程：

1. 初始化数据库
2. 导入50条训练数据
3. 训练模型并自动激活
4. 查看评估指标
5. 预测单个工单
6. 批量预测20条工单
7. 人工改判其中1条
8. 导入坏数据验证失败保护（训练失败）
9. 回滚到之前的模型版本
10. 导出复核报告
11. 导出模型对比报告

## 目录结构

```
ticket_routing_tool/
├── app.py                 # Flask应用入口
├── config.py              # 配置文件
├── requirements.txt       # 依赖列表
├── test_flow.py           # 流程测试脚本
├── commands.txt           # 命令参考
├── README.md              # 此文件
├── data/
│   ├── datasets/          # 导入的数据集
│   ├── models/            # 训练好的模型
│   ├── reports/           # 导出的报告
│   ├── samples/           # 样本数据
│   └── ticket_routing.db  # SQLite数据库
├── src/                   # 源代码
│   ├── __init__.py
│   ├── database.py        # 数据库初始化
│   ├── models.py          # 数据模型定义
│   ├── csv_importer.py    # CSV导入模块
│   ├── data_validator.py  # 数据验证模块
│   ├── classifier.py      # 模型训练模块
│   ├── evaluator.py       # 模型评估模块
│   ├── predictor.py       # 预测模块
│   ├── human_override.py  # 人工改判模块
│   ├── rollback.py        # 模型回滚模块
│   └── report_exporter.py # 报告导出模块
├── templates/             # HTML模板
│   ├── base.html
│   ├── index.html
│   ├── import.html
│   ├── train.html
│   ├── evaluate.html
│   ├── predict.html
│   ├── tickets.html
│   ├── override.html
│   ├── rollback.html
│   └── reports.html
└── static/
    └── css/
        └── style.css      # 自定义样式
```

## 注意事项

1. 训练数据必须包含 `title`, `content`, `channel`, `tags` 四列
2. tags列使用英文标签，多个标签用逗号分隔
3. 首次使用必须先导入数据并训练模型，否则无法进行预测
4. 失败的训练模型不会被自动激活，也不能被选为回滚目标
5. 人工改判会永久记录，可通过复核报告导出进行审计
