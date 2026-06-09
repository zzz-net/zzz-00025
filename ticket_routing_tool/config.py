import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, 'data', 'ticket_routing.db')

MODEL_DIR = os.path.join(BASE_DIR, 'data', 'models')

DATASET_DIR = os.path.join(BASE_DIR, 'data', 'datasets')

REPORT_DIR = os.path.join(BASE_DIR, 'data', 'reports')

SAMPLE_DIR = os.path.join(BASE_DIR, 'data', 'samples')

SUPPORTED_CHANNELS = ['email', 'phone', 'web', 'app', 'wechat', 'weibo']

QUEUE_MAPPING = {
    'technical_support': 'tech_support_queue',
    'bug_report': 'tech_support_queue',
    'billing': 'billing_queue',
    'refund_request': 'billing_queue',
    'account_management': 'account_queue',
    'security': 'account_queue',
    'cancellation': 'account_queue',
    'product_inquiry': 'product_suggestion_queue',
    'feature_request': 'product_suggestion_queue',
    'upgrade': 'product_suggestion_queue',
    'downgrade': 'product_suggestion_queue',
    'accessibility': 'product_suggestion_queue',
    'complaint': 'complaint_queue',
    'praise': 'complaint_queue',
    'general_question': 'complaint_queue',
}

QUEUE_DISPLAY_NAMES = {
    'tech_support_queue': '技术支持',
    'billing_queue': '账单咨询',
    'account_queue': '账户问题',
    'product_suggestion_queue': '产品建议',
    'complaint_queue': '投诉建议',
}

SUPPORTED_QUEUES = ['技术支持', '账单咨询', '账户问题', '产品建议', '投诉建议']

QUEUE_NAME_TO_TAG = {
    '技术支持': 'technical_support',
    '账单咨询': 'billing',
    '账户问题': 'account_management',
    '产品建议': 'feature_request',
    '投诉建议': 'complaint',
}

METRIC_EXPLANATIONS = {
    'precision': '精确率：预测为正例中实际为正例的比例，衡量模型预测的准确性',
    'recall': '召回率：实际为正例中被预测为正例的比例，衡量模型的查全能力',
    'f1': 'F1值：精确率和召回率的调和平均值，综合评价模型性能',
    'accuracy': '准确率：预测正确的样本占总样本的比例',
    'support': '支持数：该类别在数据集中的样本数量'
}
