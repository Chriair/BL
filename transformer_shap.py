import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import shap
import numpy as np

# 加载预训练的 BERT 分词器和分类模型
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)

# 示例文本
texts = ["自然语言处理很有趣", "这个任务很无聊"]

# 定义一个预测函数，用于 SHAP
def predict(texts):
    # 处理 shap 可能传入的不同类型输入
    if isinstance(texts, (int, float)):
        texts = [str(texts)]
    elif isinstance(texts, np.ndarray):
        texts = texts.tolist()
    elif isinstance(texts, list) and all(isinstance(item, (int, float)) for item in texts):
        texts = [str(item) for item in texts]
    elif isinstance(texts, (list, str)):
        pass
    else:
        raise ValueError("输入类型不支持，请传入字符串或字符串列表")

    if isinstance(texts, str):
        texts = [texts]

    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    return probabilities.numpy()

# 创建 SHAP 解释器
explainer = shap.Explainer(predict, tokenizer)

# 计算 SHAP 值
shap_values = explainer(texts)

# 打印第一个文本的 SHAP 值
print(shap_values[0])