from transformers import AutoTokenizer, AutoModel
import torch


def text_to_vectors(text):
    # 加载预训练的 BERT 分词器和模型
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    model = AutoModel.from_pretrained('bert-base-chinese')

    # 对输入文本进行分词和编码
    inputs = tokenizer(text, return_tensors='pt')

    # 运行模型得到输出
    with torch.no_grad():
        outputs = model(**inputs)

    # 获取词向量（这里取最后一层的隐藏状态）
    word_vectors = outputs.last_hidden_state[0]

    return word_vectors


# 示例文本
text = "自然语言处理很有趣"
# 将文本转换为词向量
vectors = text_to_vectors(text)
print("文本中每个词的词向量形状:", vectors.shape)
print("第一个词的词向量:", vectors[0])