import time
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity


def text_to_vectors(text):
    start_time = time.time()
    # 加载预训练的 BERT 分词器和模型
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    model = AutoModel.from_pretrained('bert-base-chinese')
    load_model_time = time.time()
    print(f"加载分词器和模型用时: {load_model_time - start_time} 秒")

    # 对输入文本进行分词和编码
    inputs = tokenizer(text, return_tensors='pt')
    tokenize_time = time.time()
    print(f"文本分词和编码用时: {tokenize_time - load_model_time} 秒")

    # 运行模型得到输出
    with torch.no_grad():
        outputs = model(**inputs)
    run_model_time = time.time()
    print(f"运行模型用时: {run_model_time - tokenize_time} 秒")

    # 获取词向量（这里取最后一层的隐藏状态）
    word_vectors = outputs.last_hidden_state[0]
    # 对词向量求平均得到文本向量
    text_vector = torch.mean(word_vectors, dim=0).numpy().reshape(1, -1)
    process_vector_time = time.time()
    print(f"处理词向量得到文本向量用时: {process_vector_time - run_model_time} 秒")

    return text_vector


# 示例文本
text1 = "自然语言处理很有趣"
text2 = "自然语言处理很有挑战性"

start_total = time.time()
# 将文本转换为向量
vector1 = text_to_vectors(text1)
vector2 = text_to_vectors(text2)
vectorize_time = time.time()
print(f"两个文本向量化总用时: {vectorize_time - start_total} 秒")

# 计算余弦相似度
similarity = cosine_similarity(vector1, vector2)[0][0]
similarity_time = time.time()
print(f"计算相似度用时: {similarity_time - vectorize_time} 秒")

print(f"文本 '{text1}' 和 '{text2}' 的相似度为: {similarity}")