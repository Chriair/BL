from sentence_transformers import SentenceTransformer
import numpy as np

# 加载预训练的中文模型
model = SentenceTransformer('shibing624/text2vec-base-chinese')

# 示例文本
text1 = "自然语言处理很有趣"
text2 = "自然语言处理很有挑战性"

# 对文本进行编码
embedding1 = model.encode(text1)
embedding2 = model.encode(text2)

# 计算余弦相似度
similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
print("文本相似度:", similarity)