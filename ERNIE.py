import paddle
from paddlenlp.transformers import AutoTokenizer, AutoModel

# 加载预训练的ERNIE 3.0模型和分词器
tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-base-zh")
model = AutoModel.from_pretrained("ernie-3.0-base-zh")

# 示例文本
text1 = "自然语言处理很有趣"
text2 = "自然语言处理很有挑战性"

# 对文本进行分词和编码
inputs1 = tokenizer(text1, return_tensors='pd')
inputs2 = tokenizer(text2, return_tensors='pd')

# 获得模型输出
outputs1 = model(**inputs1)
outputs2 = model(**inputs2)

# 取[CLS]标记的输出作为句子表示
embedding1 = outputs1[0][:, 0]
embedding2 = outputs2[0][:, 0]

# 计算余弦相似度
similarity = paddle.nn.functional.cosine_similarity(embedding1, embedding2)
print("文本相似度:", similarity.numpy()[0])