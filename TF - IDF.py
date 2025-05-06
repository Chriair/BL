import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

# 示例中文文本数据
corpus = [
    "这是第一个文档。",
    "这个文档是第二个文档。",
    "而这是第三个。",
    "这是第一个文档吗？"
]

# 对中文文本进行分词
corpus_seg = [' '.join(jieba.lcut(text)) for text in corpus]

# 创建TfidfVectorizer对象
vectorizer = TfidfVectorizer()

# 对分词后的文本数据进行TF - IDF向量化
X = vectorizer.fit_transform(corpus_seg)

# 获取特征名称（即词汇表）
feature_names = vectorizer.get_feature_names_out()

# 打印每个文档的TF - IDF向量表示
for doc_index in range(len(corpus_seg)):
    feature_index = X[doc_index, :].nonzero()[1]
    tfidf_scores = zip(feature_index, [X[doc_index, x] for x in feature_index])
    print(f"文档 {doc_index + 1}:")
    for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
        print(f"  单词: {w}, TF - IDF得分: {s}")

