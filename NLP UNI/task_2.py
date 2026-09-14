import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "Machine learning algorithms analyze structured data effectively",
    "Deep learning and neural networks excel at processing unstructured data",
    "Natural language processing helps computers understand human language",
    "Python is widely used for machine learning and data science"
]

print(documents)

doc_vectorizer = CountVectorizer()
doc_vectors = doc_vectorizer.fit_transform(documents)

print(doc_vectors)

query_vector = doc_vectorizer.transform(query)
# 
#Step 3: Compute pairwise cosine similarity
# similarity_scores = cosine_similarity(query_vector, doc_vectors)[0]
# 
# print(similarity_scores)