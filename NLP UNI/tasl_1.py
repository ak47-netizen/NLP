import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# TASK 1: Bag of Words Matrix Construction (Customer Reviews)
# =========================================================
print("=" * 60)
print("TASK 1: Bag of Words Matrix Construction")
print("=" * 60)

corpus = [
    "The product performance is amazing and fast",
    "The service was fast and performance was great",
    "Terrible customer service and bad performance"
]

# Step 1: Instantiate CountVectorizer with English stop words removed
vectorizer = CountVectorizer(stop_words='english')

# Step 2: Fit-transform the corpus
bow_matrix = vectorizer.fit_transform(corpus)

# Step 3: Extract vocabulary
vocabulary = vectorizer.get_feature_names_out()
print("\nVocabulary:", list(vocabulary))