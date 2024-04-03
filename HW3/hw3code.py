from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups_vectorized, fetch_20newsgroups
from sklearn.metrics import f1_score
import numpy as np
import os

# Fetch the category labels
categories = ['rec.autos', 'rec.motorcycles']
train_data_non_vectorized = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
test_data_non_vectorized = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))

# Fetch vectorized data without filtering categories
train_data_vectorized = fetch_20newsgroups_vectorized(subset='train', remove=('headers', 'footers', 'quotes'))
test_data_vectorized = fetch_20newsgroups_vectorized(subset='test', remove=('headers', 'footers', 'quotes'))

# Filter the vectorized data to include only 'rec.autos' and 'rec.motorcycles'
# Finding the indices of the desired categories
train_indices = [i for i, target in enumerate(train_data_vectorized.target) if target in train_data_non_vectorized.target]
test_indices = [i for i, target in enumerate(test_data_vectorized.target) if target in test_data_non_vectorized.target]

# Filter based on identified indices
x_train = train_data_vectorized.data[train_indices]
y_train = train_data_vectorized.target[train_indices]
x_test = test_data_vectorized.data[test_indices]
y_test = test_data_vectorized.target[test_indices]

# Initialize and train the classifier
clf = MultinomialNB()
clf.fit(x_train, y_train)

# Make predictions and evaluate
y_pred = clf.predict(x_test)
print(f1_score(y_test, y_pred, average='weighted'))  # Use weighted in binary classification for handling label imbalance.

# part b

# Find misclassified documents
misclassified_indices = np.where(y_test != y_pred)[0]

# If there are at least 4 misclassified documents, choose the first 4
if len(misclassified_indices) >= 4:
    sample_indices = misclassified_indices[:4]
else:
    sample_indices = misclassified_indices

# Extract the documents and their predicted and actual labels
misclassified_docs = [test_data_non_vectorized.data[i] for i in sample_indices]
actual_labels = [test_data_non_vectorized.target_names[test_data_non_vectorized.target[i]] for i in sample_indices]
predicted_labels = [test_data_non_vectorized.target_names[y_pred[i]] for i in sample_indices]

for doc, actual, predicted in zip(misclassified_docs, actual_labels, predicted_labels):
    print(f"Document: {doc}\nActual Label: {actual}\nPredicted Label: {predicted}\n")


# def read_text_files(file_paths, encoding='utf-8', errors='ignore'):
#     texts = []
#     for file_path in file_paths:
#         try:
#             with open(file_path, 'r', encoding=encoding, errors=errors) as file:
#                 texts.append(file.read())
#         except Exception as e:
#             print(f"Error reading {file_path}: {e}")
#     return texts

# clf = MultinomialNB()

# # Training data
# path = r'C:\Users\Sam\Projects\CS584\HW3\20news-bydate-train\rec.autos'
# lst_autos = [os.path.join(path, f) for f in os.listdir(path)]
# texts_autos = read_text_files(lst_autos)

# path = r'C:\Users\Sam\Projects\CS584\HW3\20news-bydate-train\rec.motorcycles'
# lst_motorcycles = [os.path.join(path, f) for f in os.listdir(path)]
# texts_motorcycles = read_text_files(lst_motorcycles)

# texts_train = texts_autos + texts_motorcycles
# y_train = [1] * len(texts_autos) + [2] * len(texts_motorcycles)

# # CountVectorizer
# cv = CountVectorizer(strip_accents='unicode', stop_words='english')
# x_train = cv.fit_transform(texts_train)

# clf.fit(x_train, y_train)

# # Test data
# path = r'C:\Users\Sam\Projects\CS584\HW3\20news-bydate-test\rec.autos'
# lst_autos_test = [os.path.join(path, f) for f in os.listdir(path)]
# texts_autos_test = read_text_files(lst_autos_test)

# path = r'C:\Users\Sam\Projects\CS584\HW3\20news-bydate-test\rec.motorcycles'
# lst_motorcycles_test = [os.path.join(path, f) for f in os.listdir(path)]
# texts_motorcycles_test = read_text_files(lst_motorcycles_test)

# texts_test = texts_autos_test + texts_motorcycles_test
# y_test = [1] * len(texts_autos_test) + [2] * len(texts_motorcycles_test)

# x_test = cv.transform(texts_test)

# y_pred = clf.predict(x_test)

# print(f1_score(y_test, y_pred, pos_label=1))
