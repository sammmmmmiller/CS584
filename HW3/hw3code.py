from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups_vectorized, fetch_20newsgroups
from sklearn.metrics import f1_score
import numpy as np
import os

# part a

categories = ['rec.autos', 'rec.motorcycles']
train_data_non_vectorized = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
test_data_non_vectorized = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))
train_data_vectorized = fetch_20newsgroups_vectorized(subset='train', remove=('headers', 'footers', 'quotes'))
test_data_vectorized = fetch_20newsgroups_vectorized(subset='test', remove=('headers', 'footers', 'quotes'))

train_indices = [i for i, target in enumerate(train_data_vectorized.target) if target in train_data_non_vectorized.target]
test_indices = [i for i, target in enumerate(test_data_vectorized.target) if target in test_data_non_vectorized.target]
x_train = train_data_vectorized.data[train_indices]
y_train = train_data_vectorized.target[train_indices]
x_test = test_data_vectorized.data[test_indices]
y_test = test_data_vectorized.target[test_indices]

clf = MultinomialNB()
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
print(f1_score(y_test, y_pred, average='weighted'))

# part b

misclassified_indices = np.where(y_test != y_pred)[0][:4]

misclassified_docs = [test_data_non_vectorized.data[i] for i in misclassified_indices]
actual = [test_data_non_vectorized.target_names[test_data_non_vectorized.target[i]] for i in misclassified_indices]
predicted = [test_data_non_vectorized.target_names[y_pred[i]] for i in misclassified_indices]

for i, (doc, actual, predicted) in enumerate(zip(misclassified_docs, actual, predicted)):
    print("Document " + str(i + 1) + f":\n{doc}\nActual: {actual}\nPredicted: {predicted}\n")

