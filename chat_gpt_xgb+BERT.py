import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Load the pre-trained BERT model from TensorFlow Hub
bert_model = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1")

# Load the dataset
df = pd.read_csv('data.csv')
X = df['text']
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Convert the training and testing data into BERT embeddings
def bert_encode(texts, model):
    all_embeddings = model(texts)
    return np.array([embedding["output_layer"][0,:,:].numpy() for embedding in all_embeddings])

X_train_embeddings = bert_encode(X_train.tolist(), bert_model)
X_test_embeddings = bert_encode(X_test.tolist(), bert_model)

# Train an XGBoost classifier on the BERT embeddings
clf = XGBClassifier(random_state=42)
clf.fit(X_train_embeddings, y_train)

# Evaluate the performance of the model on the test data
accuracy = clf.score(X_test_embeddings, y_test)
print("Accuracy: ", accuracy)


import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# Load the pre-trained BERT model from TensorFlow Hub
bert_model = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1")

# Load the dataset
df = pd.read_csv('data.csv')
X = df['text']
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Convert the training and testing data into BERT embeddings
def bert_encode(texts, model):
    all_embeddings = model(texts)
    return np.array([embedding["output_layer"][0,:,:].numpy() for embedding in all_embeddings])

X_train_embeddings = bert_encode(X_train.tolist(), bert_model)
X_test_embeddings = bert_encode(X_test.tolist(), bert_model)

# Define the parameters for the XGBoost classifier
param_grid = {
    'learning_rate': [0.1, 0.5, 1],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 500]
}

# Train an XGBoost classifier on the BERT embeddings
clf = XGBClassifier(random_state=42)
grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(X_train_embeddings, y_train)

# Evaluate the performance of the model on the test data
best_clf = grid_search.best_estimator_
accuracy = best_clf.score(X_test_embeddings, y_test)
print("Accuracy: ", accuracy)
print("Best hyperparameters: ", grid_search.best_params_)