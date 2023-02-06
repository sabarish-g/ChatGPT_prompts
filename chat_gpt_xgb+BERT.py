import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_scale

# Load the BERT module from TensorFlow Hub
module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
bert_layer = hub.KerasLayer(module_url, trainable=False)

# Load the dataset
df = pd.read_csv('imbalanced_data.csv')
X = df['text'].values
y = df['label'].values

# Preprocess the text data for input into BERT
def preprocess_text(texts):
    input_ids = []
    input_masks = []
    input_segments = []
    for text in texts:
        encoded_dict = tokenizer.encode_plus(
                            text,
                            add_special_tokens=True,
                            max_length=512,
                            pad_to_max_length=True,
                            return_attention_mask=True,
                            return_token_type_ids=True,
                            truncation=True
                        )
        
        input_ids.append(encoded_dict['input_ids'])
        input_masks.append(encoded_dict['attention_mask'])
        input_segments.append(encoded_dict['token_type_ids'])
    return [np.array(input_ids), np.array(input_masks), np.array(input_segments)]

input_ids, input_masks, input_segments = preprocess_text(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(input_ids, y, test_size=0.2, random_state=42)

# Define a function to extract features from the BERT layer
def bert_extractor(input_ids, input_masks, input_segments):
    pool_embs, _ = bert_layer([input_ids, input_masks, input_segments])
    return pool_embs.numpy()

# Extract features from the BERT layer for both training and test data
train_features = bert_extractor(X_train, input_masks[train_index], input_segments[train_index])
test_features = bert_extractor(X_test, input_masks[test_index], input_segments[test_index])

# Train an XGBoost model on the extracted features
dtrain = xgb.DMatrix(train_features, label=y_train)
dtest = xgb.DMatrix(test_features, label=y_test)

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary:logistic',
    'metric':


python

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

In this code, we use GridSearchCV from scikit-learn to perform hyperparameter tuning on the XGBoost classifier. The hyperparameters that are being tuned are defined in the param_grid dictionary. The GridSearchCV will train the XGBoost classifier using cross-validation with 5 folds, and will search for the best hyperparameters based on the accuracy of the model on the validation data. The best hyperparameters found will be saved in the best_params_ attribute of the GridSearchCV object. The best estimator (i.e., the XGBoost classifier with the best hyperparameters) will be saved in the best_estimator_ attribute.















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