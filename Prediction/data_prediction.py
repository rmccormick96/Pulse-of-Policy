
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

########## Prepare for prediction
# Create target variable
df_merged = pd.read_csv('Data/df_merged.txt')

df = df_merged.loc[df_merged.loc[:, 'introduced_date'].str[0: 4] == '2017']
df.loc[:, 'target'] = 0
df.loc[~df.loc[:, 'house_passage'].isnull(), 'target'] = 1

df.loc[:, 'target'].value_counts()
df.loc[:, 'month'] = df.loc[:, 'introduced_date'].str[5: 7]
df.loc[:, 'month'] = pd.to_numeric(df.loc[:, 'month'])

# Filter to relevant timeframe
df = df.loc[df.loc[:, 'month'] <= 10]

df = df.drop(columns = ['CVG_END_DT', 'month', 'house_passage', 
                        'sponsor_party'])

df = df.fillna(0)

df.to_csv('Data/df_prediction/df_prediction.txt', index = False)

########################################
### Benchmark Prediction - No text data

list_model = []
list_acc = []
list_recall = []
list_precision = []
list_f1 = []

df = pd.read_csv('Data/df_prediction/df_prediction.txt')


df_descr = df.describe()

cols_drop = ['TRANS_TO_AUTH', 'CAND_CONTRIB', 'CAND_LOANS', 'OTHER_LOANS', 
              'CAND_LOAN_REPAY', 'OTHER_LOAN_REPAY', 'DEBTS_OWED_BY']

cols_keep = ['TTL_RECEIPTS', 'TTL_DISB', 'TTL_INDIV_CONTRIB', 
             'OTHER_POL_CMTE_CONTRIB', 'INDIV_REFUNDS', 'cosp_dem', 'cosp_rep', 
             'cosp_ratio', 'target']

df = df.loc[:, cols_keep]

X = df.drop(columns = ['target'])

y = df.loc[:, 'target']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2, 
                                                    random_state = 9384)
# Oversampling 
ros = RandomOverSampler(random_state = 389)
X_train, y_train = ros.fit_resample(X_train, y_train)
y_train.value_counts()

# Logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

acc = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", acc)
print("Recall:", recall)
print("Precision:", precision)
print("F1: ", f1)

list_model.append("Logistic Regression")
list_acc.append(acc)
list_recall.append(recall)
list_precision.append(precision)
list_f1.append(f1)

y_test.value_counts()

# Catboost
model = CatBoostClassifier(iterations = 100,
                           depth=10,
                           learning_rate=0.001,
                           loss_function = 'Logloss',
                           verbose=True)

model.fit(X_train, y_train, verbose=False)

predictions = model.predict(X_test)

acc = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", acc)
print("Recall:", recall)
print("Precision:", precision)
print("F1: ", f1)

list_model.append("CatBoost Classifier")
list_acc.append(acc)
list_recall.append(recall)
list_precision.append(precision)
list_f1.append(f1)

# Support vector machine
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_classifier = SVC(kernel='rbf', gamma='auto')
svm_classifier.fit(X_train_scaled, y_train)

predictions = svm_classifier.predict(X_test_scaled)

acc = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", acc)
print("Recall:", recall)
print("Precision:", precision)
print("F1: ", f1)

list_model.append("Support Vector Machine")
list_acc.append(acc)
list_recall.append(recall)
list_precision.append(precision)
list_f1.append(f1)

## Random Forest
random_forest_classifier = RandomForestClassifier(n_estimators = 100,
                                                  max_depth = 5, 
                                                  random_state = 782,
                                                  class_weight = 'balanced')
random_forest_classifier.fit(X_train, y_train)
predictions = random_forest_classifier.predict(X_test)

acc = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", acc)
print("Recall:", recall)
print("Precision:", precision)

list_model.append("Random Forest Classifier")
list_acc.append(acc)
list_recall.append(recall)
list_precision.append(precision)
list_f1.append(f1)

## XGBoost
dtrain = xgb.DMatrix(X_train, label = y_train)
dtest = xgb.DMatrix(X_test, label = y_test)

params = {
    'max_depth': 3,
    'eta': 0.01,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'  # You can also use 'auc' for area under the curve
}

num_rounds = 100
watchlist = [(dtest, 'test'), (dtrain, 'train')]
bst = xgb.train(params, dtrain, num_rounds, watchlist)

preds = bst.predict(dtest)
predictions = [1 if i > 0.5 else 0 for i in preds]

acc = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", acc)
print("Recall:", recall)
print("Precision:", precision)

list_model.append("XGBoost")
list_acc.append(acc)
list_recall.append(recall)
list_precision.append(precision)
list_f1.append(f1)

## Save Benchmark results
df_results = pd.DataFrame([list_model, 
                           list_acc, 
                           list_recall, 
                           list_precision,
                           list_f1]).T
df_results.columns = ['Model', 'Accuracy', 'Recall', 'Precision', 'F1 Score']

###################################################
### Precition using Similarity with BOW methodology
list_model = []
list_acc = []
list_recall = []
list_precision = []
list_f1 = []

df = pd.read_csv('Data/df_prediction/df_prediction.txt')

path = "Data/text_indicators/bills_with_bow_similarity.csv"
df_sim = pd.read_csv(path)

df_sim = df_sim.loc[:, ['bill_id', 'bow_similarity', 'avg_media_sentiment']]

df = df.merge(df_sim, how = 'left', on = 'bill_id')


cols_drop = ['TRANS_TO_AUTH', 'CAND_CONTRIB', 'CAND_LOANS', 'OTHER_LOANS', 
              'CAND_LOAN_REPAY', 'OTHER_LOAN_REPAY', 'DEBTS_OWED_BY']

cols_keep = ['TTL_RECEIPTS', 'TTL_DISB', 'TTL_INDIV_CONTRIB', 
             'OTHER_POL_CMTE_CONTRIB', 'INDIV_REFUNDS', 'cosp_dem', 'cosp_rep', 
             'cosp_ratio', 'bow_similarity', 'avg_media_sentiment', 'target']

df = df.loc[:, cols_keep]

# X = df.drop(columns = ['introduced_date', 'target', 'OTHER_LOANS'])
X = df.drop(columns = ['target'])

y = df.loc[:, 'target']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2, 
                                                    random_state = 9384)
# Oversampling 
ros = RandomOverSampler(random_state = 389)
X_train, y_train = ros.fit_resample(X_train, y_train)
y_train.value_counts()


# Logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

acc = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", acc)
print("Recall:", recall)
print("Precision:", precision)

list_model.append("Logistic Regression")
list_acc.append(acc)
list_recall.append(recall)
list_precision.append(precision)
list_f1.append(f1)

# Catboost
model = CatBoostClassifier(iterations = 100,
                           depth=10,
                           learning_rate=0.001,
                           loss_function = 'Logloss',
                           verbose=True)

model.fit(X_train, y_train, verbose=False)

predictions = model.predict(X_test)

acc = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", acc)
print("Recall:", recall)
print("Precision:", precision)

list_model.append("CatBoost Classifier")
list_acc.append(acc)
list_recall.append(recall)
list_precision.append(precision)
list_f1.append(f1)

# Support vector machine
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_classifier = SVC(kernel='rbf', gamma='auto')
svm_classifier.fit(X_train_scaled, y_train)

predictions = svm_classifier.predict(X_test_scaled)

acc = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", acc)
print("Recall:", recall)
print("Precision:", precision)

list_model.append("Support Vector Machine")
list_acc.append(acc)
list_recall.append(recall)
list_precision.append(precision)
list_f1.append(f1)

## Random Forest
random_forest_classifier = RandomForestClassifier(n_estimators = 100,
                                                  max_depth = 5, 
                                                  random_state = 782,
                                                  class_weight = 'balanced')
random_forest_classifier.fit(X_train, y_train)
predictions = random_forest_classifier.predict(X_test)

acc = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", acc)
print("Recall:", recall)
print("Precision:", precision)

list_model.append("Random Forest Classifier")
list_acc.append(acc)
list_recall.append(recall)
list_precision.append(precision)
list_f1.append(f1)

## XGBoost
dtrain = xgb.DMatrix(X_train, label = y_train)
dtest = xgb.DMatrix(X_test, label = y_test)

params = {
    'max_depth': 3,
    'eta': 0.01,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'  # You can also use 'auc' for area under the curve
}

num_rounds = 100
watchlist = [(dtest, 'test'), (dtrain, 'train')]
bst = xgb.train(params, dtrain, num_rounds, watchlist)

preds = bst.predict(dtest)
predictions = [1 if i > 0.5 else 0 for i in preds]

acc = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", acc)
print("Recall:", recall)
print("Precision:", precision)

list_model.append("XGBoost")
list_acc.append(acc)
list_recall.append(recall)
list_precision.append(precision)
list_f1.append(f1)

## Save results from Similarity with BOW methodology 
df_results_bow = pd.DataFrame([list_model, 
                           list_acc, 
                           list_recall, 
                           list_precision,
                           list_f1]).T

df_results_bow.columns = ['Model', 'Accuracy', 'Recall', 
                          'Precision', 'F1 Score']

############################################################################
### Prediction using Similarity with Glove Embeddings Unweighted methodology
list_model = []
list_acc = []
list_recall = []
list_precision = []
list_f1 = []

import pandas as pd
df = pd.read_csv('Data/df_prediction/df_prediction.txt')

path = "Data/text_indicators/final_pca.csv"
cols = ['bill_id', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6',
'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15',
'PC16', 'PC17', 'PC18', 'PC19', 'PC20', 'PC21', 'PC22', 'PC23', 'PC24',
'PC25', 'PC26', 'PC27', 'PC28', 'PC29', 'PC30', 'PC31', 'PC32']
df_sim = pd.read_csv(path)

df_sim = df_sim.loc[:, cols]

df = df.merge(df_sim, how = 'left', on = 'bill_id')

# cols_drop = ['TRANS_TO_AUTH', 'CAND_CONTRIB', 'CAND_LOANS', 'OTHER_LOANS', 
#               'CAND_LOAN_REPAY', 'OTHER_LOAN_REPAY', 'DEBTS_OWED_BY']

cols_keep = ['TTL_RECEIPTS', 'TTL_DISB', 'TTL_INDIV_CONTRIB', 
             'OTHER_POL_CMTE_CONTRIB', 'INDIV_REFUNDS', 'cosp_dem', 'cosp_rep', 
             'cosp_ratio', 
             'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6',
             'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15',
             'PC16', 'PC17', 'PC18', 'PC19', 'PC20', 'PC21', 'PC22', 'PC23', 'PC24',
             'PC25', 'PC26', 'PC27', 'PC28', 'PC29', 'PC30', 'PC31', 'PC32',
             'target']

df = df.loc[:, cols_keep]

X = df.drop(columns = ['target'])

y = df.loc[:, 'target']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2, 
                                                    random_state = 9384)
# Oversampling 
ros = RandomOverSampler(random_state = 389)
X_train, y_train = ros.fit_resample(X_train, y_train)
y_train.value_counts()


# Logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

acc = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", acc)
print("Recall:", recall)
print("Precision:", precision)

list_model.append("Logistic Regression")
list_acc.append(acc)
list_recall.append(recall)
list_precision.append(precision)
list_f1.append(f1)

y_test.value_counts()

# Catboost
model = CatBoostClassifier(iterations = 100,
                           depth=10,
                           learning_rate=0.001,
                           loss_function = 'Logloss',
                           verbose=True)

model.fit(X_train, y_train, verbose=False)

predictions = model.predict(X_test)

acc = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", acc)
print("Recall:", recall)
print("Precision:", precision)

list_model.append("CatBoost Classifier")
list_acc.append(acc)
list_recall.append(recall)
list_precision.append(precision)
list_f1.append(f1)


# Support vector machine
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_classifier = SVC(kernel='rbf', gamma='auto')
svm_classifier.fit(X_train_scaled, y_train)

predictions = svm_classifier.predict(X_test_scaled)

acc = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", acc)
print("Recall:", recall)
print("Precision:", precision)

list_model.append("Support Vector Machine")
list_acc.append(acc)
list_recall.append(recall)
list_precision.append(precision)
list_f1.append(f1)

## Random Forest
random_forest_classifier = RandomForestClassifier(n_estimators = 100,
                                                  max_depth = 5, 
                                                  random_state = 782,
                                                  class_weight = 'balanced')
random_forest_classifier.fit(X_train, y_train)
predictions = random_forest_classifier.predict(X_test)

acc = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", acc)
print("Recall:", recall)
print("Precision:", precision)

list_model.append("Random Forest Classifier")
list_acc.append(acc)
list_recall.append(recall)
list_precision.append(precision)
list_f1.append(f1)


## XGBoost
dtrain = xgb.DMatrix(X_train, label = y_train)
dtest = xgb.DMatrix(X_test, label = y_test)

params = {
    'max_depth': 3,
    'eta': 0.01,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'  # You can also use 'auc' for area under the curve
}

num_rounds = 100
watchlist = [(dtest, 'test'), (dtrain, 'train')]
bst = xgb.train(params, dtrain, num_rounds, watchlist)

preds = bst.predict(dtest)
predictions = [1 if i > 0.5 else 0 for i in preds]

acc = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", acc)
print("Recall:", recall)
print("Precision:", precision)

list_model.append("XGBoost")
list_acc.append(acc)
list_recall.append(recall)
list_precision.append(precision)
list_f1.append(f1)

## save results of Similarity with Glove Embeddings Unweighted methodology
df_results_pca_noweights = pd.DataFrame([list_model, 
                           list_acc, 
                           list_recall, 
                           list_precision,
                           list_f1]).T

df_results_pca_noweights.columns = ['Model', 'Accuracy', 
                                    'Recall', 'Precision', 'F1 Score']

############################################################################
### Prediction using Similarity with Glove Embeddings Weighted methodology
list_model = []
list_acc = []
list_recall = []
list_precision = []
list_f1 = []

df = pd.read_csv('Data/df_prediction/df_prediction.txt')

path = "Data/text_indicators/final_pca_weighted.csv"
cols = ['bill_id', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6',
'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15',
'PC16', 'PC17', 'PC18', 'PC19', 'PC20']
df_sim = pd.read_csv(path)

df_sim = df_sim.loc[:, cols]

# igit = df_sim.iloc[0]

# df_sim.loc[:, 'PC1']

df = df.merge(df_sim, how = 'left', on = 'bill_id')


cols_drop = ['TRANS_TO_AUTH', 'CAND_CONTRIB', 'CAND_LOANS', 'OTHER_LOANS', 
              'CAND_LOAN_REPAY', 'OTHER_LOAN_REPAY', 'DEBTS_OWED_BY']

cols_keep = ['TTL_RECEIPTS', 'TTL_DISB', 'TTL_INDIV_CONTRIB', 
             'OTHER_POL_CMTE_CONTRIB', 'INDIV_REFUNDS', 'cosp_dem', 'cosp_rep', 
             'cosp_ratio', 
             'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6',
             'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15',
             'PC16', 'PC17', 'PC18', 'PC19', 'PC20',
             'target']

df = df.loc[:, cols_keep]

# X = df.drop(columns = ['introduced_date', 'target', 'OTHER_LOANS'])
X = df.drop(columns = ['target'])

y = df.loc[:, 'target']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2, 
                                                    random_state = 9384)
# Oversampling 
ros = RandomOverSampler(random_state = 389)
X_train, y_train = ros.fit_resample(X_train, y_train)
y_train.value_counts()


# Logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

acc = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", acc)
print("Recall:", recall)
print("Precision:", precision)
# print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
# print("Classification Report:\n", classification_report(y_test, predictions))

list_model.append("Logistic Regression")
list_acc.append(acc)
list_recall.append(recall)
list_precision.append(precision)
list_f1.append(f1)

y_test.value_counts()

# Catboost
model = CatBoostClassifier(iterations = 100,
                           depth=5,
                           learning_rate=0.001,
                           loss_function = 'Logloss',
                           verbose=True)

model.fit(X_train, y_train, verbose=False)

predictions = model.predict(X_test)

acc = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", acc)
print("Recall:", recall)
print("Precision:", precision)

list_model.append("CatBoost Classifier")
list_acc.append(acc)
list_recall.append(recall)
list_precision.append(precision)
list_f1.append(f1)

# Support vector machine
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_classifier = SVC(kernel='rbf', gamma='auto')
svm_classifier.fit(X_train_scaled, y_train)

predictions = svm_classifier.predict(X_test_scaled)

acc = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", acc)
print("Recall:", recall)
print("Precision:", precision)

list_model.append("Support Vector Machine")
list_acc.append(acc)
list_recall.append(recall)
list_precision.append(precision)
list_f1.append(f1)

## Random Forest
random_forest_classifier = RandomForestClassifier(n_estimators = 1000,
                                                  max_depth = 10, 
                                                  random_state = 782,
                                                  class_weight = None)
random_forest_classifier.fit(X_train, y_train)
predictions = random_forest_classifier.predict(X_test)

acc = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", acc)
print("Recall:", recall)
print("Precision:", precision)

list_model.append("Random Forest Classifier")
list_acc.append(acc)
list_recall.append(recall)
list_precision.append(precision)
list_f1.append(f1)


## XGBoost
dtrain = xgb.DMatrix(X_train, label = y_train)
dtest = xgb.DMatrix(X_test, label = y_test)

params = {
    'max_depth': 3,
    'eta': 0.01,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'  # You can also use 'auc' for area under the curve
}

num_rounds = 100
watchlist = [(dtest, 'test'), (dtrain, 'train')]
bst = xgb.train(params, dtrain, num_rounds, watchlist)

preds = bst.predict(dtest)
predictions = [1 if i > 0.5 else 0 for i in preds]

acc = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", acc)
print("Recall:", recall)
print("Precision:", precision)

list_model.append("XGBoost")
list_acc.append(acc)
list_recall.append(recall)
list_precision.append(precision)
list_f1.append(f1)

## Save results from Prediction using Similarity with Glove Embeddings Weighted methodology
df_results_pca_weights = pd.DataFrame([list_model, 
                           list_acc, 
                           list_recall, 
                           list_precision,
                           list_f1]).T

df_results_pca_weights.columns = ['Model', 'Accuracy', 
                                  'Recall', 'Precision', 'F1 Score']

#############################################################################
### Prediction using Mean Mean Text Embeddings with BERT methodology 
list_model = []
list_acc = []
list_recall = []
list_precision = []
list_f1 = []

df = pd.read_csv('Data/df_prediction/df_prediction.txt')

path = "Data/text_indicators/PABLO_MEANMEAN_FEATURES.csv"
df_sim = pd.read_csv(path)

cols = ['bill_id',
       'mean_fox_meanmean', 'median_fox_meanmean', 'standard_dev_fox_meanmean',
       '25_percentiles_fox_meanmean', '75_percentiles_fox_meanmean',
       '90_percentiles_fox_meanmean', '95_percentiles_fox_meanmean',
       '99_percentiles_fox_meanmean', 'max_fox_meanmean', 'min_fox_meanmean',
       'mean_breitbart_meanmean', 'median_breitbart_meanmean',
       'standard_dev_breitbart_meanmean', '25_percentiles_breitbart_meanmean',
       '75_percentiles_breitbart_meanmean',
       '90_percentiles_breitbart_meanmean',
       '95_percentiles_breitbart_meanmean',
       '99_percentiles_breitbart_meanmean', 'max_breitbart_meanmean',
       'min_breitbart_meanmean', 'mean_nytimes_meanmean',
       'median_nytimes_meanmean', 'standard_dev_nytimes_meanmean',
       '25_percentiles_nytimes_meanmean', '75_percentiles_nytimes_meanmean',
       '90_percentiles_nytimes_meanmean', '95_percentiles_nytimes_meanmean',
       '99_percentiles_nytimes_meanmean', 'max_nytimes_meanmean',
       'min_nytimes_meanmean', 'mean_wapo_meanmean', 'median_wapo_meanmean',
       'standard_dev_wapo_meanmean', '25_percentiles_wapo_meanmean',
       '75_percentiles_wapo_meanmean', '90_percentiles_wapo_meanmean',
       '95_percentiles_wapo_meanmean', '99_percentiles_wapo_meanmean',
       'max_wapo_meanmean', 'min_wapo_meanmean']

df_sim = df_sim.loc[:, cols]

df = df.merge(df_sim, how = 'left', on = 'bill_id')

cols_keep = ['TTL_RECEIPTS', 'TTL_DISB', 'TTL_INDIV_CONTRIB', 
             'OTHER_POL_CMTE_CONTRIB', 'INDIV_REFUNDS', 'cosp_dem', 'cosp_rep', 
             'target'] +  cols[1:]

df = df.loc[:, cols_keep]

X = df.drop(columns = ['target'])

y = df.loc[:, 'target']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2, 
                                                    random_state = 9384)
# Oversampling 
ros = RandomOverSampler(random_state = 389)
X_train, y_train = ros.fit_resample(X_train, y_train)
y_train.value_counts()


# Logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

acc = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", acc)
print("Recall:", recall)
print("Precision:", precision)
# print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
# print("Classification Report:\n", classification_report(y_test, predictions))

list_model.append("Logistic Regression")
list_acc.append(acc)
list_recall.append(recall)
list_precision.append(precision)
list_f1.append(f1)

# Catboost
model = CatBoostClassifier(iterations = 100,
                           depth=5,
                           learning_rate=0.001,
                           loss_function = 'Logloss',
                           verbose=True)

model.fit(X_train, y_train, verbose=False)

predictions = model.predict(X_test)

acc = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", acc)
print("Recall:", recall)
print("Precision:", precision)

list_model.append("CatBoost Classifier")
list_acc.append(acc)
list_recall.append(recall)
list_precision.append(precision)
list_f1.append(f1)

# Support vector machine
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_classifier = SVC(kernel='rbf', gamma='auto')
svm_classifier.fit(X_train_scaled, y_train)

predictions = svm_classifier.predict(X_test_scaled)

acc = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", acc)
print("Recall:", recall)
print("Precision:", precision)

list_model.append("Support Vector Machine")
list_acc.append(acc)
list_recall.append(recall)
list_precision.append(precision)
list_f1.append(f1)

## Random Forest
random_forest_classifier = RandomForestClassifier(n_estimators = 1000,
                                                  max_depth = 10, 
                                                  random_state = 782,
                                                  class_weight = None)
random_forest_classifier.fit(X_train, y_train)
predictions = random_forest_classifier.predict(X_test)

acc = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", acc)
print("Recall:", recall)
print("Precision:", precision)

list_model.append("Random Forest Classifier")
list_acc.append(acc)
list_recall.append(recall)
list_precision.append(precision)
list_f1.append(f1)


## XGBoost
dtrain = xgb.DMatrix(X_train, label = y_train)
dtest = xgb.DMatrix(X_test, label = y_test)

params = {
    'max_depth': 4,
    'eta': 0.01,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'  # You can also use 'auc' for area under the curve
}

num_rounds = 100
watchlist = [(dtest, 'test'), (dtrain, 'train')]
bst = xgb.train(params, dtrain, num_rounds, watchlist)

preds = bst.predict(dtest)
predictions = [1 if i > 0.5 else 0 for i in preds]

acc = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", acc)
print("Recall:", recall)
print("Precision:", precision)
print("F1:", f1)


list_model.append("XGBoost")
list_acc.append(acc)
list_recall.append(recall)
list_precision.append(precision)
list_f1.append(f1)

## Save results from Mean Mean Text Embeddings with BERT methodology  

df_results_vectors_meanmean = pd.DataFrame([list_model, 
                           list_acc, 
                           list_recall, 
                           list_precision,
                           list_f1]).T

df_results_vectors_meanmean.columns = ['Model', 'Accuracy', 
                                  'Recall', 'Precision', 'F1 Score']

###################################################################
### Prediction using Mean Max Text Embeddings with BERT methodology 
list_model = []
list_acc = []
list_recall = []
list_precision = []
list_f1 = []

df = pd.read_csv('Data/df_prediction/df_prediction.txt')

path = "Data/text_indicators/PABLO_MEANMAX_FEATURES.csv"
df_sim = pd.read_csv(path)

cols = ['bill_id', 'mean_fox_meanmax', 'median_fox_meanmax', 'standard_dev_fox_meanmax',
       '25_percentiles_fox_meanmax', '75_percentiles_fox_meanmax',
       '90_percentiles_fox_meanmax', '95_percentiles_fox_meanmax',
       '99_percentiles_fox_meanmax', 'max_fox_meanmax', 'min_fox_meanmax',
       'mean_breitbart_meanmax', 'median_breitbart_meanmax',
       'standard_dev_breitbart_meanmax', '25_percentiles_breitbart_meanmax',
       '75_percentiles_breitbart_meanmax', '90_percentiles_breitbart_meanmax',
       '95_percentiles_breitbart_meanmax', '99_percentiles_breitbart_meanmax',
       'max_breitbart_meanmax', 'min_breitbart_meanmax',
       'mean_nytimes_meanmax', 'median_nytimes_meanmax',
       'standard_dev_nytimes_meanmax', '25_percentiles_nytimes_meanmax',
       '75_percentiles_nytimes_meanmax', '90_percentiles_nytimes_meanmax',
       '95_percentiles_nytimes_meanmax', '99_percentiles_nytimes_meanmax',
       'max_nytimes_meanmax', 'min_nytimes_meanmax', 'mean_wapo_meanmax',
       'median_wapo_meanmax', 'standard_dev_wapo_meanmax',
       '25_percentiles_wapo_meanmax', '75_percentiles_wapo_meanmax',
       '90_percentiles_wapo_meanmax', '95_percentiles_wapo_meanmax',
       '99_percentiles_wapo_meanmax', 'max_wapo_meanmax', 'min_wapo_meanmax']

df_sim = df_sim.loc[:, cols]

df = df.merge(df_sim, how = 'left', on = 'bill_id')

cols_keep = ['TTL_RECEIPTS', 'TTL_DISB', 'TTL_INDIV_CONTRIB', 
             'OTHER_POL_CMTE_CONTRIB', 'INDIV_REFUNDS', 'cosp_dem', 'cosp_rep', 
             'target'] +  cols[1:]

df = df.loc[:, cols_keep]

X = df.drop(columns = ['target'])

y = df.loc[:, 'target']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2, 
                                                    random_state = 9384)
# Oversampling 
ros = RandomOverSampler(random_state = 389)
X_train, y_train = ros.fit_resample(X_train, y_train)
y_train.value_counts()


# Logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

acc = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", acc)
print("Recall:", recall)
print("Precision:", precision)
# print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
# print("Classification Report:\n", classification_report(y_test, predictions))

list_model.append("Logistic Regression")
list_acc.append(acc)
list_recall.append(recall)
list_precision.append(precision)
list_f1.append(f1)

y_test.value_counts()

list_model.append("K Nearest Neighbors")
list_acc.append(None)
list_recall.append(None)
list_precision.append(None)
list_f1.append(None)

# Catboost
model = CatBoostClassifier(iterations = 100,
                           depth=5,
                           learning_rate=0.001,
                           loss_function = 'Logloss',
                           verbose=True)

model.fit(X_train, y_train, verbose=False)

predictions = model.predict(X_test)

acc = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", acc)
print("Recall:", recall)
print("Precision:", precision)

list_model.append("CatBoost Classifier")
list_acc.append(acc)
list_recall.append(recall)
list_precision.append(precision)
list_f1.append(f1)

# Support vector machine
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_classifier = SVC(kernel='rbf', gamma='auto')
svm_classifier.fit(X_train_scaled, y_train)

predictions = svm_classifier.predict(X_test_scaled)

acc = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", acc)
print("Recall:", recall)
print("Precision:", precision)

list_model.append("Support Vector Machine")
list_acc.append(acc)
list_recall.append(recall)
list_precision.append(precision)
list_f1.append(f1)

## Random Forest
random_forest_classifier = RandomForestClassifier(n_estimators = 1000,
                                                  max_depth = 10, 
                                                  random_state = 782,
                                                  class_weight = None)
random_forest_classifier.fit(X_train, y_train)
predictions = random_forest_classifier.predict(X_test)

acc = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", acc)
print("Recall:", recall)
print("Precision:", precision)

list_model.append("Random Forest Classifier")
list_acc.append(acc)
list_recall.append(recall)
list_precision.append(precision)
list_f1.append(f1)


## XGBoost
dtrain = xgb.DMatrix(X_train, label = y_train)
dtest = xgb.DMatrix(X_test, label = y_test)

params = {
    'max_depth': 4,
    'eta': 0.01,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'  # You can also use 'auc' for area under the curve
}

num_rounds = 100
watchlist = [(dtest, 'test'), (dtrain, 'train')]
bst = xgb.train(params, dtrain, num_rounds, watchlist)

preds = bst.predict(dtest)
predictions = [1 if i > 0.5 else 0 for i in preds]

acc = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", acc)
print("Recall:", recall)
print("Precision:", precision)
print("F1:", f1)


list_model.append("XGBoost")
list_acc.append(acc)
list_recall.append(recall)
list_precision.append(precision)
list_f1.append(f1)

## Save results from Mean Max Text Embeddings with BERT methodology 
df_results_vectors_meanmax = pd.DataFrame([list_model, 
                           list_acc, 
                           list_recall, 
                           list_precision,
                           list_f1]).T

df_results_vectors_meanmax.columns = ['Model', 'Accuracy', 
                                  'Recall', 'Precision', 'F1 Score']

### Save all results to Excel files
df_results.to_excel("Results/benchmark.xlsx", index = False)
df_results_bow.to_excel("Results/similarity_bow.xlsx", index = False)
df_results_pca_weights.to_excel("Results/pca_weights.xlsx", index = False)
df_results_pca_noweights.to_excel("Results/pca_noweights.xlsx", index = False)
df_results_vectors_meanmax.to_excel("Results/bert_meanmax.xlsx", index = False)
df_results_vectors_meanmean.to_excel("Results/bert_meanmean.xlsx", index = False)

