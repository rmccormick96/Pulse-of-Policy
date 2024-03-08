
##########################################################
### Run SVM models with MeanMean 
### RUn on benchmark and Text Embeddings with BERT methodology
### The SVM model on the Text Embeddings with BERT methodology had the best results
### 100 train test splits are used to have robust results
### Density graphs are created fro F1, Recall and Precision scores
### Means scores are also calculates

import pandas as pd
import statistics
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC

from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

df = pd.read_csv('Data/df_prediction/df_prediction.txt')

######################################################
### Mean Squared Text Embeddings with BERT methodology 

# Prepare data
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

# Run training and testing
list_acc = []
list_recall = []
list_precision = []
list_f1 = []

for rand_st in range(9384, 9384 + 100):

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = 0.2, 
                                                        random_state = rand_st)
    # Oversampling 
    ros = RandomOverSampler(random_state = rand_st)
    X_train, y_train = ros.fit_resample(X_train, y_train)


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
    
    list_acc.append(acc)
    list_recall.append(recall)
    list_precision.append(precision)
    list_f1.append(f1)

mean_acc = statistics.mean(list_acc)
var_acc = statistics.variance(list_acc)

mean_recall = statistics.mean(list_recall)
var_recall = statistics.variance(list_recall)

mean_precision = statistics.mean(list_precision)
var_precision = statistics.variance(list_precision)

mean_f1 = statistics.mean(list_f1)
var_f1 = statistics.variance(list_f1)

# Plot results
fig, ax = plt.subplots(figsize = (15, 7))

sns.kdeplot(list_f1, fill = False, color = "orange", alpha = 1, 
            label = 'F1 Score')
plt.axvline(x = mean_f1, color='orange', linestyle = '--', 
            label = 'F1 Mean')

sns.kdeplot(list_recall, fill = False, color = "red", alpha = 1, 
            label='Recall Score')
plt.axvline(x = mean_recall, color = 'red', linestyle='--', 
            label = 'Recall Mean')

sns.kdeplot(list_precision, fill = False, color = "blue", alpha = 1, 
            label='Precision Score')
plt.axvline(x = mean_precision, color = 'blue', linestyle = '--', 
            label = 'Recall Mean')

plt.title('Density Plot - SVM Results With Text Vectors')
plt.xlabel('Value')
plt.ylabel('Density')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.legend()

plt.show()

######################################################
### Benchmark 

# Prepare data
df = pd.read_csv('Data/df_prediction/df_prediction.txt')

cols_drop = ['TRANS_TO_AUTH', 'CAND_CONTRIB', 'CAND_LOANS', 'OTHER_LOANS', 
              'CAND_LOAN_REPAY', 'OTHER_LOAN_REPAY', 'DEBTS_OWED_BY']

cols_keep = ['TTL_RECEIPTS', 'TTL_DISB', 'TTL_INDIV_CONTRIB', 
             'OTHER_POL_CMTE_CONTRIB', 'INDIV_REFUNDS', 'cosp_dem', 'cosp_rep', 
             'cosp_ratio', 'target']

df = df.loc[:, cols_keep]

# X = df.drop(columns = ['introduced_date', 'target', 'OTHER_LOANS'])
X = df.drop(columns = ['target'])

y = df.loc[:, 'target']

# Run training and testing
list_acc = []
list_recall = []
list_precision = []
list_f1 = []

for rand_st in range(9384, 9384 + 100):

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = 0.2, 
                                                        random_state = rand_st)
    # Oversampling 
    ros = RandomOverSampler(random_state = rand_st)
    X_train, y_train = ros.fit_resample(X_train, y_train)


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
    
    list_acc.append(acc)
    list_recall.append(recall)
    list_precision.append(precision)
    list_f1.append(f1)

mean_acc = statistics.mean(list_acc)
var_acc = statistics.variance(list_acc)

mean_recall = statistics.mean(list_recall)
var_recall = statistics.variance(list_recall)

mean_precision = statistics.mean(list_precision)
var_precision = statistics.variance(list_precision)

mean_f1 = statistics.mean(list_f1)
var_f1 = statistics.variance(list_f1)

# Plot results
fig, ax = plt.subplots(figsize = (15, 7))

sns.kdeplot(list_f1, fill = False, color = "orange", alpha = 1, 
            label = 'F1 Score')
plt.axvline(x = mean_f1, color='orange', linestyle = '--', 
            label = 'F1 Mean')

sns.kdeplot(list_recall, fill = False, color = "red", alpha = 1, 
            label='Recall Score')
plt.axvline(x = mean_recall, color = 'red', linestyle='--', 
            label = 'Recall Mean')

sns.kdeplot(list_precision, fill = False, color = "blue", alpha = 1, 
            label='Precision Score')
plt.axvline(x = mean_precision, color = 'blue', linestyle = '--', 
            label = 'Recall Mean')

plt.title('Density Plot - SVM Results')
plt.xlabel('Value')
plt.ylabel('Density')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.legend()

plt.show()
