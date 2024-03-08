Written and executed by Pablo Montenegro Helfer

# 1. data_prediction.py  

The file data_prediction.py contains the code to train and predict different machine learning models using structured legislative and financial data from Pro Publica and the Federal Election Commission, as well as the text vectors created with different methodologies, from articles of media outlets. 

The dataset without the text-related features is the benchmark. From this benchmark we compare 
- Benchmark: only Pro Publica and Federal Election Comission Data

The text-related features, extracted from the media outlets, were processed using the following methodologies, explained in detail in other sections:
- Similarity with BOW (from topic modeling and sentiment analysis)
- Similarity with Glove Embeddings Unweighted (from topic modeling and sentiment analysis)
- Similarity with Glove Embeddings Weighted (from topic modeling and sentiment analysis)
- Mean Max Text Embeddings with BERT
- MeanMean Text Embeddings with BERT

Each of these features is added, in turn, to the benchmark dataset to predict the passage of legislative bills in the House of Representatives. 

The machine learning models used for the binary prediction are the following:
- Logistic Regression
- CatBoost Classifier
- Support Vector Machine
- Random Forest Classifier
- XGBoost

Each of these prediction models are used on each dataset described above. Results are saved to Excel files. 

# 2. data_prediction_best.py
The file data_prediction_best.py contains code to predict the target variable by using the Support Vector Machine model, the one with best results on the data_prediction.py file, on the features of two datasets:
- Benchmark dataset (no text-related features)
- Dataset with text features calculated using the MeanMean Text Embeddings with BERT methodoloy, which had the best results on the data_prediction.py file. 

The difference from the data_prediction.py models is that the train-test split is done 100 times on each of these two datasets to check for consistency in the results and rule out the possibility of having good results based on chance. Density plots with means of the F1, Recall and Precision scores are created. Results show that all scores are better on the dataset and features that contain the text-related information processed with the MeanMean Text Embeddings with BERT methodoloy. 

