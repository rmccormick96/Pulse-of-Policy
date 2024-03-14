# The Pulse of Policy - Predicting Legislative Success 
## CAPP 30255 Final Project 

## 1. Sentiment Analysis

**Authors:** Robert McCormick and Robert Surridge

Utilizing a pre-trained model from the Transformers library, the sentiment analysis is performed in chunks on the text data. The outcome includes a 'Positive' column added to each dataframe, and the results are exported to CSV files.

**Rationale:** This process ensures a systematic approach to sentiment analysis across diverse news sources. By leveraging pre-trained models and preprocessing steps, the analysis becomes robust, adaptable, and transparent, enhancing reliability and comparability of results.

## 2. Legislation Collection Process

**Authors:** John Christenson and Robert Surridge

Using the ProPublica Congress API, bill data retrieval is implemented, saving it in JSON files for incremental data retrieval and backup. Selenium and Chrome WebDriver are employed to extract raw text from bills on the Congress website. Error handling mechanisms are in place to ensure data integrity. Analyzing the counts of different bill types for each Congress session (115th to 118th) and calculating the ratio of bills with a 'house_passage' field not null.

**Rationale:** This code systematically fetches bill data, complements it with raw text through web scraping, and provides quick insights into the dataset through bill count analysis. Incremental data saving ensures resilience to interruptions.

## 3. Federal Election Commission and Legislative Bill Information Data Retrieval

 **Author:** Pablo Montenegro Helfer

The `data_processing.py` file orchestrates the integration of Federal Election Commission (FEC) data from the 115th Congress (2017-2018) with pre-scraped legislative bill information from ProPublica. The script extracts pertinent FEC details and generates essential variables for a machine learning model. It employs a histogram to assess the consistency of passed bills within the specified timeframe. The final step involves merging FEC and ProPublica dataframes based on text similarity derived from bill sponsors' names.

## 4. Bill-Media Similarity Assessment Process

### a. TF-IDF with GloVe Embeddings

**Author:** Robert McCormick

This notebook aims to establish a scoring system between media topics and bill contents using TF-IDF and GloVe embeddings. Applying TF-IDF to bill text for keyword extraction, converting keywords and media topics into GloVe embeddings, and calculating cosine similarity. The process is limited to the top 50 topics, and datasets are created for evaluation with PCA applied.

**Rationale:** The approach employs TF-IDF for nuanced bill understanding and GloVe embeddings for semantic nuances. It incorporates weighted words and employs PCA for comprehensive evaluation.

### b. TF-IDF with Bag of Words

**Author:** Robert Surridge

Utilizing TF-IDF Bag of Words for computational efficiency in calculating similarity scores. Extracting keywords using TF-IDF, calculating similarity scores, and integrating them with monthly topics. The approach balances speed and accuracy, offering a foundation for further model refinement.

**Rationale:** Chosen for computational efficiency, this method provides a balanced approach between speed and accuracy in generating similarity scores.

## 5. Text Embeddings

**Author:** John Christenson

The notebook `master_key_df.ipynb` creates a master key dataframe, establishing a monthly index for bills and news sources. It facilitates cosine similarity scores by bill and news source within a 5-month window. The notebook `text_embeddings.ipynb` loads ProPublica API data, cleans and tokenizes bill text using BERT-base-uncased. The text embeddings are generated using BERT, allowing for comprehensive analysis. The notebook `text_embeddings_cosinesimilarity.ipynb` utilizes cosine similarity for text embeddings, aiding in the comparison of bills and news sources. The notebook `text_feature_creation.ipynb` reads cosine similarity scores, calculates statistics, and merges data. The final datasets are saved as CSV files.

**Rationale:** Employing BERT for text embeddings, the process considers computational efficiency and statistical insights, contributing to a comprehensive understanding of bill-media relationships. The data is accessible via Google Drive due to large file size.

## 6. Topic modeling

**Author:** Santiago Satizábal

The notebook `Topic_modeling_Pulse_pol` is designed to apply advanced machine learning techniques for topic modeling on a dataset of news articles. Through a series of defined functions and methodologies, it explores the extraction of meaningful topics from a large corpus, evaluates the sentiments associated with these topics, and visualizes the relationships between them. It
provides a comprehensive framework for topic modeling using BERTopic, equipped with preprocessing steps, topic extraction, and preparations for sentiment analysis. 

## 7. Prediction

**Author:** Pablo Montenegro Helfer

In the `data_prediction.py` script trains machine learning models on legislative data, comparing a benchmark set with Pro Publica and FEC data to enriched sets featuring text-related features from media articles. Various methodologies, such as BOW, Glove Embeddings, and BERT, are employed. The script evaluates models like Logistic Regression and XGBoost, saving results in Excel files. The subsequent script, `data_prediction_best.py`, focuses on the Support Vector Machine model, verifying its consistency by conducting 100 train-test splits on benchmark and BERT-enhanced datasets. Consistent superiority is found in the BERT-enriched dataset, emphasizing its robust predictive performance.
