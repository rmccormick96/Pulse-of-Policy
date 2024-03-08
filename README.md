# Pulse-of-Policy
## CAPP 30255 Final Project 

## Part One: Sentiment Analysis

**Authors:** Robert McCormick and Robert Surridge

Utilizing a pre-trained model from the Transformers library, the sentiment analysis is performed in chunks on the text data. The outcome includes a 'Positive' column added to each dataframe, and the results are exported to CSV files.

**Rationale:** This process ensures a systematic approach to sentiment analysis across diverse news sources. By leveraging pre-trained models and preprocessing steps, the analysis becomes robust, adaptable, and transparent, enhancing reliability and comparability of results.

## Legislation Collection Process

**Authors:** John Christenson and Robert Surridge

Using the ProPublica Congress API, bill data retrieval is implemented, saving it in JSON files for incremental data retrieval and backup. Selenium and Chrome WebDriver are employed to extract raw text from bills on the Congress website. Error handling mechanisms are in place to ensure data integrity. Analyzing the counts of different bill types for each Congress session (115th to 118th) and calculating the ratio of bills with a 'house_passage' field not null.

**Rationale:** This code systematically fetches bill data, complements it with raw text through web scraping, and provides quick insights into the dataset through bill count analysis. Incremental data saving ensures resilience to interruptions.

## Bill-Media Similarity Assessment Process

### TF-IDF with GloVe Embeddings

**Author:** Robert McCormick

This notebook aims to establish a scoring system between media topics and bill contents using TF-IDF and GloVe embeddings. Applying TF-IDF to bill text for keyword extraction, converting keywords and media topics into GloVe embeddings, and calculating cosine similarity. The process is limited to the top 50 topics, and datasets are created for evaluation with PCA applied.

**Rationale:** The approach employs TF-IDF for nuanced bill understanding and GloVe embeddings for semantic nuances. It incorporates weighted words and employs PCA for comprehensive evaluation.

### TF-IDF with Bag of Words

**Author:** Robert Surridge

Utilizing TF-IDF Bag of Words for computational efficiency in calculating similarity scores. Extracting keywords using TF-IDF, calculating similarity scores, and integrating them with monthly topics. The approach balances speed and accuracy, offering a foundation for further model refinement.

**Rationale:** Chosen for computational efficiency, this method provides a balanced approach between speed and accuracy in generating similarity scores.

## Text Embeddings

**Author:** John Christenson

The notebook `master_key_df.ipynb` creates a master key dataframe, establishing a monthly index for bills and news sources. It facilitates cosine similarity scores by bill and news source within a 5-month window. The notebook `text_embeddings.ipynb` loads ProPublica API data, cleans and tokenizes bill text using BERT-base-uncased. The text embeddings are generated using BERT, allowing for comprehensive analysis. The notebook `text_embeddings_cosinesimilarity.ipynb` utilizes cosine similarity for text embeddings, aiding in the comparison of bills and news sources. The notebook `text_feature_creation.ipynb` reads cosine similarity scores, calculates statistics, and merges data. The final datasets are saved as CSV files.

**Rationale:** Employing BERT for text embeddings, the process considers computational efficiency and statistical insights, contributing to a comprehensive understanding of bill-media relationships. The data is accessible via Google Drive due to large file size.
