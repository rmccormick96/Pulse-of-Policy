## Introduction
This Jupyter Notebook is designed to apply advanced machine learning techniques for topic modeling on a dataset of news articles. Through a series of defined functions and methodologies, it explores the extraction of meaningful topics from a large corpus, evaluates the sentiments associated with these topics, and visualizes the relationships between them.

### 1. Data Preprocessing
The notebook begins with preprocessing the textual data to prepare it for topic modeling and sentiment analysis.

*Tokenization and Cleaning:* Utilizing the sent_to_words function, the notebook tokenizes the text into words, removing punctuation and unnecessary characters to clean the data.
*Stopwords Removal:* Two functions, remove_stopwords and remove_stopwords_2, further process the text by eliminating stopwords. The former returns a single string per document, while the latter maintains the list structure of words.
### 2. Topic Modeling with BERTopic
The core of the notebook revolves around applying the BERTopic model to discover latent topics within the news articles.

*Function `datasets_topics`:* This function is a wrapper around the BERTopic model, facilitating the topic modeling process. It takes the preprocessed text, fits the BERTopic model to extract topics, and annotates each document in the dataset with its corresponding topic ID and key terms representing the topic.

### 3. Data Trimming and merge of Sentiment Analysis 
To focus on relevant articles and prepare the dataset for merging the sentiment analysis, the notebook employs data trimming.

*Function `trimDf_sentiment`:* Adjusts the DataFrame based on the news source, selects specific columns, and preprocesses the text for sentiment analysis. It ensures that the dataset is consistent across different sources and timeframes suitable for further analysis.

### 4. Extracting Words and Weights
Understanding the significance of words in topics is crucial for interpreting the results.

*Function `extract_words_and_weights`:* Extracts words and their corresponding weights from the BERTopic output, aiding in the interpretation of each topic's composition and relevance.

### 5. Visualization and Analysis
While the specific visualization and analysis steps are not detailed in the functions provided, typical workflows include generating intertopic distance maps, calculating coherence scores to evaluate topic quality, and applying sentiment analysis to gauge the emotional tone of articles related to each topic.
