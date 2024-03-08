# Bill-Media Similarity Assessment Process

#### Written and coded by Robert McCormick and Robert Surridge

## TF-IDF with GloVe Embeddings

### Overview

The notebook aims to create a scoring system between media topics and the contents of bills. To achieve this, the process involves using TF-IDF (Term Frequency-Inverse Document Frequency) for understanding the subject of the bill. The TF-IDF word frequency is then used to calculate word frequencies, and the top 10 words are extracted to create a word vector matching the number of words in a vector from media topics.

The bill keywords and the top 10 words for a given topic are converted into GloVe embeddings (Global Vectors for Word Representation) using the GloVe 6B model. Cosine similarity is calculated for each word embedding in the bill keywords vector against every word embedding in the topic. The maximum similarity for each word in the bill is chosen, and the average of these top similarity scores is calculated to represent the overall similarity of the bill with a specific topic. Weights are applied to each word from the news topics in the hope of capturing the importance of each word across all topics.

The analysis is limited to the top 50 topics from over 750 topics generated through topic modeling. Three datasets are created for evaluation: one with 100 columns of the top 50 similarity scores and 50 sentiment scores, the next with the top 20 similarity scores and 20 sentiment scores, and the last one with PCA analysis applied to the data from 100 columns.

### Process

1. **TF-IDF Process for Bill Keywords:** TF-IDF is applied to the bill text to obtain the top 10 words that represent the content of the bill.

2. **Word Embeddings and Similarity Scores:**
   - The top words from bills and media topics are converted into GloVe embeddings.
   - Cosine similarity is calculated between each word embedding in bill keywords and media topics.
   - The maximum similarity for each word in the bill is selected, and the average is taken to represent the overall similarity.

3. **Topic Modeling and Data Preparation:**
   - Over 750 topics are generated, but the analysis is focused on the top 50 topics.
   - Three datasets are created with different column combinations, and PCA is applied to reduce dimensionality.

4. **Scoring and Evaluation:**
   - Scores are applied to bills for each topic, considering both similarity and sentiment.
   - The final datasets are created for evaluation, including one with weighted scores based on word importance.

### Rationale

The approach employs TF-IDF for a nuanced understanding of individual bill subjects, leveraging its effectiveness in single-document analysis. GloVe embeddings are utilized to capture semantic nuances in word representations, while cosine similarity measures the closeness between word vectors. To manage computational complexity, the analysis focuses on the top 50 topics. Principal Component Analysis (PCA) is applied to address high-dimensionality issues and distill crucial information. Additionally, the incorporation of weighted words, reflecting their significance, enhances the model's capacity to discern pivotal terms. The scoring system, designed for a holistic evaluation, encompasses content similarity and sentiment analysis. This multi-step process involves data preprocessing, embedding creation, and dimensionality reduction to facilitate a comprehensive assessment of the relationship between bills and media topics.

## TF-IDF with Bag of Words

### Overview
The TF-IDF (Term Frequency-Inverse Document Frequency) Bag of Words method is employed to calculate similarity scores between legislative bills and monthly topics. This approach proves computationally efficient, offering results in less than a minute, as opposed to other similarity methods like TF-IDF Cosine Similarity (20 minutes) and Text Embeddings Cosine Similarity (3-5 hours). However, it faces a weakness, as the approach tends to yield relatively low scores, with a maximum observed score of 0.026, potentially impacting its significance when integrated into the final classification model.

### Process

1. **Keyword Extraction:**
   - Utilize TF-IDF to extract the top 10 keywords for each bill, considering custom stop words.
   - Implement the `get_top_words_bills` function to achieve this extraction.

2. **Similarity Calculation:**
   - Map bills to topic lists and calculate similarity scores.
   - Multiply keywords by topic weights based on media frequency.
   - Divide by the number of keywords (always 10) to normalize the scores.

3. **Integration with Monthly Topics:**
   - Merge the bill data with monthly topic information, considering the introduced date.
   - Calculate the similarity score for each bill based on keywords and weights.
   - Normalize the similarity scores to a range of 0 to 1.
   - Incorporate average media sentiment scores for each monthly topic.

4. **Results and Analysis:**
   - Examine the unique similarity scores to understand the distribution.
   - Provide statistics such as the number of bills with some similarity, minimum similarity, and maximum similarity.

### Rationale
The TF-IDF Bag of Words method is chosen for its computational efficiency, making it suitable for large datasets. However, the observed weakness in generating relatively low similarity scores necessitates caution when using it as a parameter in the final classification model. The approach ensures a balance between speed and accuracy, providing a foundation for further model refinement and analysis. The normalized scores and integration with media sentiment contribute to a comprehensive understanding of the bills in the context of monthly topics.
