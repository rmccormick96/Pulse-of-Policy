## TF-IDF with GloVe Embeddings

### Overview

### Process

### Rationale

## TF-IDF with Bag of Words

### Overview
The TF-IDF (Term Frequency-Inverse Document Frequency) Bag of Words method is employed to calculate similarity scores between legislative bills and monthly topics. This approach proves computationally efficient, offering results in less than a minute, as opposed to other similarity methods like TF-IDF Cosine Similarity (20 minutes) and Text Embeddings Cosine Similarity (3-5 hours). However, it faces a weakness, as the approach tends to yield relatively low scores, with a maximum observed score of 0.026, potentially impacting its significance when integrated into the final classification model.

### Process

1. **Data Loading and Preprocessing:**
   - Load the bill data from a CSV file (`115th_clean.csv`).
   - Prepare the data, including cleaning text and converting dates.

2. **Keyword Extraction:**
   - Utilize TF-IDF to extract the top 10 keywords for each bill, considering custom stop words.
   - Implement the `get_top_words_bills` function to achieve this extraction.

3. **Topic Summarization:**
   - Summarize 50 monthly topics using aggregated lists from the past five months.
   - Identify the top 10 keywords for each bill using TF-IDF.

4. **Similarity Calculation:**
   - Map bills to topic lists and calculate similarity scores.
   - Multiply keywords by topic weights based on media frequency.
   - Divide by the number of keywords (always 10) to normalize the scores.

5. **Integration with Monthly Topics:**
   - Merge the bill data with monthly topic information, considering the introduced date.
   - Calculate the similarity score for each bill based on keywords and weights.

6. **Score Normalization and Sentiment Integration:**
   - Normalize the similarity scores to a range of 0 to 1.
   - Incorporate average media sentiment scores for each monthly topic.

7. **Results and Analysis:**
   - Examine the unique similarity scores to understand the distribution.
   - Provide statistics such as the number of bills with some similarity, minimum similarity, and maximum similarity.

8. **Output:**
   - Save the processed bill data with similarity scores to a CSV file (`bills_with_bow_similarity.csv`).

### Rationale
The TF-IDF Bag of Words method is chosen for its computational efficiency, making it suitable for large datasets. However, the observed weakness in generating relatively low similarity scores necessitates caution when using it as a parameter in the final classification model. The approach ensures a balance between speed and accuracy, providing a foundation for further model refinement and analysis. The normalized scores and integration with media sentiment contribute to a comprehensive understanding of the bills in the context of monthly topics.
