# Sentiment Analysis 

#### Written and coded by Robert McCormick and Robert Surridge

## Process

1. Run each notebook separately to calculate sentiment on conservative and liberal media outlets.

1. **Data Processing:**
   1. Import necessary libraries and load data from CSV files for Fox, Breitbart, NYT, Washington Post, and CNN.
   2. Keep only relevant columns ('text', 'title', 'url', 'date').
   3. Format date columns to a consistent '%Y-%m-%d' format.

3. **Sentiment Analysis:**
   1. Set up sentiment analysis using a pre-trained model from the Transformers library.
   2. Perform sentiment analysis on text data in chunks.
   3. Apply sentiment analysis to each dataframe, add a 'Positive' column, and export the data to CSV files.
  
## Rationale

The outlined process ensures a systematic and standardized approach to sentiment analysis across diverse news sources. By preprocessing the data and leveraging pre-trained models, the analysis becomes more robust and adaptable. The inclusion of date formatting and handling variations in data sources enhances the reliability and comparability of results. Additionally, the demonstration and summarization steps provide transparency and insights into the sentiment landscape, facilitating meaningful interpretations of the analyzed data.
