# Sentiment Analysis Process

1. **Data Processing:**
   1. Import necessary libraries and load data from CSV files for NYT, Washington Post, and CNN. Essential for data processing.
   2. Define a function to keep only relevant columns ('text', 'title', 'url', 'date') in the dataframes. Helps streamline data for further analysis.
   3. Format date columns to a consistent '%Y-%m-%d' format for NYT, Washington Post, and CNN dataframes. Ensures uniformity in date representation.
   4. Convert CNN date column to '%Y-%m-%d' format, handling ISO8601 format. Maintains date consistency for all sources.

2. **Sentiment Analysis:**
   1. Set up sentiment analysis using a pre-trained model from the Transformers library. Prepares for sentiment analysis on the text data.
   2. Define a function to perform sentiment analysis on text data in chunks. Enables sentiment analysis on potentially long text entries.
   3. Demonstrate sentiment analysis on sample entries from NYT. Checks the functionality of sentiment analysis.
   4. Apply sentiment analysis to each dataframe, add a 'Positive' column, and export the data to CSV files. Integrates sentiment analysis results into the datasets for further exploration.
   5. Print and display the count of positive and negative sentiments for each dataset. Provides a summary of sentiment analysis outcomes.
