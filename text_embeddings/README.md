# Text Embeddings

### Written and coded by John Christenson

## `master_key_df.ipynb`
Creates a master key dataframe, that uses the text_embeddings global index for bills, 'new_index', to create a monthly index for bills and news sources.  This monthly index occurs at the Congress and individual news source level, meaning there is a separate column for the bills as a whole and each news source by month.  This dataframe allows one to utilize the text embeddings to create cosine similarity scores by bill and news source by the 5 month window size.  This file utilizes 1224 lines of code (approximately 470 lines come from the BERT_Data class initially created in text_embeddings.ipynb).


## `text_embeddings.ipynb`
Loads the ProPublica API data with the raw text of the bills.  This data is then cleaned.  Cleaning involves lower casing, fixing formatting, and removing unnecessary bill-specific lingo that creates noise to allow for a more accurate comparison between the text of the legislation and news sources.  Stopwords are kept as they are important contextual information for BERT.  

Using BERT-base-uncased, the clean text is tokenized and segmented out into segments of tokens with a max length of 510 tokens.  If the tokenized text of a bill or news article is over 510 tokens that is when the text is segmented out into blocks of 510 tokens (the total may be less in the final segment).  If the final or only segment is less than 510 tokens, then padding is added as required. A [CLS] token ID is added to the beginning and a [SEP] token ID is added to the end of each segment.  An attention mask is then applied to any padded token.  

These segments are then input into the BERT model and the text embeddings are then pulled from the last hidden state.  The user may choose a combination of truncation, mean pooling, max pooling, utilizing CLS a single segment and, if necessary, mean pooling, max pooling, or skipping if there are multiple segments if the input segment is longer than 510 tokens post tokenization.  

This embedded tensor is then stacked with the rest of the created tensors representing each bill or article from the input source and returned as a large Pytorch tensor.

This notebook utilizes 1197 lines of code.  
  
  
## `text_embeddings_cosinesimilarity.ipynb`
This juypter notebook calculates the cosine similarity between each legislative bill and every news article by news source within a 5 month window.  Normalization is available not not utilized due to it making the data unuseable: cosine similarity is approximately 0.99.  Instead, no normalization is performed.  Utilizes 1525 lines of code (approximately 470 lines come from the BERT_Data class initially created in text_embeddings.ipynb).



  
## `text_feature_creation.ipynb`
This juypter notebook reads csv files containing the cosine similarity scores where the rows are bills and the columns are the cosine similarity scores within the 5 month window of the piece of legislation into pandas dataframes.  This process occurs individually by both news source and text embedding process: mean then mean or mean then max.  We then calculate the mean, median, max, min, standard deviation, and percentiles (99th, 95th, 90th, 75th, and 25th) for the bill by news source and then merge the new data together with the dataframe, clean_df, which contains variables to be reintroduced so the created features may merge into the training/testing dataframe, or variables which allow for further data exploration.  Finally we merge data by the pooling process used on the embeddings and save the new merged mean-mean and mean-max dataframes as csv files.  The notebook utilizes 527 lines of code.
