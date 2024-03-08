# Text Embeddings

### Written by John Christenson
- *Note: for lines of code, created duplicate jupyter notebooks, deleted print statements, uploaded to GitHub for lines of code, and then deleted them.  Many of the juypter notebooks below still possess print statements*


## master_key_df.ipynb
- 1225 lines of code


## text_embeddings.ipynb
- 1197 lines of code
  



## text_embeddings_cosinesimilarity.ipynb
- 1525 lines of code



  
## text_feature_creation.ipynb
- 527 lines of code

This juypter notebook reads csv files containing the cosine similarity scores where the rows are bills and the columns are the cosine similarity scores within the 5 month window of the piece of legislation into pandas dataframes.  This process occurs individually by both news source and text embedding process: mean then mean or mean then max.  We then calculate the mean, median, max, min, standard deviation, and percentiles (99th, 95th, 90th, 75th, and 25th) for the bill by news source and then merge the new data together with the dataframe, clean_df, which contains variables to be reintroduced so the created features may merge into the training/testing dataframe, or variables which allow for further data exploration.  Finally we merge data by the pooling process used on the embeddings and save the new merged mean-mean and mean-max dataframes as csv files.
