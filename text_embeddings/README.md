# Text Embeddings

### Written and coded by John Christenson

## master_key_df.ipynb
Creates a master key dataframe, that uses the text_embeddings global index for bills, 'new_index', to create a monthly index for bills and news sources.  This monthly index occurs at the Congress and individual news source level, meaning there is a separate column for the bills as a whole and each news source by month.  This dataframe allows one to utilize the text embeddings to create cosine similarity scores by bill and news source by the 5 month window size.  Utilizes 1224 lines of code.


## text_embeddings.ipynb
Utilizes 1197 lines of code
  



## text_embeddings_cosinesimilarity.ipynb
Utilizes 1525 lines of code



  
## text_feature_creation.ipynb
This juypter notebook reads csv files containing the cosine similarity scores where the rows are bills and the columns are the cosine similarity scores within the 5 month window of the piece of legislation into pandas dataframes.  This process occurs individually by both news source and text embedding process: mean then mean or mean then max.  We then calculate the mean, median, max, min, standard deviation, and percentiles (99th, 95th, 90th, 75th, and 25th) for the bill by news source and then merge the new data together with the dataframe, clean_df, which contains variables to be reintroduced so the created features may merge into the training/testing dataframe, or variables which allow for further data exploration.  Finally we merge data by the pooling process used on the embeddings and save the new merged mean-mean and mean-max dataframes as csv files.  Utilizes 527 lines of code.


- *Note: for lines of code, created duplicate jupyter notebooks, deleted print statements, uploaded to GitHub for lines of code, and then deleted them.  Many of the juypter notebooks below still possess print statements and their outputs.  ~471 lines of code in master_key_df.ipynb, text_embeddings.ipynb, and text_embeddings_cosinesimilarity.ipynb is the BERT_Data.class.*
