# Federal Election Commission and Legislative Bill Information Data Retrieval

#### Author: Pablo Montenegro Helfer

The file `data_processing.py` contains the code that opens the Federal Election Commission data from the 115the Congress (2017-2018) and selects the relevant information. 

The file also opens the already scraped legislative bill information from ProPublica and creates relevant variables for the machine learning model. Additionally, it plots a histogram with the bills that did pass to check for consistency during the timeframe. Finally, it merges the Federal Election Commission and ProPublica dataframes by text similarity from the names of the sponsors of bills. 


