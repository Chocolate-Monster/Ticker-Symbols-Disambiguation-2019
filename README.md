# Ticker Symbols
This is a project for word sense disambiguation.

### Task definition
The task is to determine for each candidate word whether it is a ticker or not. List of candidates is given.

### Solution
1) Data extraction: non-ticker entries were obtained from twitter.com and ticker entries were obtained from stocktwits.com - about 3 thousands for each ticker candidate (see `research/1_data_retrieval.ipynb`)
Number of examples for each ticker:

![tickers](https://user-images.githubusercontent.com/82182857/121756887-5a596000-cb24-11eb-9bda-db7c146f3e67.jpg)

2) Data analysis and feature generation, such as entities, emojis, positions in text etc. (see `research/2_data_analysis_and_feature_generation.ipynb`) 
Classes are balanced: 

![target](https://user-images.githubusercontent.com/82182857/121756917-7826c500-cb24-11eb-9ac3-58e267ae7967.jpg)

3) Tf-idf + XGBoost classifier (see `research/3_XGBoost.ipynb`)

4) RNN classifier (see `4_RNN.ipynb`)

### Results

The final model is the XGBoost classifier + 1450D tf-idf vectors, because it's fast, interpretable and stable.

This model achieves following scores on my test data:

* ROC AUC = `0.99`
* precision = `0.95`
* recall = `0.92`
