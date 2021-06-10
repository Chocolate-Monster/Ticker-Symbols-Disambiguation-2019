# Ticker Symbols

This is a project for word sense disambiguation. 

The aim is to build a classifier which can distinguish ticker and non-ticker context of certain words.

## Project structure

### Evaluation

* `scripts` - contains script for model evaluation
* `models` - serialized models directory
* `output` - results

#### Research

* `1_data_retrieval.ipynb` - notebook which contains ETL code (step 1)
* `2_data_analysis_and_feature_generation.ipynb` - notebook which contains data analysis and feature generation (step 2)
* `3_XGBoost.ipynb` - notebook with XGBoost classifier
* `4_RNN.ipynb` - notebook with RNN classifier

#### Misc

* `twitter` - non-ticker entries obtained at step 1 from twitter.com
* `stocktwits_hope` - ticker entries obtained at step 1 from stocktwits.com
* `vocab.txt` and `docs.bin` - serialized Spacy documents obtained at step 2
* `all_feats_for_analysis.csv` - features generated at step 2
* `requirements.txt` - requirements

## Results

The final model is the XGBoost classifier + 1450D tf-idf vectors, because it's fast, interpretable and stable.

This model achieves following scores on my test data:

* ROC AUC = `0.99`
* precision = `0.95`
* recall = `0.92`

## How to evaluate the model

The serialized model is located at the `models` directory. `predict.py` should be ran from from the `scripts` directory in order to evaluate the serialized model.
This script takes path to the text file as it only argument. Text file should contain test examples one per line.

### Notes on model evaluation

1. Input examples may contain zero or any number of ticker words. In any case the program will return only one answer for one line.
2. If there is no ticker in line, then it will return 0. If there are several tickers in one text, the program will return label for the first ticker found.
3. The output file  s a pandas data frame with columns `[‘text’, ‘proba’, ‘label’]`.
4. The threshold is `0.5`.
