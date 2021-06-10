import pandas as pd
import numpy as np
import pickle
import sys
from preprocessing import Preprocessing


if len(sys.argv) > 1:
    input_filename = sys.argv[1]
else:
    print('Expected input file name')
    sys.exit(1)
    
print('Loading data for prediction')
with open(input_filename, 'r') as f:
    df = pd.DataFrame({'text': [t.strip() for t in f.readlines()]})
    
df_result = df.copy()

print('Loading ticker list')
with open('../tickers.txt', 'r') as f:
    ticker_list = [t.strip() for t in f.readlines()]
    
preproc = Preprocessing(ticker_list)
df_features, ticker_found_mask = preproc.evaluate(df)

model_name = '../models/model_d9.pkl'
tfidf_name = '../models/tfidf_d9.pkl'

print('Loading serialized models')
with open(model_name, 'rb') as f:
    model = pickle.load(f)
    
with open(tfidf_name, 'rb') as f:
    tfidf = pickle.load(f)

features_cols = [
    'DATE_ENT_distance',
    'MONEY_ENT_distance',
    'NUMBER_ENT_distance',
    'PERCENT_ENT_distance',
    'PERSON_ENT_distance',
    'PICTURE_ENT_distance',
    'URL_ENT_distance',
    'has_!',
    'has_?',
    'has_emoji',
    'has_special_emoji',
    'relative_pos',
    'word_count',
    'word_max',
    'word_mean',
    'word_median',
    'doc'
]

def tdidf_xgb_predict(X_val, tfidf, model):
    """ transforms doc into tfidf features and predicts the label
        input:
            * X_val
            * tfidf - fitted tfidf model
            * model - fitted xgboost model
        output:
            * p_val (probabilities)  
            * p_class (labels) 
    """
    X = X_val.copy(deep=True)
    
    tfidf_feats = tfidf.transform(X['doc'].astype(str).values).todense()
    X.drop(['doc'], axis=1, inplace=True)
    tfidf_feats_df = pd.DataFrame(
        tfidf_feats, 
        columns=['tf_idf_{}'.format(i) for i in range(tfidf_feats.shape[1])]
    )

    X = pd.concat(
        [
            X.reset_index(drop=True), 
            tfidf_feats_df.reset_index(drop=True)
        ], 
        axis=1
    )

    return model.predict_proba(X.astype(np.float32))[:, 1]

print('Making predictions')
p_val = tdidf_xgb_predict(df_features[features_cols], tfidf, model)

df_result.loc[ticker_found_mask, 'proba'] = p_val
df_result.loc[~ticker_found_mask, 'proba'] = 0

df_result.loc[ticker_found_mask, 'is_ticker'] = (p_val > 0.5).astype(np.int8)
df_result.loc[~ticker_found_mask, 'is_ticker'] = 0

print('Saving results to "../output/result.csv"')
df_result.to_csv('../output/result.csv')
