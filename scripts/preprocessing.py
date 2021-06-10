import sys
import re
import html

import pandas as pd
import numpy as np
import pickle
import tqdm

import nltk
import spacy
import emoji

from collections import defaultdict

from spacy.matcher import Matcher
from spacy.tokens import Doc
from spacy.attrs import LOWER, POS, ENT_TYPE, IS_ALPHA


class Preprocessing:
    """this class is based on ticker_list. 
       it has cleaning and features obtaining from dataframe['text'].
    """
    def __init__(self, ticker_list):
        self.url_regex = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 
            re.IGNORECASE
        )
        self.user_regex = re.compile(r'@\w[\w\d_]+')
        self.pic_regex = re.compile(r'pic.twitter(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.punct_regex = re.compile('[' + re.escape(r'"#%&\'()*+,-./:;<=>@[\\]^_`{|}~$') + ']+')
        self.number_regex = re.compile(r'\d+[\d\.\,]*')
        self.bad_name_regex = re.compile(r'[A-Z]{3,}')
        self.bad_name_set = {
            'DATE_ENT', 'NUMBER_ENT', 'URL_ENT', 'PICTURE_ENT', 'PERSON_ENT', 'MONEY_ENT', 'PERCENT_ENT'
        }
        self.tickers_set = set(ticker_list)
        
        self.nlp = spacy.load('en_core_web_sm')

        self.get_entity_matcher()
        self.get_ticker_matcher()
        
        self.special_emoji_regex = re.compile(r'[\🚀\🔥\👀\🤑\🐻\📈\💰\🤔\💲\💵\💸\€\£]+', re.UNICODE)
    
    def replace_url(self, text):
        return re.sub(self.url_regex, ' URL_ENT ', text)

    def replace_user(self, text):
        return re.sub(self.user_regex, ' PERSON_ENT ', text)

    def replace_pic(self, text):
        return re.sub(self.pic_regex, ' PICTURE_ENT ', text)

    def replace_tokens(self, doc):
        """ this function extracts entities and removes tokens from `spacy.doc`
            input: `spacy.doc` - initial document 
            output: `spacy.doc` - cleaned document
        """
        new_words = []
        idx_to_delete = []

        for index, token in enumerate(doc):

            if token.text.lower() in self.tickers_set:
                new_words.append(token.text)
                continue

            if token.ent_type_ == 'DATE':
                new_words.append('DATE_ENT')
                continue

            if (
                    token.ent_type_ == 'PERSON' 
                    and not re.match(self.bad_name_regex, token.text)
                    and token.text not in self.bad_name_set
            ):
                new_words.append('PERSON_ENT')
                continue

            if token.ent_type_ == 'PERCENT' and token.text.isdigit():
                new_words.append('PERCENT_ENT')
                continue

            if token.ent_type_ == 'MONEY' and token.text.isdigit():
                new_words.append('MONEY_ENT')
                continue

            if re.match(self.number_regex, token.text):
                new_words.append('NUMBER_ENT')
                continue

            if re.match(self.punct_regex, token.text):
                idx_to_delete.append(index)
                continue

            if token.text.lower() in self.stopwords:
                idx_to_delete.append(index)
                continue

            new_words.append(token.text)

        atrr_array = doc.to_array([LOWER, POS, ENT_TYPE, IS_ALPHA])
        atrr_array = np.delete(atrr_array, idx_to_delete, axis=0)

        new_doc = Doc(doc.vocab, words=new_words)

        new_doc.from_array([LOWER, POS, ENT_TYPE, IS_ALPHA], atrr_array)

        return new_doc
    
    def clean_text(self, text):
        """ cleans the text
            input: text as one string
            output: spacy.doc
        """
        text = html.unescape(text)
        text = self.replace_url(text)
        text = self.replace_pic(text)
        text = self.replace_user(text)
        text = re.sub('\s+', ' ', text)
        doc = self.nlp(text)
        doc = self.replace_tokens(doc)

        return doc
    
    def get_ticker_matcher(self):
        self.ticker_matcher = Matcher(self.nlp.vocab)

        for name in self.tickers_set:
            self.ticker_matcher.add(name, None, [{'LOWER': name.lower()}])

    def get_entity_matcher(self):
        
        self.entity_matcher = Matcher(self.nlp.vocab)

        for name in self.bad_name_set:
            self.entity_matcher.add(name, None, [{'LOWER': name.lower()}])   
            
    def match_ticker(self, row):
        match = self.ticker_matcher(row['doc'])

        if not match:
            return None

        match = match[0]

        result = row.copy()
        result['ticker'] = self.nlp.vocab.strings[match[0]]
        result['position'] = match[1]

        return result
    
    def has_special_emoji(self, text):
        return len(re.findall(self.special_emoji_regex, text)) > 0

    def has_emoji(self, text):
        emoji_list = []
        data = re.findall(r'\W', text)
        for word in data:
            if any(char in emoji.UNICODE_EMOJI for char in word):
                emoji_list.append(word)
        return len(emoji_list) > 0

    def get_distances_to_ents(self, doc, tic_pos):
        res_dict = defaultdict(lambda: 999)

        for res in self.entity_matcher(doc):
            key = self.nlp.vocab.strings[res[0]]
            res_dict[key] = min(res_dict[key], abs(tic_pos - res[1]))

        return res_dict

    def get_features(self, doc, tic_pos):
        doc_text = doc.text
        one_ex_dict = dict()

        one_ex_dict['word_count'] = len(doc)
        word_lens = [len(tok) for tok in doc]
        one_ex_dict['word_max'] = np.max(word_lens)
        one_ex_dict['word_mean'] = np.mean(word_lens)
        one_ex_dict['word_median'] = np.median(word_lens)

        ticker = doc[tic_pos]

        one_ex_dict['relative_pos'] = tic_pos / len(doc)

        distances_to_ents = self.get_distances_to_ents(doc, tic_pos)
        for ent in sorted(self.bad_name_set):
            one_ex_dict['{}_distance'.format(ent)] = distances_to_ents[ent]

        one_ex_dict['has_emoji'] = self.has_emoji(doc_text)
        one_ex_dict['has_special_emoji'] = self.has_special_emoji(doc_text)

        one_ex_dict['has_!'] = '!' in doc_text
        one_ex_dict['has_?'] = '?' in doc_text

        return one_ex_dict

    def build_features(self, df):
        result = []

        for _, row in df.iterrows():
            result.append(self.get_features(row['doc'], row['position']))

        return pd.DataFrame(result)
    def evaluate(self, df):
        """gets dataframe with the field 'text'
           returns new df with features and 
           indexes for not null examples (examples that contain ticker candidates)
        """
        docs = []
        df = df.copy()
        
        print('Cleaning data')
        sys.stdout.flush()
        for text in tqdm.tqdm(df['text'].values):
            docs.append(self.clean_text(text))

        df['doc'] = docs

        new_df = []
        ticker_found_mask = []

        print('Matching tickers')
        sys.stdout.flush()
        for _, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
            new_row = self.match_ticker(row)

            if new_row is not None:
                new_df.append(new_row)
                ticker_found_mask.append(True)
            else:
                ticker_found_mask.append(False)

        ticker_found_mask = np.array(ticker_found_mask, dtype=np.bool)
        df = pd.DataFrame(new_df)
        print('Tickers was found in {} examples ({:.0%})'\
                  .format(ticker_found_mask.sum(), ticker_found_mask.mean())
             )
        print('Building features')
        df = pd.concat(
            [
                df.reset_index(drop=True),
                self.build_features(df)
            ],
            axis=1
        )
        print('Done!')
        return df, ticker_found_mask
