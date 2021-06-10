from spacy.tokens import Doc
from spacy.attrs import LOWER, POS, ENT_TYPE, IS_ALPHA
from collections import Counter
from itertools import islice

doc_attrs = [LOWER, POS, ENT_TYPE, IS_ALPHA]

def serialize_doc(doc):
    return {
        'bytes': doc.to_bytes(),
        'attrs': doc.to_array(doc_attrs)
    }

def deserialize_doc(state, vocab):
    doc = Doc(vocab).from_bytes(state['bytes'])
    doc.from_array(doc_attrs, state['attrs'])
    
    return doc

def get_top_words(df, label, pos_filter, stop_words):
    result = []
    stop_words_set = set(map(lambda x: x.lower(), stop_words))
    
    for words in df.loc[df['label'] == label, 'doc']:
        result.extend(filter(
            lambda x: pos_filter(x.pos_) and x.text.lower() not in stop_words_set and x.text.isalpha(),
            words
        ))
        
    return list(map(lambda x: str(x).lower(), result))

def top_wrds(words, count=20):
    return [x for x, _ in Counter(words).most_common(count)]

def get_top_words_exclusive(words_a, words_b, count=10, exclude_count=100):
    def fst(x):
        return x[0]
    
    words_a_cnt = Counter(words_a)
    words_b_cnt = Counter(words_b)
    
    return list(islice(
        (x for x, _ in words_a_cnt.most_common() 
        if x not in set(map(fst, words_b_cnt.most_common(exclude_count)))),
        count
    ))