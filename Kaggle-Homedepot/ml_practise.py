import time
import datetime
import numpy as np
import pandas as pd
import random
import re

from sklearn.pipeline import Pipeline
from sklearn import grid_search
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, make_scorer

from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer

random.seed(1729)
start_time = time.time()
date = str(datetime.datetime.now().strftime(format='%m%d'))

print("::Start time- ", datetime.datetime.now())

snowball = SnowballStemmer('english')

print('### Importing...%s minutes ###' % (round((time.time() - start_time) / 60, 2)))

train = pd.read_csv('input/train.csv', encoding="ISO-8859-1")#[:1000]
test = pd.read_csv('input/test.csv', encoding="ISO-8859-1")#[:1000]
attributes = pd.read_csv('input/attributes.csv', encoding="ISO-8859-1")
product_desc = pd.read_csv('input/product_descriptions.csv', encoding="ISO-8859-1")#[:1000]

TrainData_Count = train.shape[0]

product_brand = attributes[attributes.name == "MFG Brand Name"][["product_uid", "value"]].rename(
    columns={"value": "brandname"})

product_details = pd.merge(product_desc, product_brand, how='left', on='product_uid')

product_details['brandname'] = product_details['brandname'].replace('^.N/A', 'N/A', False)

num_to_string = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
                 'zero': 0}


def count_common_word(str1, str2):
    count = 0
    for word in str1.split():
        if str2.find(word) >= 0:
            count += 1
    return count

def count_whole_word(str1, str2, i_):
    cnt = 0
    while i_ < len(str2):
        i_ = str2.find(str1, i_)
        if i_ == -1:
            return cnt
        else:
            cnt += 1
            i_ += len(str1)
    return cnt

def stemming(words):
    if isinstance(words, str):
        words = words.lower()
        words = words.replace("  ", " ")
        words = words.replace(" . ", " ")
        words = re.sub("a/c", "air conditioner", words)
        words = re.sub("\s?\*\s?|\s?x\s?", " cross ", words)
        words = re.sub("\s?/\s?", " by ", words)
        words = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", words)
        words = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", words)
        words = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", words)
        words = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", words)
        words = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", words)
        words = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", words)
        words = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", words)
        words = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", words)
        words = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", words)
        words = words.replace("°", " degrees ")
        words = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", words)
        words = words.replace(" v ", " volts ")
        words = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", words)
        words = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", words)
        words = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", words)
        words = words.replace("  ", " ")
        words = words.replace(" . ", " ")
        words = " ".join([str(num_to_string[z]) if z in num_to_string else z for z in words.split(" ")])
        words = words.replace("vinal", "vinyl")
        words = words.replace("vynal", "vinyl")
        words = words.replace("skill", "skil")
        return " ".join([snowball.stem(word) for word in words.split(" ")])
    else:
        return "null"

alldetails = pd.concat((train, test), axis=0, ignore_index=True)

alldetails = pd.merge(alldetails, product_details, how="left", on='product_uid')


print('### Structuring Data...%s minutes ###' % (round((time.time() - start_time) / 60, 2)))

print('...Stemming search term')
alldetails['search_term'] = alldetails['search_term'].map(lambda x: stemming(x))
print('...Stemming product title')
alldetails['product_title'] = alldetails['product_title'].map(lambda x: stemming(x))
print('...Stemming product desc')
alldetails['product_description'] = alldetails['product_description'].map(lambda x: stemming(x))
print('...Stemming brand name')
alldetails['brandname'] = alldetails['brandname'].map(lambda x: stemming(x))
print('### Stemming...%s minutes ###' % (round((time.time() - start_time) / 60, 2)))

alldetails['product_info'] = alldetails['search_term'] + "\t" + \
                             alldetails['product_title'] + "\t" + \
                             alldetails['product_description'] + "\t" + \
                             alldetails['brandname']

alldetails['word_in_title'] = alldetails['product_info']. \
    map(lambda x: count_common_word(x.split('\t')[0], x.split('\t')[1]))
alldetails['word_in_description'] = alldetails['product_info']. \
    map(lambda x: count_common_word(x.split('\t')[0], x.split('\t')[2]))
alldetails['word_in_brandname'] = alldetails['product_info']. \
    map(lambda x: count_common_word(x.split('\t')[0], x.split('\t')[3]))
print('### Counting common words...%s minutes ###' % (round((time.time() - start_time) / 60, 2)))

alldetails['len_of_query'] = alldetails['search_term'].map(lambda x: len(x.split())).astype(np.int64)
alldetails['len_of_title'] = alldetails['product_title'].map(lambda x: len(x.split())).astype(np.int64)
alldetails['len_of_description'] = alldetails['product_description'].map(lambda x: len(x.split())).astype(np.int64)
alldetails['len_of_brand'] = alldetails['brandname'].map(lambda x: len(x.split())).astype(np.int64)
print("### Len of columns: %s minutes ###" % round(((time.time() - start_time) / 60), 2))

alldetails['query_in_title'] = alldetails['product_info'].map(lambda x: count_whole_word(x.split('\t')[0],
                                                                                         x.split('\t')[1], 0))
alldetails['query_in_description'] = alldetails['product_info'].map(
    lambda x: count_whole_word(x.split('\t')[0], x.split('\t')[2], 0))
print("--- Query In: %s minutes ---" % round(((time.time() - start_time) / 60), 2))


alldetails['ratio_title'] = alldetails['word_in_title'] / alldetails['len_of_query']
alldetails['ratio_description'] = alldetails['word_in_description'] / alldetails['len_of_query']
alldetails['ratio_brand'] = alldetails['word_in_brandname'] / alldetails['len_of_brand']

alg_train = alldetails.iloc[:TrainData_Count]
alg_test = alldetails.iloc[TrainData_Count:]
alg_test_id = alg_test['id']
Y_train = alg_train['relevance'].values
X_train = alg_train[:]
X_test = alg_test[:]

print("### Columns for Pipeline:", X_train.columns)

print("### Features Set: %s minutes ###" % round(((time.time() - start_time) / 60), 2))

class cust_regression_vals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, hd_searches):
        d_col_drops = ['id', 'product_uid', 'relevance', 'search_term', 'product_title', 'product_description', 'product_info',
                       'brandname']
        hd_searches = hd_searches.drop(d_col_drops, axis=1).values
        return hd_searches



