import time
import datetime
import numpy as np
import pandas as pd
import random
import re

from sklearn import pipeline, grid_search
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

snowball_stem = SnowballStemmer('english')

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

def seg_words(str1, str2):
    str2 = str2.lower()
    str2 = re.sub("[^a-z0-9./]", " ", str2)
    str2 = [z for z in set(str2.split()) if len(z) > 2]
    words = str1.lower().split(" ")
    s = []
    for word in words:
        if len(word) > 3:
            s1 = []
            s1 += segmentit(word, str2, True)
            if len(s) > 1:
                s += [z for z in s1 if z not in ['er', 'ing', 's', 'less'] and len(z) > 1]
            else:
                s.append(word)
        else:
            s.append(word)
    return (" ".join(s))


def segmentit(s, txt_arr, t):
    st = s
    r = []
    for j in range(len(s)):
        for word in txt_arr:
            if word == s[:-j]:
                r.append(s[:-j])
                # print(s[:-j],s[len(s)-j:])
                s = s[len(s) - j:]
                r += segmentit(s, txt_arr, False)
    if t:
        i = len(("").join(r))
        if not i == len(st):
            r.append(st[i:])
    return r

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
        words = " ".join([str(num_to_string[z]) if z in num_to_string else z for z in words.split(" ")])
        return " ".join([snowball_stem.stem(word) for word in words.split(" ")])
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


alldetails['search_term'] = alldetails['product_info'].map(lambda x: seg_words(x.split('\t')[0], x.split('\t')[1]))
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


df_brand = pd.unique(alldetails.brandname.ravel())
d = {}
i = 1000
for s in df_brand:
    d[s] = i
    i += 3

#alldetails['brand_feature'] = alldetails['brandname'].map(lambda x: d[x])
#alldetails['search_term_feature'] = alldetails['search_term'].map(lambda x: len(x))


# alldetails = alldetails.drop(['search_term', 'product_title', 'product_uid',
# 'product_description', 'product_info', 'brandname'], axis=1)

alg_train = alldetails.iloc[:TrainData_Count]
alg_test = alldetails.iloc[TrainData_Count:]
alg_test_id = alg_test['id']
Y_train = alg_train['relevance'].values
X_train = alg_train[:]
X_test = alg_test[:]

print("### Columns for Regressor:", X_train.columns)


print("### Features Set: %s minutes ###" % round(((time.time() - start_time) / 60), 2))


# X_train = alg_train.drop(['id', 'relevance'], axis=1).values
# X_test = alg_test.drop(['id', 'relevance'], axis=1).values


def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions) ** 0.5
    return fmean_squared_error_


RMSE = make_scorer(fmean_squared_error, greater_is_better=False)


class cust_regression_vals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, hd_searches):
        d_col_drops = ['id', 'product_uid', 'relevance', 'search_term', 'product_title', 'product_description', 'product_info',
                       'brandname']
        hd_searches = hd_searches.drop(d_col_drops, axis=1).values
        return hd_searches

class cust_txt_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key].apply(str)

forest = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=2016, verbose=1, max_features=10, max_depth=20)
#bg = BaggingRegressor(forest, n_estimators=20, random_state=1729, verbose=10)
#bg.fit(X_train, Y_train)

tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
tsvd = TruncatedSVD(n_components=10, random_state=2016)

grid_est = pipeline.Pipeline([('union',  FeatureUnion(
        transformer_list=[
            ('drop_cols', cust_regression_vals()),
            ('tf_search', pipeline.Pipeline([('s1', cust_txt_col(key='search_term')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),
            ('tf_title', pipeline.Pipeline([('s2', cust_txt_col(key='product_title')), ('tfidf2', tfidf), ('tsvd2', tsvd)])),
            ('tf_brand', pipeline.Pipeline([('s3', cust_txt_col(key='brandname')), ('tfidf3', tfidf), ('tsvd3', tsvd)])),
            ('tf_desc', pipeline.Pipeline([('s3', cust_txt_col(key='product_description')), ('tfidf4', tfidf), ('tsvd4', tsvd)]))
        ],
        transformer_weights={
            'drop_cols': 1.0,
            'tf_search': 0.5,
            'tf_title': 0.25,
            'tf_brand': 0.5,
            'tf_desc': 0.2
        },
        n_jobs=-1
    )), ('forest', forest)])

#param_grid = {'bg__max_features': [10], 'bg__max_samples': [0.3]}
param_grid = {'forest__max_features': [10,20], 'forest__max_depth': [7, 10]}
model = grid_search.GridSearchCV(estimator=grid_est, param_grid=param_grid, n_jobs=-1, cv=2, verbose=20, scoring=RMSE)
model.fit(X_train, Y_train)

print("Best parameters found by grid search:")
print(model.best_params_)
print("Best CV score:")
print(model.best_score_)
#print(model.best_score_ + 0.47003199274)

Y_output = model.predict(X_test)
filename = 'submission_' + date + '.csv'

pd.DataFrame({"id": alg_test_id, "relevance": Y_output}).to_csv(filename, index=False)
