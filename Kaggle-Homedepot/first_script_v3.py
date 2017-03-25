import time
import datetime
start_time = time.time()
date = str(datetime.datetime.now().strftime(format='%m%d'))
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
# from sklearn import pipeline, model_selection
from sklearn import pipeline, grid_search
# from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
# from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn import decomposition
# from nltk.metrics import edit_distance
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
#stemmer = PorterStemmer()
# from nltk.stem.snowball import SnowballStemmer #0.003 improvement but takes twice as long as PorterStemmer
stemmer = SnowballStemmer('english')
import re
# import enchant
import random



random.seed(2016)
hd_train = pd.read_csv('input/train.csv', encoding="ISO-8859-1")#[:100] #update here
hd_test = pd.read_csv('input/test.csv', encoding="ISO-8859-1")#[:100] #update here
hd_pro_desc = pd.read_csv('input/product_descriptions.csv')#[:100] #update here
hd_attr = pd.read_csv('input/attributes.csv')
hd_brand = hd_attr[hd_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})
num_train = hd_train.shape[0]
all_details = pd.concat((hd_train, hd_test), axis=0, ignore_index=True)
all_details = pd.merge(all_details, hd_pro_desc, how='left', on='product_uid')
all_details = pd.merge(all_details, hd_brand, how='left', on='product_uid')
print("### Files Loaded: %s minutes ###" % round(((time.time() - start_time) / 60), 2))

stop_w = ['for', 'xbi', 'and', 'in', 'th', 'on', 'sku', 'with', 'what', 'from', 'that', 'less', 'er',
          'ing']  # 'electr','paint','pipe','light','kitchen','wood','outdoor','door','bathroom'
strNum = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}


def str_stem(s):
    if isinstance(s, str):
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s)  # Split words with a.A
        s = s.lower()
        s = s.replace("  ", " ")
        s = s.replace(",", "")  # could be number / segment later
        s = s.replace("$", " ")
        s = s.replace("?", " ")
        s = s.replace("-", " ")
        s = s.replace("//", "/")
        s = s.replace("..", ".")
        s = s.replace(" / ", " ")
        s = s.replace(" \\ ", " ")
        s = s.replace(".", " . ")
        s = re.sub(r"(^\.|/)", r"", s)
        s = re.sub(r"(\.|/)$", r"", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = s.replace(" x ", " xbi ")
        s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
        s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
        s = s.replace("*", " xbi ")
        s = s.replace(" by ", " xbi ")
        s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
        s = s.replace("°", " degrees ")
        s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
        s = s.replace(" v ", " volts ")
        s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
        s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
        s = s.replace("  ", " ")
        s = s.replace(" . ", " ")
        # s = (" ").join([z for z in s.split(" ") if z not in stop_w])
        s = (" ").join([str(strNum[z]) if z in strNum else z for z in s.split(" ")])
        s = (" ").join([stemmer.stem(z) for z in s.split(" ")])

        s = s.lower()
        s = s.replace("toliet", "toilet")
        s = s.replace("airconditioner", "air conditioner")
        s = s.replace("vinal", "vinyl")
        s = s.replace("vynal", "vinyl")
        s = s.replace("skill", "skil")
        s = s.replace("snowbl", "snow bl")
        s = s.replace("plexigla", "plexi gla")
        s = s.replace("rustoleum", "rust-oleum")
        s = s.replace("whirpool", "whirlpool")
        s = s.replace("whirlpoolga", "whirlpool ga")
        s = s.replace("whirlpoolstainless", "whirlpool stainless")
        return s
    else:
        return "null"


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


def count_commonword(str1, str2):
    words, cnt = str1.split(), 0
    for word in words:
        if str2.find(word) >= 0:
            cnt += 1
    return cnt


def count_wholeword(str1, str2, i_):
    cnt = 0
    while i_ < len(str2):
        i_ = str2.find(str1, i_)
        if i_ == -1:
            return cnt
        else:
            cnt += 1
            i_ += len(str1)
    return cnt


def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions) ** 0.5
    return fmean_squared_error_


RMSE = make_scorer(fmean_squared_error, greater_is_better=False)


class cust_regression_vals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, hd_searches):
        d_col_drops = ['id', 'relevance', 'search_term', 'product_title', 'product_description', 'product_info', 'attr',
                       'brand']
        hd_searches = hd_searches.drop(d_col_drops, axis=1).values
        return hd_searches


class cust_txt_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key].apply(str)


# comment out the lines below use all_details.csv for further grid search testing
# if adding features consider any drops on the 'cust_regression_vals' class
# *** would be nice to have a file reuse option or script chaining option on Kaggle Scripts ***
all_details['search_term'] = all_details['search_term'].map(lambda x: str_stem(x))
all_details['product_title'] = all_details['product_title'].map(lambda x: str_stem(x))
all_details['product_description'] = all_details['product_description'].map(lambda x: str_stem(x))
all_details['brand'] = all_details['brand'].map(lambda x: str_stem(x))
print("### Stemming: %s minutes ###" % round(((time.time() - start_time) / 60), 2))

all_details['product_info'] = all_details['search_term'] + "\t" + all_details['product_title'] + "\t" + all_details['product_description']
print("### Prod Info: %s minutes ###" % round(((time.time() - start_time) / 60), 2))


all_details['len_of_query'] = all_details['search_term'].map(lambda x: len(x.split())).astype(np.int64)
all_details['len_of_title'] = all_details['product_title'].map(lambda x: len(x.split())).astype(np.int64)
all_details['len_of_description'] = all_details['product_description'].map(lambda x: len(x.split())).astype(np.int64)
all_details['len_of_brand'] = all_details['brand'].map(lambda x: len(x.split())).astype(np.int64)
print("### Len of: %s minutes ###" % round(((time.time() - start_time) / 60), 2))


all_details['search_term'] = all_details['product_info'].map(lambda x: seg_words(x.split('\t')[0], x.split('\t')[1]))
# print("### Search Term Segment: %s minutes ###" % round(((time.time() - start_time)/60),2))
all_details['query_in_title'] = all_details['product_info'].map(lambda x: count_wholeword(x.split('\t')[0], x.split('\t')[1], 0))
all_details['query_in_description'] = all_details['product_info'].map(
    lambda x: count_wholeword(x.split('\t')[0], x.split('\t')[2], 0))
print("### Query In: %s minutes ###" % round(((time.time() - start_time) / 60), 2))


all_details['query_last_word_in_title'] = all_details['product_info'].map(
    lambda x: count_commonword(x.split('\t')[0].split(" ")[-1], x.split('\t')[1]))
all_details['query_last_word_in_description'] = all_details['product_info'].map(
    lambda x: count_commonword(x.split('\t')[0].split(" ")[-1], x.split('\t')[2]))
print("### Query Last Word In: %s minutes ###" % round(((time.time() - start_time) / 60), 2))


all_details['word_in_title'] = all_details['product_info'].map(lambda x: count_commonword(x.split('\t')[0], x.split('\t')[1]))
all_details['word_in_description'] = all_details['product_info'].map(
    lambda x: count_commonword(x.split('\t')[0], x.split('\t')[2]))
all_details['ratio_title'] = all_details['word_in_title'] / all_details['len_of_query']
all_details['ratio_description'] = all_details['word_in_description'] / all_details['len_of_query']
all_details['attr'] = all_details['search_term'] + "\t" + all_details['brand']
all_details['word_in_brand'] = all_details['attr'].map(lambda x: count_commonword(x.split('\t')[0], x.split('\t')[1]))
all_details['ratio_brand'] = all_details['word_in_brand'] / all_details['len_of_brand']


hd_brand = pd.unique(all_details.brand.ravel())
d = {}
i = 1000
for s in hd_brand:
    d[s] = i
    i += 3

all_details['brand_feature'] = all_details['brand'].map(lambda x: d[x])
all_details['search_term_feature'] = all_details['search_term'].map(lambda x: len(x))
#all_details.to_csv('all_details.csv')
# all_details = pd.read_csv('all_details.csv', encoding="ISO-8859-1", index_col=0)


hd_train = all_details.iloc[:num_train]
hd_test = all_details.iloc[num_train:]
id_test = hd_test['id']
y_train = hd_train['relevance'].values
X_train = hd_train[:]
X_test = hd_test[:]
print("### Features Set: %s minutes ###" % round(((time.time() - start_time) / 60), 2))


rfr = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=2016, verbose=1)
tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
tsvd = TruncatedSVD(n_components=10, random_state=2016)
# pca = decomposition.SparsePCA(n_components=10)
clf = pipeline.Pipeline([
    ('union', FeatureUnion(
        transformer_list=[
            ('cst', cust_regression_vals()),
            ('txt1', pipeline.Pipeline([('s1', cust_txt_col(key='search_term')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),
            ('txt2', pipeline.Pipeline([('s2', cust_txt_col(key='product_title')), ('tfidf2', tfidf), ('tsvd2', tsvd)])),
            ('txt3', pipeline.Pipeline([('s3', cust_txt_col(key='product_description')), ('tfidf3', tfidf), ('tsvd3', tsvd)])),
            ('txt4', pipeline.Pipeline([('s4', cust_txt_col(key='brand')), ('tfidf4', tfidf), ('tsvd4', tsvd)]))
            # ('txt1', pipeline.Pipeline([('s1', cust_txt_col(key='search_term')), ('tfidf1', tfidf), ('pca1', pca)])),
            # ('txt2', pipeline.Pipeline([('s2', cust_txt_col(key='product_title')), ('tfidf2', tfidf), ('pca2', pca)])),
            # ('txt3', pipeline.Pipeline([('s3', cust_txt_col(key='product_description')), ('tfidf3', tfidf), ('pca3', pca)])),
            # ('txt4', pipeline.Pipeline([('s4', cust_txt_col(key='brand')), ('tfidf4', tfidf), ('pca4', pca)]))
        ],
        transformer_weights={
            'cst': 1.0,
            'txt1': 0.5,
            'txt2': 0.25,
            'txt3': 0.1,
            'txt4': 0.5
        },
        n_jobs=-1
    )),
    ('rfr', rfr)])
param_grid = {'rfr__max_features': [10], 'rfr__max_depth': [20]}
model = grid_search.GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=20, scoring=RMSE)
model.fit(X_train, y_train)

print("Best parameters found by grid search:")
print(model.best_params_)
print("Best CV score:")
print(model.best_score_)
print(model.best_score_ + 0.47003199274)

y_pred = model.predict(X_test)
filename = 'submission_' + date + '.csv'
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv(filename, index=False)
print("### Training & Testing: %s minutes ###" % round(((time.time() - start_time) / 60), 2))

