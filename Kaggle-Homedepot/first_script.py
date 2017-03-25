import time
import datetime
starttime = time.time()

date = str(datetime.datetime.now().strftime(format='%Y-%m-%d'))

print ("Start time:", datetime.datetime.now())
import numpy as np

import pandas as pd
from nltk.stem.snowball import PorterStemmer
import random
random.seed(13)
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english')

print('Importing...%s minutes' % (round((time.time()-starttime)/60, 2)))

traindata = pd.read_csv('input/train.csv', encoding="ISO-8859-1")
testdata = pd.read_csv('input/test.csv', encoding="ISO-8859-1")
attributes = pd.read_csv('input/attributes.csv', encoding="ISO-8859-1")
product_desc = pd.read_csv('input/product_descriptions.csv', encoding="ISO-8859-1")

TrainData_Count = traindata.shape[0]

product_brand = attributes[attributes.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brandname"})

product_details = pd.merge(product_desc,product_brand,how='left',on='product_uid')

product_details['brandname'] = product_details['brandname'].replace('^.N/A','',False)

product_details['brandname'] = product_details['brandname'].fillna("")

def count_common_word(str1, str2):
    count = 0
    for word in str1.split():
        if str2.find(word) >= 0:
            count += 1
    return count

def lcs(S,T):
    m = len(S)
    n = len(T)
    counter = [[0]*(n+1) for x in range(m+1)]
    longest = 0
    lcs_set = set()
    for i in range(m):
        for j in range(n):
            if S[i] == T[j]:
                c = counter[i][j] + 1
                counter[i+1][j+1] = c
                if c > longest:
                    lcs_set = set()
                    longest = c
                    lcs_set.add(S[i-c+1:i+1])
                elif c == longest:
                    lcs_set.add(S[i-c+1:i+1])
    return len(lcs_set)

def stemming(s):
    return " ".join([stemmer.stem(word) for word in s.lower().split()])

alldetails = pd.concat((traindata,testdata),axis=0, ignore_index=True)

alldetails = pd.merge(alldetails,product_details,how="left",on='product_uid')

print('Structuring Data...%s minutes' % (round((time.time()-starttime)/60,2)))

alldetails['search_term'] = alldetails['search_term'].map(lambda x: stemming(x))
print('Stemming search term..')
alldetails['product_title'] = alldetails['product_title'].map(lambda x: stemming(x))
print ('Stemming product title..')
alldetails['product_description'] = alldetails['product_description'].map(lambda x: stemming(x))
print ('Stemming product desc..')
alldetails['brandname'] = alldetails['brandname'].map(lambda x: stemming(x))
print ('Stemming brand name..')

print ('Stemming...%s minutes' % (round((time.time()-starttime)/60,2)))

alldetails['len_of_query'] = alldetails['search_term'].map(lambda x:len(x.split())).astype(np.int64)

alldetails['product_info'] = alldetails['search_term'] + "\t" + \
                             alldetails['product_title'] + "\t" + \
                             alldetails['product_description'] + "\t" + \
                             alldetails['brandname']


alldetails['word_in_title'] = alldetails['product_info'].map(lambda x:count_common_word(x.split('\t')[0],x.split('\t')[1]))
alldetails['word_in_description'] = alldetails['product_info'].map(lambda x:count_common_word(x.split('\t')[0],x.split('\t')[2]))
alldetails['word_in_brandname'] = alldetails['product_info'].map(lambda x:count_common_word(x.split('\t')[0],x.split('\t')[3]))

# alldetails['LCS_search_title'] = alldetails['product_info'].map(lambda x:count_common_word(x.split('\t')[0],x.split('\t')[1]))
# alldetails['LCS_search_description'] = alldetails['product_info'].map(lambda x:count_common_word(x.split('\t')[0],x.split('\t')[2]))
# alldetails['LCS_search_brandname'] = alldetails['product_info'].map(lambda x:count_common_word(x.split('\t')[0],x.split('\t')[3]))

print('Counting common words...%s minutes' % (round((time.time()-starttime)/60, 2)))

print (alldetails.head(1))

alldetails = alldetails.drop(['search_term', 'product_title', 'product_uid',
                              'product_description', 'product_info', 'brandname'], axis=1)


alg_train = alldetails.iloc[:TrainData_Count]
alg_test = alldetails.iloc[TrainData_Count:]
alg_test_id = alg_test['id']

Y_train = alg_train['relevance'].values




X_train = alg_train.drop(['id', 'relevance'], axis=1).values
X_test = alg_test.drop(['id', 'relevance'], axis=1).values



forest = RandomForestRegressor(n_estimators=550, criterion="mse",
                               max_features=10, max_depth=15, n_jobs=-1, verbose=0)
bg = BaggingRegressor(forest, n_estimators=150, max_samples=0.1, random_state=29)
bg.fit(X_train, Y_train)
Y_output = bg.predict(X_test)

# score = forest.score(X_train, Y_train)
#
# print score

filename = 'submission_'+date+'.csv'

pd.DataFrame({"id": alg_test_id, "relevance": Y_output}).to_csv(filename,index=False)

