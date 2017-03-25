import os
import sys
import string
import csv,re,io
import nltk


os.environ['SPARK_HOME'] = "/home/sam/spark-1.6.1-bin-hadoop2.6/"
sys.path.append("/home/sam/spark-1.6.1-bin-hadoop2.6/python")
os.environ['PYSPARK_PYTHON'] = "/home/sam/anaconda3/bin/python3.5"

try:
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    from pyspark import SparkContext
    from pyspark import SparkConf
    from pyspark.mllib.feature import Normalizer
    from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
    from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
    from pyspark.mllib.linalg import Vectors
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.feature import HashingTF
    from pyspark.mllib.feature import IDF
    from pyspark.mllib.feature import StandardScaler
    from pyspark.mllib.linalg import SparseVector, DenseVector
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    from pyspark.ml.classification import LogisticRegression
    from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
    from pyspark.mllib.feature import PCA as PCAmllib

    print("Successfully imported Spark Modules")
except ImportError as e:
    print("Error importing Spark Modules", e)
    sys.exit(1)

punc = string.punctuation.replace('_', '')

punc_regex = re.compile('[%s]' % re.escape(punc))
url_regex = '(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?'
nonalpha1 = re.compile('[0-9]+[a-z]+')
nonalpha2 = re.compile('\s[0-9]+\s')
one_two = '\s+\w\w?\s+'

def remove_punc(s):
    temp = re.sub( url_regex, 'url', s).strip()  # Remove URLs from the text
    temp = re.sub('^@([A-Za-z0-9_]+)', 'at_user', temp).strip()  # Replace user names with AT_USER
    temp = nonalpha1.sub(' ',temp)
    temp = nonalpha2.sub(' ',temp)
    temp = re.sub(r'(.)\1+', r'\1\1', temp)
    temp = punc_regex.sub(' ', temp).strip()  # Replace punctuation with out '_'
    temp = re.sub(one_two, ' ', temp)
    temp = re.sub(one_two, ' ', temp)
    return re.sub( '\s+', ' ', temp).strip()


def parse_line(line):
    columns = csv.reader(io.StringIO(line))
    k = columns.__next__()
    polarity = int(k[0].strip('"'))
    tweet = k[5].strip().lower()
    tweet = remove_punc(tweet)
    return tweet, polarity






STEMMER = PorterStemmer()



STOPWORDS = set(stopwords.words('english'))

def tokenizer(line):
    tweet_words = line.strip().split(" ")
    tweet_nostopwords = [word for word in tweet_words if not word in STOPWORDS]
    tweet_stemmed = [STEMMER.stem(word) for word in tweet_nostopwords]
    return [word for word in tweet_stemmed if word]


if __name__ == '__main__':
    conf = (SparkConf().setAppName("TwitterSA").set("spark.executor.memory", "6g"))
    sc = SparkContext(conf=conf)
    input_train = "data/train.csv"
    input_test = "data/test.csv"

    raw_train = sc.textFile(input_train)
    print('## Parsing Train Data...')
    train_data = raw_train.map(parse_line)
    print('## Tokenizing Train Data...')
    train_data = train_data.map(lambda x: (tokenizer(x[0]),x[1]))
    train_data.cache()
    # print(train_data.take(3))

    raw_test = sc.textFile(input_test)
    print('## Parsing Test Data...')
    test_data = raw_test.map(parse_line)
    print('## Tokenizing Test Data...')
    test_data = test_data.map(lambda x: (tokenizer(x[0]),x[1]))
    test_data.cache()
    # print(test_data.take(3))

    hashingTF = HashingTF(1000)


    ## Start of Naive bayes

    print('## Building model on Train Data...')

    labeled_training_data = train_data.map(lambda x: LabeledPoint(x[1],hashingTF.transform(x[0])))
    labeled_training_data.persist()
    print(labeled_training_data.take(2))
    labeled_testing_data = test_data.map(lambda x: LabeledPoint(x[1],hashingTF.transform(x[0])))
    labeled_testing_data.persist()

    NBmodel = NaiveBayes.train(labeled_training_data,lambda_=1.0)

    NB_P_O_train = labeled_training_data.map(lambda p: (NBmodel.predict(p.features), p.label))
    accuracy = 1.0 * NB_P_O_train.filter(lambda x: x[0] == x[1]).count() / labeled_training_data.count()
    print("Training Data %s" % accuracy)
    predictions = NBmodel.predict(labeled_testing_data.map(lambda x: x.features))
    NB_P_O_test = labeled_testing_data.map(lambda lp:  lp.label).zip(predictions)
    #
    accuracy = 1.0 * NB_P_O_test.filter(lambda x: x[0] == x[1]).count() / labeled_testing_data.count()
    print("Testing Data %s " % accuracy)

    # NBmodel.save(sc, "model/NBModel")
    # NB_sameModel = NaiveBayesModel.load(sc, "model/NBModel")

    ## End of Naive bayes


    #
    # print('## Calculating TF*IDF for Train Data...')
    # train_tf = train_data.map(lambda x: hashingTF.transform(x[0]))
    # train_tf.cache()
    # idf = IDF().fit(train_tf)
    # train_tfidf = idf.transform(train_tf)
    # train_tf.unpersist()
    # print('## Creating Labeled points for Train Data...')
    # tfidf_index = train_tfidf.zipWithIndex()
    # training_index = train_data.zipWithIndex()
    # index_training = training_index.map(lambda line: (line[1], line[0][1]))
    # index_tfidf = tfidf_index.map(lambda l: (l[1], l[0]))
    # joined_tfidf_training = index_training.join(index_tfidf).map(lambda x: x[1])
    # # joined_tfidf_training = joined_tfidf_training.map(lambda x: (x[0], DenseVector(x[1].toArray())))
    # k = joined_tfidf_training.take(1)
    #
    #
    # # print(k[0][1].shape)
    # labeled_training_data = joined_tfidf_training.map(lambda k: LabeledPoint(k[0], k[1]))
    # print(labeled_training_data.take(2))
    # labeled_training_data.cache()
    # train_data.unpersist()
    #
    #
    #
    # print('## Calculating TF*IDF for Test Data...')
    # test_tf = test_data.map(lambda x: hashingTF.transform(x[0]))
    # test_tf.cache()
    # idf = IDF().fit(test_tf)
    # test_tfidf = idf.transform(test_tf)
    # test_tf.unpersist()
    # print('## Creating Labeled points for Test Data...')
    # tfidf_index = test_tfidf.zipWithIndex()
    # testing_index = test_data.zipWithIndex()
    # index_testing = testing_index.map(lambda line: (line[1], line[0][1]))
    # index_tfidf = tfidf_index.map(lambda l: (l[1], l[0]))
    # joined_tfidf_testing = index_testing.join(index_tfidf).map(lambda x: x[1])
    # # joined_tfidf_testing = joined_tfidf_testing.map(lambda x: (x[0], DenseVector(x[1].toArray())))
    # # print(joined_tfidf_testing.take(2))
    # labeled_testing_data = joined_tfidf_testing.map(lambda k: LabeledPoint(k[0], k[1]))
    # # print(labeled_testing_data.take(2))
    # labeled_testing_data.cache()
    # test_data.unpersist()

    # LRmodel = LogisticRegressionWithLBFGS.train(labeled_training_data)
    #
    # LRmodel.setThreshold(0.5)
    #
    # LR_P_O_train = labeled_training_data.map(lambda p: (LRmodel.predict(p.features), p.label))
    # accuracy = 1.0 * LR_P_O_train.filter(lambda x: x[1] == x[0]).count() / labeled_training_data.count()
    # print("Training Data",accuracy)
    # predictions = LRmodel.predict(labeled_testing_data.map(lambda x: x.features))
    # # LR_P_O_test = labeled_testing_data.map(lambda p: (LRmodel.predict(p.features), p.label))
    # LR_P_O_test = labeled_testing_data.map(lambda lp:  lp.label).zip(predictions)
    # accuracy = 1.0 * LR_P_O_test.filter(lambda x: x[1] == x[0]).count() / labeled_testing_data.count()
    # print("testing Data %s" % accuracy)

    # Save and load model
    # LRmodel.save(sc, "model/LRModel")
    # LR_sameModel = LogisticRegressionModel.load(sc, "model/LRModel")


    j = predictions.coalesce(int(test_data.getNumPartitions()))
    temp_testpred = test_data.zip(j)
    # print(temp_testpred.take(2))
    correctly_classified = temp_testpred.filter(lambda x: x[0][1] == x[1])
    print(correctly_classified.take(7))
    wrongly_classified = temp_testpred.filter(lambda x: x[0][1] != x[1])
    print(wrongly_classified.take(7))





    # model = DecisionTree.trainClassifier(labeled_training_data, numClasses=2, categoricalFeaturesInfo={},
    #                                      impurity='entropy', maxDepth=30, maxBins=50)
    # predictions = model.predict(labeled_training_data.map(lambda x: x.features))
    # labelsAndPredictions = labeled_training_data.map(lambda lp: lp.label).zip(predictions)
    # train_accuracy = labelsAndPredictions.filter(lambda v: v[0] == v[1]).count() / float(train_data.count())
    # print("Train accuracy is %s" % round(float(train_accuracy), 4))
    #
    # # Evaluate model on test instances and compute test error
    # predictions = model.predict(labeled_testing_data.map(lambda x: x.features))
    # labelsAndPredictions = labeled_testing_data.map(lambda lp:  lp.label).zip(predictions)
    # test_accuracy = labelsAndPredictions.filter(lambda v: v[0] == v[1]).count() / float(test_data.count())
    # print("Test accuracy is %s" % round(float(test_accuracy), 4))
    # # print(model.toDebugString())

