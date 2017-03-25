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
    from pyspark.ml.feature import HashingTF, IDF, Tokenizer
    # from pyspark.mllib.feature import IDF
    from pyspark.mllib.feature import StandardScaler
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.mllib.evaluation import BinaryClassificationMetrics
    from pyspark.mllib.evaluation import MulticlassMetrics
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import LogisticRegression
    # from pyspark.ml.feature import Tokenizer
    from pyspark.sql import SQLContext,Row
    from pyspark.sql.functions import col
    from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
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

STEMMER = PorterStemmer()

STOPWORDS = set(stopwords.words('english'))


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


def stem_nostopwords(line):
    tweet_words = line.strip().split(" ")
    tweet_nostopwords = [word for word in tweet_words if not word in STOPWORDS]
    tweet_stemmed = [STEMMER.stem(word) for word in tweet_nostopwords]
    s = (" ").join(z for z in tweet_stemmed)
    return s.strip()


class sql_to_lp():
    def fit(self, x, y=None):
        return self

    def transform(self, df):
        lp = df.select(col("features"), col("label")).map(lambda row: LabeledPoint(row.features, row.label))
        return lp


if __name__ == '__main__':
    conf = (SparkConf().setAppName("TwitterSA").set("spark.executor.memory", "6g"))
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    input_train = "data/train.csv"
    input_test = "data/test.csv"

    raw_train = sc.textFile(input_train)
    print('## Parsing Train Data...')
    train_data = raw_train.map(parse_line)
    train_data = train_data.map(lambda x: (stem_nostopwords(x[0]),float(x[1])))
    train_sql = sqlContext.createDataFrame(train_data,["features_ip", "label"])
    train_sql.cache()
    print(train_sql.take(3))

    raw_test = sc.textFile(input_test)
    print('## Parsing Train Data...')
    test_data = raw_test.map(parse_line)
    test_data = test_data.map(lambda x: (stem_nostopwords(x[0]),float(x[1])))
    test_sql = sqlContext.createDataFrame(test_data,["features_ip", "orig_label"])
    test_sql.cache()
    print(test_sql.take(3))

    # tokenizer = Tokenizer(inputCol="features_ip", outputCol="words")
    # wordsData = tokenizer.transform(train_sql)
    # hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=50000)
    # featurizedData = hashingTF.transform(wordsData)
    # idf = IDF(inputCol="rawFeatures", outputCol="features")
    # idfModel = idf.fit(featurizedData)
    # rescaledData = idfModel.transform(featurizedData)
    # for features_label in rescaledData.select("features", "label").take(3):
    #     print(features_label)
    # print(type(rescaledData))
    #
    #
    # lr = LogisticRegression()
    # grid = ParamGridBuilder().addGrid(lr.maxIter, [0, 1]).build()
    # evaluator = BinaryClassificationEvaluator()
    # cv = CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator, numFolds = 2)
    # cvModel = cv.fit(rescaledData)
    # evaluator.evaluate(cvModel.transform(rescaledData))
    #
    #  # LR_P_O_test = labeled_testing_data.map(lambda p: (LRmodel.predict(p.features), p.label))
    # # accuracy = 1.0 * LR_P_O_test.filter(lambda x: x[1] == x[0]).count() / labeled_testing_data.count()
    # # print("testing Data",accuracy)
    # print(evaluator)





'''
    # Configure an ML pipeline, which consists of tree stages:
    tokenizer, hashingTF, IDF and logistic Regression.

    tokenizer = Tokenizer(inputCol="features_ip", outputCol="words")
    hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="hashedwords")
    idf = IDF(inputCol=hashingTF.getOutputCol(), outputCol="features")
    lr = LogisticRegression(maxIter=10, regParam=0.01)
    pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, lr])

    # Fit the pipeline to training documents.
    # model = pipeline.fit(training)

    grid = ParamGridBuilder().addGrid(lr.maxIter, [100, 200]).build()
    evaluator = BinaryClassificationEvaluator()
    cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid, evaluator=evaluator, numFolds = 10)
    cvModel = cv.fit(train_sql)
    # evaluator.evaluate(cvModel.transform(train_sql))

    prediction = cvModel.transform(test_sql)
    selected = prediction.select("prediction", "orig_label").rdd
    accuracy = 1.0 * selected.filter(lambda x: x[0] == x[1]).count() / test_sql.count()
    print("Testing Data",accuracy)


    metrics = MulticlassMetrics(selected)

    print(metrics.confusionMatrix().toArray())

    print("Precision = %s" % metrics.precision())
    print("Recall = %s" % metrics.recall())
'''






























    ### Start of Naive bayes

    # print('## Building model on Train Data...')
    #
    # labeled_training_data = train_data.map(lambda x: LabeledPoint(x[1],hashingTF.transform(x[0])))
    # labeled_training_data.persist()
    # print(labeled_training_data.take(2))
    # labeled_testing_data = test_data.map(lambda x: LabeledPoint(x[1],hashingTF.transform(x[0])))
    # labeled_testing_data.persist()
    #
    # NBmodel = NaiveBayes.train(labeled_training_data,lambda_=1.0)
    #
    # NB_P_O_train = labeled_training_data.map(lambda p: (NBmodel.predict(p.features), p.label))
    # accuracy = 1.0 * NB_P_O_train.filter(lambda x: x[0] == x[1]).count() / labeled_training_data.count()
    # print("Training Data",accuracy)
    #
    # NB_P_O_test = labeled_testing_data.map(lambda p: (NBmodel.predict(p.features), p.label))
    # accuracy = 1.0 * NB_P_O_test.filter(lambda x: x[0] == x[1]).count() / labeled_testing_data.count()
    # print("Testing Data",accuracy)

    # NBmodel.save(sc, "model/NBModel")
    # NB_sameModel = NaiveBayesModel.load(sc, "model/NBModel")

    ### End of Naive bayes


    # LRmodel = LogisticRegressionWithLBFGS.train(labeled_training_data)
    #
    # LR_P_O_train = labeled_training_data.map(lambda p: (LRmodel.predict(p.features), p.label))
    # accuracy = 1.0 * LR_P_O_train.filter(lambda x: x[1] == x[0]).count() / labeled_training_data.count()
    # print("Training Data",accuracy)
    #
    # LR_P_O_test = labeled_testing_data.map(lambda p: (LRmodel.predict(p.features), p.label))
    # accuracy = 1.0 * LR_P_O_test.filter(lambda x: x[1] == x[0]).count() / labeled_testing_data.count()
    # print("testing Data",accuracy)

    # # Save and load model
    # LRmodel.save(sc, "model/LRModel")
    # LR_sameModel = LogisticRegressionModel.load(sc, "model/LRModel")



    # model = DecisionTree.trainClassifier(labeled_training_data, numClasses=2, categoricalFeaturesInfo={},
    #                                      impurity='entropy', maxDepth=3)
    #
    # # Evaluate model on test instances and compute test error
    # predictions = model.predict(labeled_testing_data.map(lambda x: x.features))
    # labelsAndPredictions = labeled_testing_data.map(lambda lp: lp.label).zip(predictions)
    # testErr = labelsAndPredictions.filter(lambda x: x[0] != x[1]).count() / float(labeled_testing_data.count())
    # print('Test Error = ' + str(testErr))
    # print('Learned classification tree model:')
    # print(model.toDebugString())


