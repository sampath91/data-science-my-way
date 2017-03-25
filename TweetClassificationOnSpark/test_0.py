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
    from pyspark.ml.feature import HashingTF
    from pyspark.mllib.feature import IDF
    from pyspark.mllib.feature import StandardScaler
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.feature import Tokenizer
    from pyspark.sql import SQLContext, Row
    from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
    print("Successfully imported Spark Modules")
except ImportError as e:
    print("Error importing Spark Modules", e)
    sys.exit(1)


conf = (SparkConf().setAppName("TwitterSA").set("spark.executor.memory", "6g"))
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
# Prepare training documents from a list of (id, text, label) tuples.
LabeledDocument = Row("id", "text", "label")
training = sqlContext.createDataFrame([
    (0, "a b c d e spark", 1.0),
    (1, "b d", 0.0),
    (2, "spark f g h", 1.0),
    (3, "hadoop mapreduce", 0.0)], ["id", "text", "label"])

# Configure an ML pipeline, which consists of tree stages: tokenizer, hashingTF, and lr.
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression(maxIter=10, regParam=0.01)
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

# Fit the pipeline to training documents.
model = pipeline.fit(training)

# Prepare test documents, which are unlabeled (id, text) tuples.
test = sqlContext.createDataFrame([
    (4, "spark i j k",1),
    (5, "l m n",0),
    (6, "mapreduce spark",1),
    (7, "apache hadoop",1)], ["id", "text","actual"])

# Make predictions on test documents and print columns of interest.
prediction = model.transform(test)
selected = prediction.select("id", "text", "prediction")
for row in selected.collect():
    print(row)