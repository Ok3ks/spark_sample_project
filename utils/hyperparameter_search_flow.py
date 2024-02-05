from metaflow import FlowSpec, Parameter, step
from pyspark.ml.param import Params

from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import numpy as np

class trainTest():
    def __init__(self):
        spark = SparkSession.builder.appName("anomaly").getOrCreate()
        df = spark.read.parquet("./data/features/train_features.pq")
        self.train = df.where(~F.col("source_ip").like("10%"))
        self.test = df.where(F.col("source_ip").like("10%"))

    def _training_data(self):
        return self.train
    
    def _testing_data(self):
        return self.test

class HyperparameterSearch(FlowSpec):
    numerical_features = ['duration', 'orig_bytes', 'resp_bytes',
                    'orig_pkts','orig_ip_bytes','resp_pkts',
                    'resp_ip_bytes', 'source_ip_count_last_min',
                    'source_ip_count_last_30_min']

    categorical_features = ["proto", "service", "conn_state"]
    categorical_features_indexed = [c + "_index" for c in categorical_features]
    input_features = numerical_features + categorical_features_indexed

    @step
    def start(self):
        "Hyperparameter Search Starts "

        self.seed = 42
        self.next(self.create_parameters)
    
    @step
    def create_parameters(self):
        "Create Parameters to be searched"
        self.trees = [30,40,50]
        self.next(self.fit_and_eval, foreach = 'trees')

    @step
    def fit_and_eval(self):
        " Fit and evaluate using the Spark method"
        
        train = trainTest()._training_data()
        test = trainTest()._testing_data()

        spark = SparkSession.builder.appName("anomaly").getOrCreate()

        indexer = StringIndexer(inputCols= self.categorical_features, outputCols=self.categorical_features_indexed, handleInvalid='skip')
        #one_hot_encoder = OneHotEncoder(inputCols= self.categorical_features_indexed, outputCols = self.categorical_features_indexed)
        assembler = VectorAssembler(inputCols=self.input_features, outputCol="features", handleInvalid="skip")

        trees = self.input
        randomforest = RandomForestClassifier(featuresCol="features", labelCol = "good", numTrees=trees, seed = self.seed)

        #randomforest.fitMultiple()
        pipeline = Pipeline(stages=[indexer, assembler, randomforest])
        pipeline = pipeline.fit(train)

        pr_util = BinaryClassificationEvaluator(labelCol="good", metricName="areaUnderPR")
        roc_util = BinaryClassificationEvaluator(labelCol="good", metricName="areaUnderROC")

        pred = pipeline.transform(test)
        self.pr = pr_util.evaluate(pred)
        self.roc = roc_util.evaluate(pred)

        self.next(self.join)
        
    @step
    def join(self, inputs):
        
        pr = [input.pr for input in inputs]
        roc = [input.roc for input in inputs]

        self.best_pr = (np.argmax(pr), roc[np.argmax(roc)])
        self.best_roc = (np.argmax(roc), roc[np.argmax(roc)])

        self.worst_pr = (np.argmin(pr), roc[np.argmin(roc)])
        self.worst_roc = (np.argmin(roc), roc[np.argmin(roc)])

        self.next(self.end)

    @step
    def end(self):
        "End of node"
        print("Process complete")

        print(f"Best Params is {self.trees[self.best_pr[0]]} with pr_score of  {self.best_pr[1]}")
        print(f"Worst Params is {self.trees[self.worst_roc[0]]} with roc_score of {self.worst_roc[1]}")

if __name__ == "__main__":
    spark = SparkSession.builder.appName("anomaly").getOrCreate()
    flow = HyperparameterSearch()