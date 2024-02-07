from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import SparkSession

from typing import Union,Any
from utils.feature_engineering import categorical_features, numerical_features, input_features
from utils.feature_engineering import categorical_features_indexed

def get_Pipeline(categorical_features: Union[Any, list] = categorical_features(), 
                 numerical_features: Union[Any, list] = numerical_features(),
                 labelCol= "good") -> Pipeline :
    """ 
    Categorical features are passed through a StringIndexer and OneHotEncoder 
    """
    #spark = SparkSession.builder.appName("anomaly").getOrCreate()

    if categorical_features :

        indexer = StringIndexer(inputCols=categorical_features, outputCols=categorical_features_indexed(categorical_features), handleInvalid='skip')
        one_hot_encoder = OneHotEncoder(inputCols=categorical_features_indexed(categorical_features), outputCols = [c + '_one_hot' for c in categorical_features])

        inputCols = input_features(categorical_features)

    if numerical_features :
        #Add numerical features transformatiojn
        pass

    if categorical_features and numerical_features:
        inputCols = input_features([c + '_one_hot' for c in categorical_features], numerical_features)

    #assert inputCols in df

    assembler = VectorAssembler(inputCols=inputCols, outputCol="features", handleInvalid="skip")
    randomforest = RandomForestClassifier(featuresCol="features", labelCol = labelCol, numTrees=100)
    pipeline = Pipeline(stages=[indexer,one_hot_encoder, assembler, randomforest])

    return pipeline

def get_evaluator(labelCol= "good", metricName = "areaUnderPR") -> BinaryClassificationEvaluator:
    """
    Returns a Binary Classification Evaluator.
    
    Specify labelCol according to your dataset, should be the column name of your target
    Specify metricName according to PySpark Docs 
    """
    return BinaryClassificationEvaluator(labelCol=labelCol, metricName=metricName)
    

if __name__ == "__main__":
    pass