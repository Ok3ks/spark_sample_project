from typing import Union
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql import functions as F 
from pyspark.ml import Pipeline

def mult_60(mins):
    """
    This function in essence multiplies by 60.
    Useful for time conversions minutes to seconds, hours to minutes
    """
    return mins * 60

def generate_window(window_in_minutes:int,
                    partition_by:str,
                    timestamp_col:str) -> Window:
    """This function generates a window, the column specified in orderBy
    is propagated to rangeBetween, Window can also work without orderBy"""
    window = (
        Window().partitionBy(F.col(partition_by))\
        .orderBy(F.col(timestamp_col).cast("long"))\
        .rangeBetween(-mult_60(window_in_minutes), -1))
    return window

def generate_rolling_aggregate(col:str,
                                partition_by: Union[str,None] = None,
                                operation: str = "count", 
                                timestamp_col:str = "dt",
                                window_in_minutes:int = 1,):
    
    """ This function generates a rolling aggregate given an 
    operation which can be 'count', 'sum' or 'avg' """
    
    if partition_by == None :
        partition_by = col
    
    if operation == "count":
        return F.count(F.col(col)).over(
            generate_window(window_in_minutes = window_in_minutes,
                            partition_by=partition_by,
                            timestamp_col= timestamp_col))
    elif operation == "sum":
        return F.sum(F.col(col)).over(
            generate_window(window_in_minutes = window_in_minutes,
                            partition_by=partition_by,
                            timestamp_col= timestamp_col))
    elif operation == "avg":
        return F.avg(F.col(col)).over(
            generate_window(window_in_minutes = window_in_minutes,
                            partition_by=partition_by,
                            timestamp_col= timestamp_col))
    else:
        raise ValueError(f"Operation {operation} is not defined")
    
def numerical_features() -> list:
    "Returns a list of numerical features"
    
    return ['duration', 'orig_bytes', 'resp_bytes',
            'orig_pkts','orig_ip_bytes','resp_pkts',
            'resp_ip_bytes', 'source_ip_count_last_min',
            'source_ip_count_last_30_min']

def categorical_features() -> list:
    "Returns a list of categorical features"

    return ["proto", "service", "conn_state"]

def categorical_features_indexed(features = categorical_features()) -> list:
    "Returns a list of categorical features indexed"

    categorical_features_indexed = [c + "_index" for c in features]
    return categorical_features_indexed

def one_hot_encoded_index(features = categorical_features()) -> list:
    "Returns one hot encoded index"
    one_hot_encoded_index = [c + "_one_hot" for c in features]
    return one_hot_encoded_index

def input_features(*features:list) -> list:
    " Returns all columns to be used as features"
    
    temp = []
    for a in features:
        temp.extend(a)
    return temp

def compute_rolling_aggregate() -> dict:
    "Computes rolling aggregate of a minutes and last 30 minutes"

    return {
        "source_ip_avg_bytes_last_min": generate_rolling_aggregate(col="orig_ip_bytes", operation="avg",timestamp_col="dt",window_in_minutes=1),
        "source_ip_avg_bytes_last_30_min": generate_rolling_aggregate(col="orig_ip_bytes", operation="avg",timestamp_col="dt",window_in_minutes=30),
        "dest_ip_count_last_min": generate_rolling_aggregate(col="destination_ip", operation="count",timestamp_col="dt",window_in_minutes=1),
        "dest_ip_count_last_30_min": generate_rolling_aggregate(col="destination_port", operation="count",timestamp_col="dt",window_in_minutes=30),
        "dest_port_count_last_min": generate_rolling_aggregate(col="destination_ip", operation="count",timestamp_col="dt",window_in_minutes=1),
        "dest_port_count_last_30_min": generate_rolling_aggregate(col="destination_port", operation="count",timestamp_col="dt",window_in_minutes=30),
        "source_ip_avg_pkts_last_min": generate_rolling_aggregate(col="orig_pkts",partition_by="source_ip", operation="avg",timestamp_col="dt",window_in_minutes=1),
        "source_ip_avg_pkts_last_30_min": generate_rolling_aggregate(col="orig_pkts",partition_by="source_ip",operation="avg",timestamp_col="dt",window_in_minutes=30)
    }

def get_feature_importance(pipeline:Pipeline) -> dict :

    return {
        "importance": list(pipeline.stages[-1].featureImportances),
        "feature": pipeline.stages[-2].getInputCols(),
    }