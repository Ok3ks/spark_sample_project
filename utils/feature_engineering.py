from typing import Union
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql import functions as F 

def mult_60(mins):
    """
    This function in essence multiplies by 60.
    Useful for time conversions minutes to seconds, hours to minutes
    """
    return mins * 60

def generate_window(window_in_minutes:int,
                    partition_by:str,
                    timestamp_col:str):
    """This function generates a window, the column specified in orderBy
    is propagated to rangeBetween, Window can also work without orderBy"""
    window = (
        Window().partitionBy(F.col(partition_by))\
        .orderBy(F.col(timestamp_col).cast("long"))\
        .rangeBetween(-mult_60(window_in_minutes), -1))
    return window

#difference between partition by and group by
#- partition by adds column, group by aggregates column
#- partition by examines in more detail with features like 
#rows between, and range between
def generate_rolling_aggregate(col:str,
                                partition_by: Union[str,None] = None,
                                operation: str = "count", 
                                timestamp_col:str = "dt",
                                window_in_minutes:int = 1,):
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