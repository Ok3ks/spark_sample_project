# spark_anomaly_detection
Repository detailing my Pyspark learnings


## Observations
- Memory management is key in Spark, It operates through with a technique called Lazy Processing
- Driver memory matters a lot for broadcast joins, column operations 
- The ability to write SQL within Spark is one I like 
- SparkContext is a portal to a lot of Spark's configurations
- Parquets are very useful because of predicate pushdown, ability to store schemas
- Equally divided parquets load faster than one giant parquet file 
- Understanding how to manage infrastructure is equally important as deep modelling methods
- Rows are unordered because spark distributes data across clusters, and processes parallely. F.monotonically_increasing_id can be used to add index but this also numbers by partitions. F.row_number() works but a window has to be specified
- Pipelining method is neat, Categorical features should be passed through a StringIndexer first, then a OneHotEncoder before model training. while categorical labels only need to be passed through a StringIndexer
- Seaborn FacetGrid to investigate distribution of categorical variable plots

## Links 
- https://sparkbyexamples.com/spark/

## To do 
Manual Hyperparameter Tuning with Metaflow
