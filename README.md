# spark_anomaly_detection
Repository detailing my Pyspark learnings. Notebooks are exploratory, utils has modularized code.

##
Unable to push the data folder to github because of the bandwidth limitation both through git and git LFS.


## Observations
- Memory management is key in Spark, It operates through with a technique called Lazy Processing.

- Driver memory matters a lot for broadcast joins, column operations. This article provides a rule of thumb for reserving memory

https://umbertogriffo.gitbook.io/apache-spark-best-practices-and-tuning/parallelism/how-to-estimate-the-number-of-partitions-executors-and-drivers-params-yarn-cluster-mode

https://www.clairvoyant.ai/blog/apache-spark-out-of-memory-issue

- The ability to write SQL within Spark is one I like.

- SparkContext is a portal to a lot of Spark's configurations.

- Parquets are very useful because of predicate pushdown, ability to store schemas.

- Equally divided parquets load faster than one giant parquet file 
- Understanding how to manage infrastructure is equally important as deep modelling methods.

- Rows are unordered because spark distributes data across clusters, and processes parallely. F.monotonically_increasing_id can be used to add index but this also numbers by partitions. F.row_number() works but a window has to be specified.

- Pipelining method is neat, Categorical features should be passed through a StringIndexer first, then a OneHotEncoder before model training. while categorical labels only need to be passed through a StringIndexer. This to prevent the model from adding numerical weights to categorical variables.

- Seaborn FacetGrid to investigate distribution of categorical variable plots

- In hyperparameter search with Metaflow, initiation of Spark drivers should all be done in one step, rather than being split over multiple steps. This is because the spark driver cannot be executed over workers, especially when the foreach feature is used.  

- Difference between partition by and group by. Partition by adds column, group by aggregates column. Partition by examines in more detail with features like rows between, and range between

## Links 
- https://sparkbyexamples.com/spark/

- #https://medium.com/illumination/managing-memory-and-disk-resources-in-pyspark-with-cache-and-persist-aa08026929e2#:~:text=uncache()%20%3A%20This%20method%20is,if%20it%20is%20needed%20again.

- https://umbertogriffo.gitbook.io/apache-spark-best-practices-and-tuning/parallelism/how-to-estimate-the-number-of-partitions-executors-and-drivers-params-yarn-cluster-mode

## To do 
Manual Hyperparameter Tuning with Metaflow
- 