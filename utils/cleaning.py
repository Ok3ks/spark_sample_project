import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F 
from pyspark.sql import Window

def preproc(filepath, output_path, save = True):

        files = os.listdir(filepath)
        print(os.path.realpath(files[0]))
        filepaths = [os.path.join(filepath, x) for x in files]

        spark = SparkSession.builder.appName("anomaly_detection").getOrCreate()
        preproc_df = spark.read.option('delimiter', "|").csv(filepaths, header = True)
        
        static_cols = ["local_orig", "local_resp", "missed_bytes", "tunnel_parents"]
        numerical_cols = [
                "duration",
                "orig_bytes",
                "resp_bytes",
                "orig_pkts",
                "orig_ip_bytes",
                "resp_pkts",
                "resp_ip_bytes"]

        categorical_cols = ["proto", "service", "conn_state","label"]

        preproc_df = preproc_df.withColumn(
                        "dt", F.from_unixtime("ts"))\
                        .withColumn("dt", F.to_timestamp("dt"))\
                .withColumnsRenamed(
                        {
                                "id.orig_h": "source_ip",
                                "id.orig_p": "source_port",
                                "id.resp_h": "destination_ip",
                                "id.resp_p": "destination_port",
                        })\
                .withColumns(
                        {
                                "day": F.date_trunc("day", F.col("dt")),
                                "hour": F.date_trunc("hour", F.col("dt")),
                                "minute": F.date_trunc("minute", F.col("dt")),
                                "second": F.date_trunc("second", F.col("dt"))
                        })\
                .drop(*static_cols)\
                .replace("-", None)\
                .withColumns({x: F.col(x).cast("double") for x in numerical_cols})\
                .fillna({x: 'missing' for x in categorical_cols})\
                .fillna({x: -999999 for x in numerical_cols})
        if save == True:
                preproc_df.write.parquet(output_path, mode = 'overwrite')
        return preproc_df

if __name__ == "__main__":
        
        import argparse
        parser = argparse.ArgumentParser(" Cleaning function")
        
        parser.add_argument("filepath")
        parser.add_argument("output_path" )

        args = parser.parse_args()

        spark = preproc(args.filepath, args.output_path)