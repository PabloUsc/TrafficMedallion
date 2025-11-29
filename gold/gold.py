import sys
import os
import logging
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ============================================
#CONFIGURACIÓN DE LOGGING
# ============================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================
# INICIALIZACIÓN Y LECTURA DE PARÁMETROS
# ============================================
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'BUCKET', 'SILVER_PATH', 'GOLD_PATH'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

GOLD_PATH = args['GOLD_PATH']
SILVER_PATH = args['SILVER_PATH']
BUCKET = args['BUCKET']

logger.info("=" * 60)
logger.info("JOB GOLD - CREACION DEL MODELO DIMENSIONAL")
logger.info("=" * 60)
logger.info(f"Bucket: {BUCKET}")
logger.info(f"Silver Path: {SILVER_PATH}")
logger.info(f"Gold Path: {GOLD_PATH}")
logger.info("=" * 60)

def process_gold_time_series():
    logger.info("--- INICIAR PROCESO GOLD (TIME SERIES) ---")
    
    # 1. READ SILVER
    silver_input_path = os.path.join(SILVER_PATH, "TRAFFIC_DATA_SILVER_PARQUET")
    df_silver = spark.read.parquet(silver_input_path)
    
    # 2. PREPARE COLUMNS
    df_transformed = df_silver.withColumn(
        "ts_string", 
        F.col("timestamp").cast("string")
    ).withColumn(
        "timestamp_obj",
        F.unix_timestamp(F.col("ts_string"), "yyyyMMddHHmmss").cast("timestamp")
    ).withColumn(
        "traffic_date",
        F.to_date(F.col("timestamp_obj"))
    ).withColumn(
        "hour_extracted",
        F.hour(F.col("timestamp_obj"))
    )
    
    # 3. FILTER HOURS OF INTEREST
    # Request: 7am, 9am, 11am, 13pm, 15pm, 17pm, 19pm, 21pm, 23pm, 1am
    target_hours = [1, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    
    #df_with_hour = df_silver.withColumn("hour_extracted", F.hour(F.col("timestamp_obj")))
    #df_filtered = df_with_hour.filter(F.col("hour_extracted").isin(target_hours))
    
    df_filtered = df_transformed.filter(F.col("hour_extracted").isin(target_hours))
    
    # 4. PIVOT TO CREATE TIME SERIES FORMAT
    df_pivoted = df_filtered.groupBy("id", "traffic_date") \
        .pivot("hour_extracted", target_hours) \
        .agg(F.avg("exponential_color_weighting")) \
        .na.fill(0)
    
    # 5. RENAME COLUMNS FOR CLARITY
    for h in target_hours:
        old_col_name = str(h)
        new_col_name = f"traffic_h{h:02d}" # e.g., traffic_h07
        df_pivoted = df_pivoted.withColumnRenamed(old_col_name, new_col_name)
        
    logger.info(f"Time Series creada. Número de filas: {df_pivoted.count()}")
    return df_pivoted

def save_gold(df, path):
    logger.info(f"Guardando Gold Time Series en: {path}")
    df.coalesce(1).write.mode("overwrite").partitionBy("traffic_date").parquet(path)

if __name__ == '__main__':
    df_gold = process_gold_time_series()
    df_gold_path = os.path.join(GOLD_PATH, "trafico_dia")
    save_gold(df_gold, df_gold_path)
    job.commit()
    logger.info("Job Gold completado exitosamente")