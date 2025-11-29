import sys
import os
import logging
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, hour, regexp_replace
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

# ============================================
# LOGGING CONFIG
# ============================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================
# INITIALIZATION
# ============================================
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'BRONZE_PATH', 'SILVER_PATH', 'CATALOGO_PATH', 'BUCKET'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

BRONZE_PATH = args['BRONZE_PATH']
SILVER_PATH = args['SILVER_PATH']
CATALOGO_PATH = args['CATALOGO_PATH']
BUCKET = args['BUCKET']

logger.info("=" * 60)
logger.info("JOB SILVER - APLICACION DE REGLAS DE NEGOCIO")
logger.info("=" * 60)
logger.info(f"Bucket: {BUCKET}")
logger.info(f"Bronze Path: {BRONZE_PATH}")
logger.info(f"Silver Path: {SILVER_PATH}")
logger.info(f"Catalogos Path: {CATALOGO_PATH}")
logger.info("=" * 60)

# ============================================
# LOGIC
# ============================================

def process_silver():
    logger.info("--- STARTING SILVER PROCESS ---")
    
    # 1. READ BRONZE (PARQUET)
    bronze_input_path = os.path.join(BRONZE_PATH, "TRAFFIC_DATA_BRONZE_PARQUET")
    logger.info(f"Cargando datos desde Bronze: {bronze_input_path}")
    
    df_bronze = spark.read.parquet(bronze_input_path)
    
    # 2. TRANSFORM TIMESTAMP
    # Convert string 'yyyyMMddHHmmss' to Timestamp
    df_bronze = df_bronze.withColumn(
        "timestamp_obj", 
        to_timestamp(col("timestamp"), "yyyyMMddHHmmss")
    )

    # 3. READ & CLEAN CATALOG (COORDINATES)
    full_catalog_path = os.path.join(CATALOGO_PATH, "locationPoints.csv")
    logger.info(f"Leyendo catalogo de: {full_catalog_path}")
    
    catalog_schema = StructType([
        StructField("id", StringType(), True),
        StructField("Coordx", StringType(), True),
        StructField("Coordy", StringType(), True)
    ])

    df_catalog = spark.read.option("header", "true").schema(catalog_schema).csv(full_catalog_path)

    df_catalog_clean = df_catalog.withColumn(
        "Coordx_clean", 
        regexp_replace(col("Coordx"), "[\",]", "").cast(DoubleType())
    ).withColumn(
        "Coordy_clean", 
        regexp_replace(col("Coordy"), "[\",]", "").cast(DoubleType())
    ).withColumn(
        "id_int",
        col("id").cast(IntegerType())
    ).select(
        col("id_int").alias("id_ref"),
        col("Coordx_clean").alias("longitude"),
        col("Coordy_clean").alias("latitude")
    )

    # 4. JOIN TRAFFIC WITH COORDINATES
    df_silver = df_bronze.join(
        df_catalog_clean,
        df_bronze.id == df_catalog_clean.id_ref,
        "left"
    ).drop("id_ref")

    logger.info(f"Silver Data Count: {df_silver.count()}")
    return df_silver

def save_to_silver(df, path):
    logger.info("Guardando datos en formato Parquet (PySpark Native)")
    try:
        df.write.mode("overwrite").partitionBy("timestamp").parquet(path)
        
        logger.info("Datos guardados exitosamente en formato Parquet")
        logger.info(f"Ubicacion: {path}")
        
    except Exception as e:
        logger.error(f"Error al guardar datos en Parquet: {str(e)}")
        raise e

if __name__ == '__main__':
    df_silver = process_silver()
    silver_output_path = os.path.join(SILVER_PATH, "TRAFFIC_DATA_SILVER_PARQUET")
    save_to_silver(df_silver, silver_output_path)
    logger.info("=" * 60)
    logger.info("RESUMEN DE PROCESAMIENTO - CAPA SILVER")
    logger.info("=" * 60)
    logger.info(f"Total de columnas: {len(df_silver.columns)}")
    logger.info("Estado: DATOS PROCESADOS EN SILVER")
    logger.info("=" * 60)
    job.commit()
    logger.info("Job Silver completado exitosamente")
    logger.info("Los datos estan listos para ser procesados en la capa Gold")