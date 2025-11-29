import sys
import os
import logging
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import SparkSession
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql.functions import input_file_name, regexp_extract, concat, lit, col, regexp_replace
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType
import glob
import shutil

#spark = SparkSession.builder.appName("AMG Traffic Unification").getOrCreate()

# ============================================
# CONFIGURACIÓN DE LOGGING
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# INICIALIZACIÓN Y LECTURA DE PARÁMETROS
# ============================================
## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, [
    'JOB_NAME',
    'BUCKET',
    'BRONZE_PATH'
])

# Inicializar contexto de Glue
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Obtener parámetros
BUCKET = args['BUCKET']
BRONZE_PATH = args['BRONZE_PATH']

logger.info("=" * 60)
logger.info("JOB BRONZE - CARGA DE DATOS EMPRESARIALES")
logger.info("=" * 60)
logger.info(f"Bucket: {BUCKET}")
logger.info(f"Raw Data Path: {BRONZE_PATH}")
logger.info("=" * 60)

# ============================================
# LEER DATOS CRUDOS Y UNIRLOS
# ============================================

def load_data(base_path: str):
    try:
        path_pattern_2025 = f"{base_path}/2025/gmap_traffic_prediction_*.csv"
        path_pattern_2024 = f"{base_path}/2024/gmap_traffic_prediction_*.csv"
        
        s3_paths_to_read = [path_pattern_2024, path_pattern_2025]
        
        logger.info(f"Rutas S3 a leer: {s3_paths_to_read}")
        
        custom_schema = StructType([
            StructField("id", StringType(), True),
            StructField("predominant_color", StringType(), True),
            StructField("exponential_color_weighting", DoubleType(), True),
            StructField("linear_color_weighting", DoubleType(), True),
            StructField("diffuse_logic_traffic", StringType(), True),
            StructField("timestamp", StringType(), True)
        ])
        
        df_raw = (
            spark.read
            .option("header", "true")
            .option("multiline", "true") # Add this for robust CSV reading
            .schema(custom_schema)
            .csv(s3_paths_to_read) # Pass the list of S3 paths/patterns
            .withColumn("input_file", F.input_file_name())
        )
        
        logger.info(f"Archivos leídos. Conteo de filas inicial: {df_raw.count()}")
        
        # Extraer el timestamp del nombre de archivo and clean/cast id
        df_final = df_raw.withColumn(
            "timestamp",
            F.regexp_extract(F.col("input_file"), r"gmap_traffic_prediction_(\d{14})\.csv", 1)
        ).filter(
            # Filter out any lingering header rows where id might be the string 'id'
            F.col("id") != F.lit("id")
        ).withColumn(
            # Explicitly cast the id column now that we've filtered out string headers
            "id", F.col("id").cast(IntegerType()) 
        ).select(
            "id",
            "predominant_color",
            "exponential_color_weighting",
            "linear_color_weighting",
            "diffuse_logic_traffic",
            "timestamp"
        ).distinct()
        
        logger.info(f"Datos de tráfico cargados exitosamente desde CSV. Filas finales: {df_final.count()}")
        return df_final
    except Exception as e:
        logger.error(f"Error al leer datos empresariales: {str(e)}")
        raise e

# ============================================
# GUARDAR EN FORMATO PARQUET
# ============================================
def save_dataframe(df, path):
    logger.info("Guardando datos en formato Parquet (PySpark Native)")
    try:
        df.write.mode("overwrite").parquet(path)
        
        logger.info("Datos guardados exitosamente en formato Parquet")
        logger.info(f"Ubicacion: {path}")
        
    except Exception as e:
        logger.error(f"Error al guardar datos en Parquet: {str(e)}")
        raise e

if __name__ == '__main__':
    # 01. Load File 
    df = load_data(BRONZE_PATH)
    # 02. save in parquet format
    FILE_PATH_SAVE = os.path.join(BRONZE_PATH, "TRAFFIC_DATA_BRONZE_PARQUET")
    save_dataframe(df, FILE_PATH_SAVE)
    # 03. End Job
    job.commit()
    logger.info("Job Bronze completado exitosamente")
    logger.info("Los datos estan listos para ser procesados en la capa Silver")