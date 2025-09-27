# -*- coding: utf-8 -*-
"""
Bronze -> Silver
- Read raw CSVs from s3://...-bronze/{customers,products,orders}/
- Apply schemas, cast types, basic cleaning
- Drop obvious duplicates
- Write Parquet to s3://...-silver/{customers,products,orders}/
- Partition orders by order_year for efficient reads
Glue job bookmarking should be enabled via "--job-bookmark-option=job-bookmark-enable"
"""

import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, trim, lower, regexp_replace, to_date, current_timestamp,
    year
)
from pyspark.sql.types import (
    StructType, StructField, IntegerType, StringType, DoubleType, BooleanType, DateType
)
from awsglue.context import GlueContext
from awsglue.job import Job

# --------- Parse parameters ----------
args = getResolvedOptions(sys.argv, [
    "JOB_NAME",
    "bronze_path",   # e.g., s3://de-lake-dev-bronze
    "silver_path"    # e.g., s3://de-lake-dev-silver
])

bronze = args["bronze_path"].rstrip("/")
silver = args["silver_path"].rstrip("/")

# --------- Spark/Glue boilerplate ----------
spark = SparkSession.builder.appName(args["JOB_NAME"]).getOrCreate()
glueContext = GlueContext(spark.sparkContext)
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

ingestion_ts_col = "ingested_at"

# --------- Explicit schemas ----------
customers_schema = StructType([
    StructField("customer_id", IntegerType(), False),
    StructField("first_name", StringType(), True),
    StructField("last_name", StringType(), True),
    StructField("email", StringType(), True),
    StructField("phone", StringType(), True),
    StructField("signup_date", StringType(), True),  # parse later
])

products_schema = StructType([
    StructField("product_id", IntegerType(), False),
    StructField("product_name", StringType(), True),
    StructField("category", StringType(), True),
    StructField("price", StringType(), True),     # parse to double
    StructField("in_stock", StringType(), True),  # parse to boolean
])

orders_schema = StructType([
    StructField("order_id", IntegerType(), False),
    StructField("customer_id", IntegerType(), True),
    StructField("product_id", IntegerType(), True),
    StructField("quantity", IntegerType(), True),
    StructField("order_date", StringType(), True),  # parse to date
    StructField("status", StringType(), True),
])

# --------- Read from bronze (CSV) ----------
customers_bronze = (
    spark.read
         .option("header", True)
         .schema(customers_schema)
         .csv(f"{bronze}/customers*.csv")  # supports customers.csv or folder/customers*.csv
)

products_bronze = (
    spark.read
         .option("header", True)
         .schema(products_schema)
         .csv(f"{bronze}/products*.csv")
)

orders_bronze = (
    spark.read
         .option("header", True)
         .schema(orders_schema)
         .csv(f"{bronze}/orders*.csv")
)

# --------- Basic cleaning / typing ----------
# Customers
customers_silver = (
    customers_bronze
      .withColumn("first_name", trim(col("first_name")))
      .withColumn("last_name", trim(col("last_name")))
      .withColumn("email", lower(trim(col("email"))))
      .withColumn("phone", regexp_replace(trim(col("phone")), r"\s+", ""))
      .withColumn("signup_date", to_date(col("signup_date"), "yyyy-MM-dd"))
      .withColumn(ingestion_ts_col, current_timestamp())
      .dropDuplicates(["customer_id"])
)

# Products
products_silver = (
    products_bronze
      .withColumn("product_name", trim(col("product_name")))
      .withColumn("category", trim(col("category")))
      .withColumn("price", regexp_replace(col("price"), ",", ""))   # safety
      .withColumn("price", col("price").cast(DoubleType()))
      .withColumn("in_stock",
                  (lower(trim(col("in_stock"))).isin("true", "t", "1", "yes")).cast(BooleanType()))
      .withColumn(ingestion_ts_col, current_timestamp())
      .dropDuplicates(["product_id"])
)

# Orders
orders_silver = (
    orders_bronze
      .withColumn("status", trim(col("status")))
      .withColumn("order_date", to_date(col("order_date"), "yyyy-MM-dd"))
      .withColumn("order_year", year(col("order_date")))
      .withColumn(ingestion_ts_col, current_timestamp())
      .filter(col("quantity").isNotNull() & (col("quantity") > 0))
      .dropDuplicates(["order_id"])
)

# --------- Write to silver (Parquet) ----------
# Use "append" so bookmarking can naturally avoid rereads; switch to "overwrite" if needed.
(customers_silver
    .write.mode("append")
    .format("parquet")
    .save(f"{silver}/customers/"))

(products_silver
    .write.mode("append")
    .format("parquet")
    .save(f"{silver}/products/"))

(orders_silver
    .write.mode("append")
    .format("parquet")
    .partitionBy("order_year")
    .format("parquet")
    .save(f"{silver}/orders/"))

job.commit()
