# -*- coding: utf-8 -*-
"""
Silver -> Gold
- Read Parquet from s3://...-silver/{customers,products,orders}/
- Build fact_orders_enriched (join orders + customers + products)
- Add total_price = quantity * price
- Create two small aggregates:
    1) daily_sales_by_category
    2) customer_lifetime_value
- Write Parquet to s3://...-gold/{fact_orders_enriched, agg_daily_sales_by_category, agg_customer_ltv}/
Glue job bookmarking should be enabled globally; this job is idempotent due to Parquet append + join keys.
"""

import sys
from awsglue.utils import getResolvedOptions
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, coalesce, lit, current_timestamp, date_format, sum as _sum, count as _count
)
from awsglue.context import GlueContext
from awsglue.job import Job

args = getResolvedOptions(sys.argv, [
    "JOB_NAME",
    "silver_path",   # e.g., s3://de-lake-dev-silver
    "gold_path"      # e.g., s3://de-lake-dev-gold
])

silver = args["silver_path"].rstrip("/")
gold = args["gold_path"].rstrip("/")

spark = SparkSession.builder.appName(args["JOB_NAME"]).getOrCreate()
glueContext = GlueContext(spark.sparkContext)
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

now_col = "processed_at"

# --------- Load silver data ----------
customers = spark.read.parquet(f"{silver}/customers/")
products  = spark.read.parquet(f"{silver}/products/")
orders    = spark.read.parquet(f"{silver}/orders/")

# --------- Enrich / conform ----------
# minimal conformance to ensure join keys are present and types correct
customers_slim = customers.select(
    col("customer_id").alias("c_customer_id"),
    "first_name", "last_name", "email", "signup_date"
)

products_slim = products.select(
    col("product_id").alias("p_product_id"),
    "product_name", "category", "price", "in_stock"
)

orders_slim = orders.select(
    "order_id", "customer_id", "product_id", "quantity", "order_date", "status"
)

# --------- Fact: orders enriched ----------
fact_orders = (
    orders_slim
      .join(customers_slim, orders_slim.customer_id == customers_slim.c_customer_id, "left")
      .join(products_slim, orders_slim.product_id == products_slim.p_product_id, "left")
      .withColumn("total_price", (col("quantity") * coalesce(col("price"), lit(0.0))))
      .withColumn(now_col, current_timestamp())
)

# --------- Aggregates ----------
# 1) Daily sales by category (sum total_price, count orders)
daily_sales_by_category = (
    fact_orders
      .groupBy("order_date", "category")
      .agg(
          _sum("total_price").alias("sum_total_price"),
          _count("order_id").alias("order_count")
      )
      .withColumn(now_col, current_timestamp())
)

# 2) Customer Lifetime Value (sum total_price per customer)
customer_ltv = (
    fact_orders
      .groupBy("customer_id", "first_name", "last_name", "email")
      .agg(_sum("total_price").alias("ltv_total"))
      .withColumn(now_col, current_timestamp())
)

# --------- Write to gold (Parquet) ----------
# Partition fact by order_date (YYYY-MM) for common analytics patterns
(fact_orders
    .withColumn("order_ym", date_format(col("order_date"), "yyyy-MM"))
    .write.mode("append")
    .partitionBy("order_ym")
    .format("parquet")
    .save(f"{gold}/fact_orders_enriched/"))

(daily_sales_by_category
    .write.mode("append")
    .format("parquet")
    .save(f"{gold}/agg_daily_sales_by_category/"))

(customer_ltv
    .write.mode("append")
    .format("parquet")
    .save(f"{gold}/agg_customer_ltv/"))

job.commit()
