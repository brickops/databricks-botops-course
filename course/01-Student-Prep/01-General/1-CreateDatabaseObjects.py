# Databricks notebook source
# MAGIC %md
# MAGIC # What's in this exercise?
# MAGIC
# MAGIC 1) Create a student-specific database<BR>

# COMMAND ----------

# MAGIC %md ### 0. Select Serverless as compute
# MAGIC
# MAGIC Press the Connect dropdown in the upper left menu, and select Serverless
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Create tripbot database in Databricks
# MAGIC
# MAGIC Specific for the student

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.1. Create schemas (AKA databases)

# COMMAND ----------

# %pip install brickops=0.3.16
# %pip install git+https://github.com/brickops/brickops.git@feature/file-cfg
%pip install brickops=0.3.16

# COMMAND ----------

# Restart python to access updated packages
dbutils.library.restartPython()

# COMMAND ----------

from brickops.datamesh.naming import dbname

# COMMAND ----------

catalog = "transport"
tripbot_db = dbname(cat=catalog, db="tripbot")
tripbot_db

# COMMAND ----------

catalog = "transport"
tripbot_db = dbname(cat=catalog, db="tripbot")
print("New db name: " + tripbot_db)
spark.sql(f"USE catalog {catalog}")
spark.sql(f"CREATE DATABASE IF NOT EXISTS {tripbot_db}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.2. Validate

# COMMAND ----------

# MAGIC %sql
# MAGIC USE CATALOG transport;
# MAGIC SHOW DATABASES;

# COMMAND ----------


