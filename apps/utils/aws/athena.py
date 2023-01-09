import os
import boto3
import logging
import awswrangler

import apps.utils.aws.s3

##############################################################################

logger = logging.getLogger(__name__)

##############################################################################

def run_query(query, database=None, ctas_approach=False, **kwargs):
    database = database or os.environ['AWS_ATHENA_DB']
    logger.debug(f"executing athena query...\n{query}")
    result = awswrangler.athena.read_sql_query(
        query,
        database=database,
        ctas_approach=ctas_approach,
        **kwargs
    )
    logger.debug(f"executing athena query: done ({result.shape[0] if result is not None else 0} rows)")
    return result

def create_parquet_table(table, s3_path, s3_bucket=None, database=None, drop_if_exists=True, drop_s3_data=False, **kwargs):
    database = database or os.environ['AWS_ATHENA_DB']
    if drop_if_exists:
        drop_table(table, database)
        if drop_s3_data:
            apps.utils.aws.s3.drop_objects(prefix=s3_path, bucket=s3_bucket)
    logger.debug(f"creating athena table '{database}.{table}' -> '{s3_path}'...")
    result = awswrangler.catalog.create_parquet_table(
        database = database,
        table = table,
        path = f's3://{s3_bucket}/{s3_path}',
        **kwargs
    )
    logger.debug(f"creating athena table '{database}.{table}' -> '{s3_path}': done")
    return result

def create_database(database=None, **kwargs):
    database = database or os.environ['AWS_ATHENA_DB']
    logger.info(f"creating athena database '{database}'...")
    result = awswrangler.catalog.create_database(name=database, **kwargs)
    logger.info(f"creating athena database '{database}': done")
    return result

def get_partitions(table, database=None, **kwargs):
    database = database or os.environ['AWS_ATHENA_DB']
    return awswrangler.catalog.get_partitions(database=database, table=table, **kwargs)

def drop_table(table, database=None):
    database = database or os.environ['AWS_ATHENA_DB']
    logger.info(f"dropping athena table '{database}.{table}'...")
    if awswrangler.catalog.does_table_exist(table=table, database=database):
        awswrangler.catalog.delete_table_if_exists(table=table, database=database)
        logger.info(f"dropping athena table '{database}.{table}': done")
    else:
        logger.info(f"dropping athena table '{database}.{table}': table doesn't exist")

##############################################################################
