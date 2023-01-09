import os
import logging
import subprocess

from collections import OrderedDict

import apps.utils.aws.s3

##############################################################################

logger = logging.getLogger(__name__)

##############################################################################

TARGET_COLUMN = 'label'
TARGET_COLUMNS = ['label']
DENSE_COLUMNS = [f'feature_{idx:02d}' for idx in range(1, 14)]
SPARSE_COLUMNS = [f'feature_{idx:02d}' for idx in range(14, 40)]
SPARSE_IDX_COLUMNS = [f'feature_{idx:02d}_idx' for idx in range(14, 40)]
ALL_COLUMNS = TARGET_COLUMNS + DENSE_COLUMNS + SPARSE_COLUMNS

##############################################################################

RAW_DATA_S3_PATH = "criteo/raw"
RAW_DATA_ATHENA_TABLE = "criteo_raw"

PARQUET_DATA_S3_PATH = "criteo/parquet"
PARQUET_DATA_ATHENA_TABLE = "criteo_parquet"

PARSED_DATA_S3_PATH = "criteo/parsed"
PARSED_DATA_ATHENA_TABLE = "criteo_parsed"

##############################################################################

def get_s3_url(s3_path, s3_prefix=None, s3_bucket=None):
    s3_bucket = s3_bucket or os.environ['AWS_S3_BUCKET']
    s3_prefix = s3_prefix or os.environ['AWS_S3_PREFIX']
    s3_fullpath = os.path.join(s3_prefix, s3_path)
    return f"s3://{s3_bucket}/{s3_fullpath}"

##############################################################################

def import_criteo_day(day, s3_fullpath, s3_bucket=None):
    s3_bucket = s3_bucket or os.environ['AWS_S3_BUCKET']
    src_url = f"https://storage.googleapis.com/criteo-cail-datasets/day_{day}.gz"
    dst_url = f"s3://{s3_bucket}/{s3_fullpath}/day={day:02d}/data.tsv.gz"
    import_cmd = f"curl --silent '{src_url}' | tqdm --desc='copying day {day}' --bytes | aws s3 cp - '{dst_url}'"
    logger.debug(f"running import cmd: '{import_cmd}'")
    subprocess.run(import_cmd, shell=True, check=True)

def import_criteo_dataset(s3_path=RAW_DATA_S3_PATH, day_from=0, day_to=23, s3_bucket=None):
    s3_prefix = os.environ['AWS_S3_PREFIX']
    s3_fullpath = os.path.join(s3_prefix, s3_path)
    for day in range(day_from, day_to+1):
        import_criteo_day(day, s3_fullpath, s3_bucket)

##############################################################################

def get_criteo_dataset_tables(s3_path=RAW_DATA_S3_PATH, s3_bucket=None):
    s3_prefix = os.environ['AWS_S3_PREFIX']
    s3_fullpath = os.path.join(s3_prefix, s3_path)
    return apps.utils.aws.s3.ls(s3_fullpath, s3_bucket)

##############################################################################

def init_database(athena_database=None):
    apps.utils.aws.athena.create_database(athena_database, exist_ok=True)

def init_raw_athena_table(s3_path=RAW_DATA_S3_PATH, athena_table_name=RAW_DATA_ATHENA_TABLE, athena_database=None, s3_bucket=None):
    s3_bucket = os.environ['AWS_S3_BUCKET']
    s3_prefix = os.environ['AWS_S3_PREFIX']
    s3_fullpath = os.path.join(s3_prefix, s3_path)
    apps.utils.aws.athena.drop_table(database=athena_database, table=athena_table_name)
    apps.utils.aws.athena.run_query(
        query=f"""
            CREATE EXTERNAL TABLE {athena_table_name} (
                label int,
                feature_01 int,
                feature_02 int,
                feature_03 int,
                feature_04 int,
                feature_05 int,
                feature_06 int,
                feature_07 int,
                feature_08 int,
                feature_09 int,
                feature_10 int,
                feature_11 int,
                feature_12 int,
                feature_13 int,
                feature_14 string,
                feature_15 string,
                feature_16 string,
                feature_17 string,
                feature_18 string,
                feature_19 string,
                feature_20 string,
                feature_21 string,
                feature_22 string,
                feature_23 string,
                feature_24 string,
                feature_25 string,
                feature_26 string,
                feature_27 string,
                feature_28 string,
                feature_29 string,
                feature_30 string,
                feature_31 string,
                feature_32 string,
                feature_33 string,
                feature_34 string,
                feature_35 string,
                feature_36 string,
                feature_37 string,
                feature_38 string,
                feature_39 string
            )
            PARTITIONED BY (day string)
            ROW FORMAT DELIMITED
                FIELDS TERMINATED BY '\\t'
                ESCAPED BY '\\\\'
                LINES TERMINATED BY '\\n'
            STORED AS
                INPUTFORMAT 'org.apache.hadoop.mapred.TextInputFormat' 
                OUTPUTFORMAT 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat'
            LOCATION 's3://{s3_bucket}/{s3_fullpath}'
        """, 
        database = athena_database
    )
    apps.utils.aws.athena.run_query(f"MSCK REPAIR TABLE {athena_table_name}", database=athena_database)

def init_parquet_athena_table(day_from=0, day_to=23, s3_path=PARQUET_DATA_S3_PATH, parquet_table=PARQUET_DATA_ATHENA_TABLE, raw_table=RAW_DATA_ATHENA_TABLE, athena_database=None, s3_bucket=None):
    s3_bucket = os.environ['AWS_S3_BUCKET']
    s3_prefix = os.environ['AWS_S3_PREFIX']
    s3_fullpath = os.path.join(s3_prefix, s3_path)
    apps.utils.aws.athena.create_parquet_table(
        database = athena_database,
        table = parquet_table,
        s3_bucket = s3_bucket,
        s3_path = s3_fullpath,
        drop_if_exists = True,
        drop_s3_data = True,
        columns_types = OrderedDict(
            [('label', 'int')] +
            [(feature_name, 'int') for feature_name in DENSE_COLUMNS] +
            [(feature_name, 'string') for feature_name in SPARSE_COLUMNS]
        ),
        partitions_types={'day': 'string'}
    )

    logger.info(f"converting raw data to parquet...")
    apps.utils.aws.athena.run_query(
        query=f"""
            INSERT INTO {parquet_table}
            SELECT * from {raw_table}
            where day >= '{day_from:02d}'
              and day <= '{day_to:02d}'
        """, 
        database = athena_database
    )
    logger.info(f"converting raw data to parquet: done")


##############################################################################

def init_parsed_athena_table(s3_path=PARSED_DATA_S3_PATH, athena_table_name=PARSED_DATA_ATHENA_TABLE, athena_database=None, s3_bucket=None):
    s3_bucket = os.environ['AWS_S3_BUCKET']
    s3_prefix = os.environ['AWS_S3_PREFIX']
    s3_fullpath = os.path.join(s3_prefix, s3_path)
    apps.utils.aws.athena.create_parquet_table(
        database = athena_database,
        table = athena_table_name,
        s3_bucket = s3_bucket,
        s3_path = s3_fullpath,
        drop_if_exists = True,
        drop_s3_data = True,
        columns_types = OrderedDict(
            [('label', 'int')] +
            [(feature_name, 'int') for feature_name in DENSE_COLUMNS] +
            [(feature_name, 'bigint') for feature_name in SPARSE_COLUMNS]
        ),
        partitions_types={'day': 'string'},
        compression='snappy'
    )

def parse_criteo_day(day, raw_table=RAW_DATA_ATHENA_TABLE, parsed_table=PARSED_DATA_ATHENA_TABLE, athena_database=None):
    logger.info(f"parsing raw data for day='{day:02d}'...")
    apps.utils.aws.athena.run_query(
        query=f"""
            INSERT INTO {parsed_table}
            SELECT
                label,
                feature_01 as feature_01,
                feature_02 as feature_02,
                feature_03 as feature_03,
                feature_04 as feature_04,
                feature_05 as feature_05,
                feature_06 as feature_06,
                feature_07 as feature_07,
                feature_08 as feature_08,
                feature_09 as feature_09,
                feature_10 as feature_10,
                feature_11 as feature_11,
                feature_12 as feature_12,
                feature_13 as feature_13,
                if(coalesce(feature_14, '') != '', from_base(feature_14, 16)) as feature_14,
                if(coalesce(feature_15, '') != '', from_base(feature_15, 16)) as feature_15,
                if(coalesce(feature_16, '') != '', from_base(feature_16, 16)) as feature_16,
                if(coalesce(feature_17, '') != '', from_base(feature_17, 16)) as feature_17,
                if(coalesce(feature_18, '') != '', from_base(feature_18, 16)) as feature_18,
                if(coalesce(feature_19, '') != '', from_base(feature_19, 16)) as feature_19,
                if(coalesce(feature_20, '') != '', from_base(feature_20, 16)) as feature_20,
                if(coalesce(feature_21, '') != '', from_base(feature_21, 16)) as feature_21,
                if(coalesce(feature_22, '') != '', from_base(feature_22, 16)) as feature_22,
                if(coalesce(feature_23, '') != '', from_base(feature_23, 16)) as feature_23,
                if(coalesce(feature_24, '') != '', from_base(feature_24, 16)) as feature_24,
                if(coalesce(feature_25, '') != '', from_base(feature_25, 16)) as feature_25,
                if(coalesce(feature_26, '') != '', from_base(feature_26, 16)) as feature_26,
                if(coalesce(feature_27, '') != '', from_base(feature_27, 16)) as feature_27,
                if(coalesce(feature_28, '') != '', from_base(feature_28, 16)) as feature_28,
                if(coalesce(feature_29, '') != '', from_base(feature_29, 16)) as feature_29,
                if(coalesce(feature_30, '') != '', from_base(feature_30, 16)) as feature_30,
                if(coalesce(feature_31, '') != '', from_base(feature_31, 16)) as feature_31,
                if(coalesce(feature_32, '') != '', from_base(feature_32, 16)) as feature_32,
                if(coalesce(feature_33, '') != '', from_base(feature_33, 16)) as feature_33,
                if(coalesce(feature_34, '') != '', from_base(feature_34, 16)) as feature_34,
                if(coalesce(feature_35, '') != '', from_base(feature_35, 16)) as feature_35,
                if(coalesce(feature_36, '') != '', from_base(feature_36, 16)) as feature_36,
                if(coalesce(feature_37, '') != '', from_base(feature_37, 16)) as feature_37,
                if(coalesce(feature_38, '') != '', from_base(feature_38, 16)) as feature_38,
                if(coalesce(feature_39, '') != '', from_base(feature_39, 16)) as feature_39,
                day
            from {raw_table}
            where day = '{day:02d}'
        """, 
        database = athena_database
    )
    logger.info(f"parsing raw data for day='{day:02d}': done")

def parse_criteo_dataset(day_from=0, day_to=23, athena_database=None, s3_bucket=None):
    init_database(athena_database)
    init_raw_athena_table(athena_database=athena_database, s3_bucket=s3_bucket)
    init_parsed_athena_table(athena_database=athena_database, s3_bucket=s3_bucket)
    for day in range(day_from, day_to+1):
        parse_criteo_day(day, athena_database=athena_database)

##############################################################################

