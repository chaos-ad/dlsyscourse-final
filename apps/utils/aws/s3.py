import os
import boto3
import logging

##############################################################################

logger = logging.getLogger(__name__)

##############################################################################

def ls(path, bucket=None):
    bucket = bucket or os.environ['AWS_S3_BUCKET']
    logger.debug(f"listing s3 objects 's3://{bucket}/{path}'...")
    s3 = boto3.resource('s3')
    result = s3.Bucket(bucket).objects.filter(Prefix=path)
    result = [obj.key for obj in result]
    logger.debug(f"listing s3 objects 's3://{bucket}/{path}': done ({len(result)} objects)")
    return result

def drop_objects(prefix, bucket=None):
    bucket = bucket or os.environ['AWS_S3_BUCKET']
    s3 = boto3.resource('s3')
    logger.info(f"deleting s3 objects 's3://{bucket}/{prefix}'...")
    result = s3.Bucket(bucket).objects.filter(Prefix=prefix).delete()
    logger.info(f"deleting s3 objects 's3://{bucket}/{prefix}': done")

##############################################################################
