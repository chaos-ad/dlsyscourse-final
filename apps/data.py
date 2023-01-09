import os
import logging
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.dataset

from needle.autograd import Tensor

import apps.etl

##############################################################################

logger = logging.getLogger(__name__)

##############################################################################

class Criteo1TBDatasetBatchIter:
    def __init__(
        self,
        s3_path,
        s3_prefix=None, 
        s3_bucket=None,
        day_from = 0,
        day_to = 23,
        batch_size=1024,
        columns = apps.etl.TARGET_COLUMNS + apps.etl.DENSE_COLUMNS,
        filter = None,
        limit_records = 0,
        limit_batches = 0,
        as_numpy = False
    ):
        logger.debug("initializing new iterator...")
        s3_bucket = s3_bucket or os.environ['AWS_S3_BUCKET']
        s3_prefix = s3_prefix or os.environ['AWS_S3_PREFIX']
        self.s3_url = f"s3://{s3_bucket}/{os.path.join(s3_prefix, s3_path)}"
        self.dataset = pyarrow.dataset.dataset(self.s3_url, partitioning="hive")
        self.filter = (pyarrow.dataset.field('day') >= day_from) & (pyarrow.dataset.field('day') <= day_to)
        self.records = self.dataset.count_rows(filter=self.filter)
        self.batch_size = batch_size
        self.columns = apps.etl.TARGET_COLUMNS + apps.etl.DENSE_COLUMNS + apps.etl.SPARSE_IDX_COLUMNS
        self.limit_batches = limit_batches
        self.limit_records = limit_records
        self.as_numpy = as_numpy

    def __iter__(self):
        logger.debug("starting iteration...")
        self.batch_iter = self.dataset.to_batches(columns=self.columns, filter=self.filter, batch_size=self.batch_size)
        self.read_batches = 0
        self.read_records = 0
        return self

    def __next__(self):
        logger.debug("fetching next batch...")
        if self.limit_batches > 0 and self.read_batches >= self.limit_batches:
            raise StopIteration()
        if self.limit_records > 0 and self.read_records >= self.limit_records:
            raise StopIteration()
        batch = next(self.batch_iter).to_pandas()
        self.read_batches += 1
        self.read_records += batch.shape[0]
        Y = batch[apps.etl.TARGET_COLUMN].to_numpy()
        X_dense = batch[apps.etl.DENSE_COLUMNS].to_numpy()
        X_sparse = batch[apps.etl.SPARSE_IDX_COLUMNS].to_numpy()
        if not self.as_numpy:
            Y = Tensor(Y, requires_grad=False)
            X_dense = Tensor(X_dense, requires_grad=False)
            X_sparse = Tensor(X_sparse, requires_grad=False)
        return (X_dense, X_sparse, Y)

    def __len__(self) -> int:
        return self.records

def read_s3_dataset(**kwargs):
    return Criteo1TBDatasetBatchIter(**kwargs)

##############################################################################
