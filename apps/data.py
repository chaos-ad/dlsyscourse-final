import os
import logging
import numpy as np
import pandas as pd

import pyarrow
import pyarrow.dataset

import torch
import torchrec.sparse.jagged_tensor

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
        target_column = apps.etl.TARGET_COLUMN,
        dense_columns = apps.etl.DENSE_COLUMNS,
        sparse_columns = apps.etl.SPARSE_IDX_COLUMNS,
        sparse_buckets = None,
        sparse_limits = None,
        filter = None,
        limit_records = 0,
        limit_batches = 0,
        output_format = "needle",
        device = None
    ):
        logger.debug("initializing new iterator...")
        s3_bucket = s3_bucket or os.environ['AWS_S3_BUCKET']
        s3_prefix = s3_prefix or os.environ['AWS_S3_PREFIX']
        self.s3_url = f"s3://{s3_bucket}/{os.path.join(s3_prefix, s3_path)}"
        self.dataset = pyarrow.dataset.dataset(self.s3_url, partitioning="hive")
        self.filter = (pyarrow.dataset.field('day') >= day_from) & (pyarrow.dataset.field('day') <= day_to)
        self.records = self.dataset.count_rows(filter=self.filter)
        self.batch_size = batch_size
        self.columns = [target_column] + dense_columns + sparse_columns
        self.limit_batches = limit_batches
        self.limit_records = limit_records
        self.output_format = output_format
        self.device = device
        self.sparse_limits = self._init_sparse_attribute(sparse_limits, sparse_columns)
        self.sparse_buckets = self._init_sparse_attribute(sparse_buckets, sparse_columns)
        num_ids_in_batch = len(apps.etl.SPARSE_IDX_COLUMNS) * batch_size
        self.lengths = torch.ones((num_ids_in_batch * 2,), dtype=torch.int32, device=self.device)
        self.offsets = torch.arange(0, num_ids_in_batch * 2 + 1, dtype=torch.int32, device=self.device)
    
    @staticmethod
    def _init_sparse_attribute(attr, sparse_columns):
        if attr is not None:
            if isinstance(attr, int):
                return [attr] * len(sparse_columns)
            else:
                assert isinstance(attr, list) and len(attr) == len(sparse_columns)
                return np.array(attr).reshape((1, len(sparse_columns)))
        else:
            return None

    def __iter__(self):
        logger.debug("starting iteration...")
        self.batch_iter = self.dataset.to_batches(columns=self.columns, filter=self.filter, batch_size=self.batch_size)
        self.read_batches = 0
        self.read_records = 0
        return self
    
    def to_torch(self, batch):
        (X_dense, X_sparse, Y_true) = batch

        batch_size = Y_true.shape[0]

        X_dense = torch.tensor(X_dense, dtype=torch.float32, device=self.device)

        CAT_FEATURE_COUNT = len(apps.etl.SPARSE_IDX_COLUMNS)
        num_ids_in_batch = CAT_FEATURE_COUNT * batch_size
        if batch_size != self.batch_size:
            lengths = torch.ones((num_ids_in_batch * 2,), dtype=torch.int32, device=self.device)
            offsets = torch.arange(0, num_ids_in_batch * 2 + 1, dtype=torch.int32, device=self.device)
        else:
            lengths = self.lengths
            offsets = self.offsets

        length_per_key = CAT_FEATURE_COUNT * [batch_size]
        offset_per_key = [batch_size * i for i in range(CAT_FEATURE_COUNT + 1)]
        index_per_key = {key: i for (i, key) in enumerate(apps.etl.SPARSE_IDX_COLUMNS)}

        X_sparse = torchrec.sparse.jagged_tensor.KeyedJaggedTensor(
            keys = apps.etl.SPARSE_IDX_COLUMNS,
            values = torch.from_numpy(X_sparse.transpose(1, 0).reshape(-1)).to(self.device),
            lengths = lengths[:num_ids_in_batch],
            offsets = offsets[: num_ids_in_batch + 1],
            stride = batch_size,
            length_per_key=length_per_key,
            offset_per_key=offset_per_key,
            index_per_key=index_per_key
        )
        Y_true = torch.tensor(Y_true.reshape(-1), dtype=torch.float32, device=self.device)

        batch = (X_dense, X_sparse, Y_true)
        return batch

    def to_needle(self, batch):
        (X_dense, X_sparse, Y_true) = batch
        Y_true = Tensor(Y_true, device=self.device, requires_grad=False)
        X_dense = Tensor(X_dense, device=self.device, requires_grad=False)
        X_sparse = Tensor(X_sparse, device=self.device, requires_grad=False)
        batch = (X_dense, X_sparse, Y_true)
        return batch

    def to_output(self, batch):
        if self.output_format == 'needle':
            return self.to_needle(batch)
        elif self.output_format == 'torch':
            return self.to_torch(batch)
        elif self.output_format == 'numpy':
            return batch
        raise Exception(f"unsupported output format {self.output_format}")

    def __next__(self):
        logger.debug("fetching next batch...")
        if self.limit_batches > 0 and self.read_batches >= self.limit_batches:
            raise StopIteration()
        if self.limit_records > 0 and self.read_records >= self.limit_records:
            raise StopIteration()
        batch = next(self.batch_iter).to_pandas()
        self.read_batches += 1
        self.read_records += batch.shape[0]
        X_dense = batch[apps.etl.DENSE_COLUMNS].to_numpy()
        X_sparse = batch[apps.etl.SPARSE_IDX_COLUMNS].to_numpy()
        Y_true = batch[apps.etl.TARGET_COLUMN].to_numpy()

        X_dense = np.log(X_dense + 3)
        if self.sparse_limits is not None:
            X_sparse[X_sparse > self.sparse_limits] = 0
        if self.sparse_buckets is not None:
            X_sparse %= self.sparse_buckets

        batch = (X_dense, X_sparse, Y_true)
        return self.to_output(batch)

    def __len__(self) -> int:
        return self.records

def read_s3_dataset(**kwargs):
    return Criteo1TBDatasetBatchIter(**kwargs)

##############################################################################

class RepeatFirstBatch():
    def __init__(self, dataset, repeat):
        self.batch = next(iter(dataset))
        self.limit_batches = repeat
    
    def __iter__(self):
        self.read_batches = 0
        return self
    
    def __next__(self):
        if self.read_batches >= self.limit_batches:
            raise StopIteration()
        self.read_batches += 1
        return self.batch
    
    def __len__(self) -> int:
        return 1

def repeat_first_batch(dataset, repeat):
    return RepeatFirstBatch(dataset, repeat)

##############################################################################
