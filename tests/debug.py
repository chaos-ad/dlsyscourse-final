import os
import sys
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("./python"))

import torch.nn
import torchrec
import torchrec.models.dlrm
import torchrec.sparse.jagged_tensor
import torcheval.metrics

import tqdm
import logging
import numpy as np
import pandas as pd
import dotenv

import needle as ndl
import needle.nn as nn

from needle.autograd import Tensor
from needle import backend_ndarray as nd

import apps.data
import apps.models

##############################################################################

logger = logging.getLogger("tests.debug")

##############################################################################

def train_torch():
    dense_arch_layer_sizes = [16]  # last dim must match the embedding size
    over_arch_layer_sizes = [32,1]
    learning_rate = 0.1 ## also try 4.0, 15.0
    weight_decay = 0.0

    device = torch.device("cpu")

    model = torchrec.models.dlrm.DLRM(
        embedding_bag_collection=torchrec.EmbeddingBagCollection(
            tables=[
                torchrec.EmbeddingBagConfig(
                    name = f"t_{feature_name}",
                    embedding_dim = 16,
                    num_embeddings = 1000,
                    feature_names=[feature_name],
                )
                for feature_idx, feature_name in enumerate(apps.etl.SPARSE_IDX_COLUMNS)
            ],
            device=device
        ),
        dense_in_features=len(apps.etl.DENSE_COLUMNS),
        dense_arch_layer_sizes=dense_arch_layer_sizes,
        over_arch_layer_sizes=over_arch_layer_sizes,
        dense_device=device
    )

    opt = torch.optim.SGD(model.parameters(), lr=learning_rate)

    auroc = torcheval.metrics.BinaryAUROC(device=device)

    loss_fn = torch.nn.BCEWithLogitsLoss()

    total_loss = 0
    total_errors = 0
    total_batches = 0
    total_examples = 0

    dataset_iter = apps.data.read_s3_dataset(
        s3_prefix="anatoly/datasets/criteo-terabyte-click-log", 
        s3_path="preprocessed/joined",
        day_from = 0,
        day_to = 0,
        batch_size = 2048,
        # limit_batches = 100,
        as_numpy=True
    )
    pbar = tqdm.tqdm(desc="reading data", total=len(dataset_iter))
    for batch_id, (X_dense, X_sparse, Y) in enumerate(dataset_iter, start=1):

        batch_size = Y.shape[0]
        # logger.debug(f"TRAIN on {batch_id=} of size {batch_size}...")
        X_dense = np.log(X_dense + 3)
        X_sparse[X_sparse > 999] = 0
        
        X_dense = torch.tensor(X_dense, dtype=torch.float32, device=device)
        # X_sparse = torch.tensor(X_sparse, dtype=torch.float32, device=device)


        CAT_FEATURE_COUNT = len(apps.etl.SPARSE_IDX_COLUMNS)
        _num_ids_in_batch = CAT_FEATURE_COUNT * (batch_size * 2)
        num_ids_in_batch = CAT_FEATURE_COUNT * batch_size
        length_per_key = CAT_FEATURE_COUNT * [batch_size]
        offset_per_key = [batch_size * i for i in range(CAT_FEATURE_COUNT + 1)]
        index_per_key = {key: i for (i, key) in enumerate(apps.etl.SPARSE_IDX_COLUMNS)}
        lengths = torch.ones((_num_ids_in_batch,), dtype=torch.int32)
        offsets = torch.arange(0, _num_ids_in_batch + 1, dtype=torch.int32)

        X_sparse = torchrec.sparse.jagged_tensor.KeyedJaggedTensor(
            keys = apps.etl.SPARSE_IDX_COLUMNS,
            values = torch.from_numpy(X_sparse.transpose(1, 0).reshape(-1)),
            lengths = lengths[:num_ids_in_batch],
            offsets = offsets[: num_ids_in_batch + 1],
            stride = batch_size,
            length_per_key=length_per_key,
            offset_per_key=offset_per_key,
            index_per_key=index_per_key
        )

        Y = torch.tensor(Y.reshape(-1), dtype=torch.float32, device=device)

        opt.zero_grad()
        out = model(X_dense, X_sparse)
        loss = loss_fn(out.squeeze(-1), Y)
        
        loss.backward()
        opt.step()

        # preds = torch.sigmoid(logits)
        # auroc.update(preds, labels)
        # auroc_result = auroc.compute().item()
        # num_samples = torch.tensor(sum(map(len, auroc.targets)), device=device)
        # print(f"AUROC over {stage} set: {auroc_result}.")
        # print(f"Number of {stage} samples: {num_samples}")

        y_prob = out.detach().numpy()
        y_pred = (y_prob > 0)
        errors = np.not_equal(y_pred, Y.numpy()).sum()

        cur_loss = float(loss.detach().numpy())
        total_loss += cur_loss * batch_size
        total_errors += errors
        total_batches += 1
        total_examples += batch_size

        avg_loss = (total_loss / total_examples)
        avg_error_rate = (total_errors / total_examples)

        pbar.update(batch_size)

        # logger.debug(f"TRAIN on {batch_id=} of size {batch_size}: done ({cur_loss=:0.4f}, {avg_loss=:0.4f}, {avg_error_rate=:0.4f})")
        

def train_needle():

    device = ndl.cpu()
    dense_layer_sizes = [512, 256, 64]
    # interaction_layer_sizes = [512,512,256,1]
    learning_rate = 0.1 ## also try 4.0, 15.0
    weight_decay = 0.0
    model = apps.models.DLRM(
        dense_in_features = len(apps.etl.DENSE_COLUMNS),
        dense_layer_sizes=[512,256,64],
        device=device
    )
    model.train()

    loss_fn = nn.BinaryCrossEntropyLoss()
    opt = ndl.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    total_loss = 0
    total_errors = 0
    total_batches = 0
    total_examples = 0

    dataset_iter = apps.data.read_s3_dataset(
        s3_prefix="anatoly/datasets/criteo-terabyte-click-log", 
        s3_path="preprocessed/joined",
        day_from = 0,
        day_to = 0,
        batch_size = 1024,
        limit_batches = 100,
        as_numpy=True
    )
    for batch_id, (X_dense, X_sparse, Y) in enumerate(dataset_iter):
        batch_size = Y.shape[0]
        logger.debug(f"TRAIN on {batch_id=} of size {batch_size}...")
        X_dense = np.log(X_dense + 3)
        X_dense = Tensor(X_dense, device=device, requires_grad=False)
        X_sparse = Tensor(X_sparse, device=device, requires_grad=False)
        Y = Tensor(Y.flatten(), device=device, requires_grad=False)

        opt.reset_grad()
        out = model(X_dense)
        out = out.reshape((out.shape[0],))
        loss = loss_fn(out, Y)
        loss.backward()
        opt.step()

        y_prob = out.numpy()
        y_pred = (y_prob > 0)
        errors = np.not_equal(y_pred, Y.numpy()).sum()

        cur_loss = loss.numpy()[0]
        total_loss += cur_loss * batch_size
        total_errors += errors
        total_batches += 1
        total_examples += batch_size

        avg_loss = (total_loss / total_examples)
        avg_error_rate = (total_errors / total_examples)

        logger.debug(f"TRAIN on {batch_id=} of size {batch_size}: done ({cur_loss=}, {total_loss=}, {total_batches=}, {avg_loss=:0.4f}, {avg_error_rate=:0.4f})")
        

def main():
    print(f"{os.getpid()=}")
    
    assert dotenv.load_dotenv(dotenv_path="conf/dev.env")
    apps.utils.common.setup_logging(config_file="conf/logging.yml")

    train_torch()

if __name__ == '__main__':
    main()