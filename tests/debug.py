import os
import sys
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("./python"))

import torch.nn
import torchrec
import torchrec.models.dlrm
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

def torch_train(model, loss_fn, optimizer, dataset, epoch_id=None, with_pbar=None):

    num_samples = 0
    num_batches = 0
    avg_loss = 0
    auroc = torcheval.metrics.BinaryAUROC()
    accuracy = torcheval.metrics.BinaryAccuracy()

    model.train()
    pbar = tqdm.tqdm(desc=f"TRAIN[{epoch_id=}]", total=len(dataset)) if with_pbar else None
    for batch_id, (X_dense, X_sparse, Y_true) in enumerate(dataset, start=1):
        batch_size = Y_true.shape[0]
        logger.debug(f"TRAIN[{epoch_id=}] on {batch_id=} of size {batch_size}...")

        optimizer.zero_grad()

        Y_logits = model(X_dense, X_sparse).squeeze(-1)
        loss = loss_fn(Y_logits, Y_true)

        loss.backward()
        optimizer.step()

        Y_pred = torch.sigmoid(Y_logits)
        auroc.update(Y_pred, Y_true)
        accuracy.update(Y_pred, Y_true)
        
        cur_loss = float(loss.detach().numpy())
        avg_loss = (avg_loss * num_batches + cur_loss) / (num_batches + 1)

        num_batches += 1
        num_samples += batch_size

        if pbar:
            pbar.update(batch_size)

        logger.debug(f"TRAIN[{epoch_id=}] on {batch_id=} of size {batch_size}: done ({cur_loss=:0.4f}, {avg_loss=:0.4f}")
    
    auroc_val = auroc.compute().item()
    accuracy_val = accuracy.compute().item()
    logger.info(f"TRAIN[{epoch_id=}] done: {avg_loss=:0.4f}, {auroc_val=:0.4f}, {accuracy_val=:0.4f}, {num_samples=}, {num_batches=}")
    return (avg_loss, auroc_val, accuracy_val, num_samples, num_batches)

def torch_eval(model, loss_fn, dataset, epoch_id=None, with_pbar=True):

    num_samples = 0
    num_batches = 0
    avg_loss = 0
    auroc = torcheval.metrics.BinaryAUROC()
    accuracy = torcheval.metrics.BinaryAccuracy()

    model.eval()
    with torch.no_grad():
        pbar = tqdm.tqdm(desc=f"EVAL[{epoch_id=}]", total=len(dataset)) if with_pbar else None
        for batch_id, (X_dense, X_sparse, Y_true) in enumerate(dataset, start=1):
            batch_size = Y_true.shape[0]
            logger.debug(f"EVAL[{epoch_id=}] on {batch_id=} of size {batch_size}...")

            Y_logits = model(X_dense, X_sparse).squeeze(-1)
            loss = loss_fn(Y_logits, Y_true)

            Y_pred = torch.sigmoid(Y_logits)
            auroc.update(Y_pred, Y_true)
            accuracy.update(Y_pred, Y_true)

            cur_loss = float(loss.detach().numpy())
            avg_loss = (avg_loss * num_batches + cur_loss) / (num_batches + 1)

            num_batches += 1
            num_samples += batch_size

            if pbar:
                pbar.update(batch_size)

            logger.debug(f"EVAL[{epoch_id=}] on {batch_id=} of size {batch_size}: done ({cur_loss=:0.4f}, {avg_loss=:0.4f}")

        auroc_val = auroc.compute().item()
        accuracy_val = accuracy.compute().item()
        logger.info(f"EVAL[{epoch_id=}] done: {avg_loss=:0.4f}, {auroc_val=:0.4f}, {accuracy_val=:0.4f}, {num_samples=}, {num_batches=}")
        return (avg_loss, auroc_val, accuracy_val, num_samples, num_batches)

def torch_dataset(day=None, **kwargs):
    kwargs['day_from'] = day if day is not None else kwargs.get('day_from', 0)
    kwargs['day_to'] = day if day is not None else kwargs.get('day_to', 23)
    kwargs['batch_size'] = kwargs.get('batch_size', 1024)
    kwargs['device'] = kwargs.get('device', torch.device("cpu"))
    
    result = apps.data.read_s3_dataset(
        s3_prefix="anatoly/datasets/criteo-terabyte-click-log", 
        s3_path="preprocessed/joined",
        output_format = "torch",
        **kwargs
    )
    result = apps.data.repeat_first_batch(result, 1)
    return result

def run_torch(
    embedding_dim = 16,
    # embedding_dim = 128,
    # num_embeddings_per_feature = [45833188,36746,17245,7413,20243,3,7114,1441,62,29275261,1572176,345138,10,2209,11267,128,4,974,14,48937457,11316796,40094537,452104,12606,104,35],
    # num_embeddings_per_feature = [2000000,36746,17245,7413,20243,3,7114,1441,62,2000000,1572176,345138,10,2209,11267,128,4,974,14,2000000,2000000,2000000,452104,12606,104,35],
    num_embeddings_per_feature = 1000,
    over_arch_layer_sizes = [32,32,16,1],
    dense_arch_layer_sizes = [64,32,16],
    # over_arch_layer_sizes = [1024,1024,512,256,1],
    # dense_arch_layer_sizes = [512,256,128],
    epochs = 1000,
    batch_size = 16384,
    learning_rate = 0.5, # 15.0,
    # change_lr = True,
    # lr_change_point = 0.65,
    # lr_after_change_point = 0.035,
    with_pbar = False
):
    device = torch.device("cpu")

    train_dataset = torch_dataset(day = 0, batch_size = batch_size, limit_batches = 100, device = device, sparse_buckets = num_embeddings_per_feature)
    eval_dataset = torch_dataset(day = 22, batch_size = batch_size, limit_batches = 100, device = device, sparse_buckets = num_embeddings_per_feature)
    test_dataset = torch_dataset(day = 23, batch_size = batch_size, limit_batches = 100, device = device, sparse_buckets = num_embeddings_per_feature)
    logger.debug(f"size of a train dataset: {len(train_dataset)}")
    logger.debug(f"size of a eval dataset: {len(eval_dataset)}")
    logger.debug(f"size of a test dataset: {len(test_dataset)}")

    model = torchrec.models.dlrm.DLRM(
        embedding_bag_collection=torchrec.EmbeddingBagCollection(
            tables=[
                torchrec.EmbeddingBagConfig(
                    name = f"t_{feature_name}",
                    embedding_dim = embedding_dim,
                    num_embeddings = num_embeddings_per_feature[feature_idx] if isinstance(num_embeddings_per_feature, list) else num_embeddings_per_feature,
                    feature_names = [feature_name],
                )
                for feature_idx, feature_name in enumerate(apps.etl.SPARSE_IDX_COLUMNS)
            ],
            device=device
        ),
        dense_in_features = len(apps.etl.DENSE_COLUMNS),
        dense_arch_layer_sizes = dense_arch_layer_sizes,
        over_arch_layer_sizes = over_arch_layer_sizes,
        dense_device = device
    )

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch_id in range(epochs):
        torch_train(model, loss_fn, optimizer, train_dataset, epoch_id=epoch_id, with_pbar=with_pbar)
        # torch_eval(model, loss_fn, eval_dataset, epoch_id=epoch_id, with_pbar=with_pbar)
    torch_eval(model, loss_fn, test_dataset, epoch_id=epoch_id, with_pbar=with_pbar)


# def train_needle():

#     device = ndl.cpu()
#     dense_layer_sizes = [512, 256, 64]
#     # interaction_layer_sizes = [512,512,256,1]
#     learning_rate = 0.1 ## also try 4.0, 15.0
#     weight_decay = 0.0
#     model = apps.models.DLRM(
#         dense_in_features = len(apps.etl.DENSE_COLUMNS),
#         dense_layer_sizes=[512,256,64],
#         device=device
#     )
#     model.train()

#     loss_fn = nn.BinaryCrossEntropyLoss()
#     opt = ndl.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

#     total_loss = 0
#     total_errors = 0
#     total_batches = 0
#     total_examples = 0

#     dataset_iter = apps.data.read_s3_dataset(
#         s3_prefix="anatoly/datasets/criteo-terabyte-click-log", 
#         s3_path="preprocessed/joined",
#         day_from = 0,
#         day_to = 0,
#         batch_size = 1024,
#         limit_batches = 100,
#         output_format = "needle"
#     )
#     for batch_id, (X_dense, X_sparse, Y) in enumerate(dataset_iter, start=1):
#         batch_size = Y.shape[0]
#         logger.debug(f"TRAIN on {batch_id=} of size {batch_size}...")
#         X_dense = np.log(X_dense + 3)
#         X_dense = Tensor(X_dense, device=device, requires_grad=False)
#         X_sparse = Tensor(X_sparse, device=device, requires_grad=False)
#         Y = Tensor(Y.flatten(), device=device, requires_grad=False)

#         opt.reset_grad()
#         out = model(X_dense)
#         out = out.reshape((out.shape[0],))
#         loss = loss_fn(out, Y)
#         loss.backward()
#         opt.step()

#         y_prob = out.numpy()
#         y_pred = (y_prob > 0)
#         errors = np.not_equal(y_pred, Y.numpy()).sum()

#         cur_loss = loss.numpy()[0]
#         total_loss += cur_loss * batch_size
#         total_errors += errors
#         total_batches += 1
#         total_examples += batch_size

#         avg_loss = (total_loss / total_examples)
#         avg_error_rate = (total_errors / total_examples)

#         logger.debug(f"TRAIN on {batch_id=} of size {batch_size}: done ({cur_loss=}, {total_loss=}, {total_batches=}, {avg_loss=:0.4f}, {avg_error_rate=:0.4f})")
        

def main():
    print(f"{os.getpid()=}")
    
    assert dotenv.load_dotenv(dotenv_path="conf/dev.env")
    apps.utils.common.setup_logging(config_file="conf/logging.yml")

    run_torch()

if __name__ == '__main__':
    main()