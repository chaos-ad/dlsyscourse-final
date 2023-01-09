import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np



class DLRMDensePart(nn.Module):
    def __init__(self, in_features, layer_sizes, device=None, dtype="float32"):
        self.layers = []
        for out_features in layer_sizes:
            layer = nn.Sequential(
                nn.Linear(in_features, out_features, device=device, dtype=dtype), 
                nn.ReLU()
            )
            self.layers.append(layer)
            in_features = out_features
        self.layers = nn.Sequential(*self.layers)

    def forward(self, X_dense):
        return self.layers(X_dense)
   

class DLRM(nn.Module):
    def __init__(self, 
        dense_in_features, 
        dense_layer_sizes, 
        device=None, 
        dtype="float32"
    ):
        super(DLRM, self).__init__()
        # self.embeddings_layer = nn.Embedding(num_embeddings=1000, embedding_dim=embedding_size, device=device, dtype=dtype)
        self.dense_layers = DLRMDensePart(
            in_features = dense_in_features,
            layer_sizes = dense_layer_sizes,
            device = device,
            dtype = dtype
        )
        self.linear_layer = nn.Linear(dense_layer_sizes[-1], 1, device=device, dtype=dtype)


    def forward(self, X_dense):
        """
        Args:
            X_dense: an input tensor of dense features of shape (batch_size, num_dense_features)
        """
        dense_embeddings = self.dense_layers(X_dense)
        logits = self.linear_layer(dense_embeddings)
        return logits



