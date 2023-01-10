import sys
sys.path.append('./python')
import numpy as np
import needle as ndl
import needle.nn as nn

#############################################################################

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

    def forward(self, input):
        return self.layers(input)

class DLRMSparsePart(nn.Module):
    def __init__(self, embedding_dim, num_embeddings_per_feature, device=None, dtype="float32"):
        self.embeddings = [
            nn.Embedding(num_embeddings, embedding_dim, device=device, dtype=dtype) 
            for num_embeddings in num_embeddings_per_feature
        ]

    def forward(self, input):
        input_list = ndl.ops.split(input, axis=1)
        output = []
        for input_idx, input_item in enumerate(input_list):
            input_item = ndl.ops.reshape(input_item, (input_item.shape[0], 1))
            output_item = self.embeddings[input_idx](input_item)
            output_item = ndl.ops.reshape(output_item, (output_item.shape[0], output_item.shape[2]))
            output.append(output_item)
        output = ndl.ops.stack(output, axis=1)
        return output

class DLRMCrossPart(nn.Module):
    def __init__(self, num_sparse_features, device=None, dtype="float32"):
        self.num_sparse_features = num_sparse_features
        self.interaction_indices = np.triu_indices(num_sparse_features+1, k=1)

    def forward(self, input_dense, input_sparse):
        # input_dense = input_dense.reshape((input_dense.shape[0], 1, input_dense.shape[1]))
        input_sparse_splitted = list(ndl.ops.split(input_sparse, axis=1))
        # input_sparse_splitted
        input_concat = ndl.ops.stack([input_dense] + input_sparse_splitted, axis=1)

        input_concat_lhs = ndl.ops.reshape(ndl.ops.transpose(input_concat, axes=(1,2)), (input_concat.shape[0] * input_concat.shape[2], input_concat.shape[1]))
        input_concat_rhs = ndl.ops.transpose(input_concat_lhs, axes=(0,1))
        # FIXME: this should be analogous to torch.bmm, multiplying B times (D,F)*(F,D) instead of doing (B*D,F)*(F,B*D)
        interactions = input_concat_lhs @ input_concat_rhs 
        interactions = ndl.ops.reshape(interactions, (input_concat.shape[0], input_concat.shape[2]))

        # dense/sparse + sparse/sparse interaction
        # size B X (F + F choose 2)
        interactions_flat = interactions[:, self.interaction_indices[0], self.interaction_indices[1]]

        result = ndl.ops.stack([input_dense, interactions_flat], axis=1)
        return result


#############################################################################

class DLRM(nn.Module):
    def __init__(self,
        embedding_dim,
        sparse_feature_embedding_nums,
        dense_in_features,
        dense_layer_sizes,
        final_layer_sizes, 
        device=None, 
        dtype="float32"
    ):
        super(DLRM, self).__init__()
        self.dense_submodel = DLRMDensePart(
            in_features = dense_in_features,
            layer_sizes = dense_layer_sizes,
            device = device,
            dtype = dtype
        )
        self.sparse_submodel = DLRMSparsePart(
            embedding_dim = embedding_dim,
            num_embeddings_per_feature = sparse_feature_embedding_nums,
            device = device,
            dtype = dtype
        )
        self.cross_submodel = DLRMCrossPart(
            num_sparse_features = len(sparse_feature_embedding_nums),
            device = device,
            dtype = dtype
        )
        # self.linear_layer = nn.Linear(dense_layer_sizes[-1], 1, device=device, dtype=dtype)


    def forward(self, input_dense, input_sparse):
        """
        Args:
            X_dense: an input tensor of dense features of shape (batch_size, num_dense_features)
        """
        dense_embeddings = self.dense_submodel(input_dense)
        sparse_embeddings = self.sparse_submodel(input_sparse)
        cross_features = self.cross_submodel(dense_embeddings, sparse_embeddings)
        # logits = self.linear_layer(dense_embeddings)
        return cross_features

#############################################################################
