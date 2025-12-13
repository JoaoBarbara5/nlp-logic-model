# util.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def gelu(x):
    """
    Gaussian Error Linear Unit (GELU) activation function.

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The output tensor after applying GELU.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def masked_softmax(vec, mask, dim=1, epsilon=1e-5):
    """
    Applies a softmax function to a tensor, masking out specified values.

    Args:
        vec (torch.Tensor): The input tensor to apply softmax to.
        mask (torch.Tensor): A boolean or float tensor of the same shape as `vec`
                             where 1s indicate values to keep and 0s indicate values to mask.
        dim (int, optional): The dimension along which softmax will be computed. Defaults to 1.
        epsilon (float, optional): A small value to prevent division by zero. Defaults to 1e-5.

    Returns:
        torch.Tensor: The masked softmax output.
    """
    exps = torch.exp(vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
    return masked_exps / masked_sums

def weighted_sum(features, weights):
    """
    Computes a weighted sum of features.

    Args:
        features (torch.Tensor): A tensor of features with shape [B, L, D].
        weights (torch.Tensor): A tensor of weights with shape [B, L].

    Returns:
        torch.Tensor: The weighted sum of features with shape [B, D].
    """
    # features: [B, L, D], weights: [B, L]
    return torch.bmm(weights.unsqueeze(1), features).squeeze(1)

def replace_masked_values(tensor, mask, replace_with):
    """
    Replaces values in a tensor where the mask is 0 with a specified value.

    Args:
        tensor (torch.Tensor): The input tensor.
        mask (torch.Tensor): A boolean or float tensor where 0s indicate positions to replace.
        replace_with (float or int): The value to replace masked elements with.

    Returns:
        torch.Tensor: The tensor with masked values replaced.
    """
    return tensor.masked_fill((1 - mask).bool(), replace_with)

class ResidualGRU(nn.Module):
    """
    Bidirectional GRU with residual connection and LayerNorm.
    Used to refine the sequence after merging Graph features.
    """
    def __init__(self, hidden_size, dropout=0.1, num_layers=2):
        """
        Initializes the ResidualGRU module.

        Args:
            hidden_size (int): The input and output feature dimension.
                               The GRU's hidden_size will be hidden_size // 2
                               because it's bidirectional.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
            num_layers (int, optional): Number of recurrent layers. Defaults to 2.
        """
        super(ResidualGRU, self).__init__()
        # Input size is hidden_size, output of BiGRU is hidden_size (hidden//2 * 2)
        self.enc_layer = nn.GRU(input_size=hidden_size, hidden_size=hidden_size // 2, 
                                num_layers=num_layers, batch_first=True, 
                                dropout=dropout, bidirectional=True)
        self.enc_ln = nn.LayerNorm(hidden_size)

    def forward(self, input):
        """
        Performs a forward pass through the ResidualGRU.

        Args:
            input (torch.Tensor): The input tensor to the GRU, shape [B, L, D].

        Returns:
            torch.Tensor: The output tensor after GRU, residual connection, and LayerNorm,
                          shape [B, L, D].
        """
        output, _ = self.enc_layer(input)
        return self.enc_ln(output + input)

class FFNLayer(nn.Module):
    """
    A Feed-Forward Network (FFN) layer with GELU activation and optional Layer Normalization.
    """
    def __init__(self, input_dim, intermediate_dim, output_dim, dropout, layer_norm=True):
        """
        Initializes the FFNLayer.

        Args:
            input_dim (int): The input feature dimension.
            intermediate_dim (int): The dimension of the intermediate layer.
            output_dim (int): The output feature dimension.
            dropout (float): Dropout probability.
            layer_norm (bool, optional): Whether to apply Layer Normalization. Defaults to True.
        """
        super(FFNLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        if layer_norm:
            self.ln = nn.LayerNorm(intermediate_dim)
        else:
            self.ln = None
        self.dropout_func = nn.Dropout(dropout)
        self.fc2 = nn.Linear(intermediate_dim, output_dim)

    def forward(self, input):
        """
        Performs a forward pass through the FFNLayer.

        Args:
            input (torch.Tensor): The input tensor to the FFN, shape [B, ..., input_dim].

        Returns:
            torch.Tensor: The output tensor after FFN, shape [B, ..., output_dim].
        """
        inter = self.fc1(self.dropout_func(input))
        inter_act = gelu(inter)
        if self.ln:
            inter_act = self.ln(inter_act)
        return self.fc2(inter_act)

class ArgumentGCN(nn.Module):
    """
    A Graph Reasoning module that performs iterative message passing on argument and punctuation graphs.
    """
    def __init__(self, node_dim, iteration_steps=1):
        """
        Initializes the ArgumentGCN module.

        Args:
            node_dim (int): The feature dimension of each node.
            iteration_steps (int, optional): Number of message passing steps. Defaults to 1.
        """
        super(ArgumentGCN, self).__init__()
        self.node_dim = node_dim
        self.iteration_steps = iteration_steps

        self._node_weight_fc = nn.Linear(node_dim, 1, bias=True)
        self._self_node_fc = nn.Linear(node_dim, node_dim, bias=True)
        
        # Separate projections for Argument edges and Punctuation edges
        self._node_fc_argument = nn.Linear(node_dim, node_dim, bias=False)
        self._node_fc_punctuation = nn.Linear(node_dim, node_dim, bias=False)

    def forward(self, node, node_mask, argument_graph, punctuation_graph):
        """
        Performs a forward pass through the ArgumentGCN.

        Args:
            node (torch.Tensor): Node features, shape [B, N, D].
            node_mask (torch.Tensor): Node mask, shape [B, N].
            argument_graph (torch.Tensor): Argument graph, shape [B, N, N].
            punctuation_graph (torch.Tensor): Punctuation graph, shape [B, N, N].

        Returns:
            torch.Tensor: Updated node features, shape [B, N, D].
        """
        # node: [B, N, D]
        # node_mask: [B, N]
        # argument_graph, punctuation_graph: [B, N, N]

        node_len = node.size(1)
        
        # Prevent self-loops in the adjacency matrices
        diagmat = torch.eye(node_len, device=node.device).unsqueeze(0).expand(node.size(0), -1, -1)
        
        # Mask out invalid nodes
        dd_graph = node_mask.unsqueeze(1) * node_mask.unsqueeze(-1) * (1 - diagmat)
        
        graph_argument = dd_graph * argument_graph
        graph_punctuation = dd_graph * punctuation_graph

        # Calculate neighbor counts for normalization
        node_neighbor_num = graph_argument.sum(-1) + graph_punctuation.sum(-1)
        node_neighbor_num_mask = (node_neighbor_num >= 1).float()
        # Avoid division by zero
        node_neighbor_num = node_neighbor_num * node_neighbor_num_mask + (1 - node_neighbor_num_mask)

        for step in range(self.iteration_steps):
            # (1) Node Relatedness Measure (Importance)
            d_node_weight = torch.sigmoid(self._node_weight_fc(node)).squeeze(-1) # [B, N]
            
            # (2) Message Propagation
            self_node_info = self._self_node_fc(node)
            
            # --- Argument Relations ---
            node_info_arg = self._node_fc_argument(node)
            # Weight neighbors by their importance (d_node_weight)
            weight_arg = replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_argument, 0
            )
            msg_arg = torch.matmul(weight_arg, node_info_arg)
            
            # --- Punctuation Relations ---
            node_info_punct = self._node_fc_punctuation(node)
            weight_punct = replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_punctuation, 0
            )
            msg_punct = torch.matmul(weight_punct, node_info_punct)

            # Aggregate
            agg_node_info = (msg_arg + msg_punct) / node_neighbor_num.unsqueeze(-1)

            # (3) Update
            node = F.relu(self_node_info + agg_node_info)
            
        return node