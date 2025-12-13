# model.py
import torch
import torch.nn as nn

class RelationalGCNLayer(nn.Module):

    """
    A single layer of a Relational Graph Convolutional Network (R-GCN).

    This layer processes node features by aggregating information from neighbors
    based on different relation types, as described in the R-GCN paper.
    It applies a separate linear transformation for each relation type before
    summing the transformed features.
    """

    def __init__(self, input_dim, output_dim, num_relations):
        """
        Initializes the RelationalGCNLayer.

        Args:
            input_dim (int): The dimensionality of input node features.
            output_dim (int): The dimensionality of output node features.
            num_relations (int): The number of different relation types in the graph.
        """
        super(RelationalGCNLayer, self).__init__()
        self.num_relations = num_relations
        # A list of linear transformations, one for each relation type.
        # Each transformation maps input features to output features.
        self.rel_linears = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_relations)])

    def forward(self, x, adj_stack):
        """
        Performs a forward pass through the Relational GCN layer.

        Aggregates features from neighbors for each relation type and sums them up.

        Args:
            x (torch.Tensor): Input node features of shape (batch_size, num_nodes, input_dim).
            adj_stack (torch.Tensor): Stacked adjacency matrices for each relation type.
                                      Shape: (batch_size, num_relations, num_nodes, num_nodes).
                                      adj_stack[b, r, i, j] is 1 if there's a relation 'r' from j to i
                                      in graph 'b', 0 otherwise.

        Returns:
            torch.Tensor: Output node features after aggregation and ReLU activation,
                          of shape (batch_size, num_nodes, output_dim).
        """
        out = 0
        for r in range(self.num_relations):
            # Apply relation-specific linear transformation to node features
            support = self.rel_linears[r](x)
            # Aggregate features: multiply adjacency matrix for relation 'r' with transformed features
            # torch.bmm performs batch matrix-matrix product
            out += torch.bmm(adj_stack[:, r, :, :], support)
        # Apply ReLU activation to the aggregated features
        return torch.relu(out)

class DAGNRelational(nn.Module):
    """
    DAGNRelational is a Graph Neural Network model
    that incorporates relational graph convolutional layers.

    It processes sequences using an LSTM, extracts node features, and then
    applies relational GCN layers to capture graph structure information.
    Finally, it uses a linear classifier for a prediction task.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout, num_relations):
        """
        Initializes the DAGNRelational model.

        Args:
            vocab_size (int): The size of the vocabulary for input sequences.
            embed_dim (int): The dimensionality of the word embeddings.
            hidden_dim (int): The dimensionality of the LSTM hidden states and GCN output.
            dropout (float): The dropout probability for regularization.
            num_relations (int): The number of different relation types in the graphs.
        """
        super(DAGNRelational, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Bidirectional LSTM to capture contextual information from sequences
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # First Relational GCN layer. Input and output dimensions are hidden_dim * 2
        # because the bidirectional LSTM output is concatenated (hidden_dim for each direction).
        self.gnn1 = RelationalGCNLayer(hidden_dim * 2, hidden_dim * 2, num_relations)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Classifier for the final prediction task
        self.classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, adjs, graph_counts):
        """
        Performs a forward pass through the DAGNRelational model.

        Args:
            x (torch.Tensor): Input sequences of shape (batch_size, sequence_length).
                              Contains token IDs.
            adjs (torch.Tensor): Stacked adjacency matrices for all graphs in the batch.
                                 Shape: (total_nodes_in_batch, num_relations, max_nodes_in_any_graph, max_nodes_in_any_graph).
                                 Note: This shape might need clarification based on how `adjs` is constructed
                                 and used with `graph_counts`. Assuming `adjs` is a list of individual
                                 adj_stacks for each graph, or a padded tensor.
                                 Based on the `bmm` in `RelationalGCNLayer`, `adjs` should be
                                 (batch_size, num_relations, num_nodes, num_nodes).
            graph_counts (torch.Tensor): A tensor indicating the number of nodes in each graph
                                         in the current batch. Shape: (batch_size,).

        Returns:
            torch.Tensor: The output predictions of the classifier, shape (batch_size, 1).
        """
        # Embed input sequences
        embeds = self.embedding(x)
        # Pass embeddings through LSTM
        # h_n shape: (num_layers * num_directions, batch_size, hidden_size)
        _, (h_n, _) = self.lstm(embeds)
        # Concatenate the final hidden states from forward and backward LSTMs
        # to get initial node features. Shape: (batch_size, hidden_dim * 2)
        node_features = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        
        num_graphs = len(graph_counts)
        # Determine the maximum number of nodes in any graph within the current batch
        max_nodes = max(graph_counts).item()
        
        # Initialize a tensor to hold batched node features, padded to max_nodes
        # This tensor will be populated with `node_features` for each graph.
        batch_nodes = torch.zeros(num_graphs, max_nodes, node_features.shape[1]).to(x.device)

    def __init__(self, input_dim, output_dim, num_relations):
        super(RelationalGCNLayer, self).__init__()
        self.num_relations = num_relations
        self.rel_linears = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_relations)])

    def forward(self, x, adj_stack):
        out = 0
        for r in range(self.num_relations):
            support = self.rel_linears[r](x)
            out += torch.bmm(adj_stack[:, r, :, :], support)
        return torch.relu(out)

class DAGNRelational(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout, num_relations):
        super(DAGNRelational, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        self.gnn1 = RelationalGCNLayer(hidden_dim * 2, hidden_dim * 2, num_relations)
        self.dropout_layer = nn.Dropout(dropout)
        
        self.classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, adjs, graph_counts):
        embeds = self.embedding(x)
        _, (h_n, _) = self.lstm(embeds)
        node_features = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        
        num_graphs = len(graph_counts)
        max_nodes = max(graph_counts).item()
        
        batch_nodes = torch.zeros(num_graphs, max_nodes, node_features.shape[1]).to(x.device)
        batch_adj = torch.zeros(num_graphs, self.gnn1.num_relations, max_nodes, max_nodes).to(x.device)
        
        current_idx = 0
        for i, count in enumerate(graph_counts):
            batch_nodes[i, :count, :] = node_features[current_idx : current_idx+count]
            adj = torch.tensor(adjs[i]).to(x.device)
            batch_adj[i, :, :count, :count] = adj
            current_idx += count
            
        gnn_out = self.gnn1(batch_nodes, batch_adj)
        gnn_out = self.dropout_layer(gnn_out)
        
        mask = torch.arange(max_nodes).expand(num_graphs, max_nodes).to(x.device) < graph_counts.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()
        
        gnn_out = gnn_out * mask + (1 - mask) * -1e9
        graph_embed, _ = torch.max(gnn_out, dim=1)
        
        scores = self.classifier(graph_embed).squeeze(-1)
        return scores.view(-1, 4)
