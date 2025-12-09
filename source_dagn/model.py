# model.py
import torch
import torch.nn as nn

class RelationalGCNLayer(nn.Module):
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