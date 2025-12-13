# dagn_model.py
import torch
import torch.nn as nn
import numpy as np
from itertools import groupby
from operator import itemgetter
from util import FFNLayer, ResidualGRU, ArgumentGCN, masked_softmax, weighted_sum

class DAGN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, dropout_prob=0.1, gcn_steps=1):
        super(DAGN, self).__init__()
        
        # --- 1. Backbone (Replaces Roberta) ---
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Bidirectional LSTM: output size = hidden_size//2 * 2 = hidden_size
        self.lstm = nn.LSTM(embed_dim, hidden_size // 2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_prob)
        
        # --- 2. Graph Reasoning Module ---
        self.gcn_steps = gcn_steps
        self._gcn = ArgumentGCN(node_dim=hidden_size, iteration_steps=gcn_steps)
        self._gcn_prj_ln = nn.LayerNorm(hidden_size)
        
        # Post-GCN refinement
        self._gcn_enc = ResidualGRU(hidden_size, dropout_prob, 2)

        # --- 3. Hierarchical Pooling & Prediction ---
        # Projection to calculate attention scores for pooling
        self._proj_sequence_h = nn.Linear(hidden_size, 1, bias=False)
        
        # Final classifier: Takes concatenated [Passage, Question, CLS] -> Logit
        self._proj_span_num = FFNLayer(3 * hidden_size, hidden_size, 1, dropout_prob)

    def split_into_spans(self, seq, seq_mask, split_ids):
        """
        Dynamically groups tokens into EDU nodes.
        split_ids: 0=None, 1=Argument Marker, 2=Punctuation.
        Logic adapted from 'split_into_spans_9' in dagn.py.
        """
        bsz, seq_len, embed_dim = seq.size()
        device = seq.device
        
        encoded_spans = []      # List of tensors (node embeddings)
        span_masks = []         # List of tensors (node masks)
        edges = []              # List of lists (edge types)
        node_indices = []       # List of lists (token indices belonging to nodes)

        for b in range(bsz):
            item_seq = seq[b]
            item_split = split_ids[b].cpu().numpy()
            valid_len = seq_mask[b].sum().item()
            
            # Identify indices where markers exist (>0)
            split_indices = np.where(item_split[:valid_len] > 0)[0].tolist()
            
            # Group consecutive markers (e.g., "on", "the", "other", "hand")
            groups = []
            for k, g in groupby(enumerate(split_indices), lambda x: x[0] - x[1]):
                groups.append(list(map(itemgetter(1), g)))
            
            item_spans_emb = []
            item_edges = []
            item_node_idxs = []
            
            prev_end = 0
            
            # Create EDUs (text spans) between markers
            for group in groups:
                marker_start = group[0]
                marker_end = group[-1]
                
                # If there is text before the marker, that's an EDU
                if marker_start > prev_end:
                    # Sum token embeddings to get Node embedding
                    span_emb = item_seq[prev_end:marker_start].sum(0) 
                    item_spans_emb.append(span_emb)
                    item_node_idxs.append(list(range(prev_end, marker_start)))
                    
                    # The marker type (1 or 2) defines the edge leaving this node
                    marker_type = item_split[marker_start]
                    item_edges.append(marker_type)
                
                prev_end = marker_end + 1
            
            # Capture the last span after the final marker
            if prev_end < valid_len:
                span_emb = item_seq[prev_end:valid_len].sum(0)
                item_spans_emb.append(span_emb)
                item_node_idxs.append(list(range(prev_end, valid_len)))
                # No outgoing edge for the last node
            
            # Handle case where no markers found (entire seq is one node)
            if not item_spans_emb:
                item_spans_emb.append(item_seq[:valid_len].sum(0))
                item_node_idxs.append(list(range(valid_len)))
            
            encoded_spans.append(torch.stack(item_spans_emb))
            span_masks.append(torch.ones(len(item_spans_emb)).to(device))
            edges.append(item_edges)
            node_indices.append(item_node_idxs)

        # Pad to create batch tensors
        max_nodes = max(len(s) for s in encoded_spans)
        padded_spans = torch.zeros(bsz, max_nodes, embed_dim).to(device)
        padded_node_masks = torch.zeros(bsz, max_nodes).to(device)
        
        for b in range(bsz):
            n_nodes = encoded_spans[b].size(0)
            padded_spans[b, :n_nodes] = encoded_spans[b]
            padded_node_masks[b, :n_nodes] = span_masks[b]
            
        return padded_spans, padded_node_masks, edges, node_indices

    def build_adjacency(self, edges, num_nodes, device):
        """
        Constructs adjacency matrices from edge lists.
        1 = Argument (Directed), 2 = Punctuation (Symmetric).
        """
        bsz = len(edges)
        arg_graph = torch.zeros(bsz, num_nodes, num_nodes).to(device)
        punct_graph = torch.zeros(bsz, num_nodes, num_nodes).to(device)
        
        for b, edge_list in enumerate(edges):
            # Edges connect Node i -> Node i+1
            for i, edge_type in enumerate(edge_list):
                if i + 1 >= num_nodes: break
                
                if edge_type == 1: # Argument
                    arg_graph[b, i, i+1] = 1
                elif edge_type == 2: # Punctuation
                    punct_graph[b, i, i+1] = 1
                    punct_graph[b, i+1, i] = 1 # Undirected
                    
        return arg_graph, punct_graph

    def scatter_nodes_to_sequence(self, node_indices, nodes, seq_size, device):
        """
        Maps updated node features back to their constituent tokens.
        """
        batch_size, seq_len, dim = seq_size
        info_vec = torch.zeros(batch_size, seq_len, dim).to(device)
        
        for b in range(batch_size):
            node_emb = nodes[b]
            idxs_list = node_indices[b]
            
            for i, token_idxs in enumerate(idxs_list):
                if i >= len(node_emb): break
                if not token_idxs: continue
                # Assign node embedding to all tokens in the span
                # Note: Using a loop here for clarity; advanced indexing can optimize
                for t_idx in token_idxs:
                    info_vec[b, t_idx] = node_emb[i]
                    
        return info_vec

    def forward(self, input_ids, attention_mask, split_ids, passage_mask, question_mask):
        """
        Forward pass for one choice option (or a batch of flattened options).
        """
        # 1. Encode Sequence
        emb = self.embedding(input_ids)
        sequence_output, _ = self.lstm(emb) # [B, L, H]
        
        # 2. Dynamic Graph Construction
        nodes, node_masks, edges, node_indices = self.split_into_spans(
            sequence_output, attention_mask, split_ids
        )
        
        # 3. Build Adjacency Matrices
        arg_graph, punct_graph = self.build_adjacency(nodes.size(1), edges, nodes.device) # Fixed args order
        
        # 4. Graph Reasoning (GCN)
        updated_nodes = self._gcn(nodes, node_masks, arg_graph, punct_graph)
        
        # 5. Scatter Back & Fuse
        gcn_info_vec = self.scatter_nodes_to_sequence(node_indices, updated_nodes, sequence_output.size(), sequence_output.device)
        
        # Fuse: (Original + GraphInfo) -> LN -> ResidualGRU
        fused_sequence = self._gcn_enc(self._gcn_prj_ln(sequence_output + gcn_info_vec))
        
        # 6. Hierarchical Pooling (Passage, Question, CLS)
        # Attention scores
        attn_scores = self._proj_sequence_h(fused_sequence).squeeze(-1) # [B, L]
        
        # Passage Vector
        passage_w = masked_softmax(attn_scores, passage_mask)
        passage_h = weighted_sum(fused_sequence, passage_w)
        
        # Question Vector
        question_w = masked_softmax(attn_scores, question_mask)
        question_h = weighted_sum(fused_sequence, question_w)
        
        # "CLS" Vector (First token in sequence)
        cls_h = fused_sequence[:, 0]
        
        # Concatenate and Classify
        features = torch.cat([passage_h, question_h, cls_h], dim=1) # [B, 3*H]
        logits = self._proj_span_num(features) # [B, 1]
        
        return logits
        
    # Helper to fix argument order in build_adjacency inside the class
    def build_adjacency(self, num_nodes, edges, device):
        bsz = len(edges)
        arg_graph = torch.zeros(bsz, num_nodes, num_nodes).to(device)
        punct_graph = torch.zeros(bsz, num_nodes, num_nodes).to(device)
        for b, edge_list in enumerate(edges):
            for i, edge_type in enumerate(edge_list):
                if i + 1 >= num_nodes: break
                if edge_type == 1:
                    arg_graph[b, i, i+1] = 1
                elif edge_type == 2:
                    punct_graph[b, i, i+1] = 1
                    punct_graph[b, i+1, i] = 1
        return arg_graph, punct_graph