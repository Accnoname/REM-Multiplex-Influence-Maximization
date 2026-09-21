# src/models/pmoe.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from src.models.experts import GATExpert

class PMoE(nn.Module):
    """
    Propagation Mixture of Experts (PMoE)
    Tổng hợp dự đoán từ nhiều GNN experts với độ sâu lan tỏa khác nhau (1-hop, 2-hop, 3-hop)
    thông qua Routing Network thích ứng.
    """
    def __init__(self, num_nodes, num_experts=8, in_channels=1, hidden_dim=64, out_channels=1, dropout=0.2):
        super(PMoE, self).__init__()
        self.num_nodes = num_nodes
        self.num_experts = num_experts

        # 1. Routing Network (Gating)
        # Gating chiếu từ node feature (hoặc trạng thái seed) sang trọng số chuyên gia
        self.gate = nn.Sequential(
            nn.Linear(in_channels, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_experts)
        )

        # 2. Experts (Đa dạng hóa về độ sâu lan truyền theo bài báo REM)
        # 1-hop experts (lan tỏa lân cận), 2-hop experts, 3-hop experts (lan tỏa tầm xa)
        self.experts = nn.ModuleList()
        for i in range(num_experts):
            depth = 1 + (i % 3)  # Chu kỳ 1-hop, 2-hop, 3-hop
            heads = 2 if depth > 1 else 1
            self.experts.append(
                GATExpert(
                    in_channels=in_channels,
                    hidden_channels=hidden_dim,
                    out_channels=out_channels,
                    num_layers=depth,
                    heads=heads,
                    dropout=dropout
                )
            )

    def forward(self, batch_data):
        """
        Input: batch_data (PyG Batch Object)
        """
        x, edge_index, batch_idx = batch_data.x, batch_data.edge_index, batch_data.batch

        # 1. Tính Routing Weights
        routing_logits = self.gate(x)
        routing_weights = F.softmax(routing_logits, dim=-1)  # [Total_Nodes, Num_Experts]

        # 2. Lấy output từ từng Expert
        expert_outputs = []
        for expert in self.experts:
            out = expert(x, edge_index)  # [Total_Nodes, 1]
            expert_outputs.append(out)

        # Stack lại: [Total_Nodes, Num_Experts, 1]
        expert_outputs = torch.stack(expert_outputs, dim=1)

        # 3. Weighted Sum (Mixture of Experts)
        weights = routing_weights.unsqueeze(-1)  # [Total_Nodes, Num_Experts, 1]
        final_output = torch.sum(expert_outputs * weights, dim=1)  # [Total_Nodes, 1]

        # 4. Reshape về lại [Batch_Size, Num_Nodes]
        out_dense, mask = to_dense_batch(final_output, batch_idx)

        return out_dense