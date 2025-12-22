import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch # <--- QUAN TRỌNG: Để reshape lại output

class PMoE(nn.Module):
    def __init__(self, num_nodes, num_experts=3, in_channels=1, hidden_dim=32, out_channels=1):
        super(PMoE, self).__init__()
        self.num_nodes = num_nodes
        self.num_experts = num_experts
        
        # 1. Routing Network (Gating)
        # Input: Node features phẳng hoặc aggregated graph features
        # Ở đây dùng Linear đơn giản projection từ feature nodes
        self.gate = nn.Linear(in_channels, num_experts) 

        # 2. Experts (List of GNNs)
        # Sử dụng GATConv hoặc GCNConv tùy config, ở đây ví dụ dùng GAT
        from torch_geometric.nn import GATConv
        self.experts = nn.ModuleList([
            GATConv(in_channels, hidden_dim, heads=1, concat=False)
            for _ in range(num_experts)
        ])
        
        # Output layer để đưa về 1 scalar (xác suất lây lan)
        self.output_layer = nn.Linear(hidden_dim, out_channels)

    def forward(self, batch_data):
        """
        Input: batch_data (PyG Batch Object)
        """
        # Unpack dữ liệu từ Batch object
        x, edge_index, batch_idx = batch_data.x, batch_data.edge_index, batch_data.batch

        # 1. Tính Routing Weights
        # x shape: [Total_Nodes_in_Batch, Features]
        routing_logits = self.gate(x) 
        routing_weights = F.softmax(routing_logits, dim=1) # [Total_Nodes, Num_Experts]

        # 2. Chạy qua từng Expert
        expert_outputs = []
        for expert in self.experts:
            # GATConv nhận input 2D chuẩn -> KHÔNG BỊ LỖI NỮA
            out = expert(x, edge_index) 
            out = F.relu(out)
            out = self.output_layer(out) # [Total_Nodes, 1]
            expert_outputs.append(out)
        
        # Stack lại: [Total_Nodes, Num_Experts, 1]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # 3. Weighted Sum (Mixture)
        # routing_weights: [Total_Nodes, Num_Experts] -> unsqueeze thành [Total_Nodes, Num_Experts, 1]
        routing_weights = routing_weights.unsqueeze(-1)
        
        # Tổng hợp kết quả từ các chuyên gia
        # final_output: [Total_Nodes, 1]
        final_output = torch.sum(expert_outputs * routing_weights, dim=1)
        
        # 4. Reshape về lại [Batch_Size, Num_Nodes] để tính Loss
        # Trả về: out_dense [Batch_Size, Num_Nodes, 1], mask [Batch_Size, Num_Nodes]
        out_dense, mask = to_dense_batch(final_output, batch_idx)
        
        return out_dense