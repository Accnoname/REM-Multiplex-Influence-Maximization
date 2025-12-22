# TODO: Implement logic here
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv

class GATExpert(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, heads=2):
        super(GATExpert, self).__init__()
        self.layers = nn.ModuleList()
        
        # Layer đầu tiên
        self.layers.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True))
        
        # Các layer ẩn ở giữa (nếu num_layers > 2)
        # Bài báo nói các experts có "varying layer depths" (độ sâu khác nhau) [cite: 145]
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True))
            
        # Layer cuối cùng (Output ra 1 giá trị xác suất nhiễm cho mỗi node)
        self.layers.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False))
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index):
        # x: Feature của node (trong bài toán này ban đầu có thể là one-hot hoặc seed vector)
        for conv in self.layers[:-1]:
            x = conv(x, edge_index)
            x = self.relu(x)
            x = self.dropout(x)
            
        # Layer cuối
        x = self.layers[-1](x, edge_index)
        # Output chưa qua Sigmoid vì PMoE sẽ tổng hợp trước
        return x