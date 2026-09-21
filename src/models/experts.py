# src/models/experts.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATExpert(nn.Module):
    """
    Expert GNN mô phỏng quá trình lan truyền với độ sâu khác nhau (varying layer depths).
    Mỗi expert chuyên biệt hóa ở một tầm lan tỏa (1-hop lân cận, 2-hop, hoặc multi-hop).
    """
    def __init__(self, in_channels=1, hidden_channels=32, out_channels=1, num_layers=2, heads=2, dropout=0.2):
        super(GATExpert, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(GATConv(in_channels, out_channels, heads=1, concat=False))
        else:
            # Layer đầu
            self.layers.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True))
            # Các layer giữa
            for _ in range(num_layers - 2):
                self.layers.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True))
            # Layer cuối chiếu về out_channels (xác suất lây lan của từng node)
            self.layers.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        if self.num_layers == 1:
            return self.layers[0](x, edge_index)

        for conv in self.layers[:-1]:
            x = conv(x, edge_index)
            x = self.relu(x)
            x = self.dropout(x)

        return self.layers[-1](x, edge_index)