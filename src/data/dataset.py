import os
import pickle
import torch
from torch.utils.data import Dataset
from torch_geometric.utils import coalesce


class REMDataset(Dataset):
    """Binary seed vectors + spread labels from .SG file."""

    def __init__(self, data_path):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data not found: {data_path}")
        try:
            with open(data_path, "rb") as f:
                self.data = pickle.load(f)
        except Exception:
            self.data = torch.load(data_path, weights_only=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return (
            x.float() if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32),
            y.float() if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32),
        )


def load_graph(cfg, device="cpu"):
    """Load multiplex graph pkl → (num_nodes, unified edge_index, graph_list)."""
    path = os.path.join(cfg["dataset"]["processed_dir"], "multiplex_graph.pkl")
    with open(path, "rb") as f:
        gd = pickle.load(f)

    raw = gd["graphs"]
    graph_list = list(raw.values()) if isinstance(raw, dict) else raw

    all_u, all_v = [], []
    for G in graph_list:
        edges = list(G.edges())
        if edges:
            u_arr, v_arr = zip(*edges)
            all_u.append(torch.tensor(u_arr, dtype=torch.long))
            all_v.append(torch.tensor(v_arr, dtype=torch.long))

    combined = torch.stack([torch.cat(all_u), torch.cat(all_v)])
    edge_index, _ = coalesce(combined, None, num_nodes=gd["num_nodes"])
    return gd["num_nodes"], edge_index.to(device), graph_list
