# TODO: Implement logic here
import torch
from torch.utils.data import Dataset
import pickle
import os

class REMDataset(Dataset):
    def __init__(self, data_path):
        """
        Args:
            data_path (str): Đường dẫn đến file .SG (ví dụ: data/processed/train_data.SG)
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Không tìm thấy file data tại: {data_path}")
            
        with open(data_path, 'rb') as f:
            # Dữ liệu là list các tuple (x_vector, y_vector)
            self.data = pickle.load(f)
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        x, y = self.data[idx]
        # Chuyển numpy array thành PyTorch Tensor (float32)
        return torch.FloatTensor(x), torch.FloatTensor(y)