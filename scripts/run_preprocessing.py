# TODO: Implement logic here
# scripts/run_preprocessing.py
import sys
import os

# Thêm thư mục gốc vào path để Python tìm thấy module 'src'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.preprocessing import DataProcessor

if __name__ == "__main__":
    processor = DataProcessor()
    
    # Bước 1: Đọc Raw Data -> Graph Object
    processor.load_raw_graph()
    
    # Bước 2: Chạy Simulation -> .SG File
    processor.generate_training_data()