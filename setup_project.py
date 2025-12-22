import os

def create_structure():
    project_name = "."
    
    # Danh sách các thư mục cần tạo
    folders = [
        "data/raw",
        "data/processed",
        "configs",
        "src/data",
        "src/models",
        "src/core",
        "src/utils",
        "experiments",
        "scripts"
    ]
    
    # Danh sách các file rỗng cần tạo (để giữ chỗ)
    files = [
        "configs/base_config.yaml",
        "configs/hyperparams.yaml",
        "src/__init__.py",
        "src/data/__init__.py",
        "src/data/preprocessing.py",
        "src/data/simulation.py",
        "src/data/dataset_loader.py",
        "src/models/__init__.py",
        "src/models/seed2vec.py",
        "src/models/experts.py",
        "src/models/pmoe.py",
        "src/core/__init__.py",
        "src/core/exploration.py",
        "src/core/trainer.py",
        "src/utils/__init__.py",
        "src/utils/logger.py",
        "scripts/run_preprocessing.py",
        "scripts/train.py",
        "requirements.txt",
        "README.md",
        ".gitignore"
    ]

    print(f"🚀 Đang khởi tạo dự án: {project_name}...")
    
    # 1. Tạo thư mục
    for folder in folders:
        path = os.path.join(project_name, folder)
        os.makedirs(path, exist_ok=True)
        # Tạo file .gitkeep để Git nhận diện thư mục rỗng
        with open(os.path.join(path, ".gitkeep"), "w") as f:
            pass
        print(f"   ✅ Created dir: {path}")

    # 2. Tạo file
    for file in files:
        path = os.path.join(project_name, file)
        if not os.path.exists(path):
            with open(path, "w") as f:
                if file.endswith(".py"):
                    f.write("# TODO: Implement logic here\n")
                elif file.endswith(".yaml"):
                    f.write("# Configuration file\n")
            print(f"   📄 Created file: {path}")
            
    print("\n🎉 HOÀN TẤT! Cấu trúc dự án đã sẵn sàng.")
    print(f"👉 Hãy copy data gốc vào: {project_name}/data/raw/")

if __name__ == "__main__":
    create_structure()