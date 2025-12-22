import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_comparison(your_result=199.02):
    # 1. Dữ liệu từ Paper (Table 1 - Celegans IC - Budget 1%)
    # 
    methods = ['ISF', 'KSN', 'MIM-Reasoner', 'REM (Paper)', 'YOUR RESULT']
    
    # Giá trị Spread (Influence Spread)
    # Lưu ý: Paper dùng budget 1% (~39 nodes), bạn dùng 50 nodes
    scores = [398.34, 398.31, 398.22, 426.69, your_result]
    
    # Màu sắc: Xám cho baseline, Xanh cho REM paper, Đỏ cho kết quả của bạn
    colors = ['#bdc3c7', '#bdc3c7', '#95a5a6', '#2ecc71', '#e74c3c']

    # 2. Thiết lập biểu đồ
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Vẽ Bar Chart
    bars = plt.bar(methods, scores, color=colors, edgecolor='black', alpha=0.8)
    
    # Thêm số liệu lên đầu cột
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 5, f'{yval:.1f}', 
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 3. Trang trí
    plt.title('Comparison on Celegans Dataset (IC Model)', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Influence Spread (Expected Infected Nodes)', fontsize=12)
    plt.xlabel('Methods', fontsize=12)
    plt.ylim(0, max(scores) * 1.2) # Tăng giới hạn trục Y để thoáng
    
    # Thêm chú thích
    plt.axhline(y=your_result, color='red', linestyle='--', alpha=0.5)
    plt.text(len(methods)-1, your_result + 20, "Current Pipeline", color='red', ha='center')

    # Lưu và hiển thị
    plt.tight_layout()
    plt.savefig('result_comparison.png', dpi=300)
    print(">> Đã lưu biểu đồ vào 'result_comparison.png'")
    plt.show()

if __name__ == "__main__":
    # Thay số 199.02 bằng số thực tế bạn vừa chạy được
    plot_comparison(your_result=199.02)