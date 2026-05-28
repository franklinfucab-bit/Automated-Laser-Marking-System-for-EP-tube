from ultralytics import YOLO
import torch

# 1. 初始化模型
# 使用 yolov8n.pt (Nano版)，它体积小、速度极快，最适合在实验室电脑上实时跑
model = YOLO('yolov8n.pt') 

if __name__ == '__main__':
    print("🚀 炼丹炉起火！正在基于你的标注训练专属 AI...")
    
    # 2. 开始训练
    model.train(
        data='data.yaml', 
        epochs=100,       # 数据量小，可以多跑几轮让它记牢
        imgsz=640,        # 输入分辨率
        batch=4,          # 每次喂 4 张图
        device='cpu',     # 如果你有显卡可以改成 0，没有就写 cpu
        workers=0         # Windows 用户建议设为 0 防止多线程报错
    )

    print("\n✅ 训练大功告成！")
    print("🎯 你的“最强大脑”已生成在: runs/detect/train/weights/best.pt")