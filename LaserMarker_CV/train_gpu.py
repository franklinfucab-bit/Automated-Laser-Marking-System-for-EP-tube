from ultralytics import YOLO
import torch

def train():
    model = YOLO('yolov8n.pt') 
    model.train(
        data='data.yaml', 
        epochs=100, 
        batch=16, 
        device=0,        # 明确使用 4060
        imgsz=640,
        mosaic=1.0,      # 开启马赛克增强，提升小目标识别
        hsv_v=0.6,       # 提高亮度抖动，让 AI 适应深色管子
        workers=0        # Windows 稳定性最高设置
    )

if __name__ == '__main__':
    train()