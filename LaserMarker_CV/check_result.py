from ultralytics import YOLO
import cv2
import os

# 1. 指向你最新的模型
model_path = r'C:\Users\Frank\LaserMarker_CV\runs\detect\train5\weights\best.pt'
model = YOLO(model_path)

# 2. 挑选一张测试图片 (选一张 val 里的或者全新的)
img_path = r'C:\Users\Frank\LaserMarker_CV\dataset\val\images\black_in_white2.jpg'

# 3. 运行预测
# save=True 会把画好框的图存在 runs/detect/predict 文件夹里
# 修改这一行，强制使用 CPU 绕过损坏的 CUDA 算子
results = model.predict(
    source=img_path, 
    save=True, 
    conf=0.15,      # 调低一点门槛，看看能不能抓到更多管子
    device='cpu'    # <--- 加入这个参数
)
# 4. 打印每一个打标点的坐标
print("\n" + "="*30)
print("🚀 激光打标路径点位预览:")
for r in results:
    for i, box in enumerate(r.boxes):
        # 获取中心点坐标 (x, y)
        xywh = box.xywh.cpu().numpy()[0]
        center_x, center_y = xywh[0], xywh[1]
        conf = box.conf.cpu().numpy()[0]
        print(f"管子 #{i+1:02d} | 像素坐标: ({int(center_x)}, {int(center_y)}) | 置信度: {conf:.2f}")
print("="*30)

print(f"\n💡 结果图已保存至: {results[0].save_dir}")