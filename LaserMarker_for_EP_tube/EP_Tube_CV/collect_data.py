import cv2
import os
import time

# ================= 配置区域 =================
# 换成你的 ESP32 IP
STREAM_URL = "http://100.100.22.159/" 
# ===========================================

# 只需要一个文件夹存放所有待标注的图片
save_dir = 'dataset/raw_images'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

print(f"尝试连接视频流: {STREAM_URL} ...")
cap = cv2.VideoCapture(STREAM_URL)

if not cap.isOpened():
    print("错误: 无法打开视频流。")
    exit()

print("=== 目标检测数据采集模式 ===")
print("调整好板子上的管子数量和位置...")
print("按 '空格键' -> 拍照保存")
print("按 'q' -> 退出")

img_count = len(os.listdir(save_dir))

while True:
    ret, frame = cap.read()
    if not ret:
        print("信号丢失，正在重连...")
        time.sleep(1)
        cap = cv2.VideoCapture(STREAM_URL)
        continue

    # 显示画面
    cv2.imshow("ESP32 Map Collector", frame)
    
    key = cv2.waitKey(1) & 0xFF

    # 按 空格键 保存
    if key == 32: # 32 是空格键的 ASCII 码
        filename = f"{save_dir}/rack_scan_{img_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 已保存布局图: {filename}")
        img_count += 1

    # 按 'q' 退出
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()