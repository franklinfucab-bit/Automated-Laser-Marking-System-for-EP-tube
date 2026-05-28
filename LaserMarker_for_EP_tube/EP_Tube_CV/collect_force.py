import cv2
import requests
import numpy as np
import os

# ================= 配置区域 =================
URL = "http://100.100.22.159/"
# ===========================================

print(f"🚀 正在尝试强力连接: {URL}")

save_dir = 'dataset/raw_images'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
img_count = len(os.listdir(save_dir))

try:
    stream = requests.get(URL, stream=True, timeout=10)
    if stream.status_code == 200:
        print("✅ 连接建立！按【空格键】拍照，按【q】退出")
        bytes_data = bytes()
        
        for chunk in stream.iter_content(chunk_size=4096):
            bytes_data += chunk
            
            # 循环处理缓冲区里的所有完整图片
            while True:
                # 1. 找开头
                a = bytes_data.find(b'\xff\xd8')
                if a == -1:
                    # 没找到开头，说明数据还在路上，保留最后一点以防断在头里
                    if len(bytes_data) > 10000: # 防溢出
                        bytes_data = bytes_data[-1024:] 
                    break 
                
                # 2. 找结尾（必须在开头之后）
                b = bytes_data.find(b'\xff\xd9', a)
                if b == -1:
                    # 有开头没结尾，说明图片还没传完，跳出等待更多数据
                    break
                
                # 3. 提取图片数据
                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:] # 移动指针到下一张
                
                # 4. 安全解码（核心修复：加了 try-except）
                if len(jpg) > 0:
                    try:
                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        
                        if frame is not None:
                            cv2.imshow('ESP32 Robust Collector', frame)
                            
                            key = cv2.waitKey(1) & 0xFF
                            if key == 32: # 空格键
                                filename = f"{save_dir}/scan_{img_count}.jpg"
                                cv2.imwrite(filename, frame)
                                print(f"📸 保存成功: {filename}")
                                img_count += 1
                            elif key == ord('q'):
                                raise StopIteration # 退出外层循环
                    except Exception as e:
                        print(f"⚠️ 跳过一帧坏图: {e}")
                        continue
                        
    else:
        print(f"❌ 连接失败，状态码: {stream.status_code}")

except StopIteration:
    print("用户退出。")
except Exception as e:
    print(f"💥 网络错误: {e}")

print("已关闭连接。")
cv2.destroyAllWindows()