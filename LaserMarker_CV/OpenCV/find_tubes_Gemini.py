import cv2
import numpy as np

image_path = 'Multi_white_tube.jpg'
img = cv2.imread(image_path)

if img is None:
    print("❌ 找不到图片，请检查文件名！")
    exit()

output = img.copy()

# 1. 依然使用无敌的“蓝通道魔法”
b_channel, _, _ = cv2.split(img)

# 稍微放宽一点二值化的阈值 (把 150 降到 80，防止边缘太暗被切掉)
_, binary_mask = cv2.threshold(b_channel, 80, 255, cv2.THRESH_BINARY)
kernel = np.ones((5,5), np.uint8)
clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print("🔍 [Debug 模式] 打印所有明显轮廓的真实数据：")
tube_count = 0

for cnt in contours:
    area = cv2.contourArea(cnt)
    
    # 门槛极低：只要不是太小的芝麻粒，全放进来看看！
    if area > 1000: 
        tube_count += 1
        
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0
        
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / float(hull_area) if hull_area > 0 else 0
        
        # 算个中心点用来画图
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # 把轮廓画出来，看看它到底抓到了什么形状
            cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)
            cv2.putText(output, str(tube_count), (cX - 10, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        print(f"👉 疑似管盖 {tube_count} | 面积: {area:.0f} | 圆度: {circularity:.2f} | 实心度: {solidity:.2f}")

print(f"\n🎯 宽容模式下，总共抓到了 {tube_count} 个物体。")

# 显示图片 (一定要看 Mask 那张图！)
height, width = output.shape[:2]
scale = 800 / height
cv2.imshow("1. Clean Mask (看看白块连在一起没？)", cv2.resize(clean_mask, (int(width * scale), 800)))
cv2.imshow("2. Debug Output", cv2.resize(output, (int(width * scale), 800)))
cv2.waitKey(0)
cv2.destroyAllWindows()