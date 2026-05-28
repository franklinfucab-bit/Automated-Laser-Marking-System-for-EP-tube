import cv2
import numpy as np

image_path = 'Multi_white_tube.jpg'
img = cv2.imread(image_path)

if img is None:
    print("❌ 找不到图片！")
    exit()

output = img.copy()

# 提取各颜色通道
b_channel, g_channel, r_channel = cv2.split(img)
blurred_b = cv2.medianBlur(b_channel, 7)

print("🎯 正在执行 [全量扫描] + [多维特征过滤]...")

# 1. 宽松扫描：宁可错杀一千，不可放过一个 (param2=22，极其宽松)
circles = cv2.HoughCircles(
    blurred_b, 
    cv2.HOUGH_GRADIENT, 
    dp=1.2, 
    minDist=90,       # 管子之间的最小距离 
    param1=50,        
    param2=22,        # 足够低，保证 10 个管子绝对全都在里面
    minRadius=25,     
    maxRadius=65      
)

tube_count = 0

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    
    for (cX, cY, r) in circles:
        # 防止越界报错
        # ✅ CRITICAL: img.shape is tuple (height, width, channels). Must use img.shape[1] for width and img.shape[0] for height
        if cX < 0 or cX >= img.shape[1] or cY < 0 or cY >= img.shape[0]:
            continue

        # ==========================================
        # 🛑 过滤网 1：尺寸过滤器 (Size Filter)
        # 真实的内环半径应该在一个稳定区间
        # ==========================================
        if r < 30 or r > 60:
            continue

        # ==========================================
        # 🛑 过滤网 2：亮度和色彩差测谎仪 (Color & Brightness Filter)
        # 橙色底座孔：极其暗 (B低)，且红蓝差值极大 (R >> B)
        # 白色EP管盖：中心明亮 (B高)，且红蓝差值极小 (R ≈ B)
        # ==========================================
        py1, py2 = max(0, cY - 5), min(img.shape[0], cY + 5)
        px1, px2 = max(0, cX - 5), min(img.shape[1], cX + 5)
        
        avg_r = int(np.mean(r_channel[py1:py2, px1:px2]))
        avg_b = int(np.mean(b_channel[py1:py2, px1:px2]))
        color_diff = avg_r - avg_b
        
        # 过滤条件：中心如果太黑（< 100），或者呈现强烈的橙红色（红蓝差 > 50），直接踢掉！
        if avg_b < 100 or color_diff > 50:
            continue
        # ==========================================

        tube_count += 1
        
        # --- 局部角度探测魔法 ---
        roi_size = int(r * 2.5) 
        
        # 修复后的边界保护 (严格指定 img.shape[0] for height, img.shape[1] for width)
        y1 = max(0, int(cY - roi_size))
        y2 = min(img.shape[0], int(cY + roi_size))
        x1 = max(0, int(cX - roi_size))
        x2 = min(img.shape[1], int(cX + roi_size))
        
        # 防止截取到空区域
        if y2 <= y1 or x2 <= x1:
            continue

        roi_b = b_channel[y1:y2, x1:x2]
        _, roi_thresh = cv2.threshold(roi_b, 130, 255, cv2.THRESH_BINARY)
        
        M = cv2.moments(roi_thresh)
        angle = 0.0
        if M["m00"] != 0:
            local_cX = int(M["m10"] / M["m00"])
            local_cY = int(M["m01"] / M["m00"])
            
            dy = (local_cY + y1) - cY
            dx = (local_cX + x1) - cX
            angle = np.rad2deg(np.arctan2(dy, dx))
            
            # 画红色指针
            line_len = r * 1.8
            endX = int(cX + line_len * np.cos(np.deg2rad(angle)))
            endY = int(cY + line_len * np.sin(np.deg2rad(angle)))
            cv2.line(output, (cX, cY), (endX, endY), (0, 0, 255), 3)

        # 画绿色外圈和蓝色靶心
        cv2.circle(output, (cX, cY), r, (0, 255, 0), 2)
        cv2.circle(output, (cX, cY), 5, (255, 0, 0), -1)
        
        print(f"✅ 锁定管盖 {tube_count:02d} | 坐标: ({cX:4d}, {cY:4d}) | 亮度: {avg_b:3d} | R-B差: {color_diff:3d} | 角度: {angle:5.1f}°")

print(f"\n🎯 多维过滤完成！完美锁定 {tube_count} 个管盖。")

# 缩小显示结果
height, width = output.shape[:2]
scale = 800 / height
output_resized = cv2.resize(output, (int(width * scale), 800))

cv2.imshow("Multi-Filter Targeting", output_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()