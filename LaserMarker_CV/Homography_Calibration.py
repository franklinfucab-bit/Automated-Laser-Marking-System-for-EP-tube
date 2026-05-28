import cv2
import numpy as np

# ================= 配置区 =================
# 替换为你实际的一张测试图片路径（找一张深色管子多的）
IMG_PATH = r'C:\Users\Frank\LaserMarker_CV\dataset\val\images\black_in_white2.jpg'

# 试管架的物理参数 (请根据你的橙色管架实际情况修改)
ROWS = 8        # 行数 (纵向)
COLS = 12       # 列数 (横向)
SPACING = 15.0  # 相邻两个孔的中心物理间距 (假设为 10mm，你可以用尺子量一下)

# 存储你鼠标点击的像素点
clicked_points = []

# ================= 核心逻辑 =================
def click_event(event, x, y, flags, param):
    global clicked_points, img_copy
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append([x, y])
            # 在你点击的地方画个红色的实心圆
            cv2.circle(img_copy, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Calibration", img_copy)
            
            print(f"📍 记录点 {len(clicked_points)}: 像素坐标 ({x}, {y})")
            
            # 当收集齐 4 个点时，立刻施展数学魔法
            if len(clicked_points) == 4:
                calculate_and_draw_grid()

def calculate_and_draw_grid():
    global img_copy
    print("\n🚀 4个锚点已集齐，正在侦测管架朝向...")
    
    # 1. 提取点击的像素坐标
    pt_TL = np.array(clicked_points[0]) # 左上
    pt_TR = np.array(clicked_points[1]) # 右上
    pt_BL = np.array(clicked_points[2]) # 左下
    pts_src = np.array(clicked_points, dtype='float32')
    
    # 2. 自动判定朝向 (利用向量的 L2 范数计算像素距离)
    dist_top_edge = np.linalg.norm(pt_TR - pt_TL)
    dist_left_edge = np.linalg.norm(pt_BL - pt_TL)
    
    # 定义标准 EP 管架的孔数 (通常是 8 x 12 = 96 孔)
    # 你可以根据实际情况修改这两个常量
    HOLES_LONG_SIDE = 12
    HOLES_SHORT_SIDE = 8
    SPACING = 10.0 # 物理孔距 (mm)
    
    if dist_top_edge > dist_left_edge:
        print("📐 侦测结果: 横向放置 (Landscape)")
        current_cols = HOLES_LONG_SIDE
        current_rows = HOLES_SHORT_SIDE
    else:
        print("📐 侦测结果: 纵向放置 (Portrait)")
        current_cols = HOLES_SHORT_SIDE
        current_rows = HOLES_LONG_SIDE
        
    # 3. 动态生成对应的物理坐标映射 (mm)
    pts_dst = np.array([
        [0, 0],                                                          # 左上
        [(current_cols-1)*SPACING, 0],                                   # 右上
        [0, (current_rows-1)*SPACING],                                   # 左下
        [(current_cols-1)*SPACING, (current_rows-1)*SPACING]             # 右下
    ], dtype='float32')
    
    # 4. 计算单应性矩阵 H
    H_phys_to_pixel, _ = cv2.findHomography(pts_dst, pts_src)
    
    # 5. 生成所有孔位并画图
    for row in range(current_rows):
        for col in range(current_cols):
            phys_x = col * SPACING
            phys_y = row * SPACING
            
            pt_phys = np.array([[[phys_x, phys_y]]], dtype='float32')
            pt_pixel = cv2.perspectiveTransform(pt_phys, H_phys_to_pixel)
            px, py = int(pt_pixel[0][0][0]), int(pt_pixel[0][0][1])
            
            # 画出计算出的中心点
            cv2.circle(img_copy, (px, py), 15, (0, 255, 0), 2)
            cv2.circle(img_copy, (px, py), 2, (0, 0, 255), -1)

    cv2.imshow("Calibration", img_copy)
    print("✅ 动态网格已完美覆盖！")
    
# ================= 主程序 =================
img = cv2.imread(IMG_PATH)
if img is None:
    print("❌ 找不到图片，请检查路径！")
    exit()

# 如果图片太大，缩小一点方便在屏幕上点
img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
img_copy = img.copy()

print("="*40)
print("请在弹出的窗口中，按以下顺序点击试管架的 4 个角：")
print("1️⃣ 左上角管子中心")
print("2️⃣ 右上角管子中心")
print("3️⃣ 左下角管子中心")
print("4️⃣ 右下角管子中心")
print("="*40)

cv2.imshow("Calibration", img_copy)
cv2.setMouseCallback("Calibration", click_event)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()