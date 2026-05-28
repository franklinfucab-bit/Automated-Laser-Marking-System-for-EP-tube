import os

def unite_labels(directory):
    if not os.path.exists(directory):
        return
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                with open(file_path, 'w') as f:
                    for line in lines:
                        # 强制把第一列改为 0
                        parts = line.split()
                        if len(parts) > 0:
                            parts[0] = '0'
                            f.write(' '.join(parts) + '\n')
    print(f"✅ {directory} 类别已全部统一为 0")

unite_labels(r'dataset/train/labels')
unite_labels(r'dataset/val/labels')