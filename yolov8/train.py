from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # 先用最轻的n版

model.train(
    data=r'E:\code\yolov8\datasets\data.yaml',
    epochs=80,            # 先80轮，配合早停更省时间
    patience=20,          # 20轮没提升自动停
    imgsz=640,            # CPU先用640，稳定；想更准再升到 960/1280
    batch=8,              # CPU建议8；慢或内存紧张就降到4
    workers=0,            # Windows/CPU 建议0
    device='cpu',         # 强制走CPU，避免奇怪的加速器冲突
    cache=True,           # 先把图片缓存到内存/磁盘，加速IO
    pretrained=True,      # 用预训练权重
    rect=True,            # 矩形训练，小数据更稳
    close_mosaic=10       # 末期关mosaic，提升收敛
)
