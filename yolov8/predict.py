from ultralytics import YOLO

# 加载训练好的 best.pt
model = YOLO(r"runs\detect\train4\weights\best.pt")

# 推理验证集，并保存预测框坐标
model.predict(
    source=r"datasets\images\val",  # 验证集图片路径
    imgsz=640,
    conf=0.25,
    save=True,      # 保存预测后的图片
    save_txt=True   # 保存预测框坐标到 labels 文件夹
)
