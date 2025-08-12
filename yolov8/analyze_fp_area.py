# analyze_fp_area.py
# 统计验证集上的“误检”框，并查看这些误检的面积占整图的比例分布
# 预测 txt 兼容 5 列(cls cx cy w h) 与 6 列(cls cx cy w h conf)

from pathlib import Path
import glob
import csv

# ====== 路径（按需修改）======
# 预测结果：Ultralytics predict 输出目录下的 labels
PRED_DIR = Path("runs/detect/predict/labels")   # 例如 runs/detect/val/labels 也可以
# 验证集标注（GT）
GT_DIR   = Path("datasets/labels/val")

# IoU 阈值：与任一 GT 的 IoU < 该值则视为误检(FP)
IOU_THR = 0.20

# 是否将误检样本导出为 CSV
SAVE_CSV = True
CSV_PATH = Path("runs/detect/predict_fp_report.csv")


def bboxes_from_txt(p: Path, is_pred: bool):
    """读取 txt 中的框（归一化坐标）。预测 txt 可能是 5 或 6 列，GT 必须是 5 列。"""
    boxes = []
    if not p.exists():
        return boxes
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            nums = list(map(float, parts))
            if is_pred:
                # 预测：Ultralytics 可能是 5 列或 6 列（conf 在最后）
                if len(nums) == 6:
                    cls, cx, cy, w, h, conf = nums
                elif len(nums) == 5:
                    cls, cx, cy, w, h = nums
                    conf = 1.0  # 没有置信度时给一个占位值
                else:
                    continue
                boxes.append({"cls": int(cls), "conf": conf, "cx": cx, "cy": cy, "w": w, "h": h})
            else:
                # GT：标准 5 列
                if len(nums) != 5:
                    continue
                cls, cx, cy, w, h = nums
                boxes.append({"cls": int(cls), "cx": cx, "cy": cy, "w": w, "h": h})
    return boxes


def iou_xywh(a, b):
    """IoU (xywh 为归一化坐标)"""
    def to_xyxy(bb):
        x1 = bb["cx"] - bb["w"] / 2
        y1 = bb["cy"] - bb["h"] / 2
        x2 = bb["cx"] + bb["w"] / 2
        y2 = bb["cy"] + bb["h"] / 2
        return x1, y1, x2, y2

    ax1, ay1, ax2, ay2 = to_xyxy(a)
    bx1, by1, bx2, by2 = to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union


def main():
    pred_txts = sorted(glob.glob(str(PRED_DIR / "*.txt")))
    if not pred_txts:
        print(f"未找到预测结果：{PRED_DIR}")
        return

    fps = []  # 收集误检
    for pred_txt in pred_txts:
        name = Path(pred_txt).name
        gt_txt = GT_DIR / name

        # 只看 DNB 类（cls==0），如果你有多类在这里改
        preds = [p for p in bboxes_from_txt(Path(pred_txt), True) if p["cls"] == 0]
        gts   = [g for g in bboxes_from_txt(gt_txt, False) if g["cls"] == 0]

        for p in preds:
            max_iou = max([iou_xywh(p, g) for g in gts], default=0.0)
            if max_iou < IOU_THR:
                area_ratio = p["w"] * p["h"]  # 归一化坐标，面积占比即 w*h
                fps.append({
                    "file": name.replace(".txt", ".jpg"),
                    "conf": round(p.get("conf", 1.0), 4),
                    "area_ratio": area_ratio
                })

    # 汇总
    print(f"总 FP 数: {len(fps)}")
    if not fps:
        return

    ratios = [x["area_ratio"] for x in fps]
    ratios_sorted = sorted(ratios)
    mean_ratio = sum(ratios) / len(ratios)
    median_ratio = ratios_sorted[len(ratios)//2]
    print(f"面积占比均值: {mean_ratio:.4f}, 中位数: {median_ratio:.4f}")

    # 粗分布
    bins = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 1.00]
    counts = [0] * len(bins)
    for r in ratios:
        for i, b in enumerate(bins):
            if r <= b:
                counts[i] += 1
                break
    print("面积占比分布（<=阈值: 数量）:")
    for b, c in zip(bins, counts):
        print(f"  <= {b:.3f}: {c}")

    # 列出“面积很小”的前若干例
    small = sorted([x for x in fps if x["area_ratio"] < 0.02], key=lambda x: x["area_ratio"])[:10]
    print("\n面积占比 < 2% 的样例（前10）:")
    for s in small:
        print({"file": s["file"], "conf": s["conf"], "area_ratio": round(s["area_ratio"], 5)})

    # 导出 CSV（可选）
    if SAVE_CSV:
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file", "conf", "area_ratio"])
            w.writeheader()
            for x in fps:
                w.writerow({"file": x["file"], "conf": x["conf"], "area_ratio": x["area_ratio"]})
        print(f"\n误检明细已导出：{CSV_PATH.resolve()}")


if __name__ == "__main__":
    main()
