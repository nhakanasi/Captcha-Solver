import numpy as np
import random
from pathlib import Path

def iou(box, clusters):
    w = np.minimum(clusters[:, 0], box[0])
    h = np.minimum(clusters[:, 1], box[1])
    inter = w * h
    box_a = box[0] * box[1]
    cl_a = clusters[:, 0] * clusters[:, 1]
    return inter / (box_a + cl_a - inter + 1e-10)

def kmeans(boxes, k, dist=np.median, max_iter=1000):
    n = boxes.shape[0]
    assert n > 0, "No boxes provided."
    k = min(k, n)
    clusters = boxes[random.sample(range(n), k)]
    last = np.full(n, -1)
    for _ in range(max_iter):
        dists = np.stack([1 - iou(b, clusters) for b in boxes], 0)  # n x k
        assign = dists.argmin(1)
        if np.all(assign == last):
            break
        for ci in range(k):
            if np.any(assign == ci):
                clusters[ci] = dist(boxes[assign == ci], axis=0)
        last = assign
    return clusters

def avg_iou(boxes, clusters):
    return np.mean([np.max(iou(b, clusters)) for b in boxes])

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent.parent
    yolo_path = script_dir / "yolo.txt"   # adjust path if needed
    if not yolo_path.exists():
        print(f"yolo.txt not found at {yolo_path}")
        raise SystemExit

    boxes = []
    with yolo_path.open("r") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 6:
                continue
            vals = tokens[1:]  # skip image path
            # groups of 5: cx cy w h cls
            for i in range(0, len(vals), 5):
                if i + 4 >= len(vals):
                    break
                try:
                    cx, cy, w, h, cls_id = map(float, vals[i:i+5])
                except ValueError:
                    continue
                if w > 0 and h > 0:
                    boxes.append([w, h])

    if not boxes:
        print("Parsed 0 boxes. Check yolo.txt format.")
        raise SystemExit

    boxes = np.array(boxes, dtype=np.float32)
    print(f"Parsed boxes: {boxes.shape[0]}")

    k = 20
    clusters = kmeans(boxes, k)
    clusters = clusters[np.argsort(clusters[:, 0])]
    print("Anchors (w,h):")
    for w, h in clusters:
        print(f"({w:.5f},{h:.5f}),")
    print(f"Average IoU: {avg_iou(boxes, clusters):.4f}")