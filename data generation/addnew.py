import pandas as pd

# Load the CSV
df = pd.read_csv(r"C:\Users\ALIENWARE\Downloads\labels_my-project-name_2025-09-15-08-44-57.csv")

# Group by image
grouped = df.groupby("image_name")

lines = []
for img_name, group in grouped:
    img_w = group.iloc[0]["image_width"]
    img_h = group.iloc[0]["image_height"]
    img_path = f"img/{img_name}"
    parts = [img_path]
    for _, row in group.iterrows():
        # Calculate normalized bbox center x, y, width, height
        x_center = (row["bbox_x"] + row["bbox_width"] / 2) / img_w
        y_center = (row["bbox_y"] + row["bbox_height"] / 2) / img_h
        w = row["bbox_width"] / img_w
        h = row["bbox_height"] / img_h
        label = int(row["label_name"])
        parts += [f"{x_center:.6f}", f"{y_center:.6f}", f"{w:.6f}", f"{h:.6f}", str(label)]
    lines.append(" ".join(parts))

# Save to yolo.txt
with open("yolo.txt", "a") as f:    
    for line in lines:
        f.write(line + "\n")