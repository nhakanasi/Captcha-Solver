import os
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms
class CaptchaYOLODataset(Dataset):
    """
    yolo_txt line format:
    img/73663.png x y w h class x y w h class ...
    All coords normalized (center-x, center-y, width, height).
    """
    def __init__(self, yolo_txt_path, img_dir, img_size=(200,80), transform=None):
        self.records = []
        self.img_dir = img_dir
        self.img_size = img_size 
        self.transform = transform or transforms.Compose([
            transforms.Resize(80),
            transforms.Grayscale(num_output_channels=1),  # True grayscale
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) 
        ])
        with open(yolo_txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                img_path = parts[0]
                nums = list(map(float, parts[1:]))
                boxes = []
                classes = []
                for i in range(0, len(nums), 5):
                    x, y, w, h, cls = nums[i:i+5]
                    boxes.append([x, y, w, h])
                    classes.append(int(cls))
                self.records.append((img_path, boxes, classes))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rel_path, boxes, classes = self.records[idx]
        img_path = os.path.join(rel_path)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        boxes = torch.tensor(boxes, dtype=torch.float32)  # (N,4)
        classes = torch.tensor(classes, dtype=torch.long) # (N,)
        return img, boxes, classes