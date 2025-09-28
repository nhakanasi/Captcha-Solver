import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms
import numpy as np

class CaptchaClassifierDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None, target_size=(28, 28)):
        """
        Dataset for character classification from YOLO annotations
        
        Args:
            annotation_file: Path to yolo.txt file
            img_dir: Directory containing images
            transform: Optional transform to be applied on a sample
            target_size: Size to resize cropped characters to
        """
        self.img_dir = img_dir
        self.transform = transform
        self.target_size = target_size
        self.samples = []
        
        # Default transform for classifier (grayscale, normalized for MNIST-like model)
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])  # Grayscale normalization
            ])
        
        # Parse annotation file
        self._parse_annotations(annotation_file)
        
    def _parse_annotations(self, annotation_file):
        """Parse yolo.txt and extract character samples"""
        with open(annotation_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 6:  # Need at least img_path + 5 values for 1 box
                continue
                
            img_path = parts[0]
            # Handle both absolute and relative paths
            if not os.path.isabs(img_path):
                img_path = os.path.join(self.img_dir, img_path)
            
            # Parse the 5 character boxes
            # Format: img_path x1 y1 w1 h1 class1 x2 y2 w2 h2 class2 x3 y3 w3 h3 class3 x4 y4 w4 h4 class4 x5 y5 w5 h5 class5
            boxes_data = parts[1:]  # Skip image path
            
            # Should have exactly 25 values (5 boxes * 5 values each)
            if len(boxes_data) != 25:
                print(f"Warning: Expected 25 values for {img_path}, got {len(boxes_data)}")
                continue
            
            # Parse each of the 5 boxes
            for i in range(5):
                start_idx = i * 5
                try:
                    x_center = float(boxes_data[start_idx])
                    y_center = float(boxes_data[start_idx + 1])
                    width = float(boxes_data[start_idx + 2])
                    height = float(boxes_data[start_idx + 3])
                    class_id = int(boxes_data[start_idx + 4])  # Class is the 5th value
                    
                    self.samples.append({
                        'img_path': img_path,
                        'class_id': class_id,
                        'bbox': (x_center, y_center, width, height),
                        'character_position': i  # 0-4 for position in CAPTCHA
                    })
                except (ValueError, IndexError) as e:
                    print(f"Error parsing box {i} in {img_path}: {e}")
                    continue
        
        print(f"Loaded {len(self.samples)} character samples from {annotation_file}")
        print(f"That's {len(self.samples)//5} images with 5 characters each")
    
    def _yolo_to_pixel_coords(self, bbox, img_width, img_height):
        """Convert YOLO format (x_center, y_center, width, height) to pixel coordinates"""
        x_center, y_center, width, height = bbox
        
        # Convert normalized coordinates to pixel coordinates
        x_center_pixel = x_center * img_width
        y_center_pixel = y_center * img_height
        width_pixel = width * img_width
        height_pixel = height * img_height
        
        # Convert to corner coordinates
        x1 = int(x_center_pixel - width_pixel / 2)
        y1 = int(y_center_pixel - height_pixel / 2)
        x2 = int(x_center_pixel + width_pixel / 2)
        y2 = int(y_center_pixel + height_pixel / 2)
        
        return x1, y1, x2, y2
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['img_path']
        class_id = sample['class_id']
        bbox = sample['bbox']
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy sample
            dummy_img = Image.new('L', self.target_size, 0)
            if self.transform:
                dummy_img = self.transform(dummy_img)
            return dummy_img, 0
        
        # Convert YOLO bbox to pixel coordinates
        x1, y1, x2, y2 = self._yolo_to_pixel_coords(bbox, image.width, image.height)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, image.width - 1))
        y1 = max(0, min(y1, image.height - 1))
        x2 = max(x1 + 1, min(x2, image.width))
        y2 = max(y1 + 1, min(y2, image.height))
        
        # Crop the character
        cropped = image.crop((x1, y1, x2, y2))
        
        # Convert to grayscale and resize
        cropped = cropped.convert('L')
        cropped = cropped.resize(self.target_size, Image.LANCZOS)
        
        # Apply transforms
        if self.transform:
            cropped = self.transform(cropped)
        
        return cropped, class_id
