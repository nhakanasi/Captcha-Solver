import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def read_yolo_annotations(yolo_file):
    """Read YOLO format annotations from file"""
    annotations = []
    with open(yolo_file, 'r') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            img_path = parts[0]
            
            # Parse bounding boxes and classes
            boxes = []
            classes = []
            coords = list(map(float, parts[1:]))
            
            # Each box has 5 values: class, x_center, y_center, width, height
            for i in range(0, len(coords), 5):
                if i + 4 < len(coords):
                    class_id = int(coords[i + 4])  # class is last
                    x_center = coords[i]
                    y_center = coords[i + 1] 
                    width = coords[i + 2]
                    height = coords[i + 3]
                    
                    boxes.append([x_center, y_center, width, height])
                    classes.append(class_id)
            
            annotations.append({
                'index': idx,
                'image_path': os.path.join('Captcha',img_path),
                'boxes': boxes,
                'classes': classes
            })
    
    return annotations

def yolo_to_pixel_coords(box, img_width, img_height):
    """Convert YOLO normalized coordinates to pixel coordinates"""
    x_center, y_center, width, height = box
    
    # Convert to pixel coordinates
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height
    
    # Calculate corner coordinates
    x1 = int(x_center_px - width_px / 2)
    y1 = int(y_center_px - height_px / 2)
    x2 = int(x_center_px + width_px / 2)
    y2 = int(y_center_px + height_px / 2)
    
    return x1, y1, x2, y2

def show_statistics(yolo_file):
    """Show statistics about the dataset"""
    annotations = read_yolo_annotations(yolo_file)
    
    # Count statistics
    total_images = len(annotations)
    total_boxes = sum(len(ann['boxes']) for ann in annotations)
    
    # Class distribution
    class_counts = {}
    for ann in annotations:
        for class_id in ann['classes']:
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
    
    # Boxes per image distribution
    boxes_per_image = [len(ann['boxes']) for ann in annotations]
    
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    print(f"Total images: {total_images}")
    print(f"Total boxes: {total_boxes}")
    print(f"Average boxes per image: {total_boxes/total_images:.2f}")
    print(f"Min boxes per image: {min(boxes_per_image)}")
    print(f"Max boxes per image: {max(boxes_per_image)}")
    
    print("\nClass distribution:")
    for class_id in sorted(class_counts.keys()):
        print(f"  Digit {class_id}: {class_counts[class_id]} boxes ({class_counts[class_id]/total_boxes*100:.1f}%)")

def visualize_specific_images(yolo_file, indices):
    """Visualize specific images by their indices"""
    annotations = read_yolo_annotations(yolo_file)
    
    fig, axes = plt.subplots(1, len(indices), figsize=(5*len(indices), 5))
    if len(indices) == 1:
        axes = [axes]
    
    colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'maroon', 'darkgreen', 'navy', 'olive']
    
    for idx, img_idx in enumerate(indices):
        if img_idx >= len(annotations):
            print(f"Index {img_idx} out of range (max: {len(annotations)-1})")
            continue
            
        ann = annotations[img_idx]
        
        try:
            # Load and display image
            img = Image.open(ann['image_path'])
            axes[idx].imshow(img)
            axes[idx].set_title(f"Index {img_idx}: {os.path.basename(ann['image_path'])}")
            
            # Print info
            print(f"Index {ann['index']:4d}: {os.path.basename(ann['image_path']):20s} - {len(ann['boxes'])} boxes - Classes: {ann['classes']}")
            
            # Draw bounding boxes
            img_width, img_height = img.size
            for box, class_id in zip(ann['boxes'], ann['classes']):
                x1, y1, x2, y2 = yolo_to_pixel_coords(box, img_width, img_height)
                
                # Create rectangle
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor=colors[class_id % len(colors)], 
                                   facecolor='none')
                axes[idx].add_patch(rect)
                
                # Add class label
                axes[idx].text(x1, y1-5, str(class_id), 
                              bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[class_id % len(colors)]),
                              fontsize=12, color='white')
            
            axes[idx].axis('off')
            
        except Exception as e:
            print(f"Error displaying image {img_idx}: {e}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    yolo_file = "Captcha\yolo.txt"
    
    # Show dataset statistics
    show_statistics(yolo_file)
    
    # Show specific images in matplotlib (change indices as needed)
    print(f"\nShowing specific images...")
    visualize_specific_images(yolo_file, [0, 1, 2])  # Show first 3 images
    
    print(f"\nVisualization complete - no images saved")