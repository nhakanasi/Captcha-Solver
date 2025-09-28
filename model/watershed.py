import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
import torch
from torchvision import transforms

class WatershedCaptchaSolver:
    def __init__(self, classifier_model=None):
        """
        Initialize watershed-based CAPTCHA solver
        
        Args:
            classifier_model: Trained classifier for character recognition
        """
        self.classifier_model = classifier_model
        
    def preprocess_image(self, image_path):
        """Preprocess CAPTCHA image for watershed segmentation"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            image = np.array(Image.open(image_path).convert('RGB'))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Threshold to create binary image
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        return image, gray, binary
    
    def create_distance_transform(self, binary_image):
        """Create distance transform for watershed seeds"""
        # Distance transform
        dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
        
        # Normalize distance transform
        dist_norm = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return dist_transform, dist_norm
    
    def find_watershed_seeds(self, dist_transform, min_distance=10, threshold_abs=0.3):
        """Find seeds for watershed algorithm using local maxima"""
        # Find local maxima in distance transform
        local_maxima = peak_local_max(
            dist_transform, 
            min_distance=min_distance,
            threshold_abs=threshold_abs * np.max(dist_transform)
        )
        
        # Create markers
        markers = np.zeros_like(dist_transform, dtype=np.int32)
        for i, (y, x) in enumerate(local_maxima):
            markers[y, x] = i + 1
            
        return markers, local_maxima
    
    def apply_watershed(self, image, markers):
        """Apply watershed algorithm"""
        # Convert to 3-channel if grayscale
        if len(image.shape) == 2:
            image_3ch = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_3ch = image.copy()
        
        # Apply watershed
        labels = watershed(-image if len(image.shape) == 2 else -cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 
                          markers, 
                          mask=image > 0 if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) > 0)
        
        return labels
    
    def extract_character_regions(self, labels, original_image, min_area=50):
        """Extract individual character regions from watershed labels"""
        regions = regionprops(labels)
        character_boxes = []
        
        for region in regions:
            # Filter by area
            if region.area < min_area:
                continue
                
            # Get bounding box
            min_row, min_col, max_row, max_col = region.bbox
            
            # Add some padding
            padding = 2
            min_row = max(0, min_row - padding)
            min_col = max(0, min_col - padding)
            max_row = min(original_image.shape[0], max_row + padding)
            max_col = min(original_image.shape[1], max_col + padding)
            
            # Extract region
            if len(original_image.shape) == 3:
                char_region = original_image[min_row:max_row, min_col:max_col]
            else:
                char_region = original_image[min_row:max_row, min_col:max_col]
            
            character_boxes.append({
                'image': char_region,
                'bbox': (min_col, min_row, max_col, max_row),
                'area': region.area,
                'centroid': region.centroid
            })
        
        # Sort by x-coordinate (left to right)
        character_boxes.sort(key=lambda x: x['bbox'][0])
        
        return character_boxes
    
    def preprocess_for_classifier(self, char_image, target_size=(28, 28)):
        """Preprocess character image for classification"""
        # Convert to PIL Image
        if isinstance(char_image, np.ndarray):
            if len(char_image.shape) == 3:
                char_pil = Image.fromarray(cv2.cvtColor(char_image, cv2.COLOR_BGR2RGB))
            else:
                char_pil = Image.fromarray(char_image)
        else:
            char_pil = char_image
        
        # Convert to grayscale and resize
        char_pil = char_pil.convert('L').resize(target_size, Image.LANCZOS)
        
        # Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        return transform(char_pil).unsqueeze(0)
    
    def classify_characters(self, character_boxes):
        """Classify extracted characters"""
        if self.classifier_model is None:
            return [{'digit': i % 10, 'confidence': 0.5} for i in range(len(character_boxes))]
        
        predictions = []
        self.classifier_model.eval()
        
        with torch.no_grad():
            for char_box in character_boxes:
                # Preprocess character image
                char_tensor = self.preprocess_for_classifier(char_box['image'])
                
                # Get prediction
                output = self.classifier_model(char_tensor)
                predicted_class = torch.argmax(output, dim=1).item()
                confidence = torch.softmax(output, dim=1).max().item()
                
                predictions.append({
                    'digit': predicted_class,
                    'confidence': confidence
                })
        
        return predictions
    
    def solve_captcha(self, image_path, min_distance=10, threshold_abs=0.3, min_area=50):
        """Complete watershed-based CAPTCHA solving pipeline"""
        # Step 1: Preprocess image
        original, gray, binary = self.preprocess_image(image_path)
        
        # Step 2: Distance transform
        dist_transform, dist_norm = self.create_distance_transform(binary)
        
        # Step 3: Find watershed seeds
        markers, local_maxima = self.find_watershed_seeds(
            dist_transform, min_distance, threshold_abs
        )
        
        # Step 4: Apply watershed
        labels = self.apply_watershed(binary, markers)
        
        # Step 5: Extract character regions
        character_boxes = self.extract_character_regions(labels, gray, min_area)
        
        # Step 6: Classify characters
        predictions = self.classify_characters(character_boxes)
        
        # Step 7: Build result
        captcha_result = ''.join([str(pred['digit']) for pred in predictions])
        
        return {
            'captcha': captcha_result,
            'characters': character_boxes,
            'predictions': predictions,
            'intermediate_images': {
                'original': original,
                'gray': gray,
                'binary': binary,
                'distance_transform': dist_norm,
                'labels': labels
            },
            'seeds': local_maxima
        }
    
    def visualize_results(self, image_path, result):
        """Visualize watershed segmentation results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(result['intermediate_images']['original'], cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Binary image
        axes[0, 1].imshow(result['intermediate_images']['binary'], cmap='gray')
        axes[0, 1].set_title('Binary Image')
        axes[0, 1].axis('off')
        
        # Distance transform
        axes[0, 2].imshow(result['intermediate_images']['distance_transform'], cmap='hot')
        axes[0, 2].set_title('Distance Transform')
        axes[0, 2].axis('off')
        
        # Watershed labels
        axes[1, 0].imshow(result['intermediate_images']['labels'], cmap='tab20')
        axes[1, 0].set_title('Watershed Labels')
        axes[1, 0].axis('off')
        
        # Seeds overlay
        axes[1, 1].imshow(result['intermediate_images']['gray'], cmap='gray')
        if result['seeds'] is not None and len(result['seeds']) > 0:
            seeds_y, seeds_x = zip(*result['seeds'])
            axes[1, 1].scatter(seeds_x, seeds_y, c='red', s=50, marker='x')
        axes[1, 1].set_title('Seeds on Grayscale')
        axes[1, 1].axis('off')
        
        # Final result with bounding boxes
        result_img = cv2.cvtColor(result['intermediate_images']['original'], cv2.COLOR_BGR2RGB).copy()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for i, (char_box, pred) in enumerate(zip(result['characters'], result['predictions'])):
            x1, y1, x2, y2 = char_box['bbox']
            color = colors[i % len(colors)]
            
            # Draw rectangle
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
            
            # Add text
            cv2.putText(result_img, f"{pred['digit']}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        axes[1, 2].imshow(result_img)
        axes[1, 2].set_title(f'Final Result: {result["captcha"]}')
        axes[1, 2].axis('off')
        
        plt.suptitle(f'Watershed CAPTCHA Solver - Result: {result["captcha"]}', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Print detailed results
        print(f"\nWatershed CAPTCHA Solver Results:")
        print(f"Image: {image_path}")
        print(f"Result: {result['captcha']}")
        print(f"Found {len(result['characters'])} characters")
        print("-" * 50)
        
        for i, (char_box, pred) in enumerate(zip(result['characters'], result['predictions'])):
            x1, y1, x2, y2 = char_box['bbox']
            print(f"Char {i+1}: {pred['digit']} (conf: {pred['confidence']:.3f}) "
                  f"at ({x1},{y1},{x2},{y2}) area: {char_box['area']}")
        
        return result_img

# Usage example
def test_watershed_solver():
    """Test the watershed CAPTCHA solver"""
    # Load classifier model (optional)
    try:
        from model.classifer import SimpleMLP
        classifier_model = SimpleMLP(num_classes=10)
        checkpoint = torch.load('checkpoints/classifiermlp_epoch_15.pth', map_location='cpu', weights_only=False)
        classifier_model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded classifier model")
    except:
        print("No classifier model loaded - using dummy predictions")
        classifier_model = None
    
    # Initialize solver
    solver = WatershedCaptchaSolver(classifier_model)
    
    # Test images
    test_images = [
        r"Self-taught\Captcha\captcha\captcha_0017.jpg",
    ]
    
    for image_path in test_images:
        print(f"\n{'='*60}")
        print(f"Processing: {image_path}")
        print('='*60)
        
        try:
            # Solve CAPTCHA
            result = solver.solve_captcha(
                image_path,
                min_distance=10,      # Minimum distance between seeds
                threshold_abs=0.65,   # Threshold for local maxima
                min_area=30          # Minimum character area
            )
            
            # Visualize results
            solver.visualize_results(image_path, result)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_watershed_solver()