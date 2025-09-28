import os
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.vocab import Vocab
from PIL import Image
import torch
import albumentations as A

class ImgAugTransformV2:
    def __init__(self, train=True):
        self.train = train
        self.aug = A.Compose(
            [
                A.InvertImg(p=0.2),
                A.ColorJitter(p=0.2),
                A.MotionBlur(blur_limit=3, p=0.2),
                A.RandomBrightnessContrast(p=0.2),
                A.Perspective(scale=(0.01, 0.05)),
            ]
        )

    def __call__(self, img):
        if not self.train:
            return img
        img = np.array(img)
        transformed = self.aug(image=img)
        img = transformed["image"]
        img = Image.fromarray(img)
        return img

def process_image(img):
    img = img.resize((200, 80), Image.LANCZOS)  # Fixed size
    img = np.asarray(img).transpose(2, 0, 1)
    img = img / 255
    return img

class OCRData(Dataset):
    def __init__(self, meta, base_dir, train=True):
        super().__init__()
        self.aug = ImgAugTransformV2(train=train)
        self.base_dir = base_dir
        self.vocab = Vocab()
        self.read_data(meta)

    def read_data(self, path):
        self.img = []
        self.labels = []
        with open(path, 'r') as f:
            for line in f:
                file, label = line.strip().split()
                if len(label) != 5:
                    raise ValueError(f"Label {label} in {file} is not 5 characters long")
                img = process_image(self.aug(Image.open(self.base_dir + file)))
                self.img.append(torch.FloatTensor(img))
                encoded = self.vocab.encode(label)
                tgt_input = encoded
                self.labels.append(tgt_input)
        self.labels = np.array(self.labels)

        self.tgt_input = self.labels[:, :-1]
        self.tgt_output = self.labels[:,1:]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.img[idx]
        tgt_input = torch.tensor(self.tgt_input[idx], dtype=torch.long)
        tgt_output = torch.tensor(self.tgt_output[idx], dtype=torch.long)
        return img, tgt_input, tgt_output
    
class PrivateDataset(Dataset):
    def __init__(self, annote_file, image_dir):
        self.image_dir = image_dir
        
        # Read annotation file
        with open(annote_file, 'r') as f:
            self.image_files = [line.strip() for line in f.readlines()]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        
        # Load image and apply same preprocessing as training
        image = Image.open(img_path).convert('RGB')
        
        # Use the same process_image function as training (no augmentation)
        processed_img = process_image(image)  # This resizes to (200, 80) and normalizes
        
        # Return filename as identifier (no label available)
        filename = self.image_files[idx]
        
        return torch.FloatTensor(processed_img), filename

# Create private dataset and dataloader
def create_private_dataloader(annote_file, image_dir, batch_size=8):
    dataset = PrivateDataset(annote_file, image_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return dataloader

# Prediction function for private set (no labels)
def predict_private_set(trainer, annote_file, image_dir, use_beam_search=True):
    """
    Generate predictions for private test set (no ground truth labels)
    """
    print("Creating private dataset...")
    private_loader = create_private_dataloader(annote_file, image_dir)
    
    print(f"Private dataset size: {len(private_loader.dataset)}")
    
    # Get predictions only (no accuracy calculation)
    print("Running predictions...")
    predictions = []
    filenames = []
    
    trainer.model.eval()
    
    with torch.no_grad():
        for batch_img, batch_filenames in private_loader:
            batch_img = batch_img.to(trainer.device)
            batch_size = batch_img.size(0)
            
            if use_beam_search:
                # Use beam search
                from utils.beam import batch_translate_beam_search
                beam_results = batch_translate_beam_search(
                    batch_img, 
                    trainer.model, 
                    beam_size=4, 
                    candidates=1, 
                    max_seq_length=7,
                    sos_token=trainer.vocab.go, 
                    eos_token=trainer.vocab.eos
                )
                
                # Decode beam search results
                for b in range(batch_size):
                    if len(beam_results) > b and len(beam_results[b]) > 0:
                        pred_tokens = beam_results[b]
                        pred_text = trainer.vocab.decode(pred_tokens)
                    else:
                        pred_text = ""
                    
                    predictions.append(pred_text)
                    filenames.append(batch_filenames[b])
            else:
                # Use autoregressive generation
                for b in range(batch_size):
                    single_img = batch_img[b:b+1]
                    pred_tokens = trainer.model.autoregressive_generate(
                        single_img,
                        max_length=7,
                        sos_token=trainer.vocab.go,
                        eos_token=trainer.vocab.eos
                    )
                    pred_text = trainer.vocab.decode(pred_tokens)
                    predictions.append(pred_text)
                    filenames.append(batch_filenames[b])
    
    print(f"\n=== PRIVATE SET PREDICTIONS ===")
    print(f"Total samples: {len(predictions)}")
    
    # Show sample predictions
    print(f"\nSample predictions:")
    for i, (filename, pred) in enumerate(zip(filenames[:10], predictions[:10])):
        print(f"  {i+1:2d}. {filename} -> '{pred}'")
    
    # Save results to submission file
    results_file = "private_predictions.txt"
    with open(results_file, 'w') as f:
        for filename, pred in zip(filenames, predictions):
            f.write(f"{filename}\t{pred}\n")
    
    print(f"\nPredictions saved to: {results_file}")
    
    return predictions, filenames
