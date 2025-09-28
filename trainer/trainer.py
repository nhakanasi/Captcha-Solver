from utils.translate import *
import torch
import numpy as np
from torch import nn
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import OneCycleLR
from optim.labelsmooth import LabelSmoothingLoss
from data.dataloader import OCRData
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.beam import batch_translate_beam_search, translate
import os

class CaptchaTrainer:
    def __init__(self):
        self.model, self.vocab = build_model()
        self.optimizer = AdamW(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09)
        self.train_gen = OCRData('annote.txt','')
        self.valid_gen = OCRData('eval_annote.txt', '')
        self.batch_size = 16
        self.epoch_step = len(self.train_gen)//16
        self.beamsearch = True
        self.scheduler = OneCycleLR(self.optimizer, max_lr=0.01, steps_per_epoch=500, epochs=10)
        self.criterion = LabelSmoothingLoss(
            len(self.vocab), padding_idx=self.vocab.pad, smoothing=0.1
        )
        self.train_loader = DataLoader(self.train_gen, batch_size=16, shuffle=True, num_workers=4)  # Changed to 0 for debugging
        self.train_loss = []
        self.iter = 0  # Initialize iteration counter
        self.device = 'cpu' 
        self.vocab_size = len(self.vocab)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]', leave=False)
        for _, (img, tgt_input, tgt_output) in enumerate(pbar):
            img, tgt_input, tgt_output = img.to(self.device), tgt_input.to(self.device), tgt_output.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(img, tgt_input)
            outputs = outputs.view(-1, self.vocab_size)
            tgt_output = tgt_output.view(-1)
            loss = self.criterion(outputs, tgt_output)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        avg_loss = total_loss / len(self.train_loader)
        self.train_loss.append(avg_loss)
        return avg_loss

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0

        with torch.no_grad():
            for step, (img, tgt_input, tgt_output) in enumerate(data_loader):
                img, tgt_input, tgt_output = img.to(self.device), tgt_input.to(self.device), tgt_output.to(self.device)
                batch_size = img.size(0)
                seq_len = tgt_output.size(1)

                outputs = self.model(img, tgt_input)
                outputs = outputs.view(seq_len * batch_size, -1)
                tgt_output = tgt_output.view(-1)

                loss = self.criterion(outputs, tgt_output)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                _, predicted = outputs.max(dim=1)
                correct_predictions += (predicted == tgt_output).sum().item()

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = 100.0 * correct_predictions / (total_samples * seq_len) if total_samples > 0 else 0.0
        return avg_loss, accuracy

    def predict(self, img=None, data_loader=None, sample=None, use_beam_search=True):
        self.model.eval()
        with torch.no_grad():
            if data_loader is not None and sample is not None:
                pred_sents = []
                actual_sents = []
                for batch_img, _, tgt_output in data_loader:
                    batch_img = batch_img.to(self.device)
                    batch_size = batch_img.size(0)
                    
                    if use_beam_search:
                        # Use beam search for prediction
                        beam_results = batch_translate_beam_search(
                            batch_img, 
                            self.model, 
                            beam_size=4, 
                            candidates=1, 
                            max_seq_length=7,  # 5 digits + SOS + EOS
                            sos_token=self.vocab.go, 
                            eos_token=self.vocab.eos
                        )
                        
                        # Decode beam search results
                        for b in range(batch_size):
                            if len(pred_sents) < sample:
                                # beam_results[b] contains the predicted sequence including SOS
                                pred_indices = beam_results[b][1:]  # Remove SOS
                                if self.vocab.eos in pred_indices:
                                    eos_idx = pred_indices.index(self.vocab.eos)
                                    pred_indices = pred_indices[:eos_idx]  # Remove EOS and everything after
                                
                                pred_sent = self.vocab.decode([self.vocab.go] + pred_indices + [self.vocab.eos])
                                actual_indices = tgt_output[b].cpu().numpy().tolist()
                                actual_sent = self.vocab.decode(actual_indices)
                                
                                # Clean up the strings
                                pred_sent = pred_sent.replace('<sos>', '').replace('<eos>', '').strip()[:5]
                                actual_sent = actual_sent.replace('<sos>', '').replace('<eos>', '').strip()[:5]
                                
                                pred_sents.append(pred_sent)
                                actual_sents.append(actual_sent)
                            
                            if len(pred_sents) >= sample:
                                break
                    else:
                        # Original autoregressive generation
                        sos_token = torch.tensor([self.vocab.go], dtype=torch.long).to(self.device)
                        tgt_inputs = sos_token.unsqueeze(0).repeat(batch_size, 1)  # [batch, 1]

                        # Generate sequence up to max length or until <eos>
                        max_len = 6  # 5 chars + <eos>
                        for step in range(max_len - 1):  # -1 because we start with <sos>
                            # Forward pass with current tgt_inputs
                            output = self.model(batch_img, tgt_inputs)
                            
                            # Mask special tokens - only allow digits (4-13) and EOS (2)
                            masked_output = output[:, -1, :].clone()
                            masked_output[:, 0] = -float('inf')  # PAD
                            masked_output[:, 1] = -float('inf')  # SOS
                            masked_output[:, 3] = -float('inf')  # MASK
                            
                            next_tokens = masked_output.argmax(dim=-1)  # [batch_size]
                            print(f"Step {step}, next_tokens: {next_tokens}")  # Debug

                            # Update tgt_inputs for next iteration
                            tgt_inputs = torch.cat([tgt_inputs, next_tokens.unsqueeze(-1)], dim=-1)

                            # Stop if all sequences in batch predict <eos>
                            if (next_tokens == self.vocab.eos).all():
                                break

                        # Decode predictions and actuals
                        for b in range(batch_size):
                            if len(pred_sents) < sample:
                                pred_indices = tgt_inputs[b, 1:].cpu().numpy().tolist()  # Exclude <sos>
                                pred_sent = self.vocab.decode(pred_indices)
                                actual_indices = tgt_output[b].cpu().numpy().tolist()
                                actual_sent = self.vocab.decode(actual_indices)
                                # Remove <sos> and <eos> for display if present
                                pred_sent = pred_sent.replace('<sos>', '').replace('<eos>', '').strip()[:5]
                                actual_sent = actual_sent.replace('<sos>', '').replace('<eos>', '').strip()[:5]
                                pred_sents.append(pred_sent)
                                actual_sents.append(actual_sent)
                            if len(pred_sents) >= sample:
                                break

                    if len(pred_sents) >= sample:
                        break

                return pred_sents, actual_sents

            elif img is not None:
                if use_beam_search:
                    # Use beam search for single image prediction
                    beam_results = batch_translate_beam_search(
                        img, 
                        self.model, 
                        beam_size=4, 
                        candidates=1, 
                        max_seq_length=7,
                        sos_token=self.vocab.go, 
                        eos_token=self.vocab.eos
                    )
                    
                    results = []
                    for b in range(img.size(0)):
                        pred_indices = beam_results[b][1:]  # Remove SOS
                        if self.vocab.eos in pred_indices:
                            eos_idx = pred_indices.index(self.vocab.eos)
                            pred_indices = pred_indices[:eos_idx]
                        
                        pred_sent = self.vocab.decode([self.vocab.go] + pred_indices + [self.vocab.eos])
                        pred_sent = pred_sent.replace('<sos>', '').replace('<eos>', '').strip()[:5]
                        results.append(pred_sent)
                    
                    return results
                else:
                    # Original autoregressive generation with masking
                    batch_size = img.size(0)
                    sos_token = torch.tensor([self.vocab.go], dtype=torch.long).to(self.device)
                    tgt_input = sos_token.unsqueeze(0).repeat(batch_size, 1)  # [batch, 1]
                    max_len = 6  # 5 chars + <eos>

                    for _ in range(max_len - 1):
                        output = self.model(img, tgt_input)
                        
                        # Mask special tokens
                        masked_output = output[:, -1, :].clone()
                        masked_output[:, 0] = -float('inf')  # PAD
                        masked_output[:, 1] = -float('inf')  # SOS
                        masked_output[:, 3] = -float('inf')  # MASK
                        
                        next_token = masked_output.argmax(dim=-1)  # [batch_size]
                        tgt_input = torch.cat([tgt_input, next_token.unsqueeze(-1)], dim=-1)
                        if (next_token == self.vocab.eos).all():
                            break
                            
                    return [self.vocab.decode(tgt_input[b, 1:].cpu().numpy().tolist()).replace('<sos>', '').replace('<eos>', '').strip()[:5] for b in range(batch_size)]
            else:
                raise ValueError("Either 'img' or 'data_loader' with 'sample' must be provided")
    def predict_with_teacher_forcing(self, data_loader, sample=5):
        """Compare teacher forcing vs autoregressive predictions"""
        self.model.eval()
        with torch.no_grad():
            for batch_img, tgt_input, tgt_output in data_loader:
                batch_img = batch_img.to(self.device)
                tgt_input = tgt_input.to(self.device)
                
                # Teacher forcing prediction
                outputs = self.model(batch_img, tgt_input)
                tf_predictions = outputs.argmax(dim=-1)
                
                print("Teacher Forcing vs Autoregressive:")
                for i in range(min(sample, batch_img.size(0))):
                    tf_pred = self.vocab.decode(tf_predictions[i].cpu().numpy().tolist())
                    actual = self.vocab.decode(tgt_output[i].cpu().numpy().tolist())
                    print(f"TF Pred: '{tf_pred}' | Actual: '{actual}'")
                break

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.iter = checkpoint["iter"]
        self.train_loss = checkpoint["train_loss"]

    def save_checkpoint(self, filename):
        state = {
            "iter": self.iter,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_loss": self.train_loss,
        }
        path, _ = os.path.split(filename)
        os.makedirs(path, exist_ok=True)
        torch.save(state, filename)