import torch
import torch.nn as nn

class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_class=10, lambda_coord=5.0, lambda_noobj=1, lambda_cls=1.0):
        super().__init__()
        self.anchors = torch.tensor(anchors)
        self.num_class = num_class
        self.bce = nn.BCEWithLogitsLoss(reduction='sum')
        self.mse = nn.MSELoss(reduction='sum')
        self.ce = nn.CrossEntropyLoss(reduction='sum')
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_cls = lambda_cls

    def forward(self, preds, targets, target_classes):
        """
        preds: (B, A, H, W, 5+num_class)
        targets: list of length B, each (Ni,4) normalized boxes (cx,cy,w,h)
        target_classes: list of length B, each (Ni,) class indices
        """
        device = preds.device
        A = self.anchors.size(0)
        B, _, H, W, _ = preds.shape
        anchors = self.anchors.to(device)

        obj_logits = preds[..., 0]
        tx = preds[..., 1]
        ty = preds[..., 2]
        tw = preds[..., 3]
        th = preds[..., 4]
        cls_logits = preds[..., 5:]  # (B, A, H, W, num_class)

        obj_target = torch.zeros((B, A, H, W), device=device)
        tx_t = torch.zeros((B, A, H, W), device=device)
        ty_t = torch.zeros((B, A, H, W), device=device)
        tw_t = torch.zeros((B, A, H, W), device=device)
        th_t = torch.zeros((B, A, H, W), device=device)
        box_mask = torch.zeros((B, A, H, W), device=device)
        cls_target = torch.full((B, A, H, W), -1, dtype=torch.long, device=device)  # -1 means ignore

        for b in range(B):
            if targets[b].numel() == 0:
                continue
            for j, (cx, cy, bw, bh) in enumerate(targets[b]):
                gx = cx * W
                gy = cy * H
                gi = int(gx)
                gj = int(gy)
                if gi < 0 or gi >= W or gj < 0 or gj >= H:
                    continue
                anchor_areas = anchors[:,0] * anchors[:,1]
                inter_w = torch.min(anchors[:,0], bw)
                inter_h = torch.min(anchors[:,1], bh)
                inter = inter_w * inter_h
                union = anchor_areas + bw*bh - inter + 1e-9
                ious = inter / union
                best_a = torch.argmax(ious).item()

                obj_target[b, best_a, gj, gi] = 1.0
                tx_t[b, best_a, gj, gi] = gx - gi
                ty_t[b, best_a, gj, gi] = gy - gj
                tw_t[b, best_a, gj, gi] = torch.log(bw / (anchors[best_a,0] + 1e-9) + 1e-9)
                th_t[b, best_a, gj, gi] = torch.log(bh / (anchors[best_a,1] + 1e-9) + 1e-9)
                box_mask[b, best_a, gj, gi] = 1.0
                cls_target[b, best_a, gj, gi] = target_classes[b][j]

        obj_loss_pos = self.bce(obj_logits[box_mask==1], obj_target[box_mask==1]) if box_mask.sum()>0 else torch.tensor(0., device=device)
        obj_loss_neg = self.bce(obj_logits[box_mask==0], obj_target[box_mask==0]) * self.lambda_noobj

        if box_mask.sum() > 0:
            x_loss = self.mse(torch.sigmoid(tx)[box_mask==1], tx_t[box_mask==1])
            y_loss = self.mse(torch.sigmoid(ty)[box_mask==1], ty_t[box_mask==1])
            w_loss = self.mse(tw[box_mask==1], tw_t[box_mask==1])
            h_loss = self.mse(th[box_mask==1], th_t[box_mask==1])
            coord_loss = self.lambda_coord * (x_loss + y_loss + w_loss + h_loss)
            # Classification loss (only for assigned boxes)
            cls_loss = self.ce(
                cls_logits[box_mask==1].view(-1, self.num_class),
                cls_target[box_mask==1]
            ) * self.lambda_cls
        else:
            coord_loss = torch.tensor(0., device=device)
            cls_loss = torch.tensor(0., device=device)

        total = coord_loss + obj_loss_pos + obj_loss_neg + cls_loss
        return total / B

class YOLOMultiScaleLoss(nn.Module):
    def __init__(self, anchors, num_class=10, lambda_coord=5.0, lambda_noobj=1, lambda_cls=1.0, 
                 scale_weights=None, num_scales=3):
        super().__init__()
        self.anchors = anchors  # Keep as list/tensor, don't convert yet
        self.num_class = num_class
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_cls = lambda_cls
        
        # Set initial scale weights
        if scale_weights is not None:
            self.scale_weights = scale_weights
            self.num_scales = len(scale_weights)
        else:
            self.num_scales = num_scales
            self.scale_weights = [1.0] * self.num_scales
        
        # Initialize with empty ModuleList - will be populated dynamically
        self.scale_losses = nn.ModuleList()

    def _ensure_loss_modules(self, num_scales):
        """Ensure we have the right number of loss modules"""
        current_modules = len(self.scale_losses)
        
        if current_modules < num_scales:
            # Add more modules
            for _ in range(num_scales - current_modules):
                self.scale_losses.append(
                    YOLOLoss(self.anchors, self.num_class, self.lambda_coord, 
                           self.lambda_noobj, self.lambda_cls)
                )
        elif current_modules > num_scales:
            # Remove excess modules
            self.scale_losses = nn.ModuleList(self.scale_losses[:num_scales])

    def forward(self, preds, targets, target_classes):
        """
        Args:
            preds: tuple/list of predictions from multiple scales OR single tensor
            targets: list of length B, each (Ni,4) normalized boxes (cx,cy,w,h)
            target_classes: list of length B, each (Ni,) class indices
        """
        # Handle both tuple input and single tensor input
        if isinstance(preds, (tuple, list)):
            pred_list = list(preds)
            actual_num_scales = len(pred_list)
        else:
            # Single prediction tensor
            pred_list = [preds]
            actual_num_scales = 1
        
        # Ensure we have the right number of loss modules
        self._ensure_loss_modules(actual_num_scales)
        
        # Adjust weights if needed
        if len(self.scale_weights) != actual_num_scales:
            if len(self.scale_weights) < actual_num_scales:
                # Extend with 1.0 weights
                self.scale_weights.extend([1.0] * (actual_num_scales - len(self.scale_weights)))
            else:
                # Truncate weights
                self.scale_weights = self.scale_weights[:actual_num_scales]
        
        # Compute loss for each scale
        total_loss = 0.0
        loss_breakdown = {}
        
        for i, pred in enumerate(pred_list):
            try:
                scale_loss = self.scale_losses[i](pred, targets, target_classes)
                
                # Apply scale weight
                weight = self.scale_weights[i]
                weighted_loss = scale_loss * weight
                
                total_loss += weighted_loss
                loss_breakdown[f'loss_scale_{i+1}'] = weighted_loss.item()
                
            except Exception as e:
                print(f"Error computing loss for scale {i+1}: {e}")
                # Skip this scale if there's an error
                loss_breakdown[f'loss_scale_{i+1}'] = 0.0
                continue
        
        loss_breakdown['total_loss'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        loss_breakdown['num_scales'] = actual_num_scales
        
        return total_loss, loss_breakdown