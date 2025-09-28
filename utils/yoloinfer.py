import torch

def nms(boxes, scores, iou_thresh=0.5):
    # boxes: (N,4) in xyxy (pixels), scores: (N,)
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)
    x1,y1,x2,y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.sort(descending=True).indices
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])
        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        remain = (iou <= iou_thresh).nonzero(as_tuple=False).squeeze(1)
        order = order[remain + 1]
    return torch.tensor(keep, device=boxes.device, dtype=torch.long)

def decode_predictions(preds, img_size, anchors, conf_thresh=0.3, iou_thresh=0.5):
    """
    Handle both single scale and multi-scale predictions
    
    Args:
        preds: Either (B,A,H,W,5+num_class) for single scale 
               OR tuple/list of predictions for multi-scale
        img_size: (W,H) or (200,80) 
        anchors: list[(w,h)] normalized - same for all scales
    
    Returns:
        Same format as before: list of (boxes, scores, sel_idx) for each batch
    """
    W_img, H_img = img_size
    device = preds[0].device if isinstance(preds, (tuple, list)) else preds.device
    anchors = torch.tensor(anchors, device=device)  # (A,2)
    
    # Handle multi-scale input
    if isinstance(preds, (tuple, list)):
        # Multi-scale predictions
        pred_list = list(preds)
        B = pred_list[0].shape[0]  # Get batch size from first scale
    else:
        # Single scale prediction
        pred_list = [preds]
        B = preds.shape[0]

    all_results = []
    for b in range(B):
        all_boxes = []
        all_scores = []
        all_indices = []
        
        # Process each scale
        for scale_idx, pred in enumerate(pred_list):
            A, H, W = pred.shape[1], pred.shape[2], pred.shape[3]
            
            obj = pred[...,0].sigmoid()
            tx  = pred[...,1].sigmoid()
            ty  = pred[...,2].sigmoid()
            tw  = pred[...,3]
            th  = pred[...,4]

            b_obj = obj[b]  # (A,H,W)
            mask = b_obj > conf_thresh
            if mask.sum() == 0:
                continue
                
            sel_idx = mask.nonzero(as_tuple=False)  # (N,3): a,y,x
            
            # Decode boxes for this scale
            for (a,y,x) in sel_idx:
                cx = (tx[b,a,y,x] + x) / W
                cy = (ty[b,a,y,x] + y) / H
                bw = anchors[a,0] * torch.exp(tw[b,a,y,x])
                bh = anchors[a,1] * torch.exp(th[b,a,y,x])
                
                # Convert to pixel coords
                x1 = (cx - bw/2) * W_img
                y1 = (cy - bh/2) * H_img
                x2 = (cx + bw/2) * W_img
                y2 = (cy + bh/2) * H_img
                
                all_boxes.append(torch.tensor([x1,y1,x2,y2], device=device))
                all_scores.append(b_obj[a,y,x])
                # Add scale info to indices for tracking
                all_indices.append(torch.tensor([a,y,x], device=device))
        
        # Combine all scales for this batch
        if len(all_boxes) == 0:
            all_results.append((
                torch.zeros((0,4), device=device), 
                torch.zeros((0,), device=device), 
                torch.zeros((0,3), dtype=torch.long, device=device)
            ))
        else:
            boxes_pix = torch.stack(all_boxes, dim=0)
            scores = torch.stack(all_scores, dim=0)
            indices = torch.stack(all_indices, dim=0)
            
            # Apply NMS across all scales
            keep = nms(boxes_pix, scores, iou_thresh=iou_thresh)
            all_results.append((boxes_pix[keep], scores[keep], indices[keep]))
    
    return all_results

