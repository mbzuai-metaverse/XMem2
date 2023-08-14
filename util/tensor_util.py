import numpy as np
import torch.nn.functional as F
import torch


def compute_tensor_iu(seg, gt):
    intersection = (seg & gt).float().sum()
    union = (seg | gt).float().sum()

    return intersection, union

def compute_tensor_iou(seg, gt):
    intersection, union = compute_tensor_iu(seg, gt)
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    return iou 

def compute_array_iou(seg, gt):
    # grayscale 2D masks, each gray shade - unique object
    seg = seg.squeeze()
    gt = gt.squeeze()

    ious = []
    for color in np.unique(seg):
        if color == 0:
            continue  # skipping background
        
        curr_object_iou = compute_tensor_iou(
            torch.tensor(seg == color),
            torch.tensor(gt == color),
        )

        ious.append(curr_object_iou)

    if not len(ious):
        # GT is pure black, let's check if the mask also doesn't have any junk
        curr_object_iou = compute_tensor_iou(
            torch.tensor(seg == 0),
            torch.tensor(gt == 0),
        )
        
        ious.append(curr_object_iou)

    return sum(ious) / len(ious)

# STM
def pad_divide_by(in_img, d):
    h, w = in_img.shape[-2:]

    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pad_array = (int(lw), int(uw), int(lh), int(uh))
    out = F.pad(in_img, pad_array)
    return out, pad_array

def unpad(img, pad):
    if len(img.shape) == 4:
        if pad[2]+pad[3] > 0:
            img = img[:,:,pad[2]:-pad[3],:]
        if pad[0]+pad[1] > 0:
            img = img[:,:,:,pad[0]:-pad[1]]
    elif len(img.shape) == 3:
        if pad[2]+pad[3] > 0:
            img = img[:,pad[2]:-pad[3],:]
        if pad[0]+pad[1] > 0:
            img = img[:,:,pad[0]:-pad[1]]
    else:
        raise NotImplementedError
    return img

def get_bbox_from_mask(mask):
    mask = torch.squeeze(mask)
    assert mask.ndim == 2
    
    nonzero = torch.nonzero(mask)
    
    min_y, min_x = nonzero.min(dim=0).values
    max_y, max_x = nonzero.max(dim=0).values
    
    return int(min_y), int(min_x), int(max_y), int(max_x)