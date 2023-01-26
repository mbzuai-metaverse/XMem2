from ast import Dict
import copy
from dataclasses import dataclass
from typing import Iterable, Optional
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


@dataclass
class SingleScaleFeatures:
    feature: torch.Tensor
    key: torch.Tensor
    shrinkage: Optional[torch.Tensor]
    selection: Optional[torch.Tensor]


class MultiscaleFeatures_16_8_4:
    def __init__(self, features: list, keys: list, shrinkages:list = None, selections: list = None) -> None:
        self.scales = (16, 8, 4)

        assert len(features) == 3
        assert len(features) == len(keys)
        if shrinkages is not None:
            assert len(features) == len(shrinkages)
        if selections is not None:
            assert len(features) == len(selections)

        self.features_by_scale: Dict[int, SingleScaleFeatures] = {}

        for i, scale in enumerate(self.scales):
            feature_map = features[i]
            key = keys[i]

            shrinkage = shrinkages[i] if shrinkages is not None else None
            selection = selections[i] if selections is not None else None

            self.features_by_scale[scale] = SingleScaleFeatures(feature_map, key, shrinkage, selection)

    def __getitem__(self, val):
            new_obj: MultiscaleFeatures_16_8_4 = copy(self)  # shallow copy
            new_obj.features_by_scale = {}

            for scale in self.scales:
                old_features: SingleScaleFeatures = self.features_by_scale[scale]
                
                new_features = SingleScaleFeatures(
                    feature=old_features.feature[val],
                    key=old_features.key[val],
                    shrinkage=old_features.shrinkage[val] if old_features.shrinkage is not None else None,
                    selection=old_features.selection[val] if old_features.selection is not None else None,
                )

                new_obj.features_by_scale[scale] = new_features
            
            return new_obj

    def deep_copy(self):
        return self[...]

    def get_all_scales(self, deep_to_shallow=True) -> Iterable[SingleScaleFeatures]:
        scales = self.scales
        if not deep_to_shallow:
            scales = scales[::-1]

        for scale in scales:
            yield self.features_by_scale[scale]


class MutliscaleValues_16_8_4:
    def __init__(self, values: list, hidden:torch.Tensor = None) -> None:
        self.scales = (16, 8, 4)

        assert len(values) == 3

        self.values_by_scale = {}
        self.hidden = hidden

        for i, scale in enumerate(self.scales):
            self.values_by_scale[scale] = values[i]

    def get_all_scales(self, deep_to_shallow=True) -> Iterable[SingleScaleFeatures]:
        scales = self.scales
        if not deep_to_shallow:
            scales = scales[::-1]

        for scale in scales:
            yield self.values_by_scale[scale]

    def __getitem__(self, val):
        new_obj: MutliscaleValues_16_8_4 = copy(self)  # shallow copy
        new_obj.values_by_scale = {}

        for scale in self.scales:
            old_values: torch.Tensor = self.values_by_scale[scale]
            new_obj.values_by_scale[scale] = old_values[val]
        
        return new_obj

    def deep_copy(self):
        return self[...]