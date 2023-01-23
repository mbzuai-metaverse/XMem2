from functools import partial
from os import access
from pathlib import Path
from sklearn.cluster import KMeans
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms.functional as FT
from torchvision.transforms import ColorJitter, Grayscale, RandomPosterize, RandomAdjustSharpness, ToTensor, RandomAffine

from util.tensor_util import get_bbox_from_mask

def select_n_frame_candidates(preds_df: pd.DataFrame, uncertainty_name: str, n=5):
    df = preds_df

    df.reset_index(drop=False, inplace=True)

    # max_frame = df['frame'].max()
    # max_entropy = df['entropy'].max()
    
    df = df[df['mask_provided'] == False]  # removing frames with masks
    df = df[df[uncertainty_name] >= df[uncertainty_name].median()] # removing low entropy parts
    
    df_backup = df.copy()
    
    df['index'] = df['index'] / df['index'].max() # scale to 0..1
    # df['entropy'] = df['entropy'] / df['entropy'].max() # scale to 0..1
    
    X = df[['index', uncertainty_name]].to_numpy()

    clusterer = KMeans(n_clusters=n)
    
    labels = clusterer.fit_predict(X)

    clusters = df_backup.groupby(labels)
    
    candidates = []

    for g, cluster in clusters:
        if g == -1:
            continue
        
        max_entropy_idx = cluster[uncertainty_name].argmax()

        res = cluster.iloc[max_entropy_idx]

        candidates.append(res)

    return candidates

def select_most_uncertain_frame(preds_df: pd.DataFrame, uncertainty_name: str):
    df = preds_df[preds_df['mask_provided'] == False] 
    df.reset_index(drop=False, inplace=True)
    return df.iloc[df[uncertainty_name].argmax()]

def select_n_frame_candidates_no_neighbours_simple(preds_df: pd.DataFrame, uncertainty_name: str, n=5, neighbourhood_size=4):
    df = preds_df
    df.reset_index(drop=False, inplace=True)

    df = df[df['mask_provided'] == False]  # removing frames with masks
    
    neighbours_indices = set()
    chosen_candidates = []
    
    df_sorted = df.sort_values(uncertainty_name, ascending=False)
    i = 0
    while len(chosen_candidates) < n:
        candidate = df_sorted.iloc[i]
        candidate_index = candidate['index']
        
        if candidate_index not in neighbours_indices:
            chosen_candidates.append(candidate)
            candidate_neighbours = range(candidate_index - neighbourhood_size, candidate_index + neighbourhood_size + 1)
            neighbours_indices.update(candidate_neighbours)
            
        i += 1
    
    return chosen_candidates


WhichAugToPick = -1

def get_determenistic_augmentations(img_size=None, mask=None, subset: str=None):
    assert subset in {'best_3', 'best_3_with_symmetrical', 'best_all', 'original_only', 'all'}
    
    bright = ColorJitter(brightness=(1.5, 1.5))
    dark = ColorJitter(brightness=(0.5, 0.5))
    gray = Grayscale(num_output_channels=3)
    reduce_bits = RandomPosterize(bits=3, p=1)
    sharp = RandomAdjustSharpness(sharpness_factor=16, p=1)
    rotate_right = RandomAffine(degrees=(30, 30))
    blur = partial(FT.gaussian_blur, kernel_size=7)
    
    if img_size is not None:
        h, w = img_size[-2:]
        translate_distance = w // 5
    else:
        translate_distance = 200
        
    translate_right = partial(FT.affine, angle=0, translate=(translate_distance, 0), scale=1, shear=0)
    
    zoom_out = partial(FT.affine, angle=0, translate=(0, 0), scale=0.5, shear=0)
    zoom_in = partial(FT.affine, angle=0, translate=(0, 0), scale=1.5, shear=0)
    shear_right = partial(FT.affine, angle=0, translate=(0, 0), scale=1, shear=20)
    
    identity = torch.nn.Identity()
    identity.name = 'identity'
    
    if mask is not None:
        if mask.any():
            min_y, min_x, max_y, max_x = get_bbox_from_mask(mask)
            h, w = mask.shape[-2:]
            crop_mask = partial(FT.resized_crop, top=min_y - 10, left=min_x - 10, height=max_y - min_y + 10, width=max_x - min_x + 10, size=(w, h))
            crop_mask.name = 'crop_mask'
        else:
            crop_mask = identity # if the mask is empty 
    else:
        crop_mask = None
      
    bright.name = 'bright'
    dark.name = 'dark'
    gray.name = 'gray'
    reduce_bits.name = 'reduce_bits'
    sharp.name = 'sharp'
    rotate_right.name = 'rotate_right'
    translate_right.name = 'translate_right'
    zoom_out.name = 'zoom_out'
    zoom_in.name = 'zoom_in'
    shear_right.name = 'shear_right'
    blur.name = 'blur'
    
 
    
    rotate_left = RandomAffine(degrees=(-30, -30))
    rotate_left.name = 'rotate_left'
    
    shear_left = partial(FT.affine, angle=0, translate=(0, 0), scale=1, shear=-20)
    shear_left.name = 'shear_left'
    
    if WhichAugToPick != -1:
        return [img_mask_augs_pairs[WhichAugToPick]]
    
    if subset == 'best_3':
        img_mask_augs_pairs = [
                # augs only applied to the image
                # (bright, identity),
                # (dark, identity),
                # (gray, identity),
                # (reduce_bits, identity),
                # (sharp, identity),
                (blur, identity),
                
                # augs requiring modifying the mask as well:
                # (rotate_right, rotate_right),
                # (rotate_left, rotate_left),
                # (translate_right, translate_right),
                # (zoom_out, zoom_out),
                (zoom_in, zoom_in),
                (shear_right, shear_right),
                # (shear_left, shear_left),
        ]
        
        return img_mask_augs_pairs
    elif subset == 'best_3_with_symmetrical':
        img_mask_augs_pairs = [
                # augs only applied to the image
                # (bright, identity),
                # (dark, identity),
                # (gray, identity),
                # (reduce_bits, identity),
                # (sharp, identity),
                (blur, identity),
                
                # augs requiring modifying the mask as well:
                # (rotate_right, rotate_right),
                # (rotate_left, rotate_left),
                # (translate_right, translate_right),
                # (zoom_out, zoom_out),
                (zoom_in, zoom_in),
                (shear_right, shear_right),
                (shear_left, shear_left),
        ]
        
        return img_mask_augs_pairs
    elif subset == 'best_all':
        img_mask_augs_pairs = [
            # augs only applied to the image
            (bright, identity),
            (dark, identity),
            # (gray, identity),
            (reduce_bits, identity),
            (sharp, identity),
            (blur, identity),
            
            # augs requiring modifying the mask as well:
            (rotate_right, rotate_right),
            (rotate_left, rotate_left),
            # (translate_right, translate_right),
            (zoom_out, zoom_out),
            (zoom_in, zoom_in),
            (shear_right, shear_right),
            (shear_left, shear_left),
        ]
        
        return img_mask_augs_pairs
    
    elif subset == 'original_only':
        img_mask_augs_pairs = [
        # augs only applied to the image
            (bright, identity),
            (dark, identity),
            (gray, identity),
            (reduce_bits, identity),
            (sharp, identity),
            (blur, identity),
            
            # augs requiring modifying the mask as well:
            # (rotate_right, rotate_right),
            # (translate_right, translate_right),
            # (zoom_out, zoom_out),
            # (zoom_in, zoom_in),
            # (shear_right, shear_right),
        ]
    else:
        img_mask_augs_pairs = [
            # augs only applied to the image
            (bright, identity),
            (dark, identity),
            (gray, identity),
            (reduce_bits, identity),
            (sharp, identity),
            (blur, identity),
            
            # augs requiring modifying the mask as well:
            (rotate_right, rotate_right),
            (rotate_left, rotate_left),
            (translate_right, translate_right),
            (zoom_out, zoom_out),
            (zoom_in, zoom_in),
            (shear_right, shear_right),
            (shear_left, shear_left),
        ]
        
        if crop_mask is not None:
            img_mask_augs_pairs.append((crop_mask, crop_mask))
        
        return img_mask_augs_pairs


def apply_aug(img_path, out_path):
    img = Image.open(img_path)
    
    bright, dark, gray, reduce_bits, sharp = get_determenistic_augmentations()
    
    img_augged = sharp(img)
    
    img_augged.save(out_path)
    

def compute_disparity(predictions, augs, images:list = None, output_save_path: str = None):
    assert len(predictions) - len(augs) == 1
    disparity_map = None
    prev = None
    
    if images is None:
        images = [None] * len(predictions)
    else:
        assert len(predictions) == len(images)
    
    if output_save_path is not None:
        p_out_disparity = Path(output_save_path)
    else:
        p_out_disparity = None
    
    try:
        aug_names = [aug.name for aug in augs]
    except AttributeError:
        aug_names = [aug._get_name() for aug in augs]
        
    names = ['original'] + aug_names
    for i, (name, img, pred) in enumerate(zip(names, images, predictions)):
        fg_mask = pred[1:2].squeeze().cpu()  # 1:2 is Foreground
        
        if disparity_map is None:
            disparity_map = torch.zeros_like(fg_mask)
        else:
            disparity_map += (prev - fg_mask).abs()
        
        pred_mask_ = FT.to_pil_image(fg_mask)
        if p_out_disparity is not None:
            p_out_save_mask = p_out_disparity / 'masks' / (f'{i}_{name}.png')
            p_out_save_image = p_out_disparity / 'images' / (f'{i}_{name}.png')
            
            if not p_out_save_mask.parent.exists():
                p_out_save_mask.parent.mkdir(parents=True)
                
            pred_mask_.save(p_out_save_mask)
            
            if not p_out_save_image.parent.exists():
                p_out_save_image.parent.mkdir(parents=True)
                
            img.save(p_out_save_image)
        
        prev = fg_mask
    
    disparity_scaled = disparity_map / (len(augs) + 1)  # 0..1; not `disparity_map.max()`, as the scale would differ across images
    disparity_avg = disparity_scaled.mean()
    disparity_large = (disparity_scaled > 0.5).sum()  # num pixels with large disparities
    
    if p_out_disparity is not None:
        disparity_img = FT.to_pil_image(disparity_scaled) 
        disparity_img.save(p_out_disparity / (f'{i+1}_absolute_disparity.png'))

    return {'full': disparity_scaled, 'avg': disparity_avg, 'large': disparity_large}

    
if __name__ == '__main__':
    img_in = '/home/maksym/RESEARCH/VIDEOS/thanks_no_ears_5_annot/JPEGImages/frame_000001.PNG'
    img_out = 'test_aug.png'
    
    apply_aug(img_in, img_out)
    