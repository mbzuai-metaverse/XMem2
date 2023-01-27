from dataclasses import asdict
from functools import partial
from os import access
from pathlib import Path
from turtle import numinput
from sklearn.cluster import KMeans
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms.functional as FT
import numpy as np
from torchvision.transforms import ColorJitter, Grayscale, RandomPosterize, RandomAdjustSharpness, ToTensor, RandomAffine
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
from umap import UMAP
from hdbscan import flat

from tqdm import tqdm
from model.memory_util import do_softmax, get_similarity

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

def _extract_keys(dataloder, processor, print_progress=False):
    frame_keys = []
    shrinkages = []
    selections = []
    device = None
    with torch.no_grad():  # just in case
        key_sum = None
        
        for ti, data in enumerate(tqdm(dataloder, disable=not print_progress, desc='Calculating key features')):
            rgb = data['rgb'].cuda()[0]
            key, shrinkage, selection = processor.encode_frame_key(rgb)
            
            if key_sum is None:
                device = key.device
                key_sum = torch.zeros_like(key, device=device, dtype=torch.float64)  # to avoid possible overflow
            
            key_sum += key.type(torch.float64)
            
            frame_keys.append(key.flatten(start_dim=2).cpu())
            shrinkages.append(shrinkage.flatten(start_dim=2).cpu())
            selections.append(selection.flatten(start_dim=2).cpu())
            
        num_frames = ti + 1  # 0 after 1 iteration, 1 after 2, etc.
        
        return frame_keys, shrinkages, selections, device, num_frames, key_sum

def calculate_proposals_for_annotations_with_average_distance(dataloader, processor, how_many_frames=9, print_progress=False):
    with torch.no_grad():  # just in case
        frame_keys, device, num_frames, key_sum = _extract_keys(dataloader, processor, print_progress)
        
        avg_key = (key_sum / num_frames).type(torch.float32)
        qk = avg_key.flatten(start_dim=2)
        
        similarities = []
        for i in tqdm(range(num_frames), desc='Computing similarity to avg frame'):  # how to run a loop for lower memory usage
            frame_key = frame_keys[i]
            similarity_per_pixel = get_similarity(frame_key.to(device), ms=None, qk=qk, qe=None)
            similarity_avg = (similarity_per_pixel < 0).sum()  # number of dissimilar pixels
            
            similarities.append(similarity_avg)
        
        # import matplotlib.pyplot as plt
        # import numpy as np
        # plt.figure(figsize=(16, 10))
        # plt.xticks(np.arange(0, num_frames, 100))
        # plt.plot([float(x) for x in similarities])
        # plt.title("Inner XMem mean similarity VS average frame")
        # plt.savefig(
        #     'output/similarities_NEG_vs_avg_frame.png'
        # )
        values, indices = torch.topk(torch.tensor(similarities), k=how_many_frames, largest=True)  # top `how_many_frames` frames LEAST similar to the avg_key 
        return indices
    
def calculate_proposals_for_annotations_with_first_distance(dataloader, processor, how_many_frames=9, print_progress=False):
    with torch.no_grad():  # just in case
        frame_keys, device, num_frames, key_sum = _extract_keys(dataloader, processor, print_progress)
        qk = frame_keys[0].flatten(start_dim=2).to(device)
        
        similarities = []
        first_similarity = None
        for i in tqdm(range(num_frames), desc='Computing similarity to avg frame'):  # how to run a loop for lower memory usage
            frame_key = frame_keys[i]
            similarity_per_pixel = get_similarity(frame_key.to(device), ms=None, qk=qk, qe=None)
            if i == 0:
                first_similarity = similarity_per_pixel
            similarity_avg = similarity_per_pixel.mean()
            
            # if i == 0 or i == 175 or i == 353 or i == 560 or i == 900:
            #     import seaborn as sns
            #     import matplotlib.pyplot as plt
            #     plt.figure(figsize=(40, 40))
            #     sns.heatmap(similarity_per_pixel.squeeze().cpu(), square=True, cmap="icefire")
            #     plt.savefig(f'output/SIMILARITY_HEATMAPS/0_vs_{i}.png')
                
            #     plt.figure(figsize=(40, 40))
            #     sns.heatmap((similarity_per_pixel - first_similarity).squeeze().cpu(), square=True, cmap="icefire")
            #     plt.savefig(f'output/SIMILARITY_HEATMAPS/0_vs_{i}_diff_with_0.png')
            similarities.append(similarity_avg)
        
        # import matplotlib.pyplot as plt
        # import numpy as np
        # plt.figure(figsize=(16, 10))
        # plt.xticks(np.arange(0, num_frames, 100))
        # plt.plot([float(x) for x in similarities])
        # plt.title("Inner XMem mean similarity VS 1st frame")
        # plt.savefig(
        #     'output/similarities_NEG_vs_1st_frame.png'
        # )
        
        # we don't need to worry about 1st frame with itself, since we take the LEAST similar frames
        values, indices = torch.topk(torch.tensor(similarities), k=how_many_frames, largest=False)  # top `how_many_frames` frames LEAST similar to the avg_key 
        return indices

 
def calculate_proposals_for_annotations_with_iterative_distance(dataloader, processor, how_many_frames=9, print_progress=False):
    with torch.no_grad():  # just in case
        frame_keys, device, num_frames, key_sum = _extract_keys(dataloader, processor, print_progress)
        
        chosen_frames = [0]
        chosen_frames_mem_keys = [frame_keys[0].flatten(start_dim=2).to(device)]
        
        for i in tqdm(range(how_many_frames), desc='Iteratively picking the most dissimilar frames'):
            similarities = []
            for j in tqdm(range(num_frames), desc='Computing similarity to avg frame', disable=True):  # how to run a loop for lower memory usage
                qk = frame_keys[j].to(device)
                
                similarities_across_mem_keys = []
                for mem_key in chosen_frames_mem_keys:
                    similarity_per_pixel = get_similarity(qk, ms=None, qk=mem_key, qe=None)
                    similarity_avg = similarity_per_pixel.mean()
                    similarities_across_mem_keys.append(similarity_avg)
                    
                similarity_max_across_all = max(similarities_across_mem_keys)
                similarities.append(similarity_max_across_all)
                
            values, indices = torch.topk(torch.tensor(similarities), k=1, largest=False)
            idx = int(indices[0])
            
            import matplotlib.pyplot as plt
            import numpy as np
            plt.figure(figsize=(16, 10))
            plt.xticks(np.arange(0, num_frames, 100))
            plt.plot([float(x) for x in similarities])
            plt.title(f"Inner XMem mean similarity VS frames {chosen_frames}")
            plt.savefig(
                f'output/iterative_similarity/{i}.png'
            )
            
            chosen_frames.append(idx)
            next_frame_to_add = frame_keys[idx]
            chosen_frames_mem_keys.append(next_frame_to_add.to(device))
            
            
        # we don't need to worry about 1st frame with itself, since we take the LEAST similar frames
        return chosen_frames


def calculate_proposals_for_annotations_with_iterative_distance_diff(dataloader, processor, how_many_frames=9, print_progress=False):
    with torch.no_grad():  # just in case
        frame_keys, device, num_frames, key_sum = _extract_keys(dataloader, processor, print_progress)
        
        chosen_frames = [0]
        chosen_frames_mem_keys = [frame_keys[0].flatten(start_dim=2).to(device)]
        
        chosen_frames_self_similarities = [get_similarity(chosen_frames_mem_keys[0], ms=None, qk=chosen_frames_mem_keys[0], qe=None)]
        for i in tqdm(range(how_many_frames), desc='Iteratively picking the most dissimilar frames', disable=not print_progress):
            dissimilarities = []
            for j in tqdm(range(num_frames), desc='Computing similarity to chosen frames', disable=not print_progress):  # how to run a loop for lower memory usage
                qk = frame_keys[j].to(device)
                
                dissimilarities_across_mem_keys = []
                for mem_key, self_sim in zip(chosen_frames_mem_keys, chosen_frames_self_similarities):
                    similarity_per_pixel = get_similarity(qk, ms=None, qk=mem_key, qe=None)

                    # basically, removing scene ambiguity and only keeping differences due to the scene change 
                    # in theory, of course
                    dissimilarity_score = (similarity_per_pixel - self_sim).abs().sum() / similarity_per_pixel.numel()
                    dissimilarities_across_mem_keys.append(dissimilarity_score)
                
                # filtering our existin or very similar frames
                dissimilarity_min_across_all = min(dissimilarities_across_mem_keys)
                dissimilarities.append(dissimilarity_min_across_all)
                
            values, indices = torch.topk(torch.tensor(dissimilarities), k=1, largest=True)
            idx = int(indices[0])
            
            # import matplotlib.pyplot as plt
            # import numpy as np
            # plt.figure(figsize=(16, 10))
            # plt.xticks(np.arange(0, num_frames, 100))
            # plt.plot([float(x) for x in dissimilarities])
            # plt.title(f"Inner XMem mean dissimilarity VS frames {chosen_frames}")
            # plt.savefig(
            #     f'output/iterative_dissimilarity/{i}.png'
            # )
            
            chosen_frames.append(idx)
            next_frame_to_add_key = frame_keys[idx].to(device)
            chosen_frames_mem_keys.append(next_frame_to_add_key)
            chosen_frames_self_similarities.append(get_similarity(next_frame_to_add_key, ms=None, qk=next_frame_to_add_key, qe=None))
            
            
        # we don't need to worry about 1st frame with itself, since we take the LEAST similar frames
        return chosen_frames

def calculate_proposals_for_annotations_with_uniform_iterative_distance_diff(dataloader, processor, how_many_frames=9, print_progress=False):
    with torch.no_grad():
        frame_keys, shrinkages, selections, device, num_frames, key_sum = _extract_keys(dataloader, processor, print_progress)
        
        space = np.linspace(0, num_frames, how_many_frames + 2, endpoint=True, dtype=int)
        ranges = zip(space, space[1:])
        
        chosen_frames = []
        chosen_frames_mem_keys = []
        chosen_frames_self_similarities = []
        chosen_frames_shrinkages = []
        
        for a, b in ranges:
            if a == 0:
                chosen_new_frame = 0  # in the first range we always pick the first frame
            else:
                dissimilarities = []
                for i in tqdm(range(b - a), desc=f'Computing similarity to chosen frames in range({a}, {b})', disable=not print_progress):  # how to run a loop for lower memory usage
                    true_frame_idx = a + i
                    qk = frame_keys[true_frame_idx].to(device)
                    selection = selections[true_frame_idx].to(device)  # query
                    
                    dissimilarities_across_mem_keys = []
                    for mem_key, shrinkage, self_sim in zip(chosen_frames_mem_keys, chosen_frames_shrinkages, chosen_frames_self_similarities):
                        similarity_per_pixel = get_similarity(mem_key, ms=shrinkage, qk=qk, qe=selection)

                        # basically, removing scene ambiguity and only keeping differences due to the scene change 
                        # in theory, of course
                        diff = (similarity_per_pixel - self_sim)
                        dissimilarity_score = diff[diff > 0].sum() / similarity_per_pixel.numel()
                        dissimilarities_across_mem_keys.append(dissimilarity_score)
                    
                    # filtering our existing or very similar frames
                    dissimilarity_min_across_all = min(dissimilarities_across_mem_keys)
                    dissimilarities.append(dissimilarity_min_across_all)
                    
                values, indices = torch.topk(torch.tensor(dissimilarities), k=1, largest=True)
                chosen_new_frame = int(indices[0]) + a

            chosen_frames.append(chosen_new_frame)
            chosen_frames_mem_keys.append(frame_keys[chosen_new_frame].to(device))
            chosen_frames_shrinkages.append(shrinkages[chosen_new_frame].to(device))
            chosen_frames_self_similarities.append(get_similarity(chosen_frames_mem_keys[-1], ms=shrinkages[chosen_new_frame].to(device), qk=chosen_frames_mem_keys[-1], qe=selections[chosen_new_frame].to(device)))
            
        # we don't need to worry about 1st frame with itself, since we take the LEAST similar frames
        return chosen_frames

def calculate_proposals_for_annotations_with_uniform_iterative_distance_cycle(dataloader, processor, how_many_frames=9, print_progress=False):
    with torch.no_grad():
        frame_keys, shrinkages, selections, device, num_frames, key_sum = _extract_keys(dataloader, processor, print_progress)
        
        space = np.linspace(0, num_frames, how_many_frames + 2, endpoint=True, dtype=int)
        ranges = zip(space, space[1:])
        
        chosen_frames = []
        chosen_frames_mem_keys = []
        # chosen_frames_self_similarities = []
        chosen_frames_shrinkages = []
        
        for a, b in ranges:
            if a == 0:
                chosen_new_frame = 0  # in the first range we always pick the first frame
            else:
                dissimilarities = []
                for i in tqdm(range(b - a), desc=f'Computing similarity to chosen frames in range({a}, {b})', disable=not print_progress):  # how to run a loop for lower memory usage
                    true_frame_idx = a + i
                    qk = frame_keys[true_frame_idx].to(device)
                    query_selection = selections[true_frame_idx].to(device)  # query
                    query_shrinkage = shrinkages[true_frame_idx].to(device)
                    
                    dissimilarities_across_mem_keys = []
                    for key_idx, mem_key, key_shrinkage in zip(chosen_frames, chosen_frames_mem_keys, chosen_frames_shrinkages):
                        mem_key = mem_key.to(device)
                        key_selection = selections[key_idx].to(device)
                        similarity_per_pixel = get_similarity(mem_key, ms=key_shrinkage, qk=qk, qe=query_selection)
                        reverse_similarity_per_pixel = get_similarity(qk, ms=query_shrinkage, qk=mem_key, qe=key_selection)
                        
                        # mapping of pixels A -> B would be very similar to B -> A if the images are similar
                        # and very different if the images are different
                        cycle_dissimilarity_per_pixel = (similarity_per_pixel - reverse_similarity_per_pixel)
                        cycle_dissimilarity_score = cycle_dissimilarity_per_pixel.abs().sum() / cycle_dissimilarity_per_pixel.numel()
                        
                        dissimilarities_across_mem_keys.append(cycle_dissimilarity_score)
                    
                    # filtering our existing or very similar frames
                    dissimilarity_min_across_all = min(dissimilarities_across_mem_keys)
                    dissimilarities.append(dissimilarity_min_across_all)
                    
                values, indices = torch.topk(torch.tensor(dissimilarities), k=1, largest=True)
                chosen_new_frame = int(indices[0]) + a

            chosen_frames.append(chosen_new_frame)
            chosen_frames_mem_keys.append(frame_keys[chosen_new_frame].to(device))
            chosen_frames_shrinkages.append(shrinkages[chosen_new_frame].to(device))
            # chosen_frames_self_similarities.append(get_similarity(chosen_frames_mem_keys[-1], ms=shrinkages[chosen_new_frame].to(device), qk=chosen_frames_mem_keys[-1], qe=selections[chosen_new_frame].to(device)))
            
        # we don't need to worry about 1st frame with itself, since we take the LEAST similar frames
        return chosen_frames


def calculate_proposals_for_annotations_with_iterative_distance_cycle(dataloader, processor, how_many_frames=9, print_progress=False):
    with torch.no_grad():
        frame_keys, shrinkages, selections, device, num_frames, key_sum = _extract_keys(dataloader, processor, print_progress)
        
        chosen_frames = [0]
        chosen_frames_mem_keys = [frame_keys[0].to(device)]
        
        for i in tqdm(range(how_many_frames), desc='Iteratively picking the most dissimilar frames', disable=not print_progress):
            dissimilarities = []
            for j in tqdm(range(num_frames), desc='Computing similarity to chosen frames', disable=not print_progress):  # how to run a loop for lower memory usage
                qk = frame_keys[j].to(device)
                query_selection = selections[j].to(device)  # query
                query_shrinkage = shrinkages[j].to(device)
                
                dissimilarities_across_mem_keys = []
                for key_idx, mem_key in zip(chosen_frames, chosen_frames_mem_keys):
                    mem_key = mem_key.to(device)
                    key_shrinkage = shrinkages[key_idx].to(device)
                    key_selection = selections[key_idx].to(device)
                    
                    similarity_per_pixel = get_similarity(mem_key, ms=None, qk=qk, qe=None)
                    reverse_similarity_per_pixel = get_similarity(qk, ms=None, qk=mem_key, qe=None)
                    
                    # mapping of pixels A -> B would be very similar to B -> A if the images are similar
                    # and very different if the images are different
                    cycle_dissimilarity_per_pixel = (similarity_per_pixel - reverse_similarity_per_pixel)
                    cycle_dissimilarity_score = cycle_dissimilarity_per_pixel.abs().sum() / cycle_dissimilarity_per_pixel.numel()
                    
                    dissimilarities_across_mem_keys.append(cycle_dissimilarity_score)
                
                # filtering our existing or very similar frames
                dissimilarity_min_across_all = min(dissimilarities_across_mem_keys)
                dissimilarities.append(dissimilarity_min_across_all)
            
            values, indices = torch.topk(torch.tensor(dissimilarities), k=1, largest=True)
            chosen_new_frame = int(indices[0])

            chosen_frames.append(chosen_new_frame)
            chosen_frames_mem_keys.append(frame_keys[chosen_new_frame].to(device))
    # chosen_frames_self_similarities.append(get_similarity(chosen_frames_mem_keys[-1], ms=shrinkages[chosen_new_frame].to(device), qk=chosen_frames_mem_keys[-1], qe=selections[chosen_new_frame].to(device)))
    
        # we don't need to worry about 1st frame with itself, since we take the LEAST similar frames
        return chosen_frames


def calculate_proposals_for_annotations_with_uniform_iterative_distance_double_diff(dataloader, processor, how_many_frames=9, print_progress=False):
    with torch.no_grad():
        frame_keys, shrinkages, selections, device, num_frames, key_sum = _extract_keys(dataloader, processor, print_progress)
        
        space = np.linspace(0, num_frames, how_many_frames + 2, endpoint=True, dtype=int)
        ranges = zip(space, space[1:])
        
        chosen_frames = []
        chosen_frames_mem_keys = []
        # chosen_frames_self_similarities = []
        chosen_frames_shrinkages = []
        
        for a, b in ranges:
            if a == 0:
                chosen_new_frame = 0  # in the first range we always pick the first frame
            else:
                dissimilarities = []
                for i in tqdm(range(b - a), desc=f'Computing similarity to chosen frames in range({a}, {b})', disable=not print_progress):  # how to run a loop for lower memory usage
                    true_frame_idx = a + i
                    qk = frame_keys[true_frame_idx].to(device)
                    query_selection = selections[true_frame_idx].to(device)  # query
                    query_shrinkage = shrinkages[true_frame_idx].to(device)
                    
                    dissimilarities_across_mem_keys = []
                    for key_idx, mem_key in zip(chosen_frames, chosen_frames_mem_keys):
                        mem_key = mem_key.to(device)
                        key_shrinkage = shrinkages[key_idx].to(device)
                        key_selection = selections[key_idx].to(device)
                        
                        similarity_per_pixel = get_similarity(mem_key, ms=key_shrinkage, qk=qk, qe=query_selection)
                        self_similarity_key = get_similarity(mem_key, ms=key_shrinkage, qk=mem_key, qe=key_selection)
                        self_similarity_query = get_similarity(qk, ms=query_shrinkage, qk=query_shrinkage, qe=query_selection)
                        
                        # mapping of pixels A -> B would be very similar to B -> A if the images are similar
                        # and very different if the images are different
                        
                        pure_similarity = 2 * similarity_per_pixel - self_similarity_key - self_similarity_query

                        dissimilarity_score = pure_similarity.abs().sum() / pure_similarity.numel()
                        
                        dissimilarities_across_mem_keys.append(dissimilarity_score)
                    
                    # filtering our existing or very similar frames
                    dissimilarity_min_across_all = min(dissimilarities_across_mem_keys)
                    dissimilarities.append(dissimilarity_min_across_all)
                    
                values, indices = torch.topk(torch.tensor(dissimilarities), k=1, largest=True)
                chosen_new_frame = int(indices[0]) + a

            chosen_frames.append(chosen_new_frame)
            chosen_frames_mem_keys.append(frame_keys[chosen_new_frame].to(device))
            chosen_frames_shrinkages.append(shrinkages[chosen_new_frame].to(device))
            # chosen_frames_self_similarities.append(get_similarity(chosen_frames_mem_keys[-1], ms=shrinkages[chosen_new_frame].to(device), qk=chosen_frames_mem_keys[-1], qe=selections[chosen_new_frame].to(device)))
            
        # we don't need to worry about 1st frame with itself, since we take the LEAST similar frames
        return chosen_frames


def calculate_proposals_for_annotations_iterative_pca_cosine(dataloader, processor, how_many_frames=9, print_progress=False):
    # might not pick 0-th frame
    np.random.seed(1)  # Just in case
    with torch.no_grad():
        frame_keys, shrinkages, selections, device, num_frames, key_sum = _extract_keys(dataloader, processor, print_progress)
        flat_keys = torch.stack([key.flatten().cpu() for key in frame_keys]).numpy()
        
        # PCA hangs at num_frames // 2: https://github.com/scikit-learn/scikit-learn/issues/22434
        pca = PCA(num_frames - 1, svd_solver='arpack')
        # pca = FastICA(num_frames - 1)
        smol_keys = pca.fit_transform(flat_keys.astype(np.float64))
        # smol_keys = flat_keys  # to disable PCA
        
        chosen_frames = [0]
        for c in range(how_many_frames):
            distances = cdist(smol_keys[chosen_frames], smol_keys, metric='euclidean')
            closest_to_mem_key_distances = distances.min(axis=0)
            most_distant_frame = np.argmax(closest_to_mem_key_distances)
            chosen_frames.append(most_distant_frame)
                    
        return chosen_frames
    
  
def calculate_proposals_for_annotations_iterative_umap_cosine(dataloader, processor, how_many_frames=9, print_progress=False):
    # might not pick 0-th frame
    np.random.seed(1)  # Just in case
    with torch.no_grad():
        frame_keys, shrinkages, selections, device, num_frames, key_sum = _extract_keys(dataloader, processor, print_progress)
        flat_keys = torch.stack([key.flatten().cpu() for key in frame_keys]).numpy()
        
        # PCA hangs at num_frames // 2: https://github.com/scikit-learn/scikit-learn/issues/22434
        pca = UMAP(n_neighbors=num_frames - 1, n_components=num_frames // 2, random_state=1)
        # pca = FastICA(num_frames - 1)
        smol_keys = pca.fit_transform(flat_keys.astype(np.float64))
        # smol_keys = flat_keys  # to disable PCA
        
        chosen_frames = [0]
        for c in range(how_many_frames):
            distances = cdist(smol_keys[chosen_frames], smol_keys, metric='euclidean')
            closest_to_mem_key_distances = distances.min(axis=0)
            most_distant_frame = np.argmax(closest_to_mem_key_distances)
            chosen_frames.append(most_distant_frame)
                    
        return chosen_frames
    
      
def calculate_proposals_for_annotations_uniform_iterative_pca_cosine(dataloader, processor, how_many_frames=9, print_progress=False):
    # might not pick 0-th frame
    with torch.no_grad():
        frame_keys, shrinkages, selections, device, num_frames, key_sum = _extract_keys(dataloader, processor, print_progress)
        flat_keys = torch.stack([key.flatten().cpu() for key in frame_keys]).numpy()
        
        # PCA hangs at num_frames // 2: https://github.com/scikit-learn/scikit-learn/issues/22434
        pca = PCA(num_frames - 1, svd_solver='arpack')
        smol_keys = pca.fit_transform(flat_keys)
        # smol_keys = flat_keys  # to disable PCA
        
        space = np.linspace(0, num_frames, how_many_frames + 2, endpoint=True, dtype=int)
        ranges = zip(space, space[1:])
        
        chosen_frames = [0]
        
        for a, b in ranges:
            if a == 0:
                # skipping the first one
                continue
            
            distances = cdist(smol_keys[chosen_frames], smol_keys[a:b], metric='cosine')
            closest_to_mem_key_distances = distances.min(axis=0)
            most_distant_frame = np.argmax(closest_to_mem_key_distances) + a
            
            chosen_frames.append(most_distant_frame)

        return chosen_frames


def calculate_proposals_for_annotations_iterative_pca_cosine_values(values, how_many_frames=9, print_progress=False):
    # might not pick 0-th frame
    np.random.seed(1)  # Just in case
    with torch.no_grad():
        num_frames = len(values)
        flat_values = torch.stack([value.flatten().cpu() for value in values]).numpy()
        
        # PCA hangs at num_frames // 2: https://github.com/scikit-learn/scikit-learn/issues/22434
        pca = PCA(num_frames - 1, svd_solver='arpack')
        # pca = FastICA(num_frames - 1)
        smol_values = pca.fit_transform(flat_values.astype(np.float64))
        # smol_keys = flat_keys  # to disable PCA
        
        chosen_frames = [0]
        for c in range(how_many_frames):
            distances = cdist(smol_values[chosen_frames], smol_values, metric='euclidean')
            closest_to_mem_key_distances = distances.min(axis=0)
            most_distant_frame = np.argmax(closest_to_mem_key_distances)
            chosen_frames.append(most_distant_frame)
                    
        return chosen_frames
    

def calculate_proposals_for_annotations_umap_hdbscan_clustering(dataloader, processor, how_many_frames=9, print_progress=False):
    # might not pick 0-th frame
    with torch.no_grad():
        frame_keys, shrinkages, selections, device, num_frames, key_sum = _extract_keys(dataloader, processor, print_progress)
        flat_keys = torch.stack([key.flatten().cpu() for key in frame_keys]).numpy()
        
        pca = UMAP(n_neighbors=num_frames - 1, n_components=num_frames // 2, random_state=1)
        smol_keys = pca.fit_transform(flat_keys)
        
        # clustering = AgglomerativeClustering(n_clusters=how_many_frames + 1, linkage='single')
        clustering = flat.HDBSCAN_flat(smol_keys, n_clusters=how_many_frames + 1)
        labels = clustering.labels_
        
        chosen_frames = []
        for c in range(how_many_frames + 1):
            vectors = smol_keys[labels == c]
            true_index_mapping = {i: int(ti) for i, ti in zip(range(len(vectors)), np.nonzero(labels == c)[0])}
            center = np.mean(vectors, axis=0)
            
            distances = cdist(vectors, [center], metric='euclidean').squeeze()
            
            closest_to_cluster_center_idx = np.argsort(distances)[0]
            
            chosen_frame_idx = true_index_mapping[closest_to_cluster_center_idx]
            chosen_frames.append(chosen_frame_idx)
        
        return chosen_frames


def calculate_proposals_for_annotations_pca_hierarchical_clustering(dataloader, processor, how_many_frames=9, print_progress=False):
    # might not pick 0-th frame
    with torch.no_grad():
        frame_keys, shrinkages, selections, device, num_frames, key_sum = _extract_keys(dataloader, processor, print_progress)
        flat_keys = torch.stack([key.flatten().cpu() for key in frame_keys]).numpy()
        
        pca = PCA(num_frames)        
        smol_keys = pca.fit_transform(flat_keys)
        
        # clustering = AgglomerativeClustering(n_clusters=how_many_frames + 1, linkage='single')
        clustering = KMeans(n_clusters=how_many_frames + 1)
        labels = clustering.fit_predict(smol_keys)
        
        chosen_frames = []
        for c in range(how_many_frames + 1):
            vectors = smol_keys[labels == c]
            true_index_mapping = {i: int(ti) for i, ti in zip(range(len(vectors)), np.nonzero(labels == c)[0])}
            center = np.mean(vectors, axis=0)
            
            distances = cdist(vectors, [center], metric='euclidean').squeeze()
            
            closest_to_cluster_center_idx = np.argsort(distances)[0]
            
            chosen_frame_idx = true_index_mapping[closest_to_cluster_center_idx]
            chosen_frames.append(chosen_frame_idx)
        
        return chosen_frames


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
