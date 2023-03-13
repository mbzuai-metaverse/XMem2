from copy import copy
from pathlib import Path
import time
from typing import Any, Dict, List, Set, Tuple, Union

import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as FT
import numpy as np
from tqdm import tqdm
from inference.frame_selection.frame_selection_utils import extract_keys

from model.memory_util import get_similarity


def first_frame_only(*args, **kwargs):
    # baseline
    return [0]


def uniformly_selected_frames(dataloader, *args, how_many_frames=10, **kwargs) -> List[int]:
    # baseline
    # TODO: debug and check if works
    num_total_frames = len(dataloader)
    return np.linspace(0, num_total_frames - 1, how_many_frames).astype(int).tolist()


def calculate_proposals_for_annotations_with_iterative_distance_cycle_MASKS(dataloader, processor, existing_masks_path: str, how_many_frames=10, print_progress=False, mult_instead=False, alpha=1.0, too_small_mask_threshold_px=9, **kwargs) -> List[int]:
    with torch.no_grad():
        frame_keys, shrinkages, selections, device, num_frames, key_sum = extract_keys(dataloader, processor, print_progress)  
        h, w = frame_keys[0].squeeze().shape[1:3]  # removing batch dimension
        p_masks_dir = Path(existing_masks_path)
        mask_sizes_px = []
        for i, p_img in enumerate(p_masks_dir.iterdir()):
            img = cv2.imread(str(p_img))
            img = cv2.resize(img, (w, h)) / 255.
            img_tensor = FT.to_tensor(img)
            mask_size_px = (img_tensor > 0).sum()
            mask_sizes_px.append(mask_size_px)

            if not mult_instead:
                composite_key = torch.cat([frame_keys[i].cpu().squeeze(), img_tensor], dim=0)  # along channels
            else:
                composite_key = frame_keys[i].cpu().squeeze() * img_tensor.max(dim=0, keepdim=True).values # all objects -> 1., background -> 0.. Keep 1 channel only
                composite_key = composite_key * alpha + frame_keys[i].cpu().squeeze() * (1 - alpha)
            frame_keys[i] = composite_key

        chosen_frames = [0]
        chosen_frames_mem_keys = [frame_keys[0].to(device)]

        for i in tqdm(range(how_many_frames - 1), desc='Iteratively picking the most dissimilar frames', disable=not print_progress):
            dissimilarities = []
            # how to run a loop for lower memory usage
            for j in tqdm(range(num_frames), desc='Computing similarity to chosen frames', disable=not print_progress):
                qk = frame_keys[j].to(device)

                if mask_sizes_px[j] < too_small_mask_threshold_px:
                    dissimilarity_min_across_all = 0
                else:
                    dissimilarities_across_mem_keys = []
                    for mem_key in chosen_frames_mem_keys:
                        mem_key = mem_key.to(device)

                        similarity_per_pixel = get_similarity(
                            mem_key, ms=None, qk=qk, qe=None)
                        reverse_similarity_per_pixel = get_similarity(
                            qk, ms=None, qk=mem_key, qe=None)

                        # mapping of pixels A -> B would be very similar to B -> A if the images are similar
                        # and very different if the images are different
                        cycle_dissimilarity_per_pixel = (
                            similarity_per_pixel - reverse_similarity_per_pixel)
                        
                        cycle_dissimilarity_score = F.relu(cycle_dissimilarity_per_pixel).sum() / \
                            cycle_dissimilarity_per_pixel.numel()
                        dissimilarities_across_mem_keys.append(
                            cycle_dissimilarity_score)

                    # filtering our existing or very similar frames
                    dissimilarity_min_across_all = min(dissimilarities_across_mem_keys)
                    
                dissimilarities.append(dissimilarity_min_across_all)

            values, indices = torch.topk(torch.tensor(
                dissimilarities), k=1, largest=True)
            chosen_new_frame = int(indices[0])

            chosen_frames.append(chosen_new_frame)
            chosen_frames_mem_keys.append(
                frame_keys[chosen_new_frame].to(device))
    # chosen_frames_self_similarities.append(get_similarity(chosen_frames_mem_keys[-1], ms=shrinkages[chosen_new_frame].to(device), qk=chosen_frames_mem_keys[-1], qe=selections[chosen_new_frame].to(device)))

        # we don't need to worry about 1st frame with itself, since we take the LEAST similar frames
        return chosen_frames


def select_next_candidates(keys: torch.Tensor, masks: List[torch.tensor], num_next_candidates: int, previously_chosen_candidates: List[int] = (0,), print_progress=False, alpha=1.0, min_mask_presence_px=9, device: torch.device = 'cuda:0', progress_callback=None, only_new_candidates=True):
    assert len(keys) == len(masks)
    assert len(keys) > 0
    assert keys[0].shape[-2:] == masks[0].shape[-2:]
    assert num_next_candidates > 0
    assert len(previously_chosen_candidates) > 0
    assert 0.0 <= alpha <= 1.0
    assert min_mask_presence_px >= 0
    assert len(previously_chosen_candidates) < len(keys)

    """
    Select candidate frames for annotation based on dissimilarity and cycle consistency.

    Parameters
    ----------
    `keys` : `List[torch.Tensor]`
        A list of "key" feature maps for all frames of the video.
    `masks` : `List[torch.Tensor]`
        A list of masks for each frame (predicted or user-provided).
    `num_next_candidates` : `int`
        The number of candidate frames to select.
    `previously_chosen_candidates` : `List[int]`, optional
        A list of previously chosen candidates. Default is (0,).
    `print_progress` : `bool`, optional
        Whether to print progress information. Default is False.
    `alpha` : `float`, optional
        The weight for cycle consistency in the candidate selection process. Default is 1.0.
    `min_mask_presence_px` : `int`, optional
        The minimum number of pixels for a valid mask. Default is 9.

    Returns
    -------
    `List[int]`
        A list of indices of the selected candidate frames.

    Notes
    -----
    This function uses a dissimilarity measure and cycle consistency to select candidate frames for the user to annotate.
    The dissimilarity measure ensures that the selected frames are as diverse as possible, while the cycle consistency
    ensures that the dissimilarity D(A->A)=0, while D(A->B)>0, and is larger the more different A and B are (pixel-wise).

    """
    with torch.no_grad():
        composite_keys = []
        keys = keys.squeeze()
        N = len(keys)
        h, w = keys[0].shape[1:3]  # removing batch dimension
        masks_validity = np.full(N, True)

        for i, mask in enumerate(masks):
            mask_size_px = (mask > 0).sum()

            if mask_size_px < min_mask_presence_px:
                masks_validity[i] = False
                composite_keys.append(None)

            composite_key = keys[i] * mask.max(dim=0, keepdim=True).values # any object -> 1., background -> 0.. Keep 1 channel only
            composite_key = composite_key * alpha + keys[i] * (1 - alpha)

            composite_keys.append(composite_key.to(device))
        
        chosen_candidates = list(previously_chosen_candidates)
        chosen_candidate_keys = [composite_keys[i] for i in chosen_candidates]

        for i in tqdm(range(num_next_candidates), desc='Iteratively picking the most dissimilar frames', disable=not print_progress):
            candidate_dissimilarities = []
            for j in tqdm(range(N), desc='Computing similarity to chosen frames', disable=not print_progress):
                qk = composite_keys[j].to(device)

                if not masks_validity[j]:
                    # ignore this potential candidate
                    dissimilarity_min_across_all = 0
                else:
                    dissimilarities_across_mem_keys = []
                    for mem_key in chosen_candidate_keys:
                        mem_key = mem_key

                        similarity_per_pixel =         get_similarity(mem_key, ms=None, qk=qk,      qe=None)
                        reverse_similarity_per_pixel = get_similarity(qk,      ms=None, qk=mem_key, qe=None)

                        # mapping of pixels A -> B would be very similar to B -> A if the images are similar
                        # and very different if the images are different
                        cycle_dissimilarity_per_pixel = (similarity_per_pixel - reverse_similarity_per_pixel)
                        
                        # Take non-negative mappings, normalize by tensor size
                        cycle_dissimilarity_score = F.relu(cycle_dissimilarity_per_pixel).sum() / cycle_dissimilarity_per_pixel.numel()

                        dissimilarities_across_mem_keys.append(cycle_dissimilarity_score)

                    # filtering our existing or very similar frames
                    # if the key has already been used or is very similar to at least one of the chosen candidates
                    # dissimilarity_min_across_all -> 0 (or close to)
                    dissimilarity_min_across_all = min(dissimilarities_across_mem_keys)
                    
                candidate_dissimilarities.append(dissimilarity_min_across_all)

            index = torch.argmax(torch.tensor(candidate_dissimilarities))
            chosen_new_frame = int(index)

            chosen_candidates.append(chosen_new_frame)
            chosen_candidate_keys.append(composite_keys[chosen_new_frame])

            if progress_callback is not None:
                progress_callback.emit(i + 1)

        if only_new_candidates:
            chosen_candidates = chosen_candidates[len(previously_chosen_candidates):]
        return chosen_candidates