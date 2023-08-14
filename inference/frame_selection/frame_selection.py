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
from torchvision.transforms import Resize, InterpolationMode

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


def select_next_candidates(keys: torch.Tensor, shrinkages, selections, masks: List[torch.tensor], num_next_candidates: int, previously_chosen_candidates: List[int] = (0,), print_progress=False, alpha=0.5, min_mask_presence_percent=0.25, device: torch.device = 'cuda:0', progress_callback=None, only_new_candidates=True, epsilon=0.5):
    assert len(keys) == len(masks)
    assert len(keys) > 0
    # assert keys[0].shape[-2:] == masks[0].shape[-2:]
    assert num_next_candidates > 0
    assert len(previously_chosen_candidates) > 0
    assert 0.0 <= alpha <= 1.0
    assert min_mask_presence_percent >= 0
    assert len(previously_chosen_candidates) < len(keys)


    """
    Select candidate frames for annotation based on dissimilarity and cycle consistency.

    Parameters
    ----------
    keys : torch.Tensor
        A list of "key" feature maps for all frames of the video (from XMem key encoder)
    shrinkages : [Type]
        A list of "shrinkage" feature maps for all frames of the video (from XMem key encoder). Used for similarity computation.
    selections : [Type]
        A list of "sellection" feature maps for all frames of the video (from XMem key encoder). Used for similarity computation.
    masks : List[torch.Tensor]
        A list of masks for each frame (predicted or user-provided).
    num_next_candidates : int
        The number of candidate frames to select.
    previously_chosen_candidates : List[int], optional
        A list of previously chosen candidate indices. Default is (0,).
    print_progress : bool, optional
        Whether to print progress information. Default is False.
    alpha : float, optional
        The weight for the masks in the candidate selection process, [0..1]. If 0 - masks will be ignored, the same frames will be chosen for the same video. If 1.0 - ONLY regions of the frames containing the mask will be compared. Default is 0.5.
        If you trust your masks and want object-specific selections, set higher. If your predictions are really bad, set lower
    min_mask_presence_percent : float, optional
        The minimum percentage of pixels for a valid mask. Default is 0.25. Used to ignore frames with a tiny mask (when heavily occluded or just some random wrong prediction)
    device : torch.device, optional
        The device to run the computation on. Default is 'cuda:0'.
    progress_callback : callable, optional
        A callback function for progress updates. Used in GUI for a progress bar. Default is None.
    only_new_candidates : bool, optional
        Whether to return only the newly selected candidates or include previous as well. Default is True.
    epsilon : float, optional
        Threshold for foreground/background [0..1]. Default is 0.5

    Returns
    -------
    List[int]
        A list of indices of the selected candidate frames.

    Notes
    -----
    This function uses a dissimilarity measure and cycle consistency to select candidate frames for the user to annotate.
    The dissimilarity measure ensures that the selected frames are as diverse as possible, while the cycle consistency
    ensures that the dissimilarity D(A->A)=0, while D(A->B)>0, and is larger the more different A and B are (pixel-wise, on feature map level - so both semantically and spatially).
    """

    with torch.no_grad():
        composite_keys = []
        keys = keys.squeeze()
        N = len(keys)
        h, w = keys[0].shape[1:3]  # removing batch dimension
        resize = Resize((h, w), interpolation=InterpolationMode.NEAREST)
        masks_validity = np.full(N, True)

        invalid = 0
        for i, mask in enumerate(masks):
            mask_3ch = mask if mask.ndim == 3 else mask.unsqueeze(0)
            mask_bin = mask_3ch.max(dim=0).values  # for multiple objects -> use them as one large mask (simplest solution)
            mask_size_px = (mask_bin > epsilon).sum()

            ratio = mask_size_px / mask_bin.numel() * 100
            if ratio < min_mask_presence_percent:  # percentages to ratio
                if i not in previously_chosen_candidates:
                    # if it's previously chosen, it's okay, we don't test for their validity
                    # e.g. we select frame #J, because we predicted something for it
                    # but in reality it's actually empty, so gt=0
                    # so next iteration will break 
                    masks_validity[i] = False
                    composite_keys.append(None)
                    invalid += 1
                    
                    continue

                # if it's previously chosen, it's okay
                # if i in previously_chosen_candidates:
                    # print(f"{i} previous candidate would be invalid (ratio perc={ratio})")
                    # raise ValueError(f"Given min_mask_presence_percent={min_mask_presence_percent}, even the previous candidates will be ignored. Reduce the value to avoid the error.")
                
            mask = resize(mask)
            composite_key = keys[i] * mask.max(dim=0, keepdim=True).values # any object -> 1., background -> 0.. Keep 1 channel only
            composite_key = composite_key * alpha + keys[i] * (1 - alpha)

            composite_keys.append(composite_key.to(dtype=keys[i].dtype, device=device))

        print(f"Frames with invalid (empty or too small) masks: {invalid} / {len(masks)}")
        chosen_candidates = list(previously_chosen_candidates)
        chosen_candidate_keys = [composite_keys[i] for i in chosen_candidates]

        for i in tqdm(range(num_next_candidates), desc='Iteratively picking the most dissimilar frames', disable=not print_progress):
            candidate_dissimilarities = []
            for j in tqdm(range(N), desc='Computing similarity to chosen frames', disable=not print_progress):

                if not masks_validity[j]:
                    # ignore this potential candidate
                    dissimilarity_min_across_all = 0
                else:
                    qk = composite_keys[j].to(device).unsqueeze(0)
                    q_shrinkage = shrinkages[j].to(device).unsqueeze(0)
                    q_selection = selections[j].to(device).unsqueeze(0)

                    dissimilarities_across_mem_keys = []
                    for mem_idx, mem_key in zip(chosen_candidates, chosen_candidate_keys):
                        mem_key = mem_key.unsqueeze(0)
                        mem_shrinkage = shrinkages[mem_idx].to(device).unsqueeze(0)
                        mem_selection = selections[mem_idx].to(device).unsqueeze(0)

                        similarity_per_pixel =         get_similarity(mem_key, ms=mem_shrinkage, qk=qk,      qe=q_selection)
                        reverse_similarity_per_pixel = get_similarity(qk,      ms=q_shrinkage,   qk=mem_key, qe=mem_selection)

                        # mapping of pixels A -> B would be very similar to B -> A if the images are similar
                        # and very different if the images are different
                        cycle_dissimilarity_per_pixel = (similarity_per_pixel - reverse_similarity_per_pixel).to(dtype=torch.float32)
                        
                        # Take non-negative mappings, normalize by tensor size
                        cycle_dissimilarity_score = F.relu(cycle_dissimilarity_per_pixel).sum() / cycle_dissimilarity_per_pixel.numel()
                        dissimilarities_across_mem_keys.append(cycle_dissimilarity_score)

                    # filtering out existing or very similar frames
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
