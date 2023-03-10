from functools import partial
from itertools import chain
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Set, Tuple, Union
import warnings

import cv2
import torch
import torchvision.transforms.functional as FT
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from umap import UMAP
from sklearn.cluster import AgglomerativeClustering
from hdbscan import flat
from tqdm import tqdm
from inference.data.mask_mapper import MaskMapper
from inference.inference_core import InferenceCore
from profilehooks import profile

from model.memory_util import get_similarity


# -----------------------------CHOSEN FRAME SELECTORS---------------------------------------

# Utility
def _extract_keys(dataloder, processor, print_progress=False, pre_flatten=False):
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
                # to avoid possible overflow
                key_sum = torch.zeros_like(
                    key, device=device, dtype=torch.float64)

            key_sum += key.type(torch.float64)

            if pre_flatten:
                frame_keys.append(key.flatten(start_dim=2).cpu())
                shrinkages.append(shrinkage.flatten(start_dim=2).cpu())
                selections.append(selection.flatten(start_dim=2).cpu())
            else:
                frame_keys.append(key.cpu())
                shrinkages.append(shrinkage.cpu())
                selections.append(selection.cpu())

        num_frames = ti + 1  # 0 after 1 iteration, 1 after 2, etc.

        return frame_keys, shrinkages, selections, device, num_frames, key_sum


def _extract_values(dataloader, processor: InferenceCore, print_progress=False):
    values = []
    mask = None

    with torch.no_grad():  # just in case
        hidden_state = None

        for ti, data in enumerate(tqdm(dataloader, disable=not print_progress, desc='Calculating value features')):
            rgb = data['rgb'].cuda()[0]
            if mask is None:
                mask = data['mask'].numpy().squeeze()
                info = data['info']
                need_resize = info['need_resize']

                # https://github.com/hkchengrex/XMem/issues/21 just make exhaustive = True
                mapper = MaskMapper()
                mask, labels = mapper.convert_mask(mask, exhaustive=True)
                mask = torch.Tensor(mask).cuda()
                processor.set_all_labels(labels)

                if min(mask.shape) == 0:  # empty mask, e.g. [1, 0, 720, 1280]
                    raise ValueError(f"EMPTY MASK!!! {info}")  # just don't add anything to the memory
                if need_resize:
                    mask = dataloader.dataset.resize_mask(mask.unsqueeze(0))[0]

                # load first frame into memory
                processor.put_to_permanent_memory(rgb, mask)

            value, hidden_state = processor.encode_frame_value(rgb, hidden_state=hidden_state, mask=mask if ti == 0 else None)
            values.append(value)
            # if ti % 100 == 0:
                # torch.cuda.empty_cache()

        return values


def first_frame_only(*args, **kwargs):
    # baseline
    return [0]


def uniformly_selected_frames(dataloader, *args, how_many_frames=10, **kwargs) -> List[int]:
    # baseline
    # TODO: debug and check if works
    num_total_frames = len(dataloader)
    return np.linspace(0, num_total_frames - 1, how_many_frames).astype(int).tolist()


def calculate_proposals_for_annotations_iterative_pca(dataloader, processor, how_many_frames=10, print_progress=False, distance_metric='euclidean', **kwargs) -> List[int]:
    assert distance_metric in {'cosine', 'euclidean'}
    # might not pick 0-th frame
    np.random.seed(1)  # Just in case
    with torch.no_grad():
        frame_keys, shrinkages, selections, device, num_frames, key_sum = _extract_keys(
            dataloader, processor, print_progress)
        flat_keys = torch.stack([key.flatten().cpu()
                                for key in frame_keys]).numpy()

        # PCA hangs at num_frames // 2: https://github.com/scikit-learn/scikit-learn/issues/22434
        pca = PCA(num_frames - 1, svd_solver='arpack')
        smol_keys = pca.fit_transform(flat_keys.astype(np.float64))
        # smol_keys = flat_keys  # to disable PCA

        chosen_frames = [0]
        for c in range(how_many_frames - 1):
            distances = cdist(smol_keys[chosen_frames],
                              smol_keys, metric=distance_metric)
            closest_to_mem_key_distances = distances.min(axis=0)
            most_distant_frame = np.argmax(closest_to_mem_key_distances)
            chosen_frames.append(int(most_distant_frame))

        return chosen_frames


def calculate_proposals_for_annotations_umap_half_hdbscan_clustering(dataloader, processor, how_many_frames=10, print_progress=False, **kwargs) -> List[int]:
    # might not pick 0-th frame
    with torch.no_grad():
        frame_keys, shrinkages, selections, device, num_frames, key_sum = _extract_keys(
            dataloader, processor, print_progress)
        flat_keys = torch.stack([key.flatten().cpu()
                                for key in frame_keys]).numpy()

        pca = UMAP(n_neighbors=num_frames - 1,
                   n_components=num_frames // 2, random_state=1)
        smol_keys = pca.fit_transform(flat_keys)

        # clustering = AgglomerativeClustering(n_clusters=how_many_frames + 1, linkage='single')
        try:
            clustering = flat.HDBSCAN_flat(smol_keys, n_clusters=how_many_frames)
        except IndexError:
            warnings.warn(f"[!!!] HDBSCAN failed, fallback to uniform baseline (video={dataloader.dataset.vid_name})", RuntimeWarning)
            return uniformly_selected_frames(dataloader, how_many_frames=how_many_frames)

        labels = clustering.labels_

        chosen_frames = []

        for c in range(how_many_frames):
            vectors = smol_keys[labels == c]
            if min(vectors.shape) == 0:
                warnings.warn(f"[!!!] HDBSCAN failed, fallback to uniform baseline (video={dataloader.dataset.vid_name})", RuntimeWarning)
                return uniformly_selected_frames(dataloader, how_many_frames=how_many_frames)

        for c in range(how_many_frames):
            vectors = smol_keys[labels == c]
            true_index_mapping = {i: int(ti) for i, ti in zip(
                range(len(vectors)), np.nonzero(labels == c)[0])}
            center = np.mean(vectors, axis=0)

            # since HDBSCAN is density-based, it makes 0 sense to use anything but euclidean distance here
            distances = cdist(vectors, [center], metric='euclidean').squeeze()

            closest_to_cluster_center_idx = np.argsort(distances)[0]

            chosen_frame_idx = true_index_mapping[closest_to_cluster_center_idx]
            chosen_frames.append(chosen_frame_idx)

        return chosen_frames


def calculate_proposals_for_annotations_with_iterative_distance_cycle(dataloader, processor, how_many_frames=10, print_progress=False, **kwargs) -> List[int]:
    with torch.no_grad():
        frame_keys, shrinkages, selections, device, num_frames, key_sum = _extract_keys(
            dataloader, processor, print_progress, pre_flatten=True)

        chosen_frames = [0]
        chosen_frames_mem_keys = [frame_keys[0].to(device)]

        for i in tqdm(range(how_many_frames - 1), desc='Iteratively picking the most dissimilar frames', disable=not print_progress):
            dissimilarities = []
            # how to run a loop for lower memory usage
            for j in tqdm(range(num_frames), desc='Computing similarity to chosen frames', disable=not print_progress):
                qk = frame_keys[j].to(device)

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
                    cycle_dissimilarity_score = cycle_dissimilarity_per_pixel.abs().sum() / \
                        cycle_dissimilarity_per_pixel.numel()

                    dissimilarities_across_mem_keys.append(
                        cycle_dissimilarity_score)

                # filtering our existing or very similar frames
                dissimilarity_min_across_all = min(
                    dissimilarities_across_mem_keys)
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


def calculate_proposals_for_annotations_with_iterative_distance_double_diff(dataloader, processor, how_many_frames=10, print_progress=False, **kwargs) -> List[int]:
    with torch.no_grad():
        frame_keys, shrinkages, selections, device, num_frames, key_sum = _extract_keys(
            dataloader, processor, print_progress, pre_flatten=True)

        chosen_frames = [0]
        chosen_frames_mem_keys = [frame_keys[0]]
        # chosen_frames_self_similarities = []
        for c in range(how_many_frames - 1):
            dissimilarities = []
            # how to run a loop for lower memory usage
            for i in tqdm(range(num_frames), desc=f'Computing similarity to chosen frames', disable=not print_progress):
                true_frame_idx = i
                qk = frame_keys[true_frame_idx].to(device)
                query_selection = selections[true_frame_idx].to(
                    device)  # query
                query_shrinkage = shrinkages[true_frame_idx].to(device)

                dissimilarities_across_mem_keys = []
                for key_idx, mem_key in zip(chosen_frames, chosen_frames_mem_keys):
                    mem_key = mem_key.to(device)
                    key_shrinkage = shrinkages[key_idx].to(device)
                    key_selection = selections[key_idx].to(device)

                    similarity_per_pixel = get_similarity(
                        mem_key, ms=key_shrinkage, qk=qk, qe=query_selection)
                    self_similarity_key = get_similarity(
                        mem_key, ms=key_shrinkage, qk=mem_key, qe=key_selection)
                    self_similarity_query = get_similarity(
                        qk, ms=query_shrinkage, qk=qk, qe=query_selection)

                    pure_similarity = 2 * similarity_per_pixel - self_similarity_key - self_similarity_query

                    # mapping of pixels A -> B would be very similar to B -> A if the images are similar
                    # and very different if the images are differentflat_keys
                    dissimilarity_score = pure_similarity.abs().sum() / pure_similarity.numel()

                    dissimilarities_across_mem_keys.append(dissimilarity_score)

                # filtering our existing or very similar frames
                dissimilarity_min_across_all = min(
                    dissimilarities_across_mem_keys)
                dissimilarities.append(dissimilarity_min_across_all)

            values, indices = torch.topk(torch.tensor(
                dissimilarities), k=1, largest=True)
            chosen_new_frame = int(indices[0])

            chosen_frames.append(chosen_new_frame)
            chosen_frames_mem_keys.append(
                frame_keys[chosen_new_frame].to(device))

        # we don't need to worry about 1st frame with itself, since we take the LEAST similar frames
        return chosen_frames

# ------------------------END CHOSEN-----------------------------------------------
# -----------------------------TARGET AWARE FRAME SELECTORS-----------------------------------

def calculate_proposals_for_annotations_iterative_pca_MASKS(dataloader, processor, existing_masks_path: str, how_many_frames=10, print_progress=False, mult_instead=False, alpha=1.0, distance_metric='euclidean', **kwargs) -> List[int]:
    assert distance_metric in {'cosine', 'euclidean'}
    # might not pick 0-th frame
    np.random.seed(1)  # Just in case
    with torch.no_grad():
        frame_keys, shrinkages, selections, device, num_frames, key_sum = _extract_keys(dataloader, processor, print_progress)

        h, w = frame_keys[0].squeeze().shape[1:3]  # removing batch dimension
        p_masks_dir = Path(existing_masks_path)
        for i, p_img in enumerate(p_masks_dir.iterdir()):
            img = cv2.imread(str(p_img))
            img = cv2.resize(img, (w, h)) / 255.

            if not mult_instead:
                composite_key = torch.cat([frame_keys[i].cpu().squeeze(), FT.to_tensor(img)], dim=0)  # along channels
            else:
                composite_key = frame_keys[i].cpu().squeeze() * FT.to_tensor(img).max(dim=0, keepdim=True).values # all objects -> 1., background -> 0.. Keep 1 channel only
                composite_key = composite_key * alpha + frame_keys[i].cpu().squeeze() * (1 - alpha)
            frame_keys[i] = composite_key

        flat_keys = torch.stack([key.flatten()
                                for key in frame_keys]).numpy()

        # PCA hangs at num_frames // 2: https://github.com/scikit-learn/scikit-learn/issues/22434
        pca = PCA(num_frames - 1, svd_solver='arpack')
        smol_keys = pca.fit_transform(flat_keys.astype(np.float64))
        # smol_keys = flat_keys  # to disable PCA

        chosen_frames = [0]
        for c in range(how_many_frames - 1):
            distances = cdist(smol_keys[chosen_frames],
                              smol_keys, metric=distance_metric)
            closest_to_mem_key_distances = distances.min(axis=0)
            most_distant_frame = np.argmax(closest_to_mem_key_distances)
            chosen_frames.append(int(most_distant_frame))

        return chosen_frames


def calculate_proposals_for_annotations_umap_hdbscan_clustering_PURE_MASKS(dataloader, processor, existing_masks_path: str, how_many_frames=10, print_progress=False, **kwargs) -> List[int]:
    # might not pick 0-th frame
    p_masks_dir = Path(existing_masks_path)
    masks = []
    for p_img in p_masks_dir.iterdir():
        img = cv2.imread(str(p_img))
        img = cv2.resize(img, (427, 240))
        masks.append(img)

    num_frames = len(masks)
    flat_masks = np.stack([mask.flatten() for mask in masks])

    dim_red = UMAP(num_frames - 1,
                n_components=num_frames // 2, random_state=1)
    smol_masks = dim_red.fit_transform(flat_masks)
    try:
        clustering = flat.HDBSCAN_flat(
            smol_masks, n_clusters=how_many_frames)
    except IndexError:
        warnings.warn(f"[!!!] HDBSCAN failed, fallback to uniform baseline (video={dataloader.dataset.vid_name})", RuntimeWarning)
        return uniformly_selected_frames(dataloader, how_many_frames=how_many_frames)
    labels = clustering.labels_

    chosen_frames = []
    distances_from_first_mask = []

    for c in range(how_many_frames):
        vectors = smol_masks[labels == c]
        if min(vectors.shape) == 0:
            warnings.warn(f"[!!!] HDBSCAN failed, fallback to uniform baseline (video={dataloader.dataset.vid_name})", RuntimeWarning)
            return uniformly_selected_frames(dataloader, how_many_frames=how_many_frames)

    for c in tqdm(range(how_many_frames), desc='Choosing cluster medoids', disable=not print_progress):
        vectors = smol_masks[labels == c]
        true_index_mapping = {i: int(ti) for i, ti in zip(
            range(len(vectors)), np.nonzero(labels == c)[0])}
        center = np.mean(vectors, axis=0)

        # since HDBSCAN is density-based, it makes 0 sense to use anything but euclidean distance here
        distances = cdist(vectors, [center], metric='euclidean').squeeze()

        closest_to_cluster_center_idx = np.argsort(distances)[0]

        dist_from_first_to_medoid = cdist([smol_masks[0]], [smol_masks[closest_to_cluster_center_idx]], metric='euclidean')
        distances_from_first_mask.append(dist_from_first_to_medoid)

        chosen_frame_idx = true_index_mapping[closest_to_cluster_center_idx]
        chosen_frames.append(chosen_frame_idx)

    # Since we already have frame 0, we need to drop the scene associated with this frame, so the closest to it
    chosen_frames.pop(int(np.argmin(distances_from_first_mask)))
    chosen_frames.insert(0, 0)

    return chosen_frames

def calculate_proposals_for_annotations_umap_hdbscan_clustering_MASKS(dataloader, processor, existing_masks_path: str, how_many_frames=10, print_progress=False, mult_instead=False, alpha=1.0, **kwargs) -> List[int]:
    # might not pick 0-th frame
    with torch.no_grad():
        p_masks_dir = Path(existing_masks_path)

        frame_keys, shrinkages, selections, device, num_frames, key_sum = _extract_keys(
            dataloader, processor, print_progress)

        h, w = frame_keys[0].squeeze().shape[1:3]  # removing batch dimension
        p_masks_dir = Path(existing_masks_path)
        for i, p_img in enumerate(p_masks_dir.iterdir()):
            img = cv2.imread(str(p_img))
            img = cv2.resize(img, (w, h)) / 255.

            if not mult_instead:
                composite_key = torch.cat([frame_keys[i].cpu().squeeze(), FT.to_tensor(img)], dim=0)  # along channels
            else:
                composite_key = frame_keys[i].cpu().squeeze() * FT.to_tensor(img).max(dim=0, keepdim=True).values
                composite_key = composite_key * alpha + frame_keys[i].cpu().squeeze() * (1 - alpha)
                 # all objects -> 1., background -> 0.. Keep 1 channel only
            frame_keys[i] = composite_key

        flat_keys = torch.stack([key.flatten().cpu()
                                    for key in frame_keys]).numpy()

        pca = UMAP(n_neighbors=num_frames - 1,
                    n_components=num_frames // 2, random_state=1)
        smol_keys = pca.fit_transform(flat_keys)
        try:
            clustering = flat.HDBSCAN_flat(
                smol_keys, n_clusters=how_many_frames)
        except IndexError:
            warnings.warn(f"[!!!] HDBSCAN failed, fallback to uniform baseline (video={dataloader.dataset.vid_name})", RuntimeWarning)
            return uniformly_selected_frames(dataloader, how_many_frames=how_many_frames)
        labels = clustering.labels_

        chosen_frames = []
        distances_from_first_mask = []

        for c in range(how_many_frames):
            vectors = smol_keys[labels == c]
            if min(vectors.shape) == 0:
                warnings.warn(f"[!!!] HDBSCAN failed, fallback to uniform baseline (video={dataloader.dataset.vid_name})", RuntimeWarning)
                return uniformly_selected_frames(dataloader, how_many_frames=how_many_frames)

        for c in tqdm(range(how_many_frames), desc='Choosing cluster medoids', disable=not print_progress):
            vectors = smol_keys[labels == c]
            true_index_mapping = {i: int(ti) for i, ti in zip(
                range(len(vectors)), np.nonzero(labels == c)[0])}
            center = np.mean(vectors, axis=0)

            # since HDBSCAN is density-based, it makes 0 sense to use anything but euclidean distance here
            distances = cdist(vectors, [center], metric='euclidean').squeeze()

            closest_to_cluster_center_idx = np.argsort(distances)[0]

            dist_from_first_to_medoid = cdist([smol_keys[0]], [smol_keys[closest_to_cluster_center_idx]], metric='euclidean')
            distances_from_first_mask.append(dist_from_first_to_medoid)

            chosen_frame_idx = true_index_mapping[closest_to_cluster_center_idx]
            chosen_frames.append(chosen_frame_idx)

        # Since we already have frame 0, we need to drop the scene associated with this frame, so the closest to it
        print(distances_from_first_mask)
        chosen_frames.pop(int(np.argmin(distances_from_first_mask)))
        chosen_frames.insert(0, 0)

        return chosen_frames

def calculate_proposals_for_annotations_umap_hdbscan_clustering_values(dataloader, processor, how_many_frames=10, print_progress=False, **kwargs):
    with torch.no_grad():
        a = perf_counter()
        values = _extract_values(dataloader, processor, print_progress=print_progress)
        num_frames = len(values)

        flat_values = torch.stack([value.flatten().cpu()
                                    for value in values]).numpy()
        
        pca = UMAP(n_neighbors=num_frames - 1,
                    n_components=num_frames // 2, random_state=1)
        smol_values = pca.fit_transform(flat_values)

        try:
            clustering = flat.HDBSCAN_flat(
                smol_values, n_clusters=how_many_frames)
        except IndexError:
            warnings.warn(f"[!!!] HDBSCAN failed, fallback to uniform baseline (video={dataloader.dataset.vid_name})", RuntimeWarning)
            return uniformly_selected_frames(dataloader, how_many_frames=how_many_frames)
        labels = clustering.labels_

        chosen_frames = []
        distances_from_first_mask = []

        for c in range(how_many_frames):
            vectors = smol_values[labels == c]
            if min(vectors.shape) == 0:
                warnings.warn(f"[!!!] HDBSCAN failed, fallback to uniform baseline (video={dataloader.dataset.vid_name})", RuntimeWarning)
                return uniformly_selected_frames(dataloader, how_many_frames=how_many_frames)

        for c in tqdm(range(how_many_frames), desc='Choosing cluster medoids', disable=not print_progress):
            vectors = smol_values[labels == c]
            true_index_mapping = {i: int(ti) for i, ti in zip(
                range(len(vectors)), np.nonzero(labels == c)[0])}
            center = np.mean(vectors, axis=0)

            # since HDBSCAN is density-based, it makes 0 sense to use anything but euclidean distance here
            distances = cdist(vectors, [center], metric='euclidean').squeeze()

            closest_to_cluster_center_idx = np.argsort(distances)[0]

            dist_from_first_to_medoid = cdist([smol_values[0]], [smol_values[closest_to_cluster_center_idx]], metric='euclidean')
            distances_from_first_mask.append(dist_from_first_to_medoid)

            chosen_frame_idx = true_index_mapping[closest_to_cluster_center_idx]
            chosen_frames.append(chosen_frame_idx)

        # Since we already have frame 0, we need to drop the scene associated with this frame, so the closest to it
        chosen_frames.pop(int(np.argmin(distances_from_first_mask)))
        chosen_frames.insert(0, 0)
        b = perf_counter()
        print(f"TOOK {b - a:.2f} seconds")

        return chosen_frames

def calculate_proposals_for_annotations_umap_agglomerative_clustering_values(dataloader, processor, how_many_frames=10, print_progress=False, **kwargs):
    with torch.no_grad():
        values = _extract_values(dataloader, processor, print_progress=print_progress)
        num_frames = len(values)

        flat_values = torch.stack([value.flatten().cpu()
                                    for value in values]).numpy()
        
        pca = UMAP(n_neighbors=num_frames - 1,
                    n_components=num_frames // 2, random_state=1)
        smol_values = pca.fit_transform(flat_values)
        clustering = AgglomerativeClustering(n_clusters=how_many_frames).fit(smol_values)

        labels = clustering.labels_

        chosen_frames = []
        distances_from_first_mask = []

        for c in tqdm(range(how_many_frames), desc='Choosing cluster medoids', disable=not print_progress):
            vectors = smol_values[labels == c]
            true_index_mapping = {i: int(ti) for i, ti in zip(
                range(len(vectors)), np.nonzero(labels == c)[0])}
            center = np.mean(vectors, axis=0)

            # since Agglomerative cluserting uses variance minimization linkage by default, it only accepts euclidean distance here
            distances = cdist(vectors, [center], metric='euclidean').squeeze()

            closest_to_cluster_center_idx = np.argsort(distances)[0]

            dist_from_first_to_medoid = cdist([smol_values[0]], [smol_values[closest_to_cluster_center_idx]], metric='euclidean')
            distances_from_first_mask.append(dist_from_first_to_medoid)

            chosen_frame_idx = true_index_mapping[closest_to_cluster_center_idx]
            chosen_frames.append(chosen_frame_idx)

        # Since we already have frame 0, we need to drop the scene associated with this frame, so the closest to it
        chosen_frames.pop(int(np.argmin(distances_from_first_mask)))
        chosen_frames.insert(0, 0)

        return chosen_frames

def calculate_proposals_for_annotations_with_iterative_distance_cycle_MASKS(dataloader, processor, existing_masks_path: str, how_many_frames=10, print_progress=False, mult_instead=False, alpha=1.0, too_small_mask_threshold_px=9, **kwargs) -> List[int]:
    import torch.nn.functional as F
    with torch.no_grad():
        frame_keys, shrinkages, selections, device, num_frames, key_sum = _extract_keys(
            dataloader, processor, print_progress)
           
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

def calculate_proposals_for_annotations_with_top_k_influence_frame_level(dataloader, processor, how_many_frames=10, print_progress=False, **kwargs):
    import torch.nn.functional as F
    a = perf_counter()
    values, *_ = _extract_keys(dataloader, processor, print_progress) #_extract_values(dataloader, processor, print_progress=print_progress)
    # ~3Mb per value
    N = len(values)
    # H, W = values[0].shape[-2:]
    G = 1 #values[0].shape[-4]  # second dim
    HW = values[0].shape[-1]

    neg_inf = float('-inf')
    # both can be 3D as well
    max_map = torch.full((N - 1, ), neg_inf, dtype=torch.float64, device='cuda:0')
    argmax_map = torch.full((N - 1, ), -1, dtype=torch.int64, device='cuda:0')

    num_total_ops = (N) * (N - 2)
    p_bar = tqdm(total=num_total_ops, desc='Computing pairwise frame values attention', disable=not print_progress)
    for mem_frame_idx in range(0, N):
        mem_frame = values[mem_frame_idx].cuda()
        # mem_frame = torch.swapaxes(mem_frame, 1, 2)  # [B, G, CV, H, W] -> [B, CV, G, H, W]
        
        for query_frame_idx in chain(range(1, mem_frame_idx), range(mem_frame_idx + 1, N)):  # skip frame#0, frame#mem_frame_idx
            query_frame = values[query_frame_idx].cuda()
            # query_frame = torch.swapaxes(query_frame, 1, 2)  # [B, G, CV, H, W] -> [B, CV, G, H, W]

            curr_affinity = get_similarity(mem_frame, None, query_frame, None)  # No selection or shrinkage used
            # try (> 0).sum() instead
            curr_affinity = F.relu(curr_affinity.squeeze()).sum()  # how much each pixel is being influenced at most
            # curr_affinity.shape = [B=1 x N=1 x HW]

            # currently averages across all object groups
            if curr_affinity > max_map[query_frame_idx - 1]:  # because max_map and argmax_map have size N - 1, not N
                max_map[query_frame_idx - 1] = curr_affinity# torch.where(affected_pixels, curr_affinity, max_map[query_frame_idx - 1])  # replace only affected pixels
                argmax_map[query_frame_idx - 1] = mem_frame_idx  # the `affected_pixels` in `query_frame` are more affected by mem_frame than other frames
            
            p_bar.update()

    argmax_map_flat = argmax_map.flatten()
    uniq, counts = torch.unique(argmax_map_flat[argmax_map_flat != -1], return_counts=True)  # ignoring pixels that are potentially unaffected at all, probably not even necessary
    result = sorted(zip(uniq, counts), key=lambda x: x[1], reverse=True)  # sort frame indices by influence, descending
    
    result_iter = iter(result)
    most_influential_frames = [0]
    while True:#len(most_influential_frames) < how_many_frames:
        try:
            next_best_frame, num_influenced_pixels = next(result_iter)
        except StopIteration:
            break
        print(f"{int(next_best_frame):>3}: {int(num_influenced_pixels)}")
        if next_best_frame == 0:
            # frame#0 is already included
            continue

        most_influential_frames.append(int(next_best_frame))

    b = perf_counter()
    if print_progress:
        print(f"TOOK {b-a:.2f} seconds for {N} frames")
    
    return most_influential_frames[:how_many_frames]
# @profile(stdout=False, immediate=False, filename='profile_influence.profile')
def calculate_proposals_for_annotations_with_top_k_influence(dataloader, processor, how_many_frames=10, print_progress=False, **kwargs):
    import torch.nn.functional as F
    a = perf_counter()
    values, *_ = _extract_keys(dataloader, processor, print_progress) #_extract_values(dataloader, processor, print_progress=print_progress)
    # ~3Mb per value
    N = len(values)
    # H, W = values[0].shape[-2:]
    G = 1 #values[0].shape[-4]  # second dim
    HW = values[0].shape[-1]

    neg_inf = float('-inf')
    # both can be 3D as well
    max_map = torch.full((N - 1, HW * G), neg_inf, dtype=torch.float64, device='cuda:0')
    argmax_map = torch.full((N - 1, HW * G), -1, dtype=torch.int64, device='cuda:0')

    num_total_ops = (N) * (N - 2)
    p_bar = tqdm(total=num_total_ops, desc='Computing pairwise frame values attention', disable=not print_progress)
    for mem_frame_idx in range(0, N):
        mem_frame = values[mem_frame_idx].cuda()
        # mem_frame = torch.swapaxes(mem_frame, 1, 2)  # [B, G, CV, H, W] -> [B, CV, G, H, W]
        
        for query_frame_idx in chain(range(1, mem_frame_idx), range(mem_frame_idx + 1, N)):  # skip frame#0, frame#mem_frame_idx
            query_frame = values[query_frame_idx].cuda()
            # query_frame = torch.swapaxes(query_frame, 1, 2)  # [B, G, CV, H, W] -> [B, CV, G, H, W]

            curr_affinity = get_similarity(mem_frame, None, query_frame, None)  # No selection or shrinkage used
            # try (> 0).sum() instead
            curr_affinity = F.relu(curr_affinity.squeeze()).sum(dim=0)  # how much each pixel is being influenced at most
            # curr_affinity.shape = [B=1 x N=1 x HW]

            # currently averages across all object groups
            affected_pixels = curr_affinity > max_map[query_frame_idx - 1]  # because max_map and argmax_map have size N - 1, not N
            max_map[query_frame_idx - 1] = torch.where(affected_pixels, curr_affinity, max_map[query_frame_idx - 1])  # replace only affected pixels
            argmax_map[query_frame_idx - 1, affected_pixels] = mem_frame_idx  # the `affected_pixels` in `query_frame` are more affected by mem_frame than other frames
            
            p_bar.update()

    argmax_map_flat = argmax_map.flatten()
    uniq, counts = torch.unique(argmax_map_flat[argmax_map_flat != -1], return_counts=True)  # ignoring pixels that are potentially unaffected at all, probably not even necessary
    result = sorted(zip(uniq, counts), key=lambda x: x[1], reverse=True)  # sort frame indices by influence, descending
    
    result_iter = iter(result)
    most_influential_frames = [0]
    while True:#len(most_influential_frames) < how_many_frames:
        try:
            next_best_frame, num_influenced_pixels = next(result_iter)
        except StopIteration:
            break
        # print(f"{int(next_best_frame):>3}: {int(num_influenced_pixels)}")
        if next_best_frame == 0:
            # frame#0 is already included
            continue

        most_influential_frames.append(int(next_best_frame))

    b = perf_counter()
    if print_progress:
        print(f"TOOK {b-a:.2f} seconds for {N} frames")
    
    return most_influential_frames[:how_many_frames]

# ------------------------END TARGET AWARE--------------------------------------------

KNOWN_ANNOTATION_PREDICTORS = {
    'PCA_EUCLIDEAN': partial(calculate_proposals_for_annotations_iterative_pca, distance_metric='euclidean'),
    'PCA_COSINE': partial(calculate_proposals_for_annotations_iterative_pca, distance_metric='cosine'),
    'UMAP_EUCLIDEAN': calculate_proposals_for_annotations_umap_half_hdbscan_clustering,
    'INTERNAL_CYCLE_CONSISTENCY': calculate_proposals_for_annotations_with_iterative_distance_cycle,
    'INTERNAL_DOUBLE_DIFF': calculate_proposals_for_annotations_with_iterative_distance_double_diff,
    'INTERNAL_INFLUENCE_MASKS': calculate_proposals_for_annotations_with_top_k_influence,

    'FIRST_FRAME_ONLY': first_frame_only, # ignores the number of candidates, baseline
    'UNIFORM': uniformly_selected_frames,  # baseline
}

KNOWN_TARGET_AWARE_ANNOTATION_PREDICTORS = {
    # 'PCA_EUCLIDEAN_MASKS': calculate_proposals_for_annotations_iterative_pca_MASKS,
    # 'PCA_EUCLIDEAN_MASKS_MULT': partial(calculate_proposals_for_annotations_iterative_pca_MASKS, mult_instead=True),

    # 'PCA_COSINE_MASKS': partial(calculate_proposals_for_annotations_iterative_pca_MASKS, distance_metric='cosine'),
    # 'PCA_COSINE_MASKS_MULT': partial(calculate_proposals_for_annotations_iterative_pca_MASKS, distance_metric='cosine', mult_instead=True),
    # 'PCA_COSINE_MASKS_MULT_BLEND': partial(calculate_proposals_for_annotations_iterative_pca_MASKS, distance_metric='cosine', mult_instead=True, alpha=0.5),

    # 'UMAP_EUCLIDEAN_PURE_MASKS': calculate_proposals_for_annotations_umap_hdbscan_clustering_PURE_MASKS,
    # 'UMAP_EUCLIDEAN_MASKS': calculate_proposals_for_annotations_umap_hdbscan_clustering_MASKS,
    # 'UMAP_EUCLIDEAN_MASKS_MULT': partial(calculate_proposals_for_annotations_umap_hdbscan_clustering_MASKS, mult_instead=True),
    # 'UMAP_EUCLIDEAN_MASKS_MULT_BLEND': partial(calculate_proposals_for_annotations_umap_hdbscan_clustering_MASKS, mult_instead=True, alpha=0.5),
    # 'UMAP_EUCLIDEAN_VALUES': calculate_proposals_for_annotations_umap_hdbscan_clustering_values,
    # 'UMAP_EUCLIDEAN_AGGLO_VALUES': calculate_proposals_for_annotations_umap_agglomerative_clustering_values,
    # 'INTERNAL_CYCLE_CONSISTENCY_MASKS': calculate_proposals_for_annotations_with_iterative_distance_cycle_MASKS,
    # 'INTERNAL_CYCLE_CONSISTENCY_MASKS_MULT': partial(calculate_proposals_for_annotations_with_iterative_distance_cycle_MASKS, mult_instead=True),

    # 'INTERNAL_CYCLE_CONSISTENCY_MASKS_MULT_BLEND': partial(calculate_proposals_for_annotations_with_iterative_distance_cycle_MASKS, mult_instead=True, alpha=0.9),
    'INTERNAL_CYCLE_CONSISTENCY_MASKS_MULT_BLEND': partial(calculate_proposals_for_annotations_with_iterative_distance_cycle_MASKS, mult_instead=True, alpha=0.2),

    'FIRST_FRAME_ONLY': first_frame_only, # ignores the number of candidates, baseline
    'UNIFORM': uniformly_selected_frames,  # baseline
}
