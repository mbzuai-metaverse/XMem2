from functools import partial
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union

import torch
import torchvision.transforms.functional as FT
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from umap import UMAP
from hdbscan import flat
from tqdm import tqdm

from model.memory_util import get_similarity


# -----------------------------CHOSEN FRAME SELECTORS---------------------------------------

# Utility
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
                # to avoid possible overflow
                key_sum = torch.zeros_like(
                    key, device=device, dtype=torch.float64)

            key_sum += key.type(torch.float64)

            frame_keys.append(key.flatten(start_dim=2).cpu())
            shrinkages.append(shrinkage.flatten(start_dim=2).cpu())
            selections.append(selection.flatten(start_dim=2).cpu())

        num_frames = ti + 1  # 0 after 1 iteration, 1 after 2, etc.

        return frame_keys, shrinkages, selections, device, num_frames, key_sum


def first_frame_only(*args, **kwargs):
    # baseline
    return [0]


def uniformly_selected_frames(dataloader, *args, how_many_frames=10, **kwargs) -> List[int]:
    # baseline
    # TODO: debug and check if works
    num_total_frames = len(dataloader)
    return np.linspace(0, num_total_frames - 1, how_many_frames).astype(int).tolist()


def calculate_proposals_for_annotations_iterative_pca(dataloader, processor, how_many_frames=10, print_progress=False, distance_metric='euclidean') -> List[int]:
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


def calculate_proposals_for_annotations_umap_half_hdbscan_clustering(dataloader, processor, how_many_frames=10, print_progress=False) -> List[int]:
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
        clustering = flat.HDBSCAN_flat(
            smol_keys, n_clusters=how_many_frames + 1)
        labels = clustering.labels_

        chosen_frames = []
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


def calculate_proposals_for_annotations_with_iterative_distance_cycle(dataloader, processor, how_many_frames=10, print_progress=False) -> List[int]:
    with torch.no_grad():
        frame_keys, shrinkages, selections, device, num_frames, key_sum = _extract_keys(
            dataloader, processor, print_progress)

        chosen_frames = [0]
        chosen_frames_mem_keys = [frame_keys[0].to(device)]

        for i in tqdm(range(how_many_frames - 1), desc='Iteratively picking the most dissimilar frames', disable=not print_progress):
            dissimilarities = []
            # how to run a loop for lower memory usage
            for j in tqdm(range(num_frames), desc='Computing similarity to chosen frames', disable=not print_progress):
                qk = frame_keys[j].to(device)
                query_selection = selections[j].to(device)  # query
                query_shrinkage = shrinkages[j].to(device)

                dissimilarities_across_mem_keys = []
                for key_idx, mem_key in zip(chosen_frames, chosen_frames_mem_keys):
                    mem_key = mem_key.to(device)
                    key_shrinkage = shrinkages[key_idx].to(device)
                    key_selection = selections[key_idx].to(device)

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


def calculate_proposals_for_annotations_with_iterative_distance_double_diff(dataloader, processor, how_many_frames=10, print_progress=False) -> List[int]:
    with torch.no_grad():
        frame_keys, shrinkages, selections, device, num_frames, key_sum = _extract_keys(
            dataloader, processor, print_progress)

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

                    # mapping of pixels A -> B would be very similar to B -> A if the images are similar
                    # and very different if the images are different

                    pure_similarity = 2 * similarity_per_pixel - \
                        self_similarity_key - self_similarity_query

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


KNOWN_ANNOTATION_PREDICTORS = {
    'PCA_EUCLIDEAN': partial(calculate_proposals_for_annotations_iterative_pca, distance_metric='euclidean'),
    'PCA_COSINE': partial(calculate_proposals_for_annotations_iterative_pca, distance_metric='cosine'),
    'UMAP_EUCLIDEAN': calculate_proposals_for_annotations_umap_half_hdbscan_clustering,
    'INTERNAL_CYCLE_CONSISTENCY': calculate_proposals_for_annotations_with_iterative_distance_cycle,
    'INTERNAL_DOUBLE_DIFF': calculate_proposals_for_annotations_with_iterative_distance_double_diff,

    'FIRST_FRAME_ONLY': first_frame_only, # ignores the number of candidates, baseline
    'UNIFORM': uniformly_selected_frames  # baseline
}

# ------------------------END CHOSEN-----------------------------------------------
