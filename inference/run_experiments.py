import os
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

from inference.frame_selection.frame_selection import KNOWN_ANNOTATION_PREDICTORS
from inference.run_on_video import predict_annotation_candidates, run_on_video

# ---------------BEGIN Inference and visualization utils --------------------------

def make_non_uniform_grid(rows_of_image_paths: List[List[str]], output_path: str, grid_size=3, resize_to: Tuple[int, int]=(854, 480)):
    assert len(rows_of_image_paths) == grid_size
    for row in rows_of_image_paths:
        assert len(row) <= grid_size

    p_out_dir = Path(output_path)
    if not p_out_dir.exists():
        p_out_dir.mkdir(parents=True)
    num_frames = None

    for row in rows_of_image_paths:
        for img_path_dir in row:
            num_frames_in_dir = len(os.listdir(img_path_dir))
            if num_frames is None:
                num_frames = num_frames_in_dir
            else:
                assert num_frames == num_frames_in_dir

    rows_of_iterators = []
    for row_of_image_dir_paths in rows_of_image_paths:
        row = []
        for image_dir_path in row_of_image_dir_paths:
            p = Path(image_dir_path)
            iterator = iter(sorted(p.iterdir()))
            row.append(iterator)
        rows_of_iterators.append(row)

    for i in tqdm(range(num_frames)):
        rows_of_frames = []
        for row in rows_of_iterators:
            frames = []
            global_h, global_w = None, None
            for iterator in row:
                frame_path = str(next(iterator))
                frame = cv2.imread(frame_path)
                h, w = frame.shape[0:2]

                if resize_to is not None:
                    desired_w, desired_h = resize_to
                    if h != desired_w or w != desired_w:
                        frame = cv2.resize(frame, (desired_w, desired_h))
                        h, w = frame.shape[0:2]

                frames.append(frame)

                if global_h is None:
                    global_h, global_w = h, w

            wide_frame = np.concatenate(frames, axis=1)

            if len(frames) < grid_size:
                pad_size = global_w * (grid_size - len(frames)) // 2
                # center the frame 
                wide_frame = np.pad(wide_frame, [(0, 0), (pad_size, pad_size), (0, 0)], mode='constant', constant_values=0)
            rows_of_frames.append(wide_frame)
        
        big_frame = np.concatenate(rows_of_frames, axis=0)
        cv2.imwrite(str(p_out_dir / f'frame_{i:06d}.png'), big_frame)


def visualize_grid(video_names: List[str], labeled=True):
    for video_name in video_names:
        p_in_general = Path(
            f'/home/maksym/RESEARCH/VIDEOS/RESULTS/XMem_memory/permanent_work_memory/AL_comparison/{video_name}/Overlay')
        if labeled:
            p_in_general /= 'Labeled'

        cycle = p_in_general / 'INTERNAL_CYCLE_CONSISTENCY'
        ddiff = p_in_general / 'INTERNAL_DOUBLE_DIFF'
        umap = p_in_general / 'UMAP_EUCLIDEAN'
        pca_euclidean = p_in_general / 'PCA_EUCLIDEAN'
        pca_cosine = p_in_general / 'PCA_COSINE'
        one_frame_only = p_in_general / 'ONLY_ONE_FRAME'
        baseline_uniform = p_in_general / 'BASELINE_UNIFORM'
        baseline_human = p_in_general / 'HUMAN_CHOSEN'
        ULTIMATE = p_in_general / 'ULTIMATE_AUTO'

        grid = [
            [cycle, ddiff, umap],
            [pca_euclidean, pca_cosine, baseline_uniform],
            [baseline_human, one_frame_only, ULTIMATE]
        ]
        if labeled:
            p_out = p_in_general.parent.parent / 'All_combined'
        else:
            p_out = p_in_general.parent / 'All_combined_unlabeled'

        make_non_uniform_grid(grid, p_out, grid_size=3)


def get_videos_info():
    return {
        'long_scene': {
            'num_annotation_candidates': 3,  # 3,
            'video_frames_path': '/home/maksym/RESEARCH/VIDEOS/long_scene/JPEGImages',
            'video_masks_path': '/home/maksym/RESEARCH/VIDEOS/long_scene/Annotations',
            'masks_out_path': '/home/maksym/RESEARCH/VIDEOS/RESULTS/XMem_memory/permanent_work_memory/AL_comparison/long_scene'
        },
        'long_scene_scale': {
            'num_annotation_candidates': 3,  # 3,
            'video_frames_path': '/home/maksym/RESEARCH/VIDEOS/long_scene_scale/JPEGImages',
            'video_masks_path': '/home/maksym/RESEARCH/VIDEOS/long_scene_scale/Annotations',
            'masks_out_path': '/home/maksym/RESEARCH/VIDEOS/RESULTS/XMem_memory/permanent_work_memory/AL_comparison/long_scene_scale'
        },
        'ariana_smile': {
            'num_annotation_candidates': 3,  # 3,
            'video_frames_path': '/home/maksym/RESEARCH/VIDEOS/Scenes_ariana_fixed_naming/smile/JPEGImages',
            'video_masks_path': '/home/maksym/RESEARCH/VIDEOS/Scenes_ariana_fixed_naming/smile/Annotations/Lips',
            'masks_out_path': '/home/maksym/RESEARCH/VIDEOS/RESULTS/XMem_memory/permanent_work_memory/AL_comparison/ariana_smile'
        },
        'ariana_blog': {
            'num_annotation_candidates': 5,  # 5,
            'video_frames_path': '/home/maksym/RESEARCH/VIDEOS/Scenes_ariana_fixed_naming/blog/JPEGImages',
            'video_masks_path': '/home/maksym/RESEARCH/VIDEOS/Scenes_ariana_fixed_naming/blog/Annotations/Together',
            'masks_out_path': '/home/maksym/RESEARCH/VIDEOS/RESULTS/XMem_memory/permanent_work_memory/AL_comparison/ariana_blog'
        },
    }


def run_multiple_frame_selectors(videos_info: Dict[str, Dict], csv_output_path: str):
    output = pd.DataFrame(columns=list(KNOWN_ANNOTATION_PREDICTORS))
    p_bar = tqdm(total=len(videos_info) * len(KNOWN_ANNOTATION_PREDICTORS))

    for video_name, info in videos_info.items():
        video_frames_path = info['video_frames_path']
        num_candidate_frames = info['num_annotation_candidates']

        results = {}
        for method_name in KNOWN_ANNOTATION_PREDICTORS:
            chosen_annotation_frames = predict_annotation_candidates(
                video_frames_path, num_candidates=num_candidate_frames, approach=method_name)
            results[method_name] = json.dumps(chosen_annotation_frames)
            p_bar.update()

        output.loc[video_name] = results

    output.index.name = 'video_name'
    output.to_csv(csv_output_path)


def run_inference_with_pre_chosen_frames(chosen_frames_csv_path: str, videos_info: Dict[str, Dict], output_path: str, only_methods_subset: Set[str] = None):
    df = pd.read_csv(chosen_frames_csv_path, index_col='video_name')
    num_runs = np.prod(df.shape)
    p_bar = tqdm(
        desc='Running inference comparing multiple different AL approaches', total=num_runs)

    for video_name, info in videos_info.items():
        video_row = df.loc[video_name]
        for method in video_row.index:
            if only_methods_subset is not None and method not in only_methods_subset:
                continue

            chosen_frames_str = video_row.loc[method]
            chosen_frames = json.loads(chosen_frames_str)
            print(chosen_frames)

            video_frames_path = info['video_frames_path']
            video_masks_path = info['video_masks_path']

            output_masks_path = Path(output_path) / video_name / method

            run_on_video(video_frames_path, video_masks_path, output_masks_path,
                         frames_with_masks=chosen_frames, compute_iou=False, print_progress=False)

            p_bar.update()


def visualize_chosen_frames(video_name: str, num_total_frames: int, data: pd.Series, output_path: str):
    def _sort_index(series):
        ll = list(series.index)
        sorted_ll = sorted(ll, key=lambda x: str(
            min(json.loads(series.loc[x]))))
        return sorted_ll

    sorted_index = _sort_index(data)
    plt.figure(figsize=(16, 10))
    plt.title(f"Chosen frames for {video_name}")
    plt.xlim(-10, num_total_frames + 10)
    num_methods = len(data.index)

    plt.ylim(-0.25, num_methods + 0.25)

    plt.xlabel('Frame number')
    plt.ylabel('AL method')

    plt.yticks([])  # disable yticks

    previous_plots = []

    for i, method_name in enumerate(sorted_index):
        chosen_frames = json.loads(data.loc[method_name])
        num_frames = len(chosen_frames)

        x = sorted(chosen_frames)
        y = [i for _ in chosen_frames]
        plt.axhline(y=i, zorder=1, xmin=0.01, xmax=0.99)

        plt.scatter(x=x, y=y, label=method_name, s=256, zorder=3, marker="v")
        if len(previous_plots) != 0:
            for i in range(num_frames):
                curr_x, curr_y = x[i], y[i]
                prev_x, prev_y = previous_plots[-1][0][i], previous_plots[-1][1][i]

                plt.plot([prev_x, curr_x], [prev_y, curr_y],
                         linewidth=1, color='gray', alpha=0.5)

        previous_plots.append((x, y))

        # texts = map(str, range(num_frames))
        # for i, txt in enumerate(texts):
        #     plt.annotate(txt, (x[i] + 2, y[i] + 0.1), zorder=4, fontproperties={'weight': 'bold'})

    plt.legend()
    p_out = Path(f'{output_path}/chosen_frames_{video_name}.png')
    if not p_out.parent.exists():
        p_out.parent.mkdir(parents=True)

    plt.savefig(p_out, bbox_inches='tight')

# -------------------------END Inference and visualization utils --------------------------


if __name__ == "__main__":
    pass

    # ## Usage examples
    # ## Run from root-level directory, e.g. in `main.py`

    # ## Running multiple frame selectors, saving their predicted frame numbers to a .csv file
    # run_multiple_frame_selectors(get_videos_info(), csv_output_path='output/al_videos_chosen_frames.csv')

    # ## Running and visualizing inference based on pre-calculated frames selected
    # run_inference_with_pre_chosen_frames(
    #     chosen_frames_csv_path='output/al_videos_chosen_frames.csv',
    #     videos_info=get_videos_info(),
    #     output_path='/home/maksym/RESEARCH/VIDEOS/RESULTS/XMem_memory/permanent_work_memory/AL_comparison/'
    # )

    # ## Concatenating multiple video results into a non-uniform grid
    # visualize_grid(
    #     names=['long_scene', 'ariana_blog', 'ariana_smile', 'long_scene_scale'],
    #     labeled=True,
    # )
