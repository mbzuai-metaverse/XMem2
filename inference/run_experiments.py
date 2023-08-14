import os
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import Image
from inference.frame_selection.frame_selection import uniformly_selected_frames
from util.metrics import batched_f_measure, batched_jaccard
from p_tqdm import p_umap

# from inference.frame_selection.frame_selection import KNOWN_ANNOTATION_PREDICTORS
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


def run_multiple_frame_selectors(videos_info: Dict[str, Dict], csv_output_path: str, predictors: Dict[str, callable] = None, load_existing_masks=False):
    output = pd.DataFrame(columns=list(predictors))
    p_bar = tqdm(total=len(videos_info) * len(predictors))

    exceptions = pd.DataFrame(columns=['video', 'method', 'error_message'])

    for video_name, info in videos_info.items():
        video_frames_path = info['video_frames_path']
        num_candidate_frames = info['num_annotation_candidates']
        if load_existing_masks:
            masks_first_frame_only = Path(info['masks_out_path']) / 'ONLY_ONE_FRAME'
        else:
            masks_first_frame_only = None

        results = {}
        for method_name, method_func in predictors.items():
            try:
                chosen_annotation_frames = predict_annotation_candidates(
                    video_frames_path, 
                    num_candidates=num_candidate_frames,
                    candidate_selection_function=method_func,
                    masks_first_frame_only=masks_first_frame_only,
                    masks_in_path=info['video_masks_path'], 
                    masks_out_path=Path(info['masks_out_path']) / 'FIRST_FRAME_ONLY' / 'masks',  # used by some target-aware algorithms
                    print_progress=False
                )
            except Exception as e:
                print(f"[!!!] ERROR ({video_name},{method_name})={e}")
                print("Resulting to uniform baseline")
                chosen_annotation_frames = predict_annotation_candidates(
                    video_frames_path,
                    num_candidates=num_candidate_frames,
                    candidate_selection_function=KNOWN_ANNOTATION_PREDICTORS['UNIFORM'],
                    masks_in_path=info['video_masks_path'],
                    print_progress=False
                )
                exceptions.append([video_name, method_name, str(e)])

            torch.cuda.empty_cache()
            results[method_name] = json.dumps(chosen_annotation_frames)
            p_bar.update()

        output.loc[video_name] = results

        # save updated after every video
        output.index.name = 'video_name'
        output.to_csv(csv_output_path)

    if min(exceptions.shape) > 0:
        exceptions.to_csv('output/exceptions.csv')


def run_inference_with_pre_chosen_frames(chosen_frames_csv_path: str, videos_info: Dict[str, Dict], output_path: str, only_methods_subset: Set[str] = None, compute_iou=False, IoU_results_save_path=None, **kwargs):
    df = pd.read_csv(chosen_frames_csv_path, index_col='video_name')
    if only_methods_subset is not None:
        num_runs = df.shape[0] * len(only_methods_subset)
    else:
        num_runs = np.prod(df.shape)

    if compute_iou:
        assert IoU_results_save_path is not None
        p_iou_dir = Path(IoU_results_save_path)

    i = 0
    p_bar = tqdm(desc='Running inference comparing multiple different AL approaches', total=num_runs)

    for video_name, info in videos_info.items():
        video_row = df.loc[video_name]
        # ious = {}
        for method in video_row.index:
            if only_methods_subset is not None and method not in only_methods_subset:
                continue
            
            chosen_frames_str = video_row.loc[method]
            chosen_frames = json.loads(chosen_frames_str)

            video_frames_path = info['video_frames_path']
            video_masks_path = info['video_masks_path']

            output_masks_path = Path(output_path) / video_name / method

            stats = run_on_video(video_frames_path, video_masks_path, output_masks_path,
                         frames_with_masks=chosen_frames, compute_iou=compute_iou, print_progress=False, **kwargs)

            if compute_iou:
                p_out_curr_video_method = p_iou_dir / video_name
                if not p_out_curr_video_method.exists():
                    p_out_curr_video_method.mkdir(parents=True)

                stats.to_csv(p_out_curr_video_method / f'{method}.csv')#f'output/AL_comparison_all_methods/{video_name}_{method}.csv')
                # print(f"Video={video_name},method={method},IoU={stats['iou'].mean():.4f}")
                # ious[f'{video_name}_{method}'] = [float(iou) for iou in stats['iou']]

            p_bar.update()
            i += 1
    
        # with open(f'output/AL_comparison_all_methods/ious_{video_name}_all_methods.json', 'wt') as f_out:
        #     json.dump(ious, f_out)

def run_inference_with_uniform_frames(videos_info: Dict[str, Dict], output_path: str, **kwargs):
    num_runs = len(videos_info)

    i = 0
    p_bar = tqdm(desc='Running inference comparing multiple different AL approaches', total=num_runs)

    for video_name, info in videos_info.items():
        frames = os.listdir(info['video_frames_path'])
        chosen_frames = uniformly_selected_frames(frames, how_many_frames=info['num_annotation_candidates'])

        video_frames_path = info['video_frames_path']
        video_masks_path = info['video_masks_path']

        output_masks_path = Path(output_path) / video_name
        try:
            stats = run_on_video(video_frames_path, video_masks_path, output_masks_path,
                        frames_with_masks=chosen_frames, compute_iou=False, print_progress=False, **kwargs)
        except ValueError as e:
            print(f"[!!!] {e}")
        p_bar.update()
        i += 1


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
# ------------------------BEGIN metrics ---------------------------------------------------

def _load_gt(p):
    return np.stack([np.array(Image.open(p_gt).convert('P')) for p_gt in sorted(p.iterdir())])


def _load_preds(p, palette: Image.Image, size: tuple):
    return np.stack([Image.open(p_gt).convert('RGB').resize(size, resample=Image.Resampling.NEAREST).quantize(palette=palette, dither=Image.Dither.NONE) for p_gt in sorted(p.iterdir())])
        
def compute_metrics_al(p_source_masks, p_preds, looped=True):
    def _proc(p_video: Path):
        video_name = p_video.name
        p_gts = p_source_masks / p_video.name
        first_mask = Image.open(next(p_gts.iterdir())).convert('P')
        w, h = first_mask.size
        gts = _load_gt(p_gts)

        stats = {
            'video_name': video_name
        }

        for p_method in p_video.iterdir():
            if not p_method.is_dir():
                continue
            method_name = p_method.name
            p_masks = p_method / 'masks'
            preds = _load_preds(p_masks, palette=first_mask, size=(w, h))

            assert preds.shape == gts.shape

            iou = batched_jaccard(gts, preds)
            avg_iou = iou.mean(axis=0)

            f_score = batched_f_measure(gts, preds)
            avg_f_score = f_score.mean(axis=0)

            stats[f'{method_name}-iou'] = float(avg_iou)
            stats[f'{method_name}-f'] = float(avg_f_score)

            if looped:
                n = iou.shape[0]
                between = int(0.9 * n)
                first_part_iou = iou[:between].mean()
                second_part_iou = iou[between:].mean()

                first_part_f_score = f_score[:between].mean()
                second_part_f_score = f_score[between:].mean()

                stats[f'{method_name}-iou-90'] = float(first_part_iou)
                stats[f'{method_name}-iou-10'] = float(second_part_iou)
                stats[f'{method_name}-f-90'] = float(first_part_f_score)
                stats[f'{method_name}-f-10'] = float(second_part_f_score)

        return stats

    list_of_stats = p_umap(_proc, list(p_preds.iterdir()), num_cpus=3)
        
    results = pd.DataFrame.from_records(list_of_stats).dropna(axis='columns').set_index('video_name')
    return results

def compute_metrics(p_source_masks, p_preds, pred_to_annot_names_lookup=None):
    list_of_stats = []
    # for p_pred_video in list(p_preds.iterdir()):
    def _proc(p_pred_video: Path):
        video_name = p_pred_video.name
        if pred_to_annot_names_lookup is not None:
            video_name = pred_to_annot_names_lookup[video_name]

        # if 'XMem' in str(p_pred_video):
        p_pred_video = Path(p_pred_video) / 'masks'
        p_gts = p_source_masks / video_name
        first_mask = Image.open(next(p_gts.iterdir())).convert('P')
        w, h = first_mask.size
        gts = _load_gt(p_gts)

        preds = _load_preds(p_pred_video, palette=first_mask, size=(w, h))

        assert preds.shape == gts.shape

        avg_iou = batched_jaccard(gts, preds).mean(axis=0)
        avg_f_score = batched_f_measure(gts, preds).mean(axis=0)
        stats = {
            'video_name': video_name,
            'iou': float(avg_iou),
            'f': float(avg_f_score),
        }

        return stats
        # list_of_stats.append(stats)
    # p_source_masks = Path('/home/maksym/RESEARCH/Datasets/MOSE/train/Annotations')
    # p_preds = Path('/home/maksym/RESEARCH/VIDEOS/RESULTS/XMem_memory/MOSE/AL_comparison')

    list_of_stats = p_umap(_proc, sorted(p_preds.iterdir(), key=lambda x: len(os.listdir(x)), reverse=True), num_cpus=4)
        
    results = pd.DataFrame.from_records(list_of_stats).dropna(axis='columns').set_index('video_name')
    return results

# -------------------------END metrics ------------------------------------------------------

def get_dataset_video_info(p_imgs_general, p_annotations_general, p_out_general, num_annotation_candidates=5):
    videos_info = {}
    
    for p_video in sorted(p_imgs_general.iterdir(), key=lambda x: len(os.listdir(x)), reverse=True):  # longest video first to avoid OOM in the future
        video_name = p_video.name
        p_masks = p_annotations_general / video_name

        videos_info[video_name] = dict(
            num_annotation_candidates=num_annotation_candidates,
            video_frames_path=p_video,
            video_masks_path=p_masks,
            masks_out_path=p_out_general / video_name
        )

    return videos_info



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
