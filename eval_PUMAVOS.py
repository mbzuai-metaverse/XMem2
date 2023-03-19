import os
from pathlib import Path
from tqdm import tqdm
from typing import Any, Dict, List, Set, Tuple, Union

from inference.run_experiments import compute_metrics_al, run_inference_with_pre_chosen_frames
from run_on_video import  run_on_video

def run_inference_with_initial_frame(videos_info: Dict[str, Dict], output_path: str, only_methods_subset: Set[str] = None, compute_iou=False, IoU_results_save_path=None, **kwargs):
    if compute_iou:
        assert IoU_results_save_path is not None
        p_iou_dir = Path(IoU_results_save_path)
    
    i = 0
    for video_name in tqdm(videos_info.items(), desc='Running inference with a single frame'):
        chosen_frames = [0]
        output_masks_path = Path(output_path) / video_name[0] / "1st_frame"
        video_frames_path = video_name[1]['video_frames_path']
        video_masks_path = video_name[1]['video_masks_path']

        method = "1st_frame"
        stats = run_on_video(video_frames_path, video_masks_path, output_masks_path,
                frames_with_masks=chosen_frames, compute_iou=compute_iou, print_progress=False, **kwargs)
        
        if compute_iou:
                p_out_curr_video_method = p_iou_dir / video_name
                if not p_out_curr_video_method.exists():
                    p_out_curr_video_method.mkdir(parents=True)
                
                stats.to_csv(p_out_curr_video_method / f'{method}.csv')#f'output/AL_comparison_all_methods/{video_name}_{method}.csv')


if __name__ == '__main__':
    videos_info = {}

    p_img_general = Path('/home/ariana/RESEARCH/Datasets/PUMAVOS/JPEGImages')
    p_masks_general = p_img_general.parent / 'Annotations'
    p_out_general = Path('/home/ariana/RESEARCH/XMem_baseline/output/Thesis/PUMAVOS_XMem_Preload')
    
    for p_video in sorted(p_img_general.iterdir(), key=lambda x: len(os.listdir(x)), reverse=True):  # longest video first to avoid OOM in the future
        video_name = p_video.name
        p_masks = p_masks_general / video_name

        videos_info[video_name] = dict(
            num_annotation_candidates=1,
            video_frames_path=p_video,
            video_masks_path=p_masks,
            masks_out_path=p_out_general / video_name
        )

    # predictors = {}  # KNOWN_ANNOTATION_PREDICTORS
    # # # predictors.update(KNOWN_TARGET_AWARE_ANNOTATION_PREDICTORS)
    # # predictors['FIRST_FRAME_ONLY'] = KNOWN_TARGET_AWARE_ANNOTATION_PREDICTORS['FIRST_FRAME_ONLY']

    # # p_out_path = Path('/home/maksym/RESEACH/VIDEOS/RESULTS/XMem_memory/permanent_work_memory/AL_comparison/OUR_VIDEOS')
    # # run_multiple_frame_selectors(videos_info, 'output/al_chosen_frames_PUMaVOS.csv', predictors)
    # # run_inference_with_pre_chosen_frames('output/al_chosen_frames_PUMaVOS.csv', videos_info, output_path=p_out_general, only_methods_subset=set(predictors.keys()), compute_iou=False)

    # predictors = {}  # KNOWN_ANNOTATION_PREDICTORS
    # predictors.update(KNOWN_TARGET_AWARE_ANNOTATION_PREDICTORS)

    # run_multiple_frame_selectors(videos_info, 'output/al_chosen_frames_PUMaVOS.csv', predictors)
    run_inference_with_initial_frame(videos_info, output_path=p_out_general, compute_iou=False)

    p_out_metrics_general = Path('output/metrics/FACS')
    p_source_masks = p_img_general.parent / 'Annotations'
    p_preds = p_out_general

    df = compute_metrics_al(p_source_masks, p_preds)
    df.to_csv(p_out_metrics_general / f'PUMaVOS_3_videos.csv')