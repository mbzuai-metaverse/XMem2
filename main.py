from functools import partial
from inference.frame_selection.frame_selection import calculate_proposals_for_annotations_with_iterative_distance_cycle_MASKS
from run_on_video import run_on_video, predict_annotation_candidates

import re
import os 
def get_numbers_from_filename(filename):
    return re.search(r'\d+', filename).group(0)

if __name__ == '__main__':
    # If pytorch cannot download the weights due to an ssl error, uncomment the following lines
    # import ssl
    # ssl._create_default_https_context = ssl._create_unverified_context

    # # Example for a fully-labeled video
    # video_frames_path = 'example_videos/DAVIS-bmx/frames'
    # video_masks_path = 'example_videos/DAVIS-bmx/masks'
    # output_masks_path_baseline = 'output/DAVIS-bmx/baseline'
    # output_masks_path_5_frames = 'output/DAVIS-bmx/5_frames'

    # num_annotation_candidates = 5
    # The following step is not necessary, you as a human can also choose suitable frames and provide annotations
    # compute_iou = True

    need_run_on_video = 0
    need_frame_estimation = 1

    n_masks = []
    video_name = "caps"
    frame_selector = "human_selector" #"human_selector" #"frame_selector"
    method = "our_method" #"original_xmem" #"our_method" 
    num_annotation_candidates = 10

    # Example for a video with only a few annotations present
    video_frames_path = f'/home/ariana/RESEARCH/Datasets/metaverse/{video_name}/JPEGImages' 
    video_masks_path = f'/home/ariana/RESEARCH/Datasets/metaverse/{video_name}/Annotations/initial' 
    for filename in os.listdir(video_masks_path):
        n_masks += [int(get_numbers_from_filename(filename))]
    n_masks = sorted(n_masks)
    output_masks_path_n_frames = f'output/xmem_memory/paper_results/{video_name}/{method}_{frame_selector}/{len(n_masks)}_frames_1_iter' 
    
    print(f"Masks used are: {len(n_masks)} and ", n_masks)
    print("Output_masks_path_n_frames: ", output_masks_path_n_frames)
    if (need_run_on_video): 
        # # 1. run inference with 1st frame, save to `output_path`
        run_on_video(video_frames_path, video_masks_path, output_masks_path_n_frames, frames_with_masks=n_masks, compute_iou=False, original_memory_mechanism=False if method=="our_method" else True)

    if (need_frame_estimation):      
        alpha = 0.5  
        if (need_frame_estimation and need_run_on_video):  
            import pdb 
            pdb.set_trace()
        
        funct_dict = {
            'INTERNAL_CYCLE_CONSISTENCY_MASKS_MULT_BLEND': partial(calculate_proposals_for_annotations_with_iterative_distance_cycle_MASKS, mult_instead=True, alpha=alpha, gui=False)
        }
        # 2. Run frame selector, give it `output_path/masks` as `masks_out_path`
        chosen_annotation_frames = predict_annotation_candidates(video_frames_path, masks_out_path=output_masks_path_n_frames+"/masks", num_candidates=num_annotation_candidates, candidate_selection_function=funct_dict['INTERNAL_CYCLE_CONSISTENCY_MASKS_MULT_BLEND'])
        print(chosen_annotation_frames)
        with open(output_masks_path_n_frames+"/"+f"frame_selector_alpha_{alpha}.txt", "w") as f:
            f.write(str(chosen_annotation_frames))


    # Run inference again

    # print(f"The following frames were chosen as annotation candidates: {chosen_annotation_frames}")

    # stats_first_frame_only = run_on_video(video_frames_path, video_masks_path, output_masks_path_baseline, frames_with_masks=[0], compute_iou=True)
    # stats_5_frames = run_on_video(video_frames_path, video_masks_path, output_masks_path_5_frames, frames_with_masks=chosen_annotation_frames, compute_iou=True)

    # print(f"Average IoU for the video: {float(stats_first_frame_only['iou'].mean())} (first frame only)")
    # print(f"Average IoU for the video: {float(stats_5_frames['iou'].mean())} ({num_annotation_candidates} chosen annotated frames)")
    # memories = ["our_method", "original_xmem"]
    # videos = [ "shirt"] #["chair", "guitar", "shirt"]
    # # for vid_name in videos: 
    # for method in memories: 
    #     n_masks = []
    #     # Example for a video with only a few annotations present
    #     video_frames_path = '/home/ariana/RESEARCH/Datasets/metaverse/blog/JPEGImages' # f'/home/ariana/RESEARCH/XMem_baseline/output/xmem_memory/paper_results/{vid_name}/JPEGImages'
    #     video_masks_path = '/home/ariana/RESEARCH/Datasets/metaverse/blog/Selected_annotations' #f'/home/ariana/RESEARCH/XMem_baseline/output/xmem_memory/paper_results/{vid_name}/Annotations' #example_videos/imbalanced-scenes/masks'

    #     for filename in os.listdir(video_masks_path):
    #         n_masks += [int(get_numbers_from_filename(filename))]
    #     n_masks = sorted(n_masks)
        
    #     # output_masks_path_baseline = 'output/imbalanced-scenes/baseline'
    #     output_masks_path_n_frames = f'output/xmem_memory/paper_results/blog/{method}/{len(n_masks)}_frames_1_iter' #f'output/xmem_memory/paper_results/{vid_name}/{method}/{len(n_masks)}_frames_1_iter'
        
    #     print(f"Masks used are: {len(n_masks)} and ", n_masks)
    #     print("Output_masks_path_n_frames: ", output_masks_path_n_frames)

    #     # run_on_video(video_frames_path, video_masks_path, output_masks_path_baseline, frames_with_masks=[0], compute_iou=False)
    #     run_on_video(video_frames_path, video_masks_path, output_masks_path_n_frames, frames_with_masks=n_masks, compute_iou=False, original_memory_mechanism=False if method=="our_method" else True)
