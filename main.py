import os
import random
from inference.run_on_video import run_on_video, select_k_next_best_annotation_candidates

if __name__ == '__main__':
    # If pytorch cannot download the weights due to an ssl error, uncomment the following lines
    # import ssl
    # ssl._create_default_https_context = ssl._create_unverified_context

    # Run inference on a video file with preselected annotated frames
    video_path = 'example_videos/chair/chair.mp4'
    masks_path = 'example_videos/chair/Annotations'
    output_path = 'output/example_video_chair_from_mp4'
    frames_with_masks = [5, 10, 15]

    run_on_video(video_path, masks_path, output_path, frames_with_masks)

    # Run inference on extracted .jpg frames with preselected annotations
    imgs_path = 'example_videos/caps/JPEGImages'
    masks_path = 'example_videos/caps/Annotations'
    output_path = 'output/example_video_caps'
    frames_with_masks = [0, 14, 33, 43, 66]

    run_on_video(imgs_path, masks_path, output_path, frames_with_masks)

    # Get proposals for the next 3 best annotation candidates using previously predicted masks
    # If you don't have previous predictions, just put `use_previously_predicted_masks=False`, the algorithm will run inference internally
    next_candidates = select_k_next_best_annotation_candidates(imgs_path, masks_path, output_path, previously_chosen_candidates=frames_with_masks, use_previously_predicted_masks=True)
    print("Next candidates for annotations are: ")
    for idx in next_candidates:
        print(f"\tFrame {idx}")
    
    # Run inference on a video with all annotations provided, compute IoU
    imgs_path = 'example_videos/chair/JPEGImages'
    masks_path = 'example_videos/chair/Annotations'
    output_path = 'output/example_video_chair'

    num_frames = len(os.listdir(imgs_path))
    frames_with_masks = random.sample(range(0, num_frames), 3)  # Give 3 random masks as GT annotations

    stats = run_on_video(imgs_path, masks_path, output_path, frames_with_masks, compute_iou=True)  #  stats: pandas DataFrame
    mean_iou = stats[stats['iou'] != -1]['iou'].mean()  # -1 is for GT annotations, we just skip them
    print(f"Average IoU: {mean_iou}")  # Should be 90%+ as a sanity check