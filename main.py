from inference.run_on_video import run_on_video, predict_annotation_candidates


if __name__ == '__main__':
    # If pytorch cannot download the weights due to an ssl error, uncomment the following lines
    # import ssl
    # ssl._create_default_https_context = ssl._create_unverified_context

    # Example for a fully-labeled video
    video_frames_path = 'example_videos/DAVIS-bmx/frames'
    video_masks_path = 'example_videos/DAVIS-bmx/masks'
    output_masks_path_baseline = 'output/DAVIS-bmx/baseline'
    output_masks_path_5_frames = 'output/DAVIS-bmx/5_frames'

    num_annotation_candidates = 5

    # The following step is not necessary, you as a human can also choose suitable frames and provide annotations
    compute_iou = True
    chosen_annotation_frames = predict_annotation_candidates(video_frames_path, num_candidates=num_annotation_candidates)

    print(f"The following frames were chosen as annotation candidates: {chosen_annotation_frames}")

    stats_first_frame_only = run_on_video(video_frames_path, video_masks_path, output_masks_path_baseline, frames_with_masks=[0], compute_iou=True)
    stats_5_frames = run_on_video(video_frames_path, video_masks_path, output_masks_path_5_frames, frames_with_masks=chosen_annotation_frames, compute_iou=True)

    print(f"Average IoU for the video: {float(stats_first_frame_only['iou'].mean())} (first frame only)")
    print(f"Average IoU for the video: {float(stats_5_frames['iou'].mean())} ({num_annotation_candidates} chosen annotated frames)")

    # Example for a video with only a few annotations present
    video_frames_path = 'example_videos/imbalanced-scenes/frames'
    video_masks_path = 'example_videos/imbalanced-scenes/masks'
    output_masks_path_baseline = 'output/imbalanced-scenes/baseline'
    output_masks_path_3_frames = 'output/imbalanced-scenes/3_frames'

    run_on_video(video_frames_path, video_masks_path, output_masks_path_baseline, frames_with_masks=[0], compute_iou=False)
    run_on_video(video_frames_path, video_masks_path, output_masks_path_3_frames, frames_with_masks=[0, 140, 830], compute_iou=False)
