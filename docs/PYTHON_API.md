# XMem++ Python API

XMem++ exposes 2 main functions you can use:
- `run_on_video` - run full inference on a video/images folder with given annotations.
- `select_k_next_best_annotation_candidates` - given a video/images folder, at least one ground truth annotation (to know which object we are even segmenting) and existing predictions [optional], select $k$ next best 

See also [main.py](../main.py).

## Inference with preselected ground truth annotations
### Using list of video frames
To run segmentation on a list of video frames (`.jpg`) with preselected annotations:
```Python
from inference.run_on_video import run_on_video
imgs_path = 'example_videos/caps/JPEGImages'
masks_path = 'example_videos/caps/Annotations'   # Should contain annotation masks for frames in `frames_with_masks`
output_path = 'output/example_video_caps'
frames_with_masks = [0, 14, 33, 43, 66]  # indices of frames for which there is an annotation mask
run_on_video(imgs_path, masks_path, output_path, frames_with_masks)
```
### Using a video file
To run segmentation on a video file (like `.mp4`) with preselected annotations:
```Python
from inference.run_on_video import run_on_video
video_path = 'example_videos/chair/chair.mp4'
masks_path = 'example_videos/chair/Annotations'  # Should contain annotation masks for rames in `frames_with_masks`
output_path = 'output/example_video_chair_from_mp4'
frames_with_masks = [5, 10, 15]  # indices of frames for which there is an annotation mask
run_on_video(video_path, masks_path, output_path, frames_with_masks)
```
## Getting next best frames to annotate
If after this you want to get proposals which frames to annotate next, add the following lines:
```Python
from inference.run_on_video import select_k_next_best_annotation_candidates
# Get proposals for the next 3 best annotation candidates using previously predicted masks
next_candidates = select_k_next_best_annotation_candidates(imgs_path, masks_path, output_path, previously_chosen_candidates=frames_with_masks, use_previously_predicted_masks=True)
print("Next candidates for annotations are: ")
for idx in next_candidates:
    print(f"\tFrame {idx}")
```
If you don't have previous predictions, just put `use_previously_predicted_masks=False`, the algorithm will run a new inference internally.

## Evaluating on a video with all ground truth masks available
If you have a fully-labeled video and want to run **XMem++** and compute IoU, run the following code:
```Python
# Run inference on a video with all annotations provided, compute IoU
import os
import random
from inference.run_on_video import run_on_video
imgs_path = 'example_videos/chair/JPEGImages'
masks_path = 'example_videos/chair/Annotations'
output_path = 'output/example_video_chair'
num_frames = len(os.listdir(imgs_path))
frames_with_masks = random.sample(range(0, num_frames), 3)  # Give 3 random masks as GT annotations
stats = run_on_video(imgs_path, masks_path, output_path, frames_with_masks, compute_iou=True)  #  stats: pandas DataFrame
mean_iou = stats[stats['iou'] != -1]['iou'].mean()  # -1 is for GT annotations, we just skip them
print(f"Average IoU: {mean_iou}")  # Should be 90%+ as a sanity check
```