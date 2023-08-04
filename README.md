# XMem++

## Production-level Video Segmentation From Few Annotated Frames

[Maksym Bekuzarov](https://www.linkedin.com/in/maksym-bekuzarov-947490165/)$^\dagger$, [Ariana Michelle Bermudez Venegas](https://www.linkedin.com/in/ariana-bermudez/)$^\dagger$, [Joon-Young Lee](https://joonyoung-cv.github.io/), [Hao Li](https://www.hao-li.com/Hao_Li/Hao_Li_-_about_me.html)

[Metaverse Lab TODO LINK]() @ [MBZUAI](https://mbzuai.ac.ae/) (Mohamed bin Zayed University of Artificial Intelligence)

[[arXiv]](https://arxiv.org/abs/2307.15958) [[PDF]](https://arxiv.org/pdf/2307.15958.pdf) [[Project Page TODO]](https://hkchengrex.github.io/XMem/)

$^\dagger$ These authors equally contributed to the work
## Demo

Inspired by movie industry use cases, **XMem++** is an Interactive Video Segmentation Tool that takes a few user-provided segmentation masks and segments very challenging use cases with minimal human supervision, such as 

- **parts** of the objects (only 6 annotated frames provided):

https://github.com/max810/XMem2/assets/29955120/d700ccc4-194e-46d8-97b2-05b0587496f4

- **fluid** objects like hair (only 5 annotated frames provided):

https://github.com/max810/XMem2/assets/29955120/06d4d8ee-3092-4fe6-a0c2-da7e3fc4a01c

- **deformable** objects like clothes (5 and 11 annotated frames used accordingly)

https://github.com/max810/XMem2/assets/29955120/a8e75648-b8cf-4312-8077-276597256289

https://github.com/max810/XMem2/assets/29955120/63e6704c-3292-4690-970e-818ab2950c56

### [[Failure Cases]](docs/FAILURE_CASES.md)

## Overview
**XMem++** is built on top of [XMem](https://github.com/hkchengrex/XMem) by [Ho Kei Cheng](https://hkchengrex.github.io/), [Alexander Schwing](https://www.alexander-schwing.de/) and improves upon it by adding the following:
1. [Permanent memory module TODO link]() that greatly improves the model's accuracy with just a few manual annotations provided (see results)
2. [Annotation candidate selection algorithm]() that selects $k$ next best frames for the user to provide annotations for.
3. We used **XMem++** to collect and annotate **PUMaVOS** - 23 video dataset with unusual and challenging annotation scenarios at 480p, 30FPS. See [Dataset](#dataset)

In addition to the following features:
* Improved GUI - references tab to see/edit what frames are in the permanent memory, candidates tab - shows candidate frames for annotation predicted by the algorithm and more.
* Negligible speed and memory usage overhead compared to XMem (if using few manually provided annotations)
* [Separate cmd script for doing segmentation on video](inference/run_on_video.py) - now you can use both GUI and Python interface easily.
* 30+ FPS on 480p footage on RTX 3090
* Come with a GUI (modified from [MiVOS](https://github.com/hkchengrex/MiVOS/tree/MiVOS-STCN)).

_We use the original weights provided by XMem, the model has not been retrained or fine-tuned in any way._

### How to use
#### Install the environment
First, install the required Python packages:

TODO conda install whatever

#### Use the GUI
To run the GUI:
```
pass
```

#### Use **XMem++** Python interface
The following examples of the Python interface usage are taken from [main.py](main.py). 

To run segmentation on a list of video frames (`.jpg`) with preselected annotations:
```Python
from inference.run_on_video import run_on_video

imgs_path = 'example_videos/caps/JPEGImages'
masks_path = 'example_videos/caps/Annotations'   # Should contain annotation masks for frames in `frames_with_masks`
output_path = 'output/example_video_caps'
frames_with_masks = [0, 14, 33, 43, 66]  # indices of frames for which there is an annotation mask

run_on_video(imgs_path, masks_path, output_path, frames_with_masks)
```

To run segmentation on a video file (like `.mp4`) with preselected annotations:
```Python
from inference.run_on_video import run_on_video

video_path = 'example_videos/chair/chair.mp4'
masks_path = 'example_videos/chair/Annotations'  # Should contain annotation masks for rames in `frames_with_masks`
output_path = 'output/example_video_chair_from_mp4'
frames_with_masks = [5, 10, 15]  # indices of frames for which there is an annotation mask

run_on_video(video_path, masks_path, output_path, frames_with_masks)
```

If after this you want to get proposals which frames to annotate next, add the following lines:
```Python
from inference.run_on_video import select_k_next_best_annotation_candidates
# Get proposals for the next 3 best annotation candidates using previously predicted masks
next_candidates = select_k_next_best_annotation_candidates(imgs_path, masks_path, output_path, previously_chosen_candidates=frames_with_masks, use_previously_predicted_masks=True)

print("Next candidates for annotations are: ")
for idx in next_candidates:
    print(f"\tFrame {idx}")
```

If you don't have previous predictions, just put `use_previously_predicted_masks=False`, the algorithm will run a new inference internally

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

### Training
For training, refer to the [original XMem repo](https://github.com/hkchengrex/XMem/blob/main/docs/TRAINING.md) (since we are using the same model with the same weights)

### Methodology

![xmem_pp_worflow_explanation](https://github.com/max810/XMem2/assets/29955120/b97f8d7b-8c59-48fd-8122-834ba04c1696)

Just like XMem (VOS), we use the two types of memory inspired by the Atkinson-Shiffrin human memory model - working memory and long-term memory. The first one stores recent convolutional feature maps with rich details, and the other - heavily compressed features for long-term dependency.

However, existing segmentation methods (XMem, TBD, AoT, DeAOT, STCN, etc.) that are using memory mechanisms to predict the segmentation mask for the current frame, typically process frames one by one, and thus suffer from a common issue - "jumps" in visual quality, when the new ground truth annotation is encountered in the video

![interpolation](https://github.com/max810/XMem2/assets/29955120/456d907e-881c-4a07-9aed-808f855e209a)

To solve this, we propose a new **permanent memory module** - same in implementations as XMem's working memory - we take all the annotations the user provides, process them and put in the permanent memory module. This way **every** ground truth annonation provided by the user can influence **any** frame in the video regardless where it is located. This increases overall segmentation accuracy and allows the model to smoothly interpolate between different appearences of the object (see figure above).

### Frame annotation candidate selector

We propose a simple algorithm to select which frames the user should annotate next to maximize performance and save time. It is based on an idea of **diversity** - to select the frames that capture the most variety of the target object's appearance - to **maximize the information** the network will get with them annotated.

It has the following properties:
- **Target-specific**: Choice of frames depends on which object you are trying to segment. 

![frame_selector_2](https://github.com/max810/XMem2/assets/29955120/f040d99b-4680-4e6f-be89-b05c1e70c4a7)

- **Model-generic**: it operates on convolutional feature maps and pixel-similarity metric (negative $\mathcal{L}_{2}$ distance), so is not specific to **XMem++**.
- **No restrictions on segmentation targets**: Some methods try to automatically estimate the visual quality of the segmentation, which puts an implicit assumption **that good-quality segmentation follows low-level image ques (edges, corners, etc.)**. However, this is not true when segmenting parts of objects, see the :

![broken_quality](https://github.com/max810/XMem2/assets/29955120/2c1246b6-e2da-4cfe-a011-8c646635493b)

- **Deterministic and simple**: It orders remaining frames by a **diversity measure** and the user just picks the top-$k$ most diverse candidates.

### Dataset



### Citation

Contact: <maksym.bekuzarov@gmail.com>, <bermudezarii@gmail.com>
