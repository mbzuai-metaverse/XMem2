from dataclasses import replace
from functools import partial
from multiprocessing import Process, Queue
from os import PathLike, path
from tempfile import TemporaryDirectory
from time import perf_counter
import time
from typing import Iterable, Optional, Union, List
from pathlib import Path
from warnings import warn

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as FT, ToTensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from inference.frame_selection.frame_selection import select_next_candidates
from model.network import XMem
from util.configuration import VIDEO_INFERENCE_CONFIG
from util.image_saver import ParallelImageSaver, create_overlay, save_image
from util.tensor_util import compute_array_iou
from inference.inference_core import InferenceCore
from inference.data.video_reader import Sample, VideoReader
from inference.data.mask_mapper import MaskMapper
from inference.frame_selection.frame_selection_utils import extract_keys, get_determenistic_augmentations

def _inference_on_video(frames_with_masks, imgs_in_path, masks_in_path, masks_out_path,
                        original_memory_mechanism=False,
                        compute_iou=False, 
                        manually_curated_masks=False, 
                        print_progress=True,
                        augment_images_with_masks=False,
                        overwrite_config: dict = None,
                        save_overlay=True,
                        object_color_if_single_object=(255, 255, 255), 
                        print_fps=False,
                        image_saving_max_queue_size=200):
    
    torch.autograd.set_grad_enabled(False)
    frames_with_masks = set(frames_with_masks)

    config = VIDEO_INFERENCE_CONFIG.copy()
    overwrite_config = {} if overwrite_config is None else overwrite_config
    overwrite_config['masks_out_path'] = masks_out_path
    config.update(overwrite_config)

    mapper, processor, vid_reader, loader = _load_main_objects(imgs_in_path, masks_in_path, config)
    vid_name = vid_reader.vid_name
    vid_length = len(loader)

    at_least_one_mask_loaded = False
    total_preloading_time = 0.0

    if original_memory_mechanism:
        # only the first frame goes into permanent memory originally
        frames_to_put_in_permanent_memory = [0]
        # the rest are going to be processed later
    else:
        # in our modification, all frames with provided masks go into permanent memory
        frames_to_put_in_permanent_memory = frames_with_masks
    at_least_one_mask_loaded, total_preloading_time = _preload_permanent_memory(frames_to_put_in_permanent_memory, vid_reader, mapper, processor, augment_images_with_masks=augment_images_with_masks)

    if not at_least_one_mask_loaded:
        raise ValueError("No valid masks provided!")

    stats = []

    total_processing_time = 0.0
    with ParallelImageSaver(config['masks_out_path'], vid_name=vid_name, overlay_color_if_b_and_w=object_color_if_single_object, max_queue_size=image_saving_max_queue_size) as im_saver:
        for ti, data in enumerate(tqdm(loader, disable=not print_progress)):
            with torch.cuda.amp.autocast(enabled=True):
                data: Sample = data  # Just for Intellisense
                # No batch dimension here, just single samples
                sample = replace(data, rgb=data.rgb.cuda())
                
                if ti in frames_with_masks:
                    msk = sample.mask
                else:
                    msk = None
                    
                # Map possibly non-continuous labels to continuous ones
                if msk is not None:
                    # https://github.com/hkchengrex/XMem/issues/21 just make exhaustive = True
                    msk, labels = mapper.convert_mask(
                        msk.numpy(), exhaustive=True)
                    msk = torch.Tensor(msk).cuda()
                    if sample.need_resize:
                        msk = vid_reader.resize_mask(msk.unsqueeze(0))[0]
                    processor.set_all_labels(list(mapper.remappings.values()))
                else:
                    labels = None

                if original_memory_mechanism:
                    # we only ignore the first mask, since it's already in the permanent memory
                    do_not_add_mask_to_memory = (ti == 0)
                else:
                    # we ignore all frames with masks, since they are already preloaded in the permanent memory
                    do_not_add_mask_to_memory = msk is not None
                # Run the model on this frame
                # 2+ channels, classes+ and background
                a = perf_counter()
                prob = processor.step(sample.rgb, msk, labels, end=(ti == vid_length-1),
                                    manually_curated_masks=manually_curated_masks, do_not_add_mask_to_memory=do_not_add_mask_to_memory)

                # Upsample to original size if needed
                out_mask = _post_process(sample, prob)
                b = perf_counter()
                total_processing_time += (b - a)

                curr_stat = {'frame': sample.frame, 'mask_provided': msk is not None}
                if compute_iou:
                    gt = sample.mask  # for IoU computations, original mask or None, NOT msk
                    if gt is not None and msk is None:  # There exists a ground truth, but the model didn't see it
                        iou = float(compute_array_iou(out_mask, gt))
                    else:
                        iou = -1  # skipping frames where the model saw the GT
                    curr_stat['iou'] = iou
                stats.append(curr_stat)

                # Save the mask and the overlay (potentially)

                if config['save_masks']:
                    out_mask = mapper.remap_index_mask(out_mask)
                    out_img = Image.fromarray(out_mask)
                    out_img = vid_reader.map_the_colors_back(out_img)

                    im_saver.save_mask(mask=out_img, frame_name=sample.frame)

                    if save_overlay:
                        original_img = sample.raw_image_pil
                        im_saver.save_overlay(orig_img=original_img, mask=out_img, frame_name=sample.frame)
        im_saver.wait_for_jobs_to_finish(verbose=True)

    if print_fps:
        print(f"TOTAL PRELOADING TIME: {total_preloading_time:.4f}s")
        print(f"TOTAL PROCESSING TIME: {total_processing_time:.4f}s")
        print(f"TOTAL TIME (excluding image saving): {total_preloading_time + total_processing_time:.4f}s")
        print(f"TOTAL PROCESSING FPS: {len(loader) / total_processing_time:.4f}")
        print(f"TOTAL FPS (excluding image saving): {len(loader) / (total_preloading_time + total_processing_time):.4f}")

    return pd.DataFrame(stats)

def _load_main_objects(imgs_in_path, masks_in_path, config):
    model_path = config['model']
    network = XMem(config, model_path, pretrained_key_encoder=False, pretrained_value_encoder=False).cuda().eval()
    if model_path is not None:
        model_weights = torch.load(model_path)
        network.load_weights(model_weights, init_as_zero_if_needed=True)
    else:
        warn('No model weights were loaded, as config["model"] was not specified.')

    mapper = MaskMapper()
    processor = InferenceCore(network, config=config)

    vid_reader, loader = _create_dataloaders(imgs_in_path, masks_in_path, config)
    return mapper,processor,vid_reader,loader


def _post_process(sample, prob):
    if sample.need_resize:
        prob = F.interpolate(prob.unsqueeze(
                    1), sample.shape, mode='bilinear', align_corners=False)[:, 0]

    # Probability mask -> index mask
    out_mask = torch.argmax(prob, dim=0)
    out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)
    return out_mask


def _create_dataloaders(imgs_in_path: Union[str, PathLike], masks_in_path: Union[str, PathLike], config: dict):
    vid_reader = VideoReader(
        "",
        imgs_in_path,  # f'/home/maksym/RESEARCH/VIDEOS/thanks_no_ears_5_annot/JPEGImages',
        masks_in_path,  # f'/home/maksym/RESEARCH/VIDEOS/thanks_no_ears_5_annot/Annotations_binarized_two_face',
        size=config['size'],
        use_all_masks=True
    )
    
    # Just return the samples as they are; only using DataLoader for preloading frames from the disk
    loader = DataLoader(vid_reader, batch_size=None, shuffle=False, num_workers=1, collate_fn=VideoReader.collate_fn_identity)

    vid_length = len(loader)
    # no need to count usage for LT if the video is not that long anyway
    config['enable_long_term_count_usage'] = (
        config['enable_long_term'] and
        (vid_length
            / (config['max_mid_term_frames']-config['min_mid_term_frames'])
            * config['num_prototypes'])
        >= config['max_long_term_elements']
    )
    
    return vid_reader,loader


def _preload_permanent_memory(frames_to_put_in_permanent_memory: List[int], vid_reader: VideoReader, mapper: MaskMapper, processor: InferenceCore, augment_images_with_masks=False):
    total_preloading_time = 0
    at_least_one_mask_loaded = False
    for j in frames_to_put_in_permanent_memory:
        sample: Sample = vid_reader[j]
        sample = replace(sample, rgb=sample.rgb.cuda())

        # https://github.com/hkchengrex/XMem/issues/21 just make exhaustive = True
        if sample.mask is None:
            raise FileNotFoundError(f"Couldn't find mask {j}! Check that the filename is either the same as for frame {j} or follows the `frame_%06d.png` format if using a video file for input.")
        msk, labels = mapper.convert_mask(sample.mask, exhaustive=True)
        msk = torch.Tensor(msk).cuda()

        if min(msk.shape) == 0:  # empty mask, e.g. [1, 0, 720, 1280]
            warn(f"Skipping adding frame {j} to permanent memory, as the mask is empty")
            continue  # just don't add anything to the memory
        if sample.need_resize:
            msk = vid_reader.resize_mask(msk.unsqueeze(0))[0]
        # sample = replace(sample, mask=msk)

        processor.set_all_labels(list(mapper.remappings.values()))
        a = perf_counter()
        processor.put_to_permanent_memory(sample.rgb, msk)
        b = perf_counter()
        total_preloading_time += (b - a)

        if not at_least_one_mask_loaded:
            at_least_one_mask_loaded = True

        if augment_images_with_masks:
            augs = get_determenistic_augmentations(
                sample.rgb.shape, msk, subset='best_all')
            rgb_raw = sample.raw_image_pil

            for img_aug, mask_aug in augs:
                # tensor -> PIL.Image -> tensor -> whatever normalization vid_reader applies
                rgb_aug = vid_reader.im_transform(img_aug(rgb_raw)).cuda()

                msk_aug = mask_aug(msk)

                processor.put_to_permanent_memory(rgb_aug, msk_aug)
    
    return at_least_one_mask_loaded, total_preloading_time


def run_on_video(
    imgs_in_path: Union[str, PathLike],
    masks_in_path: Union[str, PathLike],
    masks_out_path: Union[str, PathLike],
    frames_with_masks: Iterable[int] = (0, ),
    compute_iou=False,
    print_progress=True,
    **kwargs
) -> pd.DataFrame:
    """
    Args:
    imgs_in_path (Union[str, PathLike]): Path to the directory containing video frames in the following format: `frame_000000.png`. .jpg works too.

    masks_in_path (Union[str, PathLike]): Path to the directory containing video frames' masks in the same format, with corresponding names between video frames. Each unique object should have unique color.

    masks_out_path (Union[str, PathLike]): Path to the output directory (will be created if doesn't exist) where the predicted masks will be stored in .png format.

    frames_with_masks (Iterable[int]): A list of integers representing the frames on which the masks should be applied (default: [0], only applied to the first frame). 0-based.

    compute_iou (bool): A flag to indicate whether to compute the IoU metric (default: False, requires ALL video frames to have a corresponding mask).

    print_progress (bool): A flag to indicate whether to print a progress bar (default: True).

    Returns:
    stats (pd.Dataframe): a table containing every frame and the following information: IoU score with corresponding mask (if `compute_iou` is True)
    """

    return _inference_on_video(
        imgs_in_path=imgs_in_path,
        masks_in_path=masks_in_path,
        masks_out_path=masks_out_path,
        frames_with_masks=frames_with_masks,
        compute_iou=compute_iou,
        print_progress=print_progress,
         **kwargs
    )


def select_k_next_best_annotation_candidates(
    imgs_in_path: Union[str, PathLike],
    masks_in_path: Union[str, PathLike],  # at least the 1st frame
    masks_out_path: Optional[Union[str, PathLike]] = None,
    k: int = 5,
    print_progress=True,
    previously_chosen_candidates=[0],
    use_previously_predicted_masks=True,
    # Candidate selection hyperparameters
    alpha=0.5,
    min_mask_presence_percent=0.25,
    **kwargs
):
    """
    Selects the next best annotation candidate frames based on the provided frames and mask paths.

    Parameters:
        imgs_in_path (Union[str, PathLike]): The path to the directory containing input images.
        masks_in_path (Union[str, PathLike]): The path to the directory containing the first frame masks.
        masks_out_path (Optional[Union[str, PathLike]], optional): The path to save the generated masks.
            If not provided, a temporary directory will be used. Defaults to None.
        k (int, optional): The number of next best annotation candidate frames to select. Defaults to 5.
        print_progress (bool, optional): Whether to print progress during processing. Defaults to True.
        previously_chosen_candidates (list, optional): List of indices of frames with previously chosen candidates.
            Defaults to [0].
        use_previously_predicted_masks (bool, optional): Whether to use previously predicted masks.
            If True, `masks_out_path` must be provided. Defaults to True.
        alpha (float, optional): Hyperparameter controlling the candidate selection process. Defaults to 0.5.
        min_mask_presence_percent (float, optional): Minimum mask presence percentage for candidate selection.
            Defaults to 0.25.
        **kwargs: Additional keyword arguments to pass to `run_on_video`.

    Returns:
        list: A list of indices representing the selected next best annotation candidate frames.
    """
    mapper, processor, vid_reader, loader = _load_main_objects(imgs_in_path, masks_in_path, VIDEO_INFERENCE_CONFIG)

    # Extracting "key" feature maps
    # Could be combined with inference (like in GUI), but the code would be a mess
    frame_keys, shrinkages, selections, *_ = extract_keys(loader, processor, print_progress=print_progress, flatten=False)
    # extracting the keys and corresponding matrices 

    to_tensor = ToTensor()
    if masks_out_path is not None:
        p_masks_out = Path(masks_out_path)

    if use_previously_predicted_masks:
        print("Using existing predicted masks, no need to run inference.")
        assert masks_out_path is not None, "When `use_existing_masks=True`, you need to put the path to previously predicted masks in `masks_out_path`"
        try:
            masks = [to_tensor(Image.open(p)) for p in sorted((p_masks_out / 'masks').iterdir())]
        except Exception as e:
            warn("Loading previously predicting masks failed for `select_k_next_best_annotation_candidates`.")
            raise e
        if len(masks) != len(frame_keys):
            raise FileNotFoundError(f"Not enough masks ({len(masks)}) for {len(frame_keys)} frames provided when using `use_previously_predicted_masks=True`!")
    else:
        print("Existing predictions were not given, will run full inference and save masks in `masks_out_path` or a temporary directory if `masks_out_path` is not given.")
        if masks_out_path is None:
            d = TemporaryDirectory()
            p_masks_out = Path(d)
        
        # running inference once to obtain masks
        run_on_video(
            imgs_in_path=imgs_in_path,
            masks_in_path=masks_in_path,  # Ignored
            masks_out_path=p_masks_out,  # Used for some frame selectors
            frames_with_masks=previously_chosen_candidates,
            compute_iou=False,
            print_progress=print_progress,
            **kwargs
        )

        masks = [to_tensor(Image.open(p)) for p in sorted((p_masks_out / 'masks').iterdir())]

    keys = torch.cat(frame_keys)
    shrinkages = torch.cat(shrinkages)
    selections = torch.cat(selections)

    new_selected_candidates = select_next_candidates(keys, shrinkages=shrinkages, selections=selections, masks=masks, num_next_candidates=k, previously_chosen_candidates=previously_chosen_candidates, print_progress=print_progress, alpha=alpha, only_new_candidates=True, min_mask_presence_percent=min_mask_presence_percent)
        
    if masks_out_path is None:
        # Remove the temporary directory
        d.cleanup()

    return new_selected_candidates