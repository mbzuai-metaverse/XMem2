import os
from os import PathLike, path
from typing import Iterable, Literal, Union, List
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as FT
from torch.utils.data import DataLoader
from baal.active.heuristics import BALD
from scipy.stats import entropy
from tqdm import tqdm
from PIL import Image

from model.network import XMem
from util.image_saver import create_overlay, save_image
from util.tensor_util import compute_tensor_iou
from inference.inference_core import InferenceCore
from inference.data.video_reader import VideoReader
from inference.data.mask_mapper import MaskMapper
from inference.frame_selection.frame_selection import KNOWN_ANNOTATION_PREDICTORS
from inference.frame_selection.frame_selection_utils import disparity_func, get_determenistic_augmentations


def save_frames(dataset, frame_indices, output_folder):
    p_out = Path(output_folder)

    if not p_out.exists():
        p_out.mkdir(parents=True)

    for i in frame_indices:
        sample = dataset[i]
        rgb_raw_tensor = sample['raw_image_tensor'].cpu().squeeze()
        img = FT.to_pil_image(rgb_raw_tensor)

        img.save(p_out / f'frame_{i:06d}.png')


def _inference_on_video(frames_with_masks, imgs_in_path, masks_in_path, masks_out_path,
                        original_memory_mechanism=False,
                        compute_iou=False, compute_uncertainty=False, manually_curated_masks=False, print_progress=True,
                        augment_images_with_masks=False,
                        uncertainty_name: str = None,
                        only_predict_frames_to_annotate_and_quit=0,
                        overwrite_config: dict = None,
                        frame_selector_func: callable = None,
                        save_overlay=True,
                        b_and_w_color=(255, 0, 0)):
    torch.autograd.set_grad_enabled(False)
    frames_with_masks = set(frames_with_masks)
    config = {
        'buffer_size': 100,
        'deep_update_every': -1,
        'enable_long_term': True,
        'enable_long_term_count_usage': True,
        'fbrs_model': 'saves/fbrs.pth',
        'hidden_dim': 64,
        'images': None,
        'key_dim': 64,
        'max_long_term_elements': 10000,
        'max_mid_term_frames': 10,
        'mem_every': 10,
        'min_mid_term_frames': 5,
        'model': './saves/XMem.pth',
        'no_amp': False,
        'num_objects': 1,
        'num_prototypes': 128,
        's2m_model': 'saves/s2m.pth',
        'size': 480,
        'top_k': 30,

        'key_dim_f16': 64,
        'key_dim_f8': 32,
        'key_dim_f4': 16,

        'value_dim_f16': 512,
        'value_dim_f8': 128,
        'value_dim_f4': 32,

        # f'../VIDEOS/RESULTS/XMem_memory/thanks_two_face_5_frames/',
        'masks_out_path': masks_out_path,
        'workspace': None,
        'save_masks': True
    }

    if overwrite_config is not None:
        config.update(overwrite_config)

    if compute_uncertainty:
        assert uncertainty_name is not None
        uncertainty_name = uncertainty_name.lower()
        assert uncertainty_name in {'entropy',
                                    'bald', 'disparity', 'disparity_large'}
        compute_disparity = uncertainty_name.startswith('disparity')
    else:
        compute_disparity = False

    vid_reader = VideoReader(
        "",
        imgs_in_path,  # f'/home/maksym/RESEARCH/VIDEOS/thanks_no_ears_5_annot/JPEGImages',
        masks_in_path,  # f'/home/maksym/RESEARCH/VIDEOS/thanks_no_ears_5_annot/Annotations_binarized_two_face',
        size=config['size'],
        use_all_masks=(only_predict_frames_to_annotate_and_quit == 0)
    )

    model_path = config['model']
    network = XMem(config, model_path).cuda().eval()
    if model_path is not None:
        model_weights = torch.load(model_path)
        network.load_weights(model_weights, init_as_zero_if_needed=True)
    else:
        print('No model loaded.')

    loader = DataLoader(vid_reader, batch_size=1, shuffle=False, num_workers=8)
    vid_name = vid_reader.vid_name
    vid_length = len(loader)
    # no need to count usage for LT if the video is not that long anyway
    config['enable_long_term_count_usage'] = (
        config['enable_long_term'] and
        (vid_length
            / (config['max_mid_term_frames']-config['min_mid_term_frames'])
            * config['num_prototypes'])
        >= config['max_long_term_elements']
    )

    mapper = MaskMapper()
    processor = InferenceCore(network, config=config)
    first_mask_loaded = False

    if only_predict_frames_to_annotate_and_quit > 0:
        assert frame_selector_func is not None
        chosen_annotation_candidate_frames = frame_selector_func(
            loader, processor, print_progress=print_progress, how_many_frames=only_predict_frames_to_annotate_and_quit)

        return chosen_annotation_candidate_frames

    frames_ = []
    masks_ = []

    if original_memory_mechanism:
        # only the first frame goes into permanent memory originally
        frames_to_put_in_permanent_memory = [0]
        # the rest are going to be processed later
    else:
        # in our modification, all frames with provided masks go into permanent memory
        frames_to_put_in_permanent_memory = frames_with_masks
    for j in frames_to_put_in_permanent_memory:
        sample = vid_reader[j]
        rgb = sample['rgb'].cuda()
        rgb_raw_tensor = sample['raw_image_tensor'].cpu()
        msk = sample['mask']
        info = sample['info']
        need_resize = info['need_resize']

        # https://github.com/hkchengrex/XMem/issues/21 just make exhaustive = True
        msk, labels = mapper.convert_mask(msk, exhaustive=True)
        msk = torch.Tensor(msk).cuda()

        if min(msk.shape) == 0:  # empty mask, e.g. [1, 0, 720, 1280]
            print(f"Skipping adding frame {j} to memory, as the mask is empty")
            continue  # just don't add anything to the memory
        if need_resize:
            msk = vid_reader.resize_mask(msk.unsqueeze(0))[0]

        processor.set_all_labels(list(mapper.remappings.values()))
        processor.put_to_permanent_memory(rgb, msk)

        if not first_mask_loaded:
            first_mask_loaded = True

        frames_.append(rgb)
        masks_.append(msk)

        if augment_images_with_masks:
            augs = get_determenistic_augmentations(
                rgb.shape, msk, subset='best_all')
            rgb_raw = FT.to_pil_image(rgb_raw_tensor)

            for img_aug, mask_aug in augs:
                # tensor -> PIL.Image -> tensor -> whatever normalization vid_reader applies
                rgb_aug = vid_reader.im_transform(img_aug(rgb_raw)).cuda()

                msk_aug = mask_aug(msk)

                processor.put_to_permanent_memory(rgb_aug, msk_aug)

    if not first_mask_loaded:
        raise ValueError("No valid masks provided!")

    stats = []

    if compute_uncertainty and uncertainty_name == 'bald':
        bald = BALD()

    for ti, data in enumerate(tqdm(loader, disable=not print_progress)):
        with torch.cuda.amp.autocast(enabled=True):
            rgb = data['rgb'].cuda()[0]
            rgb_raw_tensor = data['raw_image_tensor'].cpu()[0]

            gt = data.get('mask')  # for IoU computations
            if ti in frames_with_masks:
                msk = data['mask']
            else:
                msk = None

            info = data['info']
            frame = info['frame'][0]
            shape = info['shape']
            need_resize = info['need_resize'][0]
            curr_stat = {'frame': frame, 'mask_provided': msk is not None}

            # not important anymore as long as at least one mask is in permanent memory
            if original_memory_mechanism and not first_mask_loaded:
                if msk is not None:
                    first_mask_loaded = True
                else:
                    # no point to do anything without a mask
                    continue

            # Map possibly non-continuous labels to continuous ones
            if msk is not None:
                # https://github.com/hkchengrex/XMem/issues/21 just make exhaustive = True
                msk, labels = mapper.convert_mask(
                    msk[0].numpy(), exhaustive=True)
                msk = torch.Tensor(msk).cuda()
                if need_resize:
                    msk = vid_reader.resize_mask(msk.unsqueeze(0))[0]
                processor.set_all_labels(list(mapper.remappings.values()))

            else:
                labels = None

            if (compute_uncertainty and uncertainty_name == 'bald') or compute_disparity:
                dry_run_preds = []
                augged_images = []
                augs = get_determenistic_augmentations(subset='original_only')
                rgb_raw = FT.to_pil_image(rgb_raw_tensor)
                for img_aug, mask_aug in augs:
                    # tensor -> PIL.Image -> tensor -> whatever normalization vid_reader applies
                    augged_img = img_aug(rgb_raw)
                    augged_images.append(augged_img)
                    rgb_aug = vid_reader.im_transform(augged_img).cuda()

                    # does not do anything, since original_only=True augmentations don't alter the mask at all
                    msk = mask_aug(msk)

                    dry_run_prob = processor.step(rgb_aug, msk, labels, end=(ti == vid_length-1),
                                                  manually_curated_masks=manually_curated_masks, disable_memory_updates=True)
                    dry_run_preds.append(dry_run_prob.cpu())

            if original_memory_mechanism:
                # we only ignore the first mask, since it's already in the permanent memory
                do_not_add_mask_to_memory = (ti == 0)
            else:
                # we ignore all frames with masks, since they are already preloaded in the permanent memory
                do_not_add_mask_to_memory = msk is not None
            # Run the model on this frame
            # 2+ channels, classes+ and background
            prob = processor.step(rgb, msk, labels, end=(ti == vid_length-1),
                                  manually_curated_masks=manually_curated_masks, do_not_add_mask_to_memory=do_not_add_mask_to_memory)

            if compute_uncertainty:
                if uncertainty_name == 'bald':
                    # [batch=1, num_classes, ..., num_iterations]
                    all_samples = torch.stack(
                        [x.unsqueeze(0) for x in dry_run_preds + [prob.cpu()]], dim=-1).numpy()
                    score = bald.compute_score(all_samples)
                    curr_stat['bald'] = float(np.squeeze(score).mean())
                elif compute_disparity:
                    disparity_stats = disparity_func(
                        predictions=[prob] + dry_run_preds, augs=[img_aug for img_aug, _ in augs], images=[rgb_raw] + augged_images, output_save_path=None)
                    curr_stat['disparity'] = float(disparity_stats['avg'])
                    curr_stat['disparity_large'] = float(
                        disparity_stats['large'])
                else:
                    e = entropy(prob.cpu())
                    e_mean = np.mean(e)
                    curr_stat['entropy'] = float(e_mean)

            # Upsample to original size if needed
            if need_resize:
                prob = F.interpolate(prob.unsqueeze(
                    1), shape, mode='bilinear', align_corners=False)[:, 0]

            # Probability mask -> index mask
            out_mask = torch.argmax(prob, dim=0)
            out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)

            if compute_iou:
                # mask is [0, 1]
                # gt   is [0, 255]
                # both -> [False, True]
                if gt is not None:
                    iou = float(compute_tensor_iou(torch.tensor(
                        out_mask).type(torch.bool), gt.type(torch.bool)))
                else:
                    iou = -1
                curr_stat['iou'] = iou

            # Save the mask
            if config['save_masks']:
                original_img = FT.to_pil_image(rgb_raw_tensor)

                out_mask = mapper.remap_index_mask(out_mask)
                out_img = Image.fromarray(out_mask)
                out_img = vid_reader.map_the_colors_back(out_img)
                save_image(out_img, frame, vid_name, general_dir_path=config['masks_out_path'], sub_dir_name='masks', extension='.png')

                if save_overlay:
                    overlaid_img = create_overlay(original_img, out_img, color_if_black_and_white=b_and_w_color)
                    save_image(overlaid_img, frame, vid_name, general_dir_path=config['masks_out_path'], sub_dir_name='overlay', extension='.jpg')

            if False:  # args.save_scores:
                np_path = path.join(args.output, 'Scores', vid_name)
                os.makedirs(np_path, exist_ok=True)
                if ti == len(loader)-1:
                    hkl.dump(mapper.remappings, path.join(
                        np_path, f'backward.hkl'), mode='w')
                if args.save_all or info['save'][0]:
                    hkl.dump(prob, path.join(
                        np_path, f'{frame[:-4]}.hkl'), mode='w', compression='lzf')

            stats.append(curr_stat)

    return pd.DataFrame(stats)


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
        compute_uncertainty=False,
        compute_iou=compute_iou,
        print_progress=print_progress,
        manually_curated_masks=False,
         **kwargs
    )


def predict_annotation_candidates(
    imgs_in_path: Union[str, PathLike],
    approach: str,
    num_candidates: int = 1,
    print_progress=True,
) -> List[int]:
    """
    Args:
    imgs_in_path (Union[str, PathLike]): Path to the directory containing video frames in the following format: `frame_000000.png` .jpg works too.

    if num_candidates == 1:
        return [0]  # First frame is hard-coded to always be used

    #         p_bar.update()

    Returns:
    annotation_candidates (List[int]): A list of frames indices (0-based) chosen as annotation candidates, sorted by importance (most -> least). Always contains [0] - first frame - at index 0.
    """

    candidate_selection_function = KNOWN_ANNOTATION_PREDICTORS[approach]

    assert num_candidates >= 1

    if num_candidates == 1:
        return [0]  # First frame is hard-coded to always be used

    return _inference_on_video(
        imgs_in_path=imgs_in_path,
        masks_in_path=imgs_in_path,  # Ignored
        masks_out_path=None,  # Ignored
        frames_with_masks=[0],  # Ignored
        compute_uncertainty=False,
        compute_iou=False,
        print_progress=print_progress,
        manually_curated_masks=False,
        only_predict_frames_to_annotate_and_quit=num_candidates,
        frame_selector_func=candidate_selection_function
    )
