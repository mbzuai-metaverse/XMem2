import csv
from typing import Iterable, Union, List
from util.tensor_util import compute_tensor_iou
from inference.inference_core import InferenceCore
from model.network import XMem
from inference.data.video_reader import VideoReader
from inference.data.mask_mapper import MaskMapper
from inference.data.test_datasets import LongTestDataset, DAVISTestDataset, YouTubeVOSTestDataset
from inference.active_learning import calculate_proposals_for_annotations_iterative_umap_cosine, calculate_proposals_for_annotations_iterative_pca_cosine, calculate_proposals_for_annotations_iterative_pca_cosine_values, calculate_proposals_for_annotations_pca_hierarchical_clustering, calculate_proposals_for_annotations_umap_hdbscan_clustering, calculate_proposals_for_annotations_uniform_iterative_pca_cosine, calculate_proposals_for_annotations_with_average_distance, calculate_proposals_for_annotations_with_first_distance, calculate_proposals_for_annotations_with_iterative_distance, calculate_proposals_for_annotations_with_iterative_distance_cycle, calculate_proposals_for_annotations_with_iterative_distance_diff, calculate_proposals_for_annotations_with_uniform_iterative_distance_cycle, calculate_proposals_for_annotations_with_uniform_iterative_distance_diff, calculate_proposals_for_annotations_with_uniform_iterative_distance_double_diff, get_determenistic_augmentations, select_most_uncertain_frame, select_n_frame_candidates, compute_disparity as compute_disparity_func, select_n_frame_candidates_no_neighbours_simple
import torchvision.transforms.functional as FT
from baal.active.heuristics import BALD
from scipy.stats import entropy
from tqdm import tqdm
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import pandas as pd
import shutil
from pathlib import Path
from argparse import ArgumentParser
from os import PathLike, path
import math
from collections import defaultdict
import os
from torchvision.transforms import functional as FT


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
                       overwrite_config: dict = None):
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
        'value_dim': 512,
        'masks_out_path': masks_out_path,  # f'../VIDEOS/RESULTS/XMem_memory/thanks_two_face_5_frames/',
        'workspace': None,
        'save_masks': True
    }

    if overwrite_config is not None:
        config.update(overwrite_config)

    if compute_uncertainty:
        assert uncertainty_name is not None
        uncertainty_name = uncertainty_name.lower()
        assert uncertainty_name in {'entropy', 'bald', 'disparity', 'disparity_large'}
        compute_disparity = uncertainty_name.startswith('disparity')
    else:
        compute_disparity = False

    vid_reader = VideoReader(
        "",
        imgs_in_path,  # f'/home/maksym/RESEARCH/VIDEOS/thanks_no_ears_5_annot/JPEGImages',
        masks_in_path,  # f'/home/maksym/RESEARCH/VIDEOS/thanks_no_ears_5_annot/Annotations_binarized_two_face',
        size=config['size'],
        use_all_mask=True
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
        iterative_frames = calculate_proposals_for_annotations_iterative_pca_cosine(loader, processor, print_progress=print_progress, how_many_frames=only_predict_frames_to_annotate_and_quit)
        
        return iterative_frames

    frames_ = []
    masks_ = []
    
    if original_memory_mechanism:
        frames_to_put_in_permanent_memory = [0]  # only the first frame goes into permanent memory originally
        # the rest are going to be processed later
    else:
        frames_to_put_in_permanent_memory = frames_with_masks  # in our modification, all frames with provided masks go into permanent memory
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
        
        frames_.append(rgb)
        masks_.append(msk)

        if augment_images_with_masks:
            augs = get_determenistic_augmentations(rgb.shape, msk, subset='best_all')
            rgb_raw = FT.to_pil_image(rgb_raw_tensor)

            for img_aug, mask_aug in augs:
                # tensor -> PIL.Image -> tensor -> whatever normalization vid_reader applies
                rgb_aug = vid_reader.im_transform(img_aug(rgb_raw)).cuda()

                msk_aug = mask_aug(msk)

                processor.put_to_permanent_memory(rgb_aug, msk_aug)

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

            if not first_mask_loaded:
                if msk is not None:
                    first_mask_loaded = True
                else:
                    # no point to do anything without a mask
                    continue

            # Map possibly non-continuous labels to continuous ones
            if msk is not None:
                # https://github.com/hkchengrex/XMem/issues/21 just make exhaustive = True
                msk, labels = mapper.convert_mask(msk[0].numpy(), exhaustive=True)
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

                    msk = mask_aug(msk)  # does not do anything, since original_only=True augmentations don't alter the mask at all

                    dry_run_prob = processor.step(rgb_aug, msk, labels, end=(ti == vid_length-1),
                                                  manually_curated_masks=manually_curated_masks, disable_memory_updates=True)
                    dry_run_preds.append(dry_run_prob.cpu())

            if original_memory_mechanism:
                do_not_add_mask_to_memory = (ti == 0)  # we only ignore the first mask, since it's already in the permanent memory
            else:
                do_not_add_mask_to_memory = msk is not None  # we ignore all frames with masks, since they are already preloaded in the permanent memory
            # Run the model on this frame
            # 2+ channels, classes+ and background
            prob = processor.step(rgb, msk, labels, end=(ti == vid_length-1),
                                  manually_curated_masks=manually_curated_masks, do_not_add_mask_to_memory=do_not_add_mask_to_memory)
            
            if compute_uncertainty:
                if uncertainty_name == 'bald':
                    # [batch=1, num_classes, ..., num_iterations]
                    all_samples = torch.stack([x.unsqueeze(0) for x in dry_run_preds + [prob.cpu()]], dim=-1).numpy()
                    score = bald.compute_score(all_samples)
                    # TODO: can also return the exact pixels for every frame? As a suggestion on what to label
                    curr_stat['bald'] = float(np.squeeze(score).mean())
                elif compute_disparity:
                    disparity_stats = compute_disparity_func(
                        predictions=[prob] + dry_run_preds, augs=[img_aug for img_aug, _ in augs], images=[rgb_raw] + augged_images, output_save_path=None)
                    curr_stat['disparity'] = float(disparity_stats['avg'])
                    curr_stat['disparity_large'] = float(disparity_stats['large'])
                else:
                    e = entropy(prob.cpu())
                    e_mean = np.mean(e)
                    curr_stat['entropy'] = float(e_mean)

            # Upsample to original size if needed
            if need_resize:
                prob = F.interpolate(prob.unsqueeze(1), shape, mode='bilinear', align_corners=False)[:, 0]

            # Probability mask -> index mask
            out_mask = torch.argmax(prob, dim=0)
            out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)

            if compute_iou:
                # mask is [0, 1]
                # gt   is [0, 255]
                # both -> [False, True]
                if gt is not None:
                    iou = float(compute_tensor_iou(torch.tensor(out_mask).type(torch.bool), gt.type(torch.bool)))
                else:
                    iou = -1
                curr_stat['iou'] = iou

            # Save the mask
            if config['save_masks']:
                this_out_path = path.join(config['masks_out_path'], vid_name)
                os.makedirs(this_out_path, exist_ok=True)
                out_mask = mapper.remap_index_mask(out_mask)
                out_img = Image.fromarray(out_mask)
                if vid_reader.get_palette() is not None:
                    out_img.putpalette(vid_reader.get_palette())
                out_img.save(os.path.join(this_out_path, frame[:-4]+'.png'))

            if False:  # args.save_scores:
                np_path = path.join(args.output, 'Scores', vid_name)
                os.makedirs(np_path, exist_ok=True)
                if ti == len(loader)-1:
                    hkl.dump(mapper.remappings, path.join(np_path, f'backward.hkl'), mode='w')
                if args.save_all or info['save'][0]:
                    hkl.dump(prob, path.join(np_path, f'{frame[:-4]}.hkl'), mode='w', compression='lzf')

            stats.append(curr_stat)

    return pd.DataFrame(stats)


def run_active_learning(imgs_in_path, masks_in_path, masks_out_path, num_extra_frames: int, uncertainty_name: str, csv_out_path: str = None, mode='batched', use_cache=False, **kwargs):
    """
    mode:str
        Possible values:
            'uniform': uniformly distributed indices np.linspace(0, `num_total_frames` - 1, `num_extra_frames` + 1).astype(int)
            'random': pick `num_extra_frames` random frames (cannot include first or last ones)
            'batched': Pick only `num_extra_frames` best frames
            'iterative': Pick only 1 best frame instead of `num_extra_frames`, repeat `num_extra_frames` times
    """
    if mode.startswith('uniform_random'):
        pass
    else:
        assert mode in {'uniform', 'random', 'batched', 'iterative', 'umap_half_cosine', 'umap_hdbscan_clustering', 'pca_max_cosine_values_arpack'}
    assert uncertainty_name in {'entropy', 'bald', 'disparity', 'disparity_large'}

    num_total_frames = len(os.listdir(imgs_in_path))
    if mode == 'uniform':
        # linspace is [a, b] (inclusive)
        frames_with_masks = np.linspace(0, num_total_frames - 1, num_extra_frames + 1).astype(int)
    elif mode == 'random':
        np.random.seed(1)
        extra_frames = np.random.choice(np.arange(1, num_total_frames), size=num_extra_frames, replace=False).tolist()
        frames_with_masks = sorted([0] + extra_frames)
    elif mode.startswith('uniform_random'):
        seed = int(mode.split('_')[-1])
        chosen_frames = []
        space = np.linspace(0, num_total_frames, num_extra_frames + 2, endpoint=True, dtype=int)
        ranges = zip(space, space[1:])
        np.random.seed(seed)
        
        for a, b in ranges:
            if a == 0:
                chosen_frames.append(0)
            else:
                extra_frame = int(np.random.choice(np.arange(a, b), replace=False))
                chosen_frames.append(extra_frame)
        frames_with_masks = chosen_frames
        
    elif mode == 'batched':
        # we save baseline results here, with just 1 annotation
        baseline_out = Path(masks_out_path).parent.parent / 'baseline'
        df = _inference_on_video(
            imgs_in_path=imgs_in_path,
            masks_in_path=masks_in_path,
            masks_out_path=baseline_out / 'masks',
            frames_with_masks=[0],
            compute_uncertainty=True,
            compute_iou=True,
            uncertainty_name=uncertainty_name,
            manually_curated_masks=False,
            print_progress=False,
            overwrite_config={'save_masks': True},
        )

        df.to_csv(baseline_out / 'stats.csv', index=False)
        if uncertainty_name == 'disparity_large':
            candidates = select_n_frame_candidates_no_neighbours_simple(df, n=num_extra_frames, uncertainty_name=uncertainty_name)
        else:
            candidates = select_n_frame_candidates(df, n=num_extra_frames, uncertainty_name=uncertainty_name)

        extra_frames = [int(candidate['index']) for candidate in candidates]

        frames_with_masks = sorted([0] + extra_frames)
    elif mode == 'iterative':
        extra_frames = []
        for i in range(num_extra_frames):
            df = _inference_on_video(
                imgs_in_path=imgs_in_path,
                masks_in_path=masks_in_path,
                masks_out_path=masks_out_path,
                frames_with_masks=[0] + extra_frames,
                compute_uncertainty=True,
                compute_iou=False,
                uncertainty_name=uncertainty_name,
                manually_curated_masks=False,
                print_progress=False,
                overwrite_config={'save_masks': False}, 
            )

            max_frame = select_most_uncertain_frame(df, uncertainty_name=uncertainty_name)
            extra_frames.append(max_frame['index'])

        # keep unsorted to preserve order of the choices
        frames_with_masks = [0] + extra_frames  
    elif mode == 'umap_hdbscan_clustering' or mode == 'umap_half_cosine':
        frames_with_masks = _inference_on_video(
                imgs_in_path=imgs_in_path,
                masks_in_path=masks_in_path,
                masks_out_path=masks_out_path,
                frames_with_masks=[0],
                compute_uncertainty=True,
                compute_iou=False,
                manually_curated_masks=False,
                print_progress=False,
                uncertainty_name=uncertainty_name,
                overwrite_config={'save_masks': False},
                only_predict_frames_to_annotate_and_quit=num_extra_frames,  #  ONLY THIS WILL RUN ANYWAY
        )
    elif mode == 'pca_max_cosine_values_arpack':
        # getting all the values
        _, values = _inference_on_video(
                imgs_in_path=imgs_in_path,
                masks_in_path=masks_in_path,
                masks_out_path=masks_out_path,
                frames_with_masks=[0],
                compute_uncertainty=False,
                compute_iou=False,
                manually_curated_masks=False,
                print_progress=True,
                uncertainty_name=uncertainty_name,
                return_all_values=True, ## The key argument
                overwrite_config={'save_masks': False},
        )
        
        frames_with_masks = calculate_proposals_for_annotations_iterative_pca_cosine_values(values, how_many_frames=num_extra_frames, print_progress=False)
    if use_cache and os.path.exists(csv_out_path):
        final_df = pd.read_csv(csv_out_path)
    else:
        final_df = _inference_on_video(
            imgs_in_path=imgs_in_path,
            masks_in_path=masks_in_path,
            masks_out_path=masks_out_path,
            frames_with_masks=frames_with_masks,
            compute_uncertainty=True,
            compute_iou=True,
            print_progress=False,
            uncertainty_name=uncertainty_name,
            **kwargs
        )

        if csv_out_path is not None:
            p_csv_out = Path(csv_out_path)

            if not p_csv_out.parent.exists():
                p_csv_out.parent.mkdir(parents=True)

            final_df.to_csv(p_csv_out, index=False)

    return final_df, frames_with_masks


def eval_active_learning(dataset_path: str, out_path: str, num_extra_frames: int, uncertainty_name: str, modes: list = None, **kwargs):
    assert uncertainty_name in {'entropy', 'bald', 'disparity', 'disparity_large'}

    if modes is None:
        modes = ['uniform', 'random', 'uniform_random', 'batched', 'iterative', 'umap_half_cosine', 'umap_hdbscan_clustering', 'pca_max_cosine_values_arpack']

    p_in_ds = Path(dataset_path)
    p_out = Path(out_path)

    big_stats = defaultdict(list)
    for i, p_video_imgs_in in enumerate(tqdm(sorted((p_in_ds / 'JPEGImages').iterdir()))):
        video_name = p_video_imgs_in.stem
        p_video_masks_in = p_in_ds / 'Annotations_binarized' / video_name

        p_video_out_general = p_out / f'Active_learning_{uncertainty_name}' / video_name / f'{num_extra_frames}_extra_frames'

        for mode in modes:
            curr_video_stat = {'video': video_name}
            p_out_masks = p_video_out_general / mode / 'masks'
            p_out_stats = p_video_out_general / mode / 'stats.csv'

            stats, frames_with_masks = run_active_learning(p_video_imgs_in, p_video_masks_in, p_out_masks,
                                                           num_extra_frames=num_extra_frames, csv_out_path=p_out_stats, mode=mode, uncertainty_name=uncertainty_name, use_cache=False, **kwargs)

            stats = stats[stats['mask_provided'] == False]  # remove stats for frames with given masks
            for i in range(1, len(frames_with_masks) + 1):
                curr_video_stat[f'extra_frame_{i}'] = frames_with_masks[i - 1]

            curr_video_stat[f'mean_iou'] = stats['iou'].mean()
            curr_video_stat[f'mean_{uncertainty_name}'] = stats[uncertainty_name].mean()

            big_stats[mode].append(curr_video_stat)

    for mode, mode_stats in big_stats.items():
        df_mode_stats = pd.DataFrame(mode_stats)
        df_mode_stats.to_csv(p_out / f'Active_learning_{uncertainty_name}' / f'stats_{mode}_all_videos.csv', index=False)


def run_on_video(
        imgs_in_path: Union[str, PathLike],
        masks_in_path: Union[str, PathLike],
        masks_out_path: Union[str, PathLike],
        frames_with_masks: Iterable[int] = (0, ),
        compute_iou=False,
        print_progress=True,
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
        manually_curated_masks=False
    )


def predict_annotation_candidates(
        imgs_in_path: Union[str, PathLike],
        num_candidates: int = 1,
        print_progress=True,
    ) -> List[int]:

    """
    Args:
    imgs_in_path (Union[str, PathLike]): Path to the directory containing video frames in the following format: `frame_000000.png` .jpg works too.

    num_candidates (int, default: 1): How many annotations candidates to predict.

    print_progress (bool): A flag to indicate whether to print a progress bar (default: True).

    Returns:
    annotation_candidates (List[int]): A list of frames indices (0-based) chosen as annotation candidates, sorted by importance (most -> least). Always contains [0] - first frame - at index 0.
    """

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
    )

