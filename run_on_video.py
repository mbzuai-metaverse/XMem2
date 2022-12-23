from collections import defaultdict
import math
import os
from os import path
from argparse import ArgumentParser
from pathlib import Path
import shutil

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.stats import entropy
from baal.active.heuristics import BALD
import torchvision.transforms.functional as FT

from inference.active_learning import get_determenistic_augmentations, select_most_uncertain_frame, select_n_frame_candidates
from inference.data.test_datasets import LongTestDataset, DAVISTestDataset, YouTubeVOSTestDataset
from inference.data.mask_mapper import MaskMapper
from inference.data.video_reader import VideoReader
from model.network import XMem
from inference.inference_core import InferenceCore
from util.tensor_util import compute_tensor_iou


def inference_on_video(frames_with_masks, imgs_in_path, masks_in_path, masks_out_path, 
                        original_memory_mechanism=False,
                        compute_iou = False, compute_uncertainty = False, manually_curated_masks=False, print_progress=True,
                        uncertainty_name: str = None,
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
        # 'video': '../VIDEOS/Maksym_frontal_simple.mp4',
        'masks_out_path': masks_out_path,#f'../VIDEOS/RESULTS/XMem_feedback/thanks_two_face_5_frames/',
        # 'masks_out_path': f'../VIDEOS/RESULTS/XMem/WhichFramesWithPreds/1/{dir_name}/{"_".join(map(str, frames_with_masks))}_frames_provided',
        # 'masks_out_path': f'../VIDEOS/RESULTS/XMem/DAVIS_2017/WhichFrames/1/{dir_name}/{len(frames_with_masks) - 1}_extra_frames',
        'workspace': None,
        'save_masks': True
    }

    if overwrite_config is not None:
        config.update(overwrite_config)

    vid_reader = VideoReader(
        "", 
        imgs_in_path, #f'/home/maksym/RESEARCH/VIDEOS/thanks_no_ears_5_annot/JPEGImages',
        masks_in_path, #f'/home/maksym/RESEARCH/VIDEOS/thanks_no_ears_5_annot/Annotations_binarized_two_face',
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

    if original_memory_mechanism:
        frames_to_put_in_permanent_memory = [0] # only the first frame goes into permanent memory originally
        # the rest are going to be processed later
    else:
        frames_to_put_in_permanent_memory = frames_with_masks  # in our modification, all frames with provided masks go into permanent memory
    for j in frames_to_put_in_permanent_memory:
        sample = vid_reader[j]
        rgb = sample['rgb'].cuda()
        msk = sample['mask']
        info = sample['info']
        need_resize = info['need_resize']

        # https://github.com/hkchengrex/XMem/issues/21 just make exhaustive = True
        msk, labels = mapper.convert_mask(msk, exhaustive=True)
        msk = torch.Tensor(msk).cuda()
        if need_resize:
            msk = vid_reader.resize_mask(msk.unsqueeze(0))[0]

        processor.set_all_labels(list(mapper.remappings.values()))
        processor.put_to_permanent_memory(rgb, msk)
        
    stats = []

    if compute_uncertainty:
        assert uncertainty_name is not None
        uncertainty_name = uncertainty_name.lower()
        assert uncertainty_name in {'entropy', 'bald'}
    
    if uncertainty_name == 'bald':
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

            if False:
                rgb = torch.flip(rgb, dims=[-1])
                msk = torch.flip(msk, dims=[-1]) if msk is not None else None

            # Map possibly non-continuous labels to continuous ones
            # TODO: What are labels? Debug
            if msk is not None:
                # https://github.com/hkchengrex/XMem/issues/21 just make exhaustive = True
                msk, labels = mapper.convert_mask(msk[0].numpy(), exhaustive=True)
                msk = torch.Tensor(msk).cuda()
                if need_resize:
                    msk = vid_reader.resize_mask(msk.unsqueeze(0))[0]
                processor.set_all_labels(list(mapper.remappings.values()))
    
            else:
                labels = None
            
            if compute_uncertainty and uncertainty_name == 'bald':
                dry_run_preds = []
                augs = get_determenistic_augmentations()
                for aug in augs:
                    # tensor -> PIL.Image -> tensor -> whatever normalization vid_reader applies
                    rgb_raw = FT.to_pil_image(rgb_raw_tensor)
                    rgb_aug = vid_reader.im_transform(aug(rgb_raw)).cuda()
                                    
                    dry_run_prob = processor.step(rgb_aug, msk, labels, end=(ti==vid_length-1), manually_curated_masks=manually_curated_masks, disable_memory_updates=True)
                    dry_run_preds.append(dry_run_prob.cpu())
                
            if original_memory_mechanism:
                do_not_add_mask_to_memory = (ti == 0)  # we only ignore the first mask, since it's already in the permanent memory
            else:
                do_not_add_mask_to_memory = msk is not None  # we ignore all frames with masks, since they are already preloaded in the permanent memory
            # Run the model on this frame
            # TODO: still running inference even on frames with masks?
            # 2+ channels, classes+ and background
            prob = processor.step(rgb, msk, labels, end=(ti==vid_length-1), manually_curated_masks=manually_curated_masks, do_not_add_mask_to_memory=do_not_add_mask_to_memory)
            
            if compute_uncertainty:
                if uncertainty_name == 'bald':
                    # [batch=1, num_classes, ..., num_iterations]
                    all_samples = torch.stack([x.unsqueeze(0) for x in dry_run_preds + [prob.cpu()]], dim=-1).numpy()
                    score = bald.compute_score(all_samples)
                    # TODO: can also return the exact pixels for every frame? As a suggestion on what to label
                    curr_stat['bald'] = float(np.squeeze(score).mean())
                else:
                    e = entropy(prob.cpu())
                    e_mean = np.mean(e)
                    curr_stat['entropy'] = float(e_mean)

            # Upsample to original size if needed
            if need_resize:
                prob = F.interpolate(prob.unsqueeze(1), shape, mode='bilinear', align_corners=False)[:,0]

            if False:
                prob = torch.flip(prob, dims=[-1])

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
            if False:
                prob = (prob.detach().cpu().numpy()*255).astype(np.uint8)

            # Save the mask
            if config['save_masks']:
                this_out_path = path.join(config['masks_out_path'], vid_name)
                os.makedirs(this_out_path, exist_ok=True)
                out_mask = mapper.remap_index_mask(out_mask)
                out_img = Image.fromarray(out_mask)
                if vid_reader.get_palette() is not None:
                    out_img.putpalette(vid_reader.get_palette())
                out_img.save(os.path.join(this_out_path, frame[:-4]+'.png'))

            if False: #args.save_scores:
                np_path = path.join(args.output, 'Scores', vid_name)
                os.makedirs(np_path, exist_ok=True)
                if ti==len(loader)-1:
                    hkl.dump(mapper.remappings, path.join(np_path, f'backward.hkl'), mode='w')
                if args.save_all or info['save'][0]:
                    hkl.dump(prob, path.join(np_path, f'{frame[:-4]}.hkl'), mode='w', compression='lzf')

            stats.append(curr_stat)

    return pd.DataFrame(stats)


def run_active_learning(imgs_in_path, masks_in_path, masks_out_path, num_extra_frames: int, uncertainty_name: str, csv_out_path: str = None, mode='batched', use_cache=False):
    """
    mode:str
        Possible values:
            'uniform': uniformly distributed indices (np.linspace(0, num_total_frames, `num_extra_frames`).astype(int))
            'random': pick `num_extra_frames` random frames (cannot include first or last ones)
            'batched': Pick only `num_extra_frames` best frames
            'iterative': Pick only 1 best frame instead of `num_extra_frames`, repeat `num_extra_frames` times
    """

    assert mode in {'uniform', 'random', 'batched', 'iterative'}
    assert uncertainty_name in {'entropy', 'bald'}
    
    if mode == 'uniform':
        num_total_frames = len(os.listdir(imgs_in_path))
        # linspace is [a, b] (inclusive)
        frames_with_masks = np.linspace(0, num_total_frames - 1, num_extra_frames).astype(int)
    elif mode == 'random':
        num_total_frames = len(os.listdir(imgs_in_path))
        extra_frames = np.random.choice(np.arange(1, num_total_frames), size=num_extra_frames, replace=False).tolist()
        frames_with_masks = sorted([0] + extra_frames)
    elif mode == 'batched':
        # we save baseline results here, with just 1 annotation
        baseline_out= Path(masks_out_path).parent.parent / 'baseline'
        df = inference_on_video(
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

        candidates = select_n_frame_candidates(df, n=num_extra_frames, uncertainty_name=uncertainty_name)

        extra_frames = [int(candidate['index']) for candidate in candidates]

        frames_with_masks = sorted([0] + extra_frames)
    elif mode == 'iterative':
        extra_frames = []
        for i in range(num_extra_frames):
            df = inference_on_video(
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

            max_frame = select_most_uncertain_frame(df)
            extra_frames.append(max_frame['index'])

        # keep unsorted to preserve order of the choices
        frames_with_masks = [0] + extra_frames
    if use_cache and os.path.exists(csv_out_path):
        final_df = pd.read_csv(csv_out_path)
    else:    
        final_df = inference_on_video(
                imgs_in_path=imgs_in_path,
                masks_in_path=masks_in_path,
                masks_out_path=masks_out_path,
                frames_with_masks=frames_with_masks,
                compute_uncertainty=True,
                compute_iou=True,
                print_progress=False,
                uncertainty_name=uncertainty_name,
                manually_curated_masks=False,                
        )

        if csv_out_path is not None:
            p_csv_out = Path(csv_out_path)

            if not p_csv_out.parent.exists():
                p_csv_out.parent.mkdir(parents=True)

            final_df.to_csv(p_csv_out, index=False)

    return final_df, frames_with_masks


def eval_active_learning(dataset_path: str, out_path: str, num_extra_frames: int, uncertainty_name: str):
    assert uncertainty_name in {'entropy', 'bald'}
    
    p_in_ds = Path(dataset_path)
    p_out = Path(out_path)

    big_stats = defaultdict(list)
    for p_video_imgs_in in tqdm(sorted((p_in_ds / 'JPEGImages').iterdir())):
        video_name = p_video_imgs_in.stem
        p_video_masks_in = p_in_ds / 'Annotations_binarized' / video_name

        p_video_out_general = p_out / f'Active_learning_{uncertainty_name}' / video_name / f'{num_extra_frames}_extra_frames'

        for mode in ['uniform', 'random', 'batched', 'iterative']:
            curr_video_stat = {'video': video_name}
            p_out_masks = p_video_out_general / mode / 'masks'
            p_out_stats = p_video_out_general / mode / 'stats.csv'

            stats, frames_with_masks = run_active_learning(p_video_imgs_in, p_video_masks_in, p_out_masks, 
                    num_extra_frames=num_extra_frames, csv_out_path=p_out_stats, mode=mode, use_cache=True)

            stats = stats[stats['mask_provided'] == False]  # remove stats for frames with given masks
            for i in range(1, len(frames_with_masks) + 1):
                curr_video_stat[f'extra_frame_{i}'] = frames_with_masks[i - 1]

            curr_video_stat[f'mean_iou'] = stats['iou'].mean()
            curr_video_stat[f'mean_{uncertainty_name}'] = stats[uncertainty_name].mean()

            big_stats[mode].append(curr_video_stat)

    for mode, mode_stats in big_stats.items():
        df_mode_stats = pd.DataFrame(mode_stats)
        df_mode_stats.to_csv(p_out / f'Active_learning_{uncertainty_name}' / f'stats_{mode}_all_videos.csv', index=False)


if __name__ == '__main__':
    pass
    # eval_active_learning('/home/maksym/RESEARCH/VIDEOS/LVOS_dataset/valid', 
    #                      '/home/maksym/RESEARCH/VIDEOS/RESULTS/XMem_feedback/permanent_work_memory/LVOS',
                        #  5)


    # res, frames_with_masks = run_active_learning('/home/maksym/RESEARCH/VIDEOS/LVOS_dataset/valid/JPEGImages/0tCWPOrc',
    #  '/home/maksym/RESEARCH/VIDEOS/LVOS_dataset/valid/Annotations_binarized/0tCWPOrc', 
    #  '/home/maksym/RESEARCH/VIDEOS/RESULTS/XMem_feedback/permanent_work_memory/LVOS/JUNK/masks',
    #  num_extra_frames=5, 
    #  csv_out_path='/home/maksym/RESEARCH/VIDEOS/RESULTS/XMem_feedback/permanent_work_memory/LVOS/JUNK/stats.csv', mode='iterative')

    # print(frames_with_masks)
    # pass
    # bald_df = inference_on_video([0],
    #                    '/home/maksym/RESEARCH/VIDEOS/thanks_no_ears_5_annot/JPEGImages', 
    #                    '/home/maksym/RESEARCH/VIDEOS/thanks_no_ears_5_annot/Annotations_binarized',
    #                    'JUNK',
    #                    compute_iou=False,
    #                    compute_uncertainty=True,
    #                    use_bald=True) # for t.hanks style video
    
    # bald_df.to_csv('output/bald_thanks_0_frame.csv', index=False)
    
    # df = inference_on_video(
    #             imgs_in_path='/home/maksym/RESEARCH/VIDEOS/thanks_no_ears_5_annot/JPEGImages',
    #             masks_in_path='/home/maksym/RESEARCH/VIDEOS/thanks_no_ears_5_annot/Annotations_binarized',
    #             frames_with_masks=[0, 259, 621, 785, 1401],
    #             masks_out_path='../VIDEOS/RESULTS/XMem_feedback/BASELINE_REIMPLEMENTED/5_annotated_frames_new_mem',
    #             compute_uncertainty=False,
    #             compute_iou=False,
    #             manually_curated_masks=False,
    #             original_memory_mechanism=False,
    #             overwrite_config={'save_masks': True})

    # df = inference_on_video(
    #         imgs_in_path='/home/maksym/RESEARCH/VIDEOS/LVOS_dataset/valid/JPEGImages/vjG0jbkQ',
    #         masks_in_path='/home/maksym/RESEARCH/VIDEOS/LVOS_dataset/valid/Annotations/vjG0jbkQ',
    #         masks_out_path='JUNK',
    #         frames_with_masks=[0],
    #         compute_entropy=True,
    #         compute_iou=True,
    #         manually_curated_masks=False,
    #         overwrite_config={'save_masks': False})
    
    # print(df.shape)
    # df.to_csv('junk.csv', index=False)
    # p_in = Path('/home/maksym/RESEARCH/VIDEOS/DAVIS-2017-trainval-480p/DAVIS/2017_train_val_split/val/JPEGImages_chosen')
    # p_in = Path('/home/maksym/RESEARCH/VIDEOS/DAVIS-2017-trainval-480p/DAVIS/2017_train_val_split/val/JPEGImages_chosen')
    # num_frames_mapping = {}

    # for p_dir in sorted(p for p in p_in.iterdir() if p.is_dir()):
    #     dir_name = p_dir.name
    #     num_frames = sum(1 for _ in p_dir.iterdir())
    #     num_frames_mapping[dir_name] = num_frames # math.ceil(num_frames/2)
    # # print(extra_frames_ranges)
    # # exit(0)
    # p_bar = tqdm(total=sum(num_frames_mapping.values()), desc='% extra frames DAVIS 2017 val')
    # for dir_name, total_frames in num_frames_mapping.items():
    #     for how_many_extra_frames in range(0, math.ceil(total_frames // 2)):
    #         # frames_with_masks = set([0, frame_with_mask])
    #         frames_with_masks = set(np.linspace(0, num_frames, how_many_extra_frames+2)[0:-1].astype(int))
    #         inference_on_video(frames_with_masks, dir_name)

    #         p_bar.update()

    # num_runs = 90
    # p_bar = tqdm(total=num_runs)
    # for how_many_extra_frames in range(0, 90):
    #     # for j in range(0, 181 - 1):
    #         # e.g. [0, 10, 20, ..., 180] without 180
    #         frames_with_masks = set(np.linspace(0, 180, how_many_extra_frames+2)[0:-1].astype(int))
    #         # frames_with_masks = set([0, i, j])
    #         inference_on_video(frames_with_masks)

    #         p_bar.update()
